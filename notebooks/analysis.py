"""Load a run's chains and derive the windowed mass and spin quantities the notebooks plot.

    import analysis as A
    r = A.load_run('ppc_gwtc5_k5_short')
    r.p_mass, r.P_win, r.MOM, r.summary ...

Import after the thread / platform environment is set (the notebooks set JAX_PLATFORMS=cpu and the single-thread XLA
flags before importing this), since this imports jax through pixelpop.

* Mass: the saved ``log_p_mass`` is unwindowed; ``p_mass`` adds the m2 > 3 Msun window on bin centres and renormalizes.
* Spins: rebuilt from the saved spin latents of any spin model (field, split, gauss; exactly as model.py builds them) on
  every draw, and cached per chain in results/post_processing/for_plotting/<run>/chain_<c>.h5 (built on first use,
  or ahead of time with ``python notebooks/analysis.py <run> [<chain> ...]``): the windowed population marginal, the conditional's per-(m1, q) mean and
  sd of chi_eff and chi_p, p(chi_eff | m1) and p(chi_p | m1) averaged over the windowed p(q | m1), and the per-cell
  probability of each chi_eff slice (CHI_EFF_SLICES), from which p(m1, q | chi_eff in S) follows.
"""
import glob
import os
import pickle
import re
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_M, MAX_M = 3., 200.
B = 100
lnm_edges = np.linspace(np.log(MIN_M), np.log(MAX_M), B + 1)
q_edges = np.linspace(0., 1., B + 1)
e_edges = np.linspace(-1., 1., B + 1)
p_edges = np.linspace(0., 1., B + 1)
centres = lambda e: 0.5 * (e[1:] + e[:-1])
lnm_c, q_c, e_c, p_c = map(centres, (lnm_edges, q_edges, e_edges, p_edges))
dlnm, dq, de, dp = (np.diff(e)[0] for e in (lnm_edges, q_edges, e_edges, p_edges))
m_c = np.exp(lnm_c)
SKIP = ('_eigenbasis_sites', '_spin_sites', '_spin0_sites', '_spin1_sites', '_gauss_sites')     # white noise
GP = ('mu_eff', 'mu_p', 'ln_sd_eff', 'ln_sd_p', 'atanh_rho')
LABELS = {'log_ranges': ('ln m1', 'q'), 'log_ranges_spin': ('ln m1 mode', 'q mode', 'chi_eff', 'chi_p'),
          'log_ranges_spin0': ('chi_eff', 'chi_p'), 'log_ranges_spin1': ('ln m1 mode', 'q mode', 'chi_eff', 'chi_p'),
          'gauss_offsets': GP, 'lnsigma_gauss': GP,
          'log_ranges_gauss': tuple(f'{g}, {a}' for g in GP for a in ('ln m1', 'q'))}
HYPER = ('lnsigma', 'log_nu_spde', 'log_ranges', 'lnsigma_spin', 'log_nu_spin', 'log_ranges_spin',
         'lnsigma_spin0', 'log_nu_spin0', 'log_ranges_spin0', 'lnsigma_spin1', 'log_nu_spin1', 'log_ranges_spin1',
         'gauss_offsets', 'lnsigma_gauss', 'log_nu_gauss', 'log_ranges_gauss', 'lamb',
         'log_likelihood', 'log_likelihood_variance', 'log_likelihood_correction', 'selection_neff')
MOM_NAMES = ('mean chi_eff', 'sd chi_eff', 'mean chi_p', 'sd chi_p')
EVENT_RE = re.compile(r'GW\d{6}(?:_\d{6})?')
DATA_DIRS = {'gwtc5': os.path.join(REPO, 'data', 'gwtc5_ifar1_chieffchip', 'data')}


def band(x, axis=0):
    return np.percentile(x, [5, 50, 95], axis=axis)


def mass_window():
    """log of the m2 > 3 Msun window on the (ln m1, q) bin centres (0 inside, about -1000 outside)."""
    from pixelpop.models.gwpop_models import PlanckWindow_SecondaryMass
    return np.asarray(PlanckWindow_SecondaryMass({'log_mass_1': lnm_c[:, None], 'mass_ratio': q_c[None]}, MIN_M, 0.))


def event_names(catalog):
    directory = DATA_DIRS.get(catalog, f'/home/jack.heinzel/public_html/mass-gap-investigation/data/'
                                       f'{catalog}_o1o4b_chieffchip/data')
    with open(os.path.join(directory, 'posteriors.pkl'), 'rb') as f:
        return [EVENT_RE.findall(os.path.basename(str(n)))[-1] for n in pickle.load(f)['event_names']]


def summarize(S):
    """Median, 90% interval, ESS and split R-hat of the hyperparameters and diagnostics, over all chains."""
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
    rows = []
    for k in [k for k in HYPER if k in S] + ['min event Neff']:
        x = S['event_neff'].min(-1) if k == 'min event Neff' else S[k]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        for j in range(x.shape[2]):
            v = x[:, :, j]
            rows.append(dict(parameter=k if x.shape[2] == 1 else f'{k}[{LABELS.get(k, range(x.shape[2]))[j]}]',
                             median=np.median(v), lo5=np.percentile(v, 5), hi95=np.percentile(v, 95),
                             ess=float(effective_sample_size(v)),
                             rhat=float(split_gelman_rubin(v)) if v.shape[1] >= 4 else np.nan))
    return pd.DataFrame(rows).set_index('parameter')


# chi_eff slices for p(m1, q | chi_eff in S); their edges fall on spin-bin edges (width 0.02)
# |chi_eff| below this is the 'core': MOM_CORE holds the chi_eff moments restricted to it (the flexible field puts
# prior-like spikes at chi_eff -> -1 in some draws)
CHI_CORE = 0.8

CHI_EFF_SLICES = {r'0 < \chi_\mathrm{eff} < 0.1': lambda e: (e > 0) & (e < 0.1),
                  r'|\chi_\mathrm{eff}| < 0.2': lambda e: np.abs(e) < 0.2,
                  r'|\chi_\mathrm{eff}| > 0.2': lambda e: np.abs(e) > 0.2}


def _spin_fns(km, logW, spin_model):
    """Jitted spin rebuild for one draw of the given spin model; returns P_raw, P_win (population marginal,
    unwindowed / windowed mass weights, unnormalized), MOM (per-cell moments), PE_M, PP_M (p(chi_eff | m1),
    p(chi_p | m1): per-bin probabilities, averaged over the windowed p(q | m1)) and SLICE (per-cell
    P(chi_eff in S | m1, q) for CHI_EFF_SLICES)."""
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp as LSE
    import pixelpop
    from model import (NEG, axis_eigs, gauss_log_kernel, matern_spectrum, spin_window_fraction, split_spin_field,
                       to_grid)

    (w_ax, v_ax), = axis_eigs([pixelpop.utils.create_CAR_coupling_matrix(B, 1, isVisible=False)])
    window = spin_window_fraction(q_edges, e_edges, p_edges)
    log_window = jnp.asarray(np.where(window > 0, np.log(np.maximum(window, 1e-300)), NEG))
    log_support = jnp.where(log_window > NEG / 2, 0., NEG)
    pin = np.zeros((km, km, B, B), bool); pin[:, :, 0, 0] = True; pin = jnp.asarray(pin)
    Vm = Vq = v_ax[:, :km]
    ec, pc, lW, logdV = jnp.asarray(e_c), jnp.asarray(p_c), jnp.asarray(logW), np.log(dlnm * dq)
    masks = jnp.asarray(np.stack([f(e_c) for f in CHI_EFF_SLICES.values()], -1).astype(float))    # (Be, n_slices)
    core = jnp.asarray((np.abs(e_c) < CHI_CORE).astype(float))                                    # (Be,)

    def stats(slab_T, log_p_mass):
        lw = log_p_mass + logdV
        w_raw = jnp.exp(lw); lw_win = lw + lW; w_win = jnp.exp(lw_win - LSE(lw_win))

        def slab(m):
            T = slab_T(m) + log_support
            P = jnp.exp(T - LSE(T + log_window, axis=(1, 2))[:, None, None] + log_window)       # (Bq, Be, Bp)
            pe, pq = P.sum(2), P.sum(1)
            me, mp = pe @ ec, pq @ pc
            mom = jnp.stack([me, jnp.sqrt(jnp.maximum(pe @ ec ** 2 - me ** 2, 0.)),
                             mp, jnp.sqrt(jnp.maximum(pq @ pc ** 2 - mp ** 2, 0.))], -1)
            wq = w_win[m] / jnp.maximum(w_win[m].sum(), 1e-300)                                # p(q | m1), windowed
            # chi_eff restricted to |chi_eff| < CHI_CORE: mean, sd, and the probability outside it
            pe_in = pe * core; inside = jnp.maximum(pe_in.sum(1), 1e-300)
            mu_in = (pe_in @ ec) / inside
            mom_core = jnp.stack([mu_in, jnp.sqrt(jnp.maximum((pe_in @ ec ** 2) / inside - mu_in ** 2, 0.)),
                                  1. - pe_in.sum(1)], -1)
            return (jnp.einsum('q,qep->ep', w_raw[m], P), jnp.einsum('q,qep->ep', w_win[m], P), mom,
                    wq @ pe, wq @ pq, pe @ masks, mom_core)
        raw, win, mom, pe_m, pp_m, sl, mom_core = jax.lax.map(slab, jnp.arange(B), batch_size=10)
        return raw.sum(0), win.sum(0), mom, pe_m, pp_m, sl, mom_core

    @jax.jit
    def field(white, lnsig, lognu, logr, log_p_mass):
        G = to_grid(matern_spectrum([w_ax[:km], w_ax[:km], w_ax, w_ax], lnsig, logr, jnp.exp(lognu), B ** 4, pin=pin),
                    white, [jnp.eye(km), jnp.eye(km), v_ax, v_ax])
        return stats(lambda m: jnp.einsum('k,ql,klep->qep', Vm[m], Vq, G), log_p_mass)

    @jax.jit
    def split(w0, l0, n0, r0, w1, l1, n1, r1, log_p_mass):
        G = split_spin_field(w0, l0, jnp.exp(n0), r0, w1, l1, jnp.exp(n1), r1, ((w_ax[:km], Vm), (w_ax[:km], Vq)),
                             ((w_ax, v_ax), (w_ax, v_ax)), B ** 4)
        return stats(lambda m: jnp.einsum('k,ql,klep->qep', Vm[m], Vq, G), log_p_mass)

    @jax.jit
    def gauss(theta, log_p_mass):
        return stats(lambda m: gauss_log_kernel(theta[:, m][:, :, None, None], ec[None, :, None], pc[None, None, :]),
                     log_p_mass)

    keys = {'field': ('_spin_sites', 'lnsigma_spin', 'log_nu_spin', 'log_ranges_spin', 'log_p_mass'),
            'split': ('_spin0_sites', 'lnsigma_spin0', 'log_nu_spin0', 'log_ranges_spin0',
                      '_spin1_sites', 'lnsigma_spin1', 'log_nu_spin1', 'log_ranges_spin1', 'log_p_mass'),
            'gauss': ('gauss_params', 'log_p_mass')}[spin_model]
    return {'field': field, 'split': split, 'gauss': gauss}[spin_model], keys


REBUILT = ('P_raw', 'P_win', 'MOM', 'PE_M', 'PP_M', 'SLICE', 'MOM_CORE')   # MOM_CORE: (mean, sd, P outside) at |chi_eff| < CHI_CORE
POST_DIR = os.path.join(REPO, 'results', 'post_processing', 'for_plotting')


def post_path(run, chain_file):
    """results/post_processing/for_plotting/<run>/chain_<c>.h5 for results/<run>/chain_<c>_samples.h5."""
    return os.path.join(POST_DIR, run, os.path.basename(chain_file).replace('_samples.h5', '.h5'))


def spin_model_of(keys):
    return 'split' if 'lnsigma_spin1' in keys else 'gauss' if 'gauss_offsets' in keys else 'field'


def rebuild_chain(run, chain_file, n_draws=None, log_every=100, block=25):
    """Rebuild the spin quantities (REBUILT) on every draw of one chain from its saved spin latents, exactly as
    model.py builds them, and write them (float32, normalized per draw) with the per-draw total variation between the
    rebuilt and the saved unwindowed marginal (``tv``, a check of the rebuild) to post_path(run, chain_file).
    Reads the chain ``block`` draws at a time (one read per dataset and block: draw-by-draw reads of the multi-GB chain
    files crawl on ceph, and a whole k = 10 chain, 4 GB, does not fit the login nodes' 8 GB per user) and writes as it
    goes, so memory stays near 1 GB."""
    import time
    import jax.numpy as jnp
    logW = mass_window()
    path = post_path(run, chain_file)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.partial'
    t0 = time.time()
    with h5py.File(chain_file, 'r') as h, h5py.File(tmp, 'w') as g:
        spin_model = spin_model_of(h)
        km = (h['_spin_sites'].shape[1] if spin_model == 'field' else h['_spin1_sites'].shape[1] if spin_model == 'split'
              else 1)
        n = len(h['log_likelihood']) if n_draws is None else n_draws
        fn, keys = _spin_fns(km, logW, spin_model)
        g.attrs.update(n_draws=n, spin_model=spin_model, k=km, source=chain_file)
        out = None
        tv = g.create_dataset('tv', (n,))
        for b in range(0, n, block):
            data = {k: h[k][b:min(b + block, n)] for k in keys + ('log_p_spin_marginal',)}
            for j in range(len(data['log_p_mass'])):
                i = b + j
                res = [np.array(x) for x in fn(*(jnp.asarray(data[k][j], dtype=float) for k in keys))]
                res[0] /= res[0].sum(); res[1] /= res[1].sum()
                if out is None:
                    out = {k: g.create_dataset(k, (n,) + v.shape, 'f4', chunks=(1,) + v.shape, compression='gzip',
                                               compression_opts=1) for k, v in zip(REBUILT, res)}
                saved = np.exp(data['log_p_spin_marginal'][j].astype(float))
                tv[i] = 0.5 * np.abs(res[0] - saved / saved.sum()).sum()
                for k, v in zip(REBUILT, res):
                    out[k][i] = v
                if log_every and (i + 1) % log_every == 0:
                    print(f'{run} {os.path.basename(chain_file)}: {i + 1}/{n} draws, '
                          f'{(time.time() - t0) / (i + 1):.2f} s each', flush=True)
    os.replace(tmp, path)
    return path


def load_rebuilt(run, files, n_draws):
    """The rebuilt spin quantities of every chain, first n_draws draws each, rebuilding any chain whose file is
    missing or shorter than n_draws (slow on CPU: about 1 s per draw; ``python notebooks/analysis.py <run>`` does it
    ahead of time)."""
    out = {k: [] for k in REBUILT + ('tv',)}
    for f in files:
        path = post_path(run, f)
        have = 0
        if os.path.exists(path):
            with h5py.File(path, 'r') as g:
                have = g.attrs['n_draws'] if all(k in g for k in REBUILT) else 0     # older caches lack keys
        if have < n_draws:
            print(f'rebuilding spins of {run} {os.path.basename(f)} -> {path}', flush=True)
            rebuild_chain(run, f)
        with h5py.File(path, 'r') as g:
            for k in out:
                out[k].append(g[k][:n_draws])
    return {k: np.concatenate(v) for k, v in out.items()}


def load_run(run):
    """Everything the notebooks need from results/<run>/, as a namespace."""
    files = sorted(glob.glob(os.path.join(REPO, 'results', run, 'chain_*_samples.h5')))
    if not files:
        raise FileNotFoundError(f'no chains in results/{run}')
    chains = []
    for f in files:
        with h5py.File(f, 'r') as h:
            chains.append({k: h[k][()] for k in h if k not in SKIP})
    n_draws = min(len(c['log_likelihood']) for c in chains)
    S = {k: np.stack([c[k][:n_draws] for c in chains]) for k in chains[0]}
    flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in S.items()}
    r = SimpleNamespace(run=run, catalog=run.split('_')[1], files=files, n_chains=len(files), n_draws=n_draws, S=S,
                        flat=flat)
    r.spin_model = spin_model_of(S)
    r.k = int(re.search(r'_k(\d+)', run).group(1))
    r.summary = summarize(S)

    # mass, windowed
    r.logW = mass_window()
    lp = flat['log_p_mass'].astype(float) + r.logW[None]
    mx = lp.max((1, 2), keepdims=True)
    lp -= np.log(np.exp(lp - mx).sum((1, 2), keepdims=True) * dlnm * dq) + mx
    r.p_mass = np.exp(lp)                                          # (draws, ln m1, q), density per d ln m1 dq
    r.p_lnm = r.p_mass.sum(2) * dq
    r.p_q = r.p_mass.sum(1) * dlnm
    wm = np.exp(flat['log_p_mass'].astype(float)) * dlnm * dq
    r.window_removed = (wm * (r.logW < -1)[None]).sum((1, 2))

    # spins, windowed: rebuilt from the spin latents on every draw (the per-mass and chi_eff-slice quantities are not
    # saved by any model), cached in results/post_processing/for_plotting/<run>/
    reb = load_rebuilt(run, files, n_draws)
    r.P_raw, r.P_win, r.MOM, r.PE_M, r.PP_M, r.SLICE, r.MOM_CORE = (reb[k] for k in REBUILT)
    r.rebuild_tv = float(reb['tv'].max())
    r.spin_source = (f'rebuilt from the saved spin latents on all {n_draws} draws per chain '
                     f'(results/post_processing/for_plotting/{run}/)')
    # mass in chi_eff slices: p(ln m1, q | chi_eff in S) ~ p_mass(ln m1, q) P(S | m1, q), on the rebuilt draws
    # float32 from here on: these are only plotted as bands, and two long runs in float64 do not fit the login nodes' 8 GB
    joint = (r.p_mass * (dlnm * dq)).astype(np.float32)[..., None] * r.SLICE           # (draws, Bm, Bq, n_slices)
    r.slice_share = joint.sum((1, 2), dtype=float)                                  # P(chi_eff in S)
    joint /= np.maximum(r.slice_share[:, None, None], 1e-300).astype(np.float32) * np.float32(dlnm * dq)
    r.p_mass_slice = joint
    lq = flat['log_p_spin_q1_m']
    r.P_q1 = np.exp(lq - lq.max((2, 3), keepdims=True))
    r.P_q1 /= r.P_q1.sum((2, 3), keepdims=True)
    try:
        names = event_names(r.catalog)
    except FileNotFoundError:
        names = []
    if len(names) != flat['event_neff'].shape[1]:
        names = [str(i) for i in range(flat['event_neff'].shape[1])]
    r.neff = pd.DataFrame({'median': np.median(flat['event_neff'], 0), 'lo5': np.percentile(flat['event_neff'], 5, 0),
                           'frac < 100': (flat['event_neff'] < 100).mean(0)}, index=names)
    return r


def spin_1d(P):
    """p(chi_eff), p(chi_p) densities from per-bin probabilities (draws, Be, Bp)."""
    return P.sum(2) / de, P.sum(1) / dp


def spin_moments_table(P):
    """Population moments of chi_eff and chi_p from per-bin probabilities, per draw."""
    p_e, p_p = spin_1d(P)
    me = (p_e * e_c).sum(1) * de
    return pd.DataFrame({'mean chi_eff': me, 'sd chi_eff': np.sqrt((p_e * e_c ** 2).sum(1) * de - me ** 2),
                         'P(chi_eff < 0)': (p_e * (e_c < 0)).sum(1) * de, 'mean chi_p': (p_p * p_c).sum(1) * dp})


def fmt(x):
    lo, med, hi = np.percentile(x, [5, 50, 95])
    return f'{med:.3f} [{lo:.3f}, {hi:.3f}]'


def q_averaged_moments(r, q_range=None, names=('chi_eff', 'chi_p'), core=False):
    """Spin moments given m1 alone: the conditional averaged over the windowed p(q | m1), per draw and m1 bin.
    For chi_eff and chi_p: 'mean', 'sd' (of p(chi | m1)), and the two parts of sd^2 by the law of total variance,
    'within' = sqrt(E_q[sd(m1, q)^2]) (the width at fixed q) and 'between' = sqrt(Var_q[mean(m1, q)]) (the spread of
    the per-q means), so sd^2 = within^2 + between^2. Each is (draws, Bm). With ``q_range`` = (lo, hi), only the q
    bins with centres in [lo, hi] (p(chi | m1, q in range)); NaN where the range has no support. ``names`` limits
    the spins computed. float32 throughout (two long runs must fit the login nodes' 8 GB). With ``core``, chi_eff
    only, restricted to |chi_eff| < CHI_CORE (r.MOM_CORE): each cell is weighted by its probability inside the core,
    so this is p(chi_eff | m1, |chi_eff| < CHI_CORE)."""
    if core:
        names = ('chi_eff',)
    w = r.p_mass.astype(np.float32)
    if core:
        w *= 1 - r.MOM_CORE[..., 2]
    if q_range is not None:
        w *= ((q_c >= q_range[0]) & (q_c <= q_range[1]))[None, None]
    tot = w.sum(2, keepdims=True)
    w /= np.maximum(tot, 1e-30)                                                  # windowed p(q | m1), (draws, Bm, Bq)
    wq, empty = w, tot[..., 0] <= 0
    out = {}
    for name, (i_mu, i_sd) in (('chi_eff', (0, 1)), ('chi_p', (2, 3))):
        if name not in names:
            continue
        mu, sd = (r.MOM_CORE[..., 0], r.MOM_CORE[..., 1]) if core else (r.MOM[..., i_mu], r.MOM[..., i_sd])
        mean = (wq * mu).sum(2)
        within2 = (wq * sd ** 2).sum(2)
        between2 = (wq * (mu - mean[..., None]) ** 2).sum(2)
        out[name] = {k: np.where(empty, np.nan, v) for k, v in
                     (('mean', mean), ('sd', np.sqrt(within2 + between2)), ('within', np.sqrt(within2)),
                      ('between', np.sqrt(between2)))}
    return out


# the chi_eff width contrasts: the gap below the 35 Msun peak and above it, against the peak
WIDTH_REF = 32.
WIDTH_AT = (10., 19., 45., 60., 80.)


def width_contrasts(Q, ref=WIDTH_REF, at=WIDTH_AT, name='chi_eff', key='sd'):
    """Per-draw differences Q[name][key](m1) - Q[name][key](ref) at the m1 bins nearest ``at``: median [5%, 95%] and
    P(> 0)."""
    x = Q[name][key]
    i0 = int(np.argmin(np.abs(m_c - ref)))
    rows = {}
    for m in at:
        i = int(np.argmin(np.abs(m_c - m)))
        d = x[:, i] - x[:, i0]
        rows[f'{m_c[i]:.0f} minus {m_c[i0]:.0f} Msun'] = {'difference': fmt(d), 'P(> 0)': f'{np.mean(d > 0):.2f}'}
    return pd.DataFrame(rows).T


PEAK_WINDOWS = ((7., 14.), (25., 45.))


def p_m1_peaks(r):
    """The m1 bins where the median windowed p(m1) is largest within each of PEAK_WINDOWS (Msun)."""
    med = np.median(r.p_lnm / m_c[None], 0)
    return [float(m_c[np.argmax(np.where((m_c > a) & (m_c < b), med, -np.inf))]) for a, b in PEAK_WINDOWS]


# column headers of summary_figure, by spin model
_P_SPIN = r'p(\chi_\mathrm{eff}, \chi_p \mid m_1, q)'
SPIN_MODEL_TITLES = {'field': rf'Flexible ${_P_SPIN}$', 'split': rf'Split ${_P_SPIN}$',
                     'gauss': rf'Gaussian ${_P_SPIN}$'}

SUMMARY_M_GRID = (10., 30., 100.)        # summary_figure's vertical grid lines and labelled m1 ticks

# the two mass-ratio ranges of summary_figure: unequal (where the broad-chi_eff systems sit) and near-equal masses
SUMMARY_Q_RANGES = ((0.3, 0.65), (0.8, 1.0))


def p_m1_given_q_range(r, q_range):
    """p(m1 | q in [lo, hi]) per draw, (draws, Bm), density per Msun, from the windowed mass field (q bin centres)."""
    sel = (q_c >= q_range[0]) & (q_c <= q_range[1])
    joint = r.p_mass[:, :, sel].sum(2) * dq                                        # p(ln m1, q in range)
    out = joint / (joint.sum(1, keepdims=True) * dlnm) / m_c[None]
    return np.where(joint > 0, out, np.nan)                         # NaN where m2 > 3 Msun excludes the whole range


def summary_figure(runs, labels=None, q_ranges=SUMMARY_Q_RANGES, m_min=MIN_M, show_mean=False, core=False):
    """Mass ratio and chi_eff on a shared m1 axis, one column per run (namespaces from load_run or load_light), with
    shared y ranges and colour scales across columns, rows of equal height:
    p(m1): the 90% interval over all q (black lines) and p(m1 | q in each of ``q_ranges``) (90% bands, medians);
    p(q | m1): the median, with the 90% (solid white) and 99% (dashed white) intervals of the mean q given m1;
    with ``show_mean``, mu_chi_eff(m1), the mean of p(chi_eff | m1), drawn like the next row;
    sigma_chi_eff(m1), the sd of p(chi_eff | m1): over all q (black lines, 90%) and over each q range (bands).
    With ``core``, the chi_eff rows use |chi_eff| < CHI_CORE only (r.MOM_CORE, q_averaged_moments(core=True)).
    Cells with q m1 < 3 Msun are blank, and so are the q-range curves where m2 > 3 Msun excludes the range. The m1
    axis starts at ``m_min``, and the y ranges cover what is visible. Columns are headed by ``labels`` (default:
    SPIN_MODEL_TITLES). Returns the figure and each run's p(m1) peaks (p_m1_peaks; printed, not drawn)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import plot_style
    labels = labels or [SPIN_MODEL_TITLES[r.spin_model] for r in runs]
    ok = q_c[None, :] * m_c[:, None] >= MIN_M                                        # (Bm, Bq): cells with support
    m_edges = np.exp(lnm_edges)
    cols = [f'C{i}' for i in range(len(q_ranges))]
    rng = lambda a, b: rf'${a:g} < q < {b:g}$'
    chi = lambda r, qr=None: q_averaged_moments(r, qr, names=('chi_eff',), core=core)['chi_eff']
    D = []
    for r in runs:
        pq = r.p_mass / r.p_mass.sum(2, keepdims=True)                               # p(q | m1), per q bin
        allq, byq = chi(r), [chi(r, qr) for qr in q_ranges]
        D.append(dict(p_m=r.p_lnm / m_c[None], p_mq=[p_m1_given_q_range(r, qr) for qr in q_ranges],
                      mq=(pq * q_c).sum(2), img_q=np.where(ok, np.median(pq, 0) / dq, np.nan),
                      mean=allq['mean'], mean_q=[x['mean'] for x in byq], sd=allq['sd'], sd_q=[x['sd'] for x in byq],
                      peaks=p_m1_peaks(r)))
        del pq, allq, byq
    vq = max(np.nanpercentile(d['img_q'], 99.5) for d in D)
    q5 = lambda x: np.nanpercentile(x, [5, 50, 95], axis=0)
    # curves on bin centres, extended flat to the outer bin edges so they reach the axis ends (3 and 200 Msun)
    mx = np.concatenate([[m_edges[0]], m_c, [m_edges[-1]]])
    ext = lambda y: np.concatenate([y[..., :1], y, y[..., -1:]], -1)
    cut = rf', $|\chi_\mathrm{{eff}}| < {CHI_CORE:g}$' if core else ''
    rows = ['pm', 'pq'] + (['mean'] if show_mean else []) + ['sd']
    n = len(runs)
    fig, axes = plt.subplots(len(rows), n, figsize=(7.5 * n + 1, 3.83 * len(rows)), sharex=True, sharey='row',
                             squeeze=False)
    A_ = {k: axes[i] for i, k in enumerate(rows)}
    for j, (d, lab) in enumerate(zip(D, labels)):
        for key, allq, byq in (('pm', d['p_m'], d['p_mq']), ('mean', d['mean'], d['mean_q']), ('sd', d['sd'], d['sd_q'])):
            if key not in A_:
                continue
            ax = A_[key][j]
            for y, col in zip(byq, cols):
                lo, med, hi = ext(q5(y))
                ax.fill_between(mx, lo, hi, color=col, alpha=0.3, lw=0)
                ax.plot(mx, med, color=col, lw=1.3)
            lo, _, hi = ext(q5(allq))
            ax.plot(mx, lo, color='k', lw=1); ax.plot(mx, hi, color='k', lw=1)
        if 'mean' in A_:
            A_['mean'][j].axhline(0, color='0.5', lw=0.8, ls=':')
        A_['pm'][j].set_yscale('log')
        A_['pm'][j].set_title(lab, pad=10)                        # column headers (user, 2026-10-22)
        ax = A_['pq'][j]
        pc_q = ax.pcolormesh(m_edges, q_edges, d['img_q'].T, cmap='viridis', shading='flat', rasterized=True,
                             vmin=0, vmax=vq)
        for (a, b), ls in (((5, 95), '-'), ((0.5, 99.5), '--')):
            for f in (a, b):
                ax.plot(mx, ext(np.percentile(d['mq'], f, 0)), color='w', lw=1, ls=ls)
        for ax in axes[:, j]:                                     # grid at m1 = 10, 30, 100 (user, 2026-10-22)
            for m in SUMMARY_M_GRID:
                ax.axvline(m, color='0.6', lw=0.6, alpha=0.7, zorder=3)
        axes[-1, j].set_xscale('log'); axes[-1, j].set_xlim(m_min, MAX_M); axes[-1, j].set_xlabel(r'$m_1\,[M_\odot]$')
        axes[-1, j].set_xticks(SUMMARY_M_GRID, [f'${m:g}$' for m in SUMMARY_M_GRID])
    vis = m_c >= m_min
    band_hi = lambda ys: max(np.nanmax(np.nanpercentile(y, 95, 0)[vis]) for d in D for y in ys(d))
    band_lo = lambda ys: min(np.nanmin(np.nanpercentile(y, 5, 0)[vis]) for d in D for y in ys(d))
    lo_all = min(np.nanpercentile(np.nanpercentile(d['p_m'], 5, 0)[vis], 2) for d in D)
    A_['pm'][0].set_ylim(lo_all / 3, band_hi(lambda d: [d['p_m']] + d['p_mq']) * 3)
    A_['pq'][0].set_ylim(0, 1)
    A_['sd'][0].set_ylim(0, 1.05 * band_hi(lambda d: [d['sd']] + d['sd_q']))
    A_['pm'][0].set_ylabel(r'$p(m_1)\,[M_\odot^{-1}]$'); A_['pq'][0].set_ylabel(r'$q$')
    A_['sd'][0].set_ylabel(r'$\sigma_{\chi_\mathrm{eff}}(m_1)$' + cut)
    if 'mean' in A_:
        lo_m, hi_m = band_lo(lambda d: [d['mean']] + d['mean_q']), band_hi(lambda d: [d['mean']] + d['mean_q'])
        pad = 0.05 * (hi_m - lo_m)
        A_['mean'][0].set_ylim(lo_m - pad, hi_m + pad)
        A_['mean'][0].set_ylabel(r'$\mu_{\chi_\mathrm{eff}}(m_1)$' + cut)
    # legends in every column (user, 2026-10-22): top row top right, p(q | m1) row bottom left, sigma row top left;
    # the mean row repeats the sigma row's artists, so it has none
    handles = [Line2D([], [], color='k', lw=1)] + [(Patch(color=c, alpha=0.3, lw=0), Line2D([], [], color=c, lw=1.3))
                                                   for c in cols]
    names = ['all $q$ (90\\%)'] + [rng(*qr) for qr in q_ranges]
    dark = lambda ls: Line2D([], [], color='k', lw=1, ls=ls)
    for j in range(n):
        A_['pm'][j].legend(handles, names, loc='upper right', **plot_style.LEG_KW)
        A_['sd'][j].legend(handles, names, loc='upper left', **plot_style.LEG_KW)
        # standard light legend (user, 2026-10-22); the samples are black because white would not show on it
        A_['pq'][j].legend([dark('-'), dark('--')], [r'$\langle q \rangle$, 90\%', r'$\langle q \rangle$, 99\%'],
                           loc='lower left', **plot_style.LEG_KW)
    cb = fig.colorbar(pc_q, cax=A_['pq'][-1].inset_axes([1.015, 0, 0.025, 1])); cb.set_label(r'$p(q \mid m_1)$')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.06, wspace=0.04)                  # tighter than tight_layout (user, 2026-10-22)
    return fig, [d['peaks'] for d in D]

def load_light(run):
    """What summary_figure needs, without load_run's full set of derived arrays (two long runs then fit the login
    nodes' 8 GB): the windowed mass field and the chi_eff moments per cell (MOM[..., :2]; chi_p is left out, so
    q_averaged_moments needs names=('chi_eff',)). Needs the rebuilt spin cache (load_run or python analysis.py <run>)."""
    files = sorted(glob.glob(os.path.join(REPO, 'results', run, 'chain_*_samples.h5')))
    with h5py.File(files[0], 'r') as h:
        spin_model = spin_model_of(h)
    n_draws = min(len(h5py.File(f, 'r')['log_likelihood']) for f in files)
    lp = np.concatenate([h5py.File(f, 'r')['log_p_mass'][:n_draws] for f in files]).astype(float) + mass_window()[None]
    mx = lp.max((1, 2), keepdims=True)
    lp -= np.log(np.exp(lp - mx).sum((1, 2), keepdims=True) * dlnm * dq) + mx
    r = SimpleNamespace(run=run, spin_model=spin_model, k=int(re.search(r'_k(\d+)', run).group(1)), n_draws=n_draws,
                        p_mass=np.exp(lp).astype(np.float32))
    del lp, mx
    r.p_lnm = r.p_mass.sum(2) * dq
    r.MOM = np.concatenate([h5py.File(post_path(run, f), 'r')['MOM'][:n_draws, ..., :2] for f in files])
    if all('MOM_CORE' in h5py.File(post_path(run, f), 'r') for f in files):
        r.MOM_CORE = np.concatenate([h5py.File(post_path(run, f), 'r')['MOM_CORE'][:n_draws] for f in files])
    return r


if __name__ == '__main__':
    # python notebooks/analysis.py <run> [<chain> ...]: rebuild and cache the spin quantities of all (or the given)
    # chains. Pin it first (taskset, single-thread XLA flags; see build_run_nb.py), or run it on a GPU.
    import sys
    sys.path.insert(0, REPO)
    run, chains = sys.argv[1], sys.argv[2:]
    files = sorted(glob.glob(os.path.join(REPO, 'results', run, 'chain_*_samples.h5')))
    if chains:
        files = [f for f in files if os.path.basename(f).split('_')[1] in chains]
    for f in files:
        print('wrote', rebuild_chain(run, f), flush=True)
