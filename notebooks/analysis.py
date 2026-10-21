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
            return (jnp.einsum('q,qep->ep', w_raw[m], P), jnp.einsum('q,qep->ep', w_win[m], P), mom,
                    wq @ pe, wq @ pq, pe @ masks)
        raw, win, mom, pe_m, pp_m, sl = jax.lax.map(slab, jnp.arange(B), batch_size=10)
        return raw.sum(0), win.sum(0), mom, pe_m, pp_m, sl

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


REBUILT = ('P_raw', 'P_win', 'MOM', 'PE_M', 'PP_M', 'SLICE')
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
                have = g.attrs['n_draws']
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
    r.P_raw, r.P_win, r.MOM, r.PE_M, r.PP_M, r.SLICE = (reb[k] for k in REBUILT)
    r.rebuild_tv = float(reb['tv'].max())
    r.spin_source = (f'rebuilt from the saved spin latents on all {n_draws} draws per chain '
                     f'(results/post_processing/for_plotting/{run}/)')
    # mass in chi_eff slices: p(ln m1, q | chi_eff in S) ~ p_mass(ln m1, q) P(S | m1, q), on the rebuilt draws
    joint = r.p_mass[..., None] * r.SLICE * dlnm * dq                                     # (draws, Bm, Bq, n_slices)
    r.slice_share = joint.sum((1, 2))                                               # P(chi_eff in S)
    r.p_mass_slice = joint / np.maximum(r.slice_share[:, None, None], 1e-300) / (dlnm * dq)
    r.P_q1 = (lambda P: P / P.sum((2, 3), keepdims=True))(np.exp(flat['log_p_spin_q1_m'].astype(float)))
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
