"""Load a run's chains and derive the windowed mass and spin quantities the notebooks plot.

    import analysis as A
    r = A.load_run('ppc_gwtc5_k5_short')
    r.p_mass, r.P_win, r.MOM, r.summary ...

Import after the thread / platform environment is set (the notebooks set JAX_PLATFORMS=cpu and the single-thread XLA
flags before importing this), since this imports jax through pixelpop.

* Mass: the saved ``log_p_mass`` is unwindowed; ``p_mass`` adds the m2 > 3 Msun window on bin centres and renormalizes.
* Spins: rebuilt from the saved spin latents of any spin model (field, split, gauss; exactly as model.py builds them) on
  up to ``max_rebuild`` evenly spaced draws: the windowed population marginal, the conditional's per-(m1, q) mean and
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

REPO = '/home/jack.heinzel/public_html/bbh-pixelpop-constraints'
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


def _rebuild(files, n_draws, draws, logW, spin_model, flat):
    """Spin quantities from the saved latents of any spin model, on ``draws`` of each chain:
    P_raw, P_win (population marginal, unwindowed / windowed mass weights), MOM (per-cell moments), PE_M, PP_M
    (p(chi_eff | m1), p(chi_p | m1): per-bin probabilities, averaged over the windowed p(q | m1)), SLICE (per-cell
    P(chi_eff in S | m1, q) for CHI_EFF_SLICES), and the largest total variation between the rebuilt and the saved
    unwindowed marginal (a check of the rebuild)."""
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
    with h5py.File(files[0], 'r') as h:
        km = (h['_spin_sites'].shape[1] if spin_model == 'field' else h['_spin1_sites'].shape[1] if spin_model == 'split'
              else h['_gauss_sites'].shape[1])
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

    out = {k: [] for k in ('P_raw', 'P_win', 'MOM', 'PE_M', 'PP_M', 'SLICE')}
    tv = []
    for c, f in enumerate(files):
        with h5py.File(f, 'r') as h:
            g = lambda k, i: jnp.asarray(h[k][i], dtype=float)
            for i in draws:
                if spin_model == 'field':
                    res = field(g('_spin_sites', i), g('lnsigma_spin', i), g('log_nu_spin', i), g('log_ranges_spin', i),
                                g('log_p_mass', i))
                elif spin_model == 'split':
                    res = split(*(g(k, i) for k in ('_spin0_sites', 'lnsigma_spin0', 'log_nu_spin0', 'log_ranges_spin0',
                                                    '_spin1_sites', 'lnsigma_spin1', 'log_nu_spin1', 'log_ranges_spin1',
                                                    'log_p_mass')))
                else:
                    res = gauss(jnp.asarray(flat['gauss_params'][c * n_draws + i], dtype=float), g('log_p_mass', i))
                raw, win, mom, pe_m, pp_m, sl = (np.asarray(x) for x in res)
                saved = np.exp(h['log_p_spin_marginal'][i].astype(float))
                tv.append(0.5 * np.abs(raw / raw.sum() - saved / saved.sum()).sum())
                for k, v in zip(out, (raw / raw.sum(), win / win.sum(), mom, pe_m, pp_m, sl)):
                    out[k].append(np.asarray(v, dtype=np.float32))
    out = {k: np.array(v) for k, v in out.items()}
    return out, float(np.max(tv))


def load_run(run, max_rebuild=150):
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
    r.spin_model = 'split' if 'lnsigma_spin1' in S else 'gauss' if 'gauss_offsets' in S else 'field'
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

    # spins, windowed: always rebuilt (the per-mass and chi_eff-slice quantities are not saved by any model)
    draws = np.unique(np.linspace(0, n_draws - 1, min(n_draws, max_rebuild)).round().astype(int))
    reb, r.rebuild_tv = _rebuild(files, n_draws, draws, r.logW, r.spin_model, flat)
    r.P_raw, r.P_win, r.MOM, r.PE_M, r.PP_M, r.SLICE = (reb[k] for k in ('P_raw', 'P_win', 'MOM', 'PE_M', 'PP_M', 'SLICE'))
    r.spin_source = f'rebuilt from the saved spin latents on {len(draws)} of {n_draws} draws per chain'
    # mass in chi_eff slices: p(ln m1, q | chi_eff in S) ~ p_mass(ln m1, q) P(S | m1, q), on the rebuilt draws
    pm = r.p_mass.reshape(r.n_chains, n_draws, B, B)[:, draws].reshape(-1, B, B)
    joint = pm[..., None] * r.SLICE * dlnm * dq                                     # (draws, Bm, Bq, n_slices)
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
