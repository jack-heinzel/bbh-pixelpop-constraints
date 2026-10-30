"""Parametric check of the mass-ratio-dependent mass spectrum (user, 2026-10-29): a power law plus two Gaussian peaks in
m1 above and below a mass-ratio split, with completely separate hyperparameters,

    p(m1, q) = zeta p_hi(m1) p_hi(q | m1) + (1 - zeta) p_lo(m1) p_lo(q | m1),

p_r(m1) a power law plus two truncated Gaussians, tapered at its own m_min (pixelpop's m_smoother), normalized on
[MIN_M, MAX_M]; p_r(q | m1) a power law in q on range r (q > Q_SPLIT or q < Q_SPLIT), intersected with m2 > MIN_M and
normalized at every m1. The low-q spectrum is zero below MIN_M / Q_SPLIT, where its q range is empty. The spins are a
mass-independent truncated Gaussian in (chi_eff, chi_p) on the physical window, binned exactly as model.py's
spin_model='gauss' with constant fields; redshift, the m2 window and the likelihood (Neff floor, selection-bias
correction) are model.py's. Run with ``run.py --parametric``.

``spin_split`` (user, 2026-10-30): each range has its own mass-independent spin Gaussian (offsets 'gauss_offsets_hi',
'gauss_offsets_lo', same priors), so p(chi | m1, q) jumps at the split; ``run.py --spin-split``.
"""
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import logsumexp as LSE
from pixelpop.models.base_models import log_powerlaw_norm, m_smoother, trunc_gaussian

from data import MAX_M, MIN_M
from model import (GAUSS_OFFSET_BOUNDS, GAUSS_SD_FLOOR, NEG, gauss_log_kernel, population_likelihood, spin_bins,
                   spin_window_fraction)

Q_SPLIT = 0.7
RANGES = ('hi', 'lo')                    # q > Q_SPLIT, q < Q_SPLIT
M_GRID = np.linspace(MIN_M, MAX_M, 4000)  # normalization of p_r(m1) (Riemann sum, as pixelpop's mass models)
# priors, the same in both ranges (peak 1 below peak 2 only through their prior ranges, which overlap at 25-30 Msun)
PRIORS = {'alpha': (-4., 12.), 'mmin': (MIN_M, 10.), 'delta_m': (0., 10.), 'mpp_1': (5., 30.), 'sigpp_1': (0.5, 10.),
          'mpp_2': (25., 100.), 'sigpp_2': (1., 20.), 'beta': (-10., 10.)}
MASS_SITES = tuple(f'{k}_{r}' for r in RANGES for k in PRIORS) + tuple(f'lam_{r}' for r in RANGES) + ('zeta',)


def log_p_m1(m1, alpha, mmin, delta_m, lam, mpp_1, sigpp_1, mpp_2, sigpp_2, m_floor=MIN_M):
    """Unnormalized log density of a power law (slope -alpha) plus two Gaussian peaks (truncated to [MIN_M, MAX_M]),
    mixed with fractions lam = (lam_pl, lam_1, lam_2), tapered at mmin over delta_m, zero below m_floor."""
    pl = -alpha * jnp.log(m1) - log_powerlaw_norm(-alpha, MIN_M, MAX_M)
    g1 = trunc_gaussian(m1, mpp_1, sigpp_1, MIN_M, MAX_M)
    g2 = trunc_gaussian(m1, mpp_2, sigpp_2, MIN_M, MAX_M)
    p = LSE(jnp.stack([jnp.log(lam[0]) + pl, jnp.log(lam[1]) + g1, jnp.log(lam[2]) + g2]), axis=0)
    p = p + m_smoother(m1, mmin, delta_m)
    return jnp.where((m1 >= m_floor) & (m1 <= MAX_M), p, NEG)


def log_p_q(q, m1, beta, q_lo, q_hi):
    """log p(q | m1): q^beta on [max(q_lo, MIN_M / m1), q_hi], normalized at each m1 (NEG outside)."""
    lo = jnp.maximum(q_lo, MIN_M / m1)
    ok = (q >= lo) & (q <= q_hi) & (lo < q_hi)
    hi_safe = jnp.where(ok, q_hi, lo + 1.)
    return jnp.where(ok, beta * jnp.log(q) - log_powerlaw_norm(beta, lo, hi_safe), NEG)


def setup_parametric(pixelpop_data, n_spin_bins=(100, 100), likelihood_correction=True, record_weights=False,
                     gauss_sd_floor=GAUSS_SD_FLOOR, q_split=Q_SPLIT, spin_split=False):
    """NumPyro model and initial values of the q-split power law plus two peaks (module docstring), on
    ``pixelpop_data``'s posteriors and injections (data.build_pixelpop_data)."""
    pp = pixelpop_data
    Be, Bp = (int(b) for b in n_spin_bins)
    q_edges = np.asarray(pp.bin_axes[1])
    e_edges, p_edges = np.linspace(-1., 1., Be + 1), np.linspace(0., 1., Bp + 1)
    e_c = jnp.asarray(0.5 * (e_edges[1:] + e_edges[:-1])); p_c = jnp.asarray(0.5 * (p_edges[1:] + p_edges[:-1]))
    # spin bins, the physical window per q bin and the zero-weight flag for unreached spin bins, as model.py
    ev_e, ev_p = spin_bins(pp.posteriors, e_edges, p_edges)
    in_e, in_p = spin_bins(pp.injections, e_edges, p_edges)
    covered = np.zeros((Be, Bp), bool); covered[in_e, in_p] = True
    flag = ~covered[ev_e, ev_p]
    if flag.any():
        print(f'[parametric] {flag.mean():.2e} of PE samples are in spin bins with no injections; zero weight')
        pp.posteriors['log_prior'] = jnp.where(jnp.asarray(flag), jnp.inf, pp.posteriors['log_prior'])
    window = spin_window_fraction(q_edges, e_edges, p_edges)                           # (Bq, Be, Bp)
    log_window = jnp.asarray(np.where(window > 0, np.log(np.maximum(window, 1e-300)), NEG))
    log_support = jnp.where(log_window > NEG / 2, 0., NEG)
    Bq = len(q_edges) - 1
    q_bin = lambda d: jnp.asarray(np.clip(np.digitize(np.asarray(d['mass_ratio']), q_edges) - 1, 0, Bq - 1))
    ev_idx = (q_bin(pp.posteriors), jnp.asarray(ev_e), jnp.asarray(ev_p))
    in_idx = (q_bin(pp.injections), jnp.asarray(in_e), jnp.asarray(in_p))
    m_grid = jnp.asarray(M_GRID); log_dm = float(np.log(M_GRID[1] - M_GRID[0]))
    bounds = {'hi': (q_split, 1.), 'lo': (0., q_split)}
    floors = {'hi': MIN_M, 'lo': MIN_M / q_split}

    def mass_log_p(data, theta):
        """(2, n): each range's term of the density in (ln m1, q), weight included (RANGES order)."""
        m1 = jnp.exp(data['log_mass_1']); q = data['mass_ratio']
        terms = []
        for r in RANGES:
            t = theta[r]
            args = (t['alpha'], t['mmin'], t['delta_m'], t['lam'], t['mpp_1'], t['sigpp_1'], t['mpp_2'], t['sigpp_2'])
            norm = LSE(log_p_m1(m_grid, *args, m_floor=floors[r])) + log_dm
            w = jnp.log(theta['zeta']) if r == 'hi' else jnp.log1p(-theta['zeta'])
            terms.append(w + log_p_m1(m1, *args, m_floor=floors[r]) - norm + log_p_q(q, m1, t['beta'], *bounds[r]))
        return jnp.stack(terms) + data['log_mass_1']                                       # density in (ln m1, q)

    def spin_log_p(offsets):
        """log p(chi | q) at the binned samples, per q bin normalized on the physical window, for one Gaussian."""
        kern = gauss_log_kernel(offsets[:, None, None], e_c[:, None], p_c[None, :], gauss_sd_floor)   # (Be, Bp)
        log_z = LSE(kern[None] + log_support + log_window, axis=(1, 2))                 # (Bq,)
        return lambda iq, ie, ip: kern[ie, ip] + log_support[iq, ie, ip] - log_z[iq]

    def probabilistic_model(posteriors, injections):
        theta = {'zeta': numpyro.sample('zeta', dist.Uniform(0., 1.))}
        for r in RANGES:
            theta[r] = {k: numpyro.sample(f'{k}_{r}', dist.Uniform(*b)) for k, b in PRIORS.items()}
            theta[r]['lam'] = numpyro.sample(f'lam_{r}', dist.Dirichlet(jnp.ones(3)))
        lo, hi = (jnp.asarray(b, dtype=float) for b in GAUSS_OFFSET_BOUNDS)
        if spin_split:
            spins = [spin_log_p(numpyro.sample(f'gauss_offsets_{r}', dist.Uniform(lo, hi))) for r in RANGES]
        else:
            spins = [spin_log_p(numpyro.sample('gauss_offsets', dist.Uniform(lo, hi)))] * len(RANGES)
        joint = lambda data, idx: LSE(mass_log_p(data, theta) + jnp.stack([f(*idx) for f in spins]), axis=0)
        event_weights = posteriors['ln_dVTc'] - posteriors['log_prior'] + joint(posteriors, ev_idx)
        inj_weights = injections['ln_dVTc'] - injections['log_prior'] + joint(injections, in_idx)
        population_likelihood(pp, posteriors, injections, event_weights, inj_weights, likelihood_correction,
                              record_weights)

    lo, hi = (np.asarray(b) for b in GAUSS_OFFSET_BOUNDS)
    g0 = jnp.asarray(np.clip([0.05, 0.25, np.log(0.1), np.log(0.1), 0.], lo + 1e-3, hi - 1e-3))
    init = {'zeta': jnp.asarray(0.5)}
    init.update({f'gauss_offsets_{r}': g0 for r in RANGES} if spin_split else {'gauss_offsets': g0})
    start = {'alpha': 3., 'mmin': 5., 'delta_m': 3., 'mpp_1': 10., 'sigpp_1': 2., 'mpp_2': 35., 'sigpp_2': 5., 'beta': 1.}
    for r in RANGES:
        init.update({f'{k}_{r}': jnp.asarray(v) for k, v in start.items()})
        init[f'lam_{r}'] = jnp.asarray([0.6, 0.3, 0.1])
    for key, (args, distribution) in pp.priors.items():      # redshift hyperparameters at their prior means
        if distribution.__name__ != 'Delta':
            init[key] = jnp.asarray(distribution(*args).mean)
    return probabilistic_model, init
