"""parametric.py: each range's p(m1) p(q | m1) integrates to 1 over m1 in [MIN_M, MAX_M], q in its range with
m2 > MIN_M (CPU; see test_variants.py)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp as LSE

from data import MAX_M, MIN_M
from parametric import M_GRID, Q_SPLIT, log_p_m1, log_p_q


def test_ranges_normalized():
    args = (2.5, 4., 3., jnp.asarray([0.6, 0.3, 0.1]), 10., 2., 35., 5.)
    lnm = np.linspace(np.log(MIN_M), np.log(MAX_M), 1500); q = np.linspace(0., 1., 1501)[1:]
    m1 = np.exp(lnm)[:, None]
    dlnm, dq = lnm[1] - lnm[0], q[1] - q[0]
    for (q_lo, q_hi), floor, beta in (((Q_SPLIT, 1.), MIN_M, 1.5), ((0., Q_SPLIT), MIN_M / Q_SPLIT, -1.)):
        norm = LSE(log_p_m1(jnp.asarray(M_GRID), *args, m_floor=floor)) + np.log(M_GRID[1] - M_GRID[0])
        logp = log_p_m1(jnp.asarray(m1), *args, m_floor=floor) - norm + log_p_q(jnp.asarray(q[None]), jnp.asarray(m1),
                                                                                   beta, q_lo, q_hi) + np.log(m1)
        total = float(jnp.exp(logp).sum() * dlnm * dq)
        np.testing.assert_allclose(total, 1., atol=0.02)
        # p(q | m1) alone integrates to 1 wherever the range is open
        pq = np.asarray(jnp.exp(log_p_q(jnp.asarray(q[None]), jnp.asarray(m1), beta, q_lo, q_hi)).sum(1) * dq)
        open_ = np.maximum(q_lo, MIN_M / m1[:, 0]) < q_hi - 0.05
        np.testing.assert_allclose(pq[open_], 1., atol=0.02)


def test_spin_split_reduces_to_shared_spins():
    """spin_split with equal offsets in both ranges gives the shared-spin likelihood; different offsets change it."""
    sys.path.insert(0, os.path.dirname(__file__))
    from parametric import RANGES, setup_parametric
    from test_model import synthetic, trace_at
    pp, kw = synthetic()
    a, init_a = setup_parametric(pp, n_spin_bins=(10, 8))
    pp, kw = synthetic()
    b, init_b = setup_parametric(pp, n_spin_bins=(10, 8), spin_split=True)
    g = jnp.asarray([0.1, 0.3, np.log(0.2), np.log(0.15), 0.3])
    lat_a = dict(init_a, gauss_offsets=g)
    lat_b = dict(init_b, **{f'gauss_offsets_{r}': g for r in RANGES})
    lnl = lambda m, lat: float(trace_at(m, lat, **kw)['log_likelihood']['value'])
    np.testing.assert_allclose(lnl(b, lat_b), lnl(a, lat_a), rtol=1e-6, atol=1e-4)
    lat_b['gauss_offsets_lo'] = jnp.asarray([-0.1, 0.5, np.log(0.4), np.log(0.3), -0.3])
    assert abs(lnl(b, lat_b) - lnl(a, lat_a)) > 1e-3


def test_shared_continuum_reduces_to_separate():
    """shared_continuum equals the separate model with the same continuum in both ranges; a different alpha_lo in the
    separate model changes the likelihood."""
    sys.path.insert(0, os.path.dirname(__file__))
    from parametric import CONTINUUM, RANGES, setup_parametric
    from test_model import synthetic, trace_at
    pp, kw = synthetic()
    a, init_a = setup_parametric(pp, n_spin_bins=(10, 8))
    pp, kw = synthetic()
    b, init_b = setup_parametric(pp, n_spin_bins=(10, 8), shared_continuum=True)
    assert not any(f'{k}_{r}' in init_b for k in CONTINUUM for r in RANGES)
    cont = {'alpha': 2.2, 'mmin': 4.5, 'delta_m': 2.5}
    lat_a = dict(init_a, **{f'{k}_{r}': jnp.asarray(v) for k, v in cont.items() for r in RANGES})
    lat_b = dict(init_b, **{k: jnp.asarray(v) for k, v in cont.items()})
    lnl = lambda m, lat: float(trace_at(m, lat, **kw)['log_likelihood']['value'])
    np.testing.assert_allclose(lnl(b, lat_b), lnl(a, lat_a), rtol=1e-6, atol=1e-4)
    lat_a['alpha_lo'] = jnp.asarray(5.)
    assert abs(lnl(b, lat_b) - lnl(a, lat_a)) > 1e-3
