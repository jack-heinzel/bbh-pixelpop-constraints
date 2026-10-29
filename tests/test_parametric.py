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
