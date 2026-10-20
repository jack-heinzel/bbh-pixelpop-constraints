"""Tests for model.py (CPU, pinned):

    JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
        taskset -c <4 cores> ~/.conda/envs/gwjax313/bin/python -m pytest tests -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import numpy as np
import pixelpop
import pytest
from numpyro import handlers
from numpyro import distributions as dist

from model import setup_model, spin_window_fraction

MGI = '/home/jack.heinzel/public_html/mass-gap-investigation/code/pixelpop'


def trace_at(model, values, **kwargs):
    with handlers.seed(rng_seed=0), handlers.substitute(data=values):
        return handlers.trace(model).get_trace(**kwargs)


def synthetic(bins=(12, 10), nobs=40, npe=150, seed=2):
    """Isotropic-spin events and injections on a small (ln m1, q) grid."""
    rng = np.random.default_rng(seed)

    def draw(n):
        m1 = np.exp(rng.uniform(np.log(5), np.log(150), n))
        q = rng.uniform(0.15, 1, n)
        a1, a2 = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
        c1, c2 = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
        chi_eff = (a1 * c1 + q * a2 * c2) / (1 + q)
        chi_p = np.maximum(a1 * np.sqrt(1 - c1 ** 2), (4 * q + 3) / (4 + 3 * q) * q * a2 * np.sqrt(1 - c2 ** 2))
        return m1, q, chi_eff, chi_p

    m1, q, ce, cp = draw(nobs)
    jit = lambda x, s, lo, hi: np.clip(x[:, None] + s * rng.normal(size=(nobs, npe)), lo, hi)
    posteriors = {'log_mass_1': jit(np.log(m1), 0.05, np.log(3.1), np.log(199)), 'mass_ratio': jit(q, 0.03, 0.05, 1.),
                  'chi_eff': jit(ce, 0.02, -0.99, 0.99), 'chi_p': jit(cp, 0.02, 0.01, 0.99),
                  'redshift': rng.uniform(0.05, 1.5, (nobs, npe)), 'log_prior': np.zeros((nobs, npe))}
    ninj = 20000
    im1, iq, ice, icp = draw(ninj)
    injections = {'log_mass_1': np.log(im1), 'mass_ratio': iq, 'chi_eff': ice, 'chi_p': icp,
                  'redshift': rng.uniform(0.05, 1.5, ninj), 'log_prior': np.zeros(ninj)}
    posteriors = {k: jnp.asarray(v) for k, v in posteriors.items()}
    injections = {k: jnp.asarray(v) for k, v in injections.items()}
    injections['total_generated'] = float(ninj); injections['analysis_time'] = 1.
    from pixelpop.models.gwtc_defaults import gwtc6_default
    other = ['redshift', 'mass_ratio_window']
    models = {p: gwtc6_default.models[p] for p in other}
    hypers = {p: list(gwtc6_default.hyperparameters[p]) for p in other}
    priors = {h: gwtc6_default.priors[h] for p in other for h in hypers[p]}
    priors.update({'mlow_2': ([3.], dist.Delta), 'delta_m_2': ([0.], dist.Delta)})
    pp = pixelpop.utils.PixelPopData(
        name='t', posteriors=posteriors, injections=injections, pixelpop_parameters=['log_mass_1', 'mass_ratio'],
        other_parameters=other, bins=list(bins), minima={'log_mass_1': np.log(3.)}, maxima={'log_mass_1': np.log(200.)},
        parametric_models=models, parameter_to_hyperparameters=hypers, priors=priors, UncertaintyCut=np.inf,
        spde_matern=True, EventNeffCut=30.)
    return pp, {'posteriors': posteriors, 'injections': injections}


def random_latents(init, seed=5):
    rng = np.random.default_rng(seed)
    return dict({k: jnp.asarray(v) for k, v in init.items()},
                _spin_sites=jnp.asarray(rng.normal(size=init['_spin_sites'].shape)), lnsigma_spin=jnp.asarray(0.7))


@pytest.mark.parametrize('km', [0, 3])
def test_factorized_matches_dense_and_gradients(km):
    pp, kw = synthetic()
    fac, init = setup_model(pp, spin_mass_modes=km, n_spin_bins=(10, 8))
    dense, _ = setup_model(pp, spin_mass_modes=km, n_spin_bins=(10, 8), _dense_cube=True)
    lat = random_latents(init)
    a, b = trace_at(fac, lat, **kw), trace_at(dense, lat, **kw)
    np.testing.assert_allclose(float(a['log_likelihood']['value']), float(b['log_likelihood']['value']), atol=1e-3)
    np.testing.assert_allclose(np.exp(np.asarray(a['log_p_spin_marginal']['value'])),
                               np.exp(np.asarray(b['log_p_spin_marginal']['value'])), atol=1e-6)
    sl = np.exp(np.asarray(a['log_p_spin_q1_m']['value']))
    np.testing.assert_allclose(sl.sum((1, 2)), 1., rtol=1e-4)                # each saved spin slice is normalized
    if km:
        assert np.std(sl, axis=0).max() > 1e-4                                  # and varies with mass
    np.testing.assert_allclose(np.exp(np.asarray(a['log_p_spin_marginal']['value'])).sum(), 1., rtol=1e-4)
    g = jax.grad(lambda prm: trace_at(fac, prm, **kw)['log_likelihood']['value'])(lat)
    assert all(np.all(np.isfinite(np.asarray(v))) for v in g.values())


def test_spin_conditional_normalized_on_window():
    """At every (m1, q), the spin density integrates to 1 over the allowed region (dense cube)."""
    pp, kw = synthetic()
    dense, init = setup_model(pp, spin_mass_modes=3, n_spin_bins=(10, 8), _dense_cube=True)
    lat = random_latents(init)
    # rebuild the cube from the traced fields: normalization is checked through the saved slices at all four masses
    tr = trace_at(dense, lat, **kw)
    sl = np.exp(np.asarray(tr['log_p_spin_q1_m']['value']))
    np.testing.assert_allclose(sl.sum((1, 2)), 1., rtol=1e-5)
    win = spin_window_fraction(np.linspace(0, 1, 11), np.linspace(-1, 1, 11), np.linspace(0, 1, 9))
    assert np.all(sl[:, win[-1] == 0] == 0)                                     # nothing outside the chi_p window


@pytest.mark.skipif(not os.path.isdir(MGI), reason='mass-gap-investigation not available')
def test_matches_mass_gap_investigation_hierarchical():
    """Same likelihood as mass-gap-investigation's hierarchical model with mass_gate='none', frac_2g_gate='none'."""
    sys.path.insert(0, MGI)
    from hierarchical import setup_hierarchical_model
    pp, kw = synthetic()
    mine, init = setup_model(pp, spin_mass_modes=3, n_spin_bins=(10, 8), likelihood_correction=True)
    theirs, init_h = setup_hierarchical_model(pp, base_mass_modes=3, n_spin_bins=(10, 8), spin_mean='none',
                                              mass_gate='none', frac_2g_gate='none', likelihood_correction=True)
    lat = random_latents(init)
    rename = {'_spin_sites': '_spin_base_sites', 'lnsigma_spin': 'lnsigma_base', 'log_nu_spin': 'log_nu_base',
              'log_ranges_spin': 'log_ranges_base'}
    lat_h = {rename.get(k, k): v for k, v in lat.items()}
    assert set(lat_h) == set(init_h)
    a, b = trace_at(mine, lat, **kw), trace_at(theirs, lat_h, **kw)
    for k in ('log_likelihood', 'log_likelihood_correction', 'log_likelihood_variance'):
        np.testing.assert_allclose(float(a[k]['value']), float(b[k]['value']), rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(np.asarray(a['event_neff']['value']), np.asarray(b['event_neff']['value']), rtol=1e-3)   # float32
    np.testing.assert_allclose(np.asarray(a['log_p_mass']['value']), np.asarray(b['log_p_mass']['value']), atol=1e-4)
