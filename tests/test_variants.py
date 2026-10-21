"""Tests for the spin_model variants of model.py ('split', 'gauss'), on test_model.py's synthetic data (CPU, pinned):

    JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
        taskset -c <4 cores> ~/.conda/envs/gwjax313/bin/python -m pytest tests -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.infer.util import log_density

from model import GAUSS_PARAMS, GAUSS_SD_FLOOR, NEG, setup_model, spin_window_fraction
from test_model import synthetic, trace_at

SPIN_BINS = (10, 8)
KM = 3


def common_latents(init, seed=7):
    """Mass field and redshift latents shared between models, at random non-trivial values."""
    rng = np.random.default_rng(seed)
    out = {k: jnp.asarray(v) for k, v in init.items()}
    out['_eigenbasis_sites'] = jnp.asarray(rng.normal(size=np.shape(init['_eigenbasis_sites'])))
    out['lnsigma'] = jnp.asarray(0.4)
    return out


def spin_grid():
    e_edges, p_edges = np.linspace(-1, 1, SPIN_BINS[0] + 1), np.linspace(0, 1, SPIN_BINS[1] + 1)
    return e_edges, p_edges, 0.5 * (e_edges[1:] + e_edges[:-1]), 0.5 * (p_edges[1:] + p_edges[:-1])


def test_split_without_mass_variation_is_the_mass_independent_field():
    """sites of G1 = 0: the split model is the km = 0 field model with G0 as its field."""
    pp, kw = synthetic()
    split, init_s = setup_model(pp, spin_mass_modes=KM, n_spin_bins=SPIN_BINS, spin_model='split')
    flat, init_f = setup_model(pp, spin_mass_modes=0, n_spin_bins=SPIN_BINS)
    rng = np.random.default_rng(3)
    w0 = jnp.asarray(rng.normal(size=SPIN_BINS))
    hyp = dict(lnsigma=jnp.asarray(0.6), nu=jnp.asarray(0.2), ranges=jnp.asarray([1.2, 0.9]))
    lat_s = dict(common_latents(init_s), _spin0_sites=w0, lnsigma_spin0=hyp['lnsigma'], log_nu_spin0=hyp['nu'],
                 log_ranges_spin0=hyp['ranges'], _spin1_sites=jnp.zeros_like(init_s['_spin1_sites']))
    lat_f = dict(common_latents(init_f), _spin_sites=w0, lnsigma_spin=hyp['lnsigma'], log_nu_spin=hyp['nu'],
                 log_ranges_spin=hyp['ranges'])
    a, b = trace_at(split, lat_s, **kw), trace_at(flat, lat_f, **kw)
    for k in ('log_likelihood', 'log_likelihood_variance'):
        np.testing.assert_allclose(np.asarray(a[k]['value']), np.asarray(b[k]['value']), rtol=1e-4, atol=1e-4)
    # the spin marginal on reachable bins (unreachable ones are -inf on the factorized path, a -2e4 floor on the dense)
    ma, mb = np.asarray(a['log_p_spin_marginal']['value']), np.asarray(b['log_p_spin_marginal']['value'])
    reach = mb > -1e3
    np.testing.assert_allclose(ma[reach], mb[reach], rtol=1e-4, atol=1e-4)
    # and the conditional does not depend on mass: every (m1, q ~ 1) slice is the same
    q1 = np.asarray(a['log_p_spin_q1_m']['value'])
    np.testing.assert_allclose(q1 - q1[:1], 0., atol=1e-4)


def test_split_mass_variation_changes_the_conditional():
    pp, kw = synthetic()
    split, init = setup_model(pp, spin_mass_modes=KM, n_spin_bins=SPIN_BINS, spin_model='split')
    rng = np.random.default_rng(4)
    lat = dict(common_latents(init), _spin1_sites=jnp.asarray(rng.normal(size=np.shape(init['_spin1_sites']))),
               lnsigma_spin1=jnp.asarray(1.))
    mom = np.asarray(trace_at(split, lat, **kw)['spin_moments']['value'])
    assert np.ptp(mom[:, -1, 0]) > 1e-2                     # mean chi_eff at q ~ 1 varies along m1


def gauss_conditional_numpy(theta, q_edges):
    """log p(chi bin | m1, q) of spin_model='gauss' (density on the window, as model.py), (Bm, Bq, Be, Bp)."""
    e_edges, p_edges, e_c, p_c = spin_grid()
    mu_e, mu_p, ls_e, ls_p, z = (np.asarray(t, dtype=float)[:, :, None, None] for t in theta)
    sd_e, sd_p = np.sqrt(np.exp(2 * ls_e) + GAUSS_SD_FLOOR ** 2), np.sqrt(np.exp(2 * ls_p) + GAUSS_SD_FLOOR ** 2)
    rho = np.tanh(z)
    x, y = (e_c[:, None] - mu_e) / sd_e, (p_c[None, :] - mu_p) / sd_p
    logn = -0.5 * (x ** 2 - 2 * rho * x * y + y ** 2) / (1 - rho ** 2)
    window = spin_window_fraction(q_edges, e_edges, p_edges)[None]                    # (1, Bq, Be, Bp)
    support = np.where(window > 0, 0., -np.inf)
    with np.errstate(divide='ignore'):
        log_w = np.log(window)
    t = logn + log_w
    logz = np.log(np.exp(t - t.max((2, 3), keepdims=True)).sum((2, 3), keepdims=True)) + t.max((2, 3), keepdims=True)
    return logn + support - logz, window


def test_gauss_event_weights_and_moments():
    pp, kw = synthetic()
    gauss, init_g = setup_model(pp, spin_mass_modes=KM, n_spin_bins=SPIN_BINS, spin_model='gauss', record_weights=True)
    flat, init_f = setup_model(pp, spin_mass_modes=0, n_spin_bins=SPIN_BINS, record_weights=True)
    rng = np.random.default_rng(6)
    lat_g = dict(common_latents(init_g), _gauss_sites=jnp.asarray(rng.normal(size=np.shape(init_g['_gauss_sites']))),
                 gauss_offsets=jnp.asarray([0.1, 0.3, np.log(0.15), np.log(0.2), 0.4]),
                 lnsigma_gauss=jnp.asarray([np.log(0.1), np.log(0.1), np.log(0.3), np.log(0.3), np.log(0.3)]))
    lat_f = dict(common_latents(init_f), _spin_sites=jnp.zeros(SPIN_BINS))            # flat spins on the window
    tg, tf = trace_at(gauss, lat_g, **kw), trace_at(flat, lat_f, **kw)
    theta = np.asarray(tg['gauss_params']['value'])
    assert theta.shape == (len(GAUSS_PARAMS), *pp.bins)

    q_edges = np.asarray(pp.bin_axes[1])
    log_p, window = gauss_conditional_numpy(theta, q_edges)
    e_edges, p_edges, e_c, p_c = spin_grid()
    flat_log_p = -np.log(window.sum((2, 3), keepdims=True))[0]                         # (Bq, 1, 1)
    im, iq = (np.asarray(x) for x in pp.event_bins)
    inside = (im >= 0) & (im < pp.bins[0]) & (iq >= 0) & (iq < pp.bins[1])      # off-grid samples have prior = +inf
    im, iq = np.clip(im, 0, pp.bins[0] - 1), np.clip(iq, 0, pp.bins[1] - 1)
    ie = np.clip(np.digitize(np.asarray(pp.posteriors['chi_eff']), e_edges) - 1, 0, SPIN_BINS[0] - 1)
    ip = np.clip(np.digitize(np.asarray(pp.posteriors['chi_p']), p_edges) - 1, 0, SPIN_BINS[1] - 1)
    want = log_p[im, iq, ie, ip] - flat_log_p[iq, 0, 0]
    got = np.asarray(tg['event_log_weights']['value']) - np.asarray(tf['event_log_weights']['value'])
    ok = inside & np.isfinite(want) & (np.asarray(pp.posteriors['log_prior']) < np.inf)
    assert ok.mean() > 0.5
    np.testing.assert_allclose(got[ok], want[ok], rtol=1e-4, atol=1e-3)

    # per-cell moments and the q ~ 1 slices from the same numpy conditional
    P = np.exp(np.where(np.isfinite(log_p), log_p, -np.inf)) * window
    pe, pq = P.sum(3), P.sum(2)
    me = pe @ e_c
    mom = np.asarray(tg['spin_moments']['value'])
    np.testing.assert_allclose(mom[..., 0], me, atol=1e-4)
    np.testing.assert_allclose(mom[..., 1], np.sqrt(pe @ e_c ** 2 - me ** 2), atol=1e-4)
    np.testing.assert_allclose(mom[..., 2], pq @ p_c, atol=1e-4)

    # windowed spin marginal = the conditional averaged with the windowed mass probabilities
    lp_mass = np.asarray(tg['log_p_mass']['value']) + float(np.sum(pp.logdV))
    lnm_c = 0.5 * (np.asarray(pp.bin_axes[0])[1:] + np.asarray(pp.bin_axes[0])[:-1])
    q_c = 0.5 * (q_edges[1:] + q_edges[:-1])
    w = np.exp(lp_mass) * (np.exp(lnm_c)[:, None] * q_c[None] > 3.)
    w /= w.sum()
    marg = np.einsum('mq,mqep->ep', w, P)
    np.testing.assert_allclose(np.exp(np.asarray(tg['log_p_spin_marginal_windowed']['value'])), marg,
                               rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize('spin_model', ['split', 'gauss'])
def test_variant_gradients_finite(spin_model):
    pp, kw = synthetic()
    model, init = setup_model(pp, spin_mass_modes=KM, n_spin_bins=SPIN_BINS, spin_model=spin_model)
    val, grad = jax.value_and_grad(lambda p: log_density(model, (), kw, p)[0])(init)
    assert np.isfinite(float(val))
    for k, g in grad.items():
        assert np.all(np.isfinite(np.asarray(g))), k


def test_gauss_with_all_mass_modes():
    """spin_model='gauss' with k = the full q axis of the synthetic grid (its mass fields then use every mode along
    q): finite likelihood and gradients, and moments inside the physical range."""
    pp, kw = synthetic()
    k_full = min(int(b) for b in pp.bins)
    model, init = setup_model(pp, spin_mass_modes=k_full, n_spin_bins=SPIN_BINS, spin_model='gauss')
    assert init['_gauss_sites'].shape == (len(GAUSS_PARAMS), k_full, k_full)
    val, grad = jax.value_and_grad(lambda p: log_density(model, (), kw, p)[0])(init)
    assert np.isfinite(float(val)) and all(np.all(np.isfinite(np.asarray(g))) for g in grad.values())
    mom = np.asarray(trace_at(model, init, **kw)['spin_moments']['value'])
    assert np.all(np.abs(mom[..., 0]) < 1) and np.all((mom[..., 2] > 0) & (mom[..., 2] < 1))


@pytest.mark.parametrize('spin_model', ['field', 'split', 'gauss'])
def test_prior_draw_is_a_valid_start(spin_model):
    """run.prior_draw (the default initialization since the Rusty merge) gives a start with finite potential energy and
    gradient in NUTS's unconstrained space, for every spin model."""
    import run
    from numpyro.infer.util import potential_energy, unconstrain_fn
    pp, kw = synthetic()
    model, init = setup_model(pp, spin_mass_modes=KM, n_spin_bins=SPIN_BINS, spin_model=spin_model)
    draw = run.prior_draw(model, init, kw, seed=3, chain=1)
    assert set(draw) == set(init)
    u = unconstrain_fn(model, (), kw, draw)
    pe, grad = jax.value_and_grad(lambda z: potential_energy(model, (), kw, z))(u)
    assert np.isfinite(float(pe)) and all(np.all(np.isfinite(np.asarray(g))) for g in grad.values())
