"""Tests for the redshift-dependent spin conditional of model.py (spin_z_modes > 0), on test_model.py's synthetic data
(CPU, pinned; see test_variants.py)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import jax.numpy as jnp
import numpy as np

import model as M
from model import NEG, gauss_log_kernel, setup_model, spin_bins, spin_window_fraction
from test_model import synthetic, trace_at
from test_variants import common_latents, spin_grid

SPIN_BINS = (10, 8)
KM, KQ, KZ, BZ = 3, 2, 3, 6


def lnl(model, lat, kw):
    return float(trace_at(model, lat, **kw)['log_likelihood']['value'])


def test_constant_fields_match_the_mass_only_models():
    """All white sites zero: the conditional is the same at every (m1, q, z), so the z-dependent models give the
    likelihood of their mass-only versions."""
    pp, kw = synthetic()
    for spin_model in ('field', 'gauss'):
        a, init_a = setup_model(pp, spin_mass_modes=KM, n_spin_bins=SPIN_BINS, spin_model=spin_model)
        b, init_b = setup_model(pp, spin_mass_modes=KM, n_spin_bins=SPIN_BINS, spin_model=spin_model,
                                spin_q_modes=KQ, spin_z_modes=KZ, z_bins=BZ)
        site = '_gauss_sites' if spin_model == 'gauss' else '_spin_sites'
        lat_a = dict(common_latents(init_a), **{site: jnp.zeros_like(init_a[site])})
        lat_b = dict(common_latents(init_b), **{site: jnp.zeros_like(init_b[site])})
        if spin_model == 'gauss':
            lat_a['gauss_offsets'] = lat_b['gauss_offsets'] = jnp.asarray([0.1, 0.3, np.log(0.2), np.log(0.15), 0.3])
        np.testing.assert_allclose(lnl(b, lat_b, kw), lnl(a, lat_a, kw), rtol=1e-6, atol=1e-4)


def test_gauss_per_sample_conditional_by_brute_force():
    """The z-dependent Gaussian's per-sample spin log density, from its saved parameter fields (gauss_params) and the
    sample's (m1, q, z, chi) bins, normalized over the (chi_eff, chi_p) grid on the window."""
    pp, kw = synthetic()
    zm, init = setup_model(pp, spin_mass_modes=KM, n_spin_bins=SPIN_BINS, spin_model='gauss', spin_q_modes=KQ,
                           spin_z_modes=KZ, z_bins=BZ, record_weights=True)
    flat, init_f = setup_model(pp, spin_mass_modes=0, n_spin_bins=SPIN_BINS, record_weights=True)
    rng = np.random.default_rng(11)
    lat = dict(common_latents(init), _gauss_sites=jnp.asarray(rng.normal(size=np.shape(init['_gauss_sites']))),
               lnsigma_gauss=jnp.asarray([-1.5, -1.5, -0.5, -0.5, -0.5]))
    tz = trace_at(zm, lat, **kw)
    tf = trace_at(flat, dict(common_latents(init_f), _spin_sites=jnp.zeros_like(init_f['_spin_sites'])), **kw)
    theta = np.asarray(tz['gauss_params']['value'])                                   # (5, Bm, Bq, Bz)
    assert theta.shape[-1] == BZ and np.ptp(theta[0], axis=-1).max() > 1e-3          # varies along z
    e_edges, p_edges, e_c, p_c = spin_grid()
    window = spin_window_fraction(np.asarray(pp.bin_axes[1]), e_edges, p_edges)
    log_window = np.where(window > 0, np.log(np.maximum(window, 1e-300)), NEG)
    support = np.where(log_window > NEG / 2, 0., NEG)
    # samples outside the grid carry an out-of-range bin (zero weight; jax clamps the index): clip, and drop them below
    im, iq = (np.clip(np.asarray(x), 0, n - 1) for x, n in zip(pp.event_bins, pp.bins))
    ie, ip = spin_bins(pp.posteriors, e_edges, p_edges)
    z_edges = np.linspace(0., 1.9, BZ + 1)
    iz = np.clip(np.digitize(np.asarray(pp.posteriors['redshift']), z_edges) - 1, 0, BZ - 1)
    par = theta[:, im, iq, iz]                                                         # (5, Nobs, NPE)
    kern = np.asarray(gauss_log_kernel(jnp.asarray(par)[..., None, None], e_c[:, None], p_c[None, :]))
    log_z = np.log(np.exp(kern + (support + log_window)[iq]).sum((-2, -1)))
    expect = kern[np.arange(im.shape[0])[:, None], np.arange(im.shape[1])[None], ie, ip] + support[iq, ie, ip] - log_z
    # the flat model's per-sample spin term is -ln(window area), the rest of the weight is shared
    flat_spin = -np.log(np.exp(log_window[iq]).sum((-2, -1)))
    got = np.asarray(tz['event_log_weights']['value']) - (np.asarray(tf['event_log_weights']['value']) - flat_spin)
    ok = np.isfinite(got) & (np.asarray(pp.posteriors['log_prior']) < np.inf)
    np.testing.assert_allclose(got[ok], expect[ok], rtol=1e-5, atol=1e-4)


def test_field_chunking_and_redshift_variation(monkeypatch):
    """The field's per-sample evaluation does not depend on its chunk size, and z modes change the conditional
    along z."""
    pp, kw = synthetic()
    rng = np.random.default_rng(12)

    def build():
        m, init = setup_model(pp, spin_mass_modes=KM, n_spin_bins=SPIN_BINS, spin_q_modes=KQ, spin_z_modes=KZ,
                              z_bins=BZ)
        return m, dict(common_latents(init), _spin_sites=jnp.asarray(rng.normal(size=np.shape(init['_spin_sites']))),
                       lnsigma_spin=jnp.asarray(1.))
    m, lat = build()
    t = trace_at(m, lat, **kw)
    a = float(t['log_likelihood']['value'])
    mom = np.asarray(t['spin_moments']['value'])                                        # (Bm, Bq, Bz, 4)
    assert mom.shape[2] == BZ and np.ptp(mom[:, -1, :, 0], axis=1).max() > 1e-2
    monkeypatch.setattr(M, 'Z_CHUNK', 37)
    m2, _ = setup_model(pp, spin_mass_modes=KM, n_spin_bins=SPIN_BINS, spin_q_modes=KQ, spin_z_modes=KZ, z_bins=BZ)
    np.testing.assert_allclose(lnl(m2, lat, kw), a, rtol=1e-6, atol=1e-4)


def test_additive_gauss():
    """spin_z_additive: with G(z) = 0 it is the mass-only Gaussian; otherwise every (m1, q) gets the same shift
    G(z), and the per-sample conditional is the z-dependent Gaussian's at theta = F + G."""
    pp, kw = synthetic()
    a, init_a = setup_model(pp, spin_mass_modes=KM, n_spin_bins=SPIN_BINS, spin_model='gauss')
    b, init_b = setup_model(pp, spin_mass_modes=KM, n_spin_bins=SPIN_BINS, spin_model='gauss', spin_z_modes=KZ,
                            z_bins=BZ, spin_z_additive=True)
    rng = np.random.default_rng(13)
    white = jnp.asarray(rng.normal(size=np.shape(init_a['_gauss_sites'])))
    lat_a = dict(common_latents(init_a), _gauss_sites=white)
    lat_b = dict(common_latents(init_b), _gauss_sites=white, _gauss_z_sites=jnp.zeros_like(init_b['_gauss_z_sites']))
    np.testing.assert_allclose(lnl(b, lat_b, kw), lnl(a, lat_a, kw), rtol=1e-6, atol=1e-4)
    lat_b['_gauss_z_sites'] = jnp.asarray(rng.normal(size=np.shape(init_b['_gauss_z_sites'])))
    t = trace_at(b, lat_b, **kw)
    F, G = np.asarray(t['gauss_params']['value']), np.asarray(t['gauss_params_z']['value'])
    assert F.shape[1:] == tuple(pp.bins) and G.shape == (5, BZ) and np.ptp(G, axis=1).min() > 1e-3
    np.testing.assert_allclose(G.mean(1), 0., atol=1e-6)                               # mean mode pinned
    assert 'spin_moments' not in t
    assert np.isfinite(float(t['log_likelihood']['value']))
