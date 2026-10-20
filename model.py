r"""Weakly modelled binary black hole population: a PixelPop mass field times a slowly evolving spin conditional.

Probability-space population over (ln m1, q, chi_eff, chi_p):

    p(ln m1, q, chi_eff, chi_p) = p_mass(ln m1, q) p(chi_eff, chi_p | ln m1, q)
    p_mass ∝ exp L(ln m1, q)                                         (PixelPop's Matern field, all modes)
    p(chi | m1, q) ∝ exp F(ln m1, q, chi) on the physical window chi_p < chi_p,max(chi_eff, q)

* No gates, no families, no second-generation template: any structure (gaps, peaks, a high-spin population at some
  masses) has to appear in the two fields.
* F is a 4D Matern field factorized over the mass plane: F = sum_kl v_k(ln m1) v_l(q) G_kl(chi), the lowest
  ``spin_mass_modes`` (k) Laplacian eigenmodes along ln m1 and along q, all modes along chi_eff and chi_p. So the spin
  distribution evolves slowly with mass (k = 10: features down to ~1/10 of each mass axis), with no mass-independent
  template. Modes with zero spin wavenumber are constant in chi and cancel in the per-(m1, q) normalization, so they
  are pinned to zero.
* The spin field is a log density on the allowed region: each (chi_eff, chi_p) bin's allowed-area fraction f
  (averaged over q within the q bin) enters the normalization only, sum_b rho_b f_b = 1; a zero field is flat on the
  allowed region (no isotropic or prior mean).
* Hyperpriors are PixelPop's: ln sigma ~ N(0, 2), ln nu ~ N(0, 0.5), ranges log-uniform from 1 bin (mass axes) or
  ``spin_range_floor`` in chi (spin axes) to the axis length.
* Likelihood: rate-marginalized (pixelpop's hierarchical_likelihood), with the event Neff taper and, optionally, the
  selection-bias correction N(N+1) sigma_sel^2 / 2; redshift and the m2 > 3 Msun window are PixelPopData's parametric
  models.

Extracted from mass-gap-investigation/code/pixelpop/hierarchical.py (mass_gate='none', frac_2g_gate='none', spin_mean
'none', spin_window 'analytic'); tests/test_model.py checks the two give the same likelihood.
"""
from functools import reduce

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import logsumexp as LSE

import pixelpop
from pixelpop.models.base_models import smooth
from pixelpop.models.gwpop_models import hierarchical_likelihood
from pixelpop.models.reparameterization import ordered_pair_bounds

# finite floor for unreachable spin bins: exp(NEG) is exactly 0, while logaddexp of two floors keeps finite gradients
NEG = -1e4
SPIN_REF_MASSES = (10., 18., 35., 60.)     # primary masses (Msun) of the saved spin slices at q ~ 1
L_SITES = ('lnsigma', 'log_nu_spde', 'log_ranges')
SPIN_SITES = ('lnsigma_spin', 'log_nu_spin', 'log_ranges_spin')


# ---------------------------------------------------------------- Matern fields on the grid's Laplacian eigenbasis
def axis_eigs(adj_matrices):
    """Per-axis path-graph Laplacian eigenpairs, DC eigenvalue pinned to 0."""
    eigs = []
    for adj in adj_matrices:
        adj = np.asarray(adj.toarray())
        w, v = np.linalg.eigh(np.diag(adj.sum(axis=1)) - adj)
        w[0] = 0.
        eigs.append((jnp.asarray(w), jnp.asarray(v)))
    return eigs


def matern_spectrum(eigvals, log_sigma, log_ranges, nu, n_sites, pin=None):
    """Per-mode Matern SPDE amplitude, (1 + sum_d rho_d^2 lambda^(d) / (8 nu))^(-(nu + D/2)/2), DC mode (and ``pin``)
    zeroed, scaled so the field's site-averaged variance over ``n_sites`` sites is exp(2 log_sigma)."""
    dim = len(eigvals)
    log_ranges = jnp.broadcast_to(log_ranges, (dim,))
    g = jnp.exp(2 * log_ranges) / (8 * nu)
    lam = reduce(jnp.add.outer, [g[i] * w for i, w in enumerate(eigvals)])
    spectrum = (1. + lam) ** (-(nu + dim / 2) / 2)
    spectrum = spectrum.at[(0,) * dim].set(0.)
    if pin is not None:
        spectrum = jnp.where(pin, 0., spectrum)
    return jnp.exp(log_sigma) * jnp.sqrt(n_sites / jnp.sum(spectrum ** 2)) * spectrum


def to_grid(amplitude, white, eigvecs):
    """Inverse transform: sum_k amplitude_k white_k (v_k1 x v_k2 x ...)."""
    res = amplitude * white
    for v in eigvecs:
        res = jnp.tensordot(res, v, axes=(0, 1))
    return res


# ---------------------------------------------------------------- spin support
def chi_p_max(chi_eff, q, amax=1.):
    """Largest chi_p reachable at (chi_eff, q): sqrt(amax^2 - x^2), x = clip((1 + q)|chi_eff| - amax q, 0, amax)
    (Eq. 12 of arXiv:2412.14551)."""
    x = np.clip((1 + q) * np.abs(chi_eff) - amax * q, 0., amax)
    return np.sqrt(amax ** 2 - x ** 2)


def spin_window_fraction(q_edges, chi_eff_edges, chi_p_edges, n_sub_e=64, n_sub_q=8):
    """(n_q, n_chieff, n_chip): fraction of each (chi_eff, chi_p) bin's area below chi_p_max(chi_eff, q), averaged over q
    within each q bin. Zero only where no q in the bin reaches the bin."""
    t = lambda n: (np.arange(n) + 0.5) / n
    e_sub = chi_eff_edges[:-1, None] + np.diff(chi_eff_edges)[:, None] * t(n_sub_e)[None]
    p_lo, dp = chi_p_edges[:-1], np.diff(chi_p_edges)
    out = []
    for qlo, qhi in zip(q_edges[:-1], q_edges[1:]):
        fq = [np.clip((chi_p_max(e_sub, q)[:, :, None] - p_lo) / dp, 0., 1.).mean(1)
              for q in qlo + (qhi - qlo) * np.linspace(0, 1, n_sub_q)]
        out.append(np.mean(fq, 0))
    return np.stack(out)


def spin_bins(data, chi_eff_edges, chi_p_edges):
    """(chi_eff, chi_p) bin indices of every sample."""
    ie = np.clip(np.digitize(np.asarray(data['chi_eff']), chi_eff_edges) - 1, 0, len(chi_eff_edges) - 2)
    ip = np.clip(np.digitize(np.asarray(data['chi_p']), chi_p_edges) - 1, 0, len(chi_p_edges) - 2)
    return ie, ip


# ---------------------------------------------------------------- the model
def setup_model(pixelpop_data, spin_mass_modes=10, n_spin_bins=(100, 100), spin_range_floor=0.05,
                likelihood_correction=True, record_weights=False, _dense_cube=False):
    """NumPyro model and initial values. ``pixelpop_data`` pixelates the mass plane (log_mass_1, mass_ratio) with
    ``spde_matern=True``; its posteriors and injections carry chi_eff and chi_p, binned here onto ``n_spin_bins`` over
    [-1, 1] x [0, 1]. ``spin_mass_modes`` = 0 makes the spins independent of mass (a 2D field on chi). ``_dense_cube``
    builds the (Bm, Bq, Be, Bp) cube (test hook)."""
    pp = pixelpop_data
    if list(pp.pixelpop_parameters) != ['log_mass_1', 'mass_ratio']:
        raise ValueError(f'pixelpop_parameters must be [log_mass_1, mass_ratio], got {pp.pixelpop_parameters}')
    if ordered_pair_bounds(pp.priors):
        raise NotImplementedError('pin mlow_1 / mlow_2 with Delta priors')
    Bm, Bq = (int(b) for b in pp.bins)
    Be, Bp = (int(b) for b in n_spin_bins)
    km = int(spin_mass_modes)
    if km > min(Bm, Bq):
        raise ValueError(f'spin_mass_modes {km} exceeds the mass grid {Bm} x {Bq}')

    spin_adj = [pixelpop.utils.create_CAR_coupling_matrix(n, 1, isVisible=False) for n in (Be, Bp)]
    (wm, vm), (wq, vq), (we, ve), (wp, vp) = axis_eigs(pp.adj_matrices) + axis_eigs(spin_adj)
    edges = [np.asarray(a) for a in pp.bin_axes] + [np.linspace(-1., 1., Be + 1), np.linspace(0., 1., Bp + 1)]
    lnm_c = 0.5 * (edges[0][1:] + edges[0][:-1])

    # spin bins of every sample; samples in spin bins no injection reaches get zero weight
    ev_e, ev_p = spin_bins(pp.posteriors, edges[2], edges[3])
    in_e, in_p = spin_bins(pp.injections, edges[2], edges[3])
    covered = np.zeros((Be, Bp), bool); covered[in_e, in_p] = True
    flag = ~covered[ev_e, ev_p]
    if flag.any():
        print(f'[model] {flag.mean():.2e} of PE samples are in spin bins with no injections; giving them zero weight')
        pp.posteriors['log_prior'] = jnp.where(jnp.asarray(flag), jnp.inf, pp.posteriors['log_prior'])
    event_index = (*pp.event_bins, *(jnp.asarray(x) for x in (ev_e, ev_p)))
    inj_index = (*pp.inj_bins, *(jnp.asarray(x) for x in (in_e, in_p)))

    # physical window: log of each bin's allowed-area fraction (normalization only) and the support (0 / NEG)
    window = spin_window_fraction(edges[1], edges[2], edges[3])                    # (Bq, Be, Bp)
    log_window = jnp.asarray(np.where(window > 0, np.log(np.maximum(window, 1e-300)), NEG))
    log_support = jnp.where(log_window > NEG / 2, 0., NEG)
    for what, (iq, ie, ip) in (('PE samples', (pp.event_bins[1], ev_e, ev_p)), ('injections', (pp.inj_bins[1], in_e, in_p))):
        out = np.asarray(window[np.clip(np.asarray(iq), 0, Bq - 1), np.asarray(ie), np.asarray(ip)] == 0)
        if out.any():
            print(f'[model] {out.mean():.2e} of {what} fall outside the chi_p window')

    # spin range priors: log-uniform from a physical floor (in chi) to the axis length, in bins
    spin_lo = [float(np.log(max(spin_range_floor / d, 1.))) for d in (2. / Be, 1. / Bp)]
    pin = np.zeros((km, km, Be, Bp), bool) if km else None
    if km:
        pin[:, :, 0, 0] = True
        pin = jnp.asarray(pin)
    Vm, Vq = jnp.asarray(vm[:, :km]), jnp.asarray(vq[:, :km])
    sel_m = jnp.asarray([int(np.argmin(np.abs(np.exp(lnm_c) - m))) for m in SPIN_REF_MASSES])

    def matern_field(white_site, shape, vals, range_vecs, lsigma, nu_site, range_site, n_sites, pin=None, lo=None,
                     grid_vecs=None):
        white = numpyro.sample(white_site, dist.Normal(0., 1.).expand(shape))
        nu = jnp.exp(numpyro.sample(nu_site, pp.smoothness_prior[1](*pp.smoothness_prior[0])))
        hi = jnp.log(jnp.asarray([len(v) for v in range_vecs], dtype=float))
        lo = jnp.zeros_like(hi) if lo is None else jnp.asarray(lo, dtype=float)
        log_ranges = numpyro.sample(range_site, dist.Uniform(lo, hi))
        amp = matern_spectrum(vals, lsigma, log_ranges, nu, n_sites, pin=pin)
        return to_grid(amp, white, range_vecs if grid_vecs is None else grid_vecs)

    def parametric(posteriors, injections, event_weights, inj_weights):
        sample = {}
        for key, (args, distribution) in pp.priors.items():
            sample[key] = args[0] if distribution.__name__ == 'Delta' else numpyro.sample(key, distribution(*args))
        for p in pp.other_parameters:
            hyp = [sample[h] for h in pp.parameter_to_hyperparameters[p]]
            event_weights += pp.parametric_models[p](posteriors, *hyp)
            inj_weights += pp.parametric_models[p](injections, *hyp)
        return event_weights, inj_weights

    def probabilistic_model(posteriors, injections):
        # ---- mass: PixelPop's field (DC mode dropped; probability space, so no rate offset) ----
        lnsigma = numpyro.sample('lnsigma', pp.coupling_prior[1](*pp.coupling_prior[0]))
        L = matern_field('_eigenbasis_sites', [Bm, Bq], [wm, wq], [vm, vq], lnsigma, 'log_nu_spde', 'log_ranges', Bm * Bq)
        log_tot = LSE(L)
        numpyro.deterministic('log_p_mass', L - log_tot - pp.logdV[0] - pp.logdV[1])
        w_mass = jnp.exp(L - log_tot)                                                  # (Bm, Bq), sums to 1

        # ---- spin: chi field, conditional on (ln m1, q) through km x km mass modes ----
        lnsigma_s = numpyro.sample('lnsigma_spin', pp.coupling_prior[1](*pp.coupling_prior[0]))
        if km:
            G = matern_field('_spin_sites', [km, km, Be, Bp], [wm[:km], wq[:km], we, wp], [Vm, Vq, ve, vp], lnsigma_s,
                             'log_nu_spin', 'log_ranges_spin', Bm * Bq * Be * Bp, pin=pin, lo=[0., 0.] + spin_lo,
                             grid_vecs=[jnp.eye(km), jnp.eye(km), ve, vp])               # (km, km, Be, Bp)
        else:
            G = matern_field('_spin_sites', [Be, Bp], [we, wp], [ve, vp], lnsigma_s, 'log_nu_spin', 'log_ranges_spin',
                             Be * Bp, lo=spin_lo)                                         # (Be, Bp)

        if km and not _dense_cube:
            # ln Z(m, q) = LSE_chi [F + support + window], one m slab (Bq, Be, Bp) at a time; the population spin
            # marginal accumulates in the same pass
            @jax.checkpoint
            def slab(m):
                T = jnp.einsum('k,ql,klep->qep', Vm[m], Vq, G) + log_support
                lz = LSE(T + log_window, axis=(1, 2))
                return lz, jnp.einsum('q,qep->ep', w_mass[m], jnp.exp(T - lz[:, None, None] + log_window))
            log_z, contribs = jax.lax.map(slab, jnp.arange(Bm), batch_size=10)            # (Bm, Bq), (Bm, Be, Bp)

            def log_p_at(idx):
                im, iq, ie, ip = idx
                f = jnp.einsum('...k,...l,kl...->...', Vm[im], Vq[iq], G[:, :, ie, ip])
                return L[im, iq] - log_tot + f + log_support[iq, ie, ip] - log_z[im, iq]
            log_spin_marg = jnp.log(contribs.sum(0) + 1e-300)
            T_sel = jnp.einsum('mk,l,klep->mep', Vm[sel_m], Vq[-1], G) + log_support[-1][None]
            numpyro.deterministic('log_p_spin_q1_m', T_sel - log_z[sel_m, -1][:, None, None] + log_window[-1][None])
        else:
            F = (jnp.einsum('mk,ql,klep->mqep', Vm, Vq, G) if km else G[None, None]) + log_support[None]
            log_ps = F - LSE(F + log_window[None], axis=(2, 3), keepdims=True)          # (Bm or 1, Bq, Be, Bp)
            log_p = (L - log_tot)[:, :, None, None] + log_ps
            log_spin_marg = LSE(log_p + log_window[None], axis=(0, 1))
            log_p_at = lambda idx: log_p[idx] if km else (L - log_tot)[idx[0], idx[1]] + log_ps[0][idx[1:]]
            numpyro.deterministic('log_p_spin_q1_m', jnp.broadcast_to(log_ps[:, -1] + log_window[-1][None],
                                                                      (len(SPIN_REF_MASSES), Be, Bp)) if not km
                                  else log_ps[sel_m, -1] + log_window[-1][None])
        numpyro.deterministic('log_p_spin_marginal', log_spin_marg)

        event_weights = posteriors['ln_dVTc'] - posteriors['log_prior'] + log_p_at(event_index)
        inj_weights = injections['ln_dVTc'] - injections['log_prior'] + log_p_at(inj_index)
        event_weights, inj_weights = parametric(posteriors, injections, event_weights, inj_weights)
        if record_weights:     # post-processing only (error statistics)
            numpyro.deterministic('event_log_weights', event_weights)
            numpyro.deterministic('injection_log_weights', inj_weights)

        lnL, var, _, (pe_v, vt_v) = hierarchical_likelihood(
            event_weights, inj_weights, injections['total_generated'], live_time=injections['analysis_time'],
            rate_likelihood=False, return_likelihood_info=True, event_counts=pp.event_counts)
        taper = smooth(var, pp.UncertaintyCut ** 2, 0.1) if np.isfinite(pp.UncertaintyCut) else 0.
        cnt = jnp.asarray(pp.event_counts, dtype=float)
        num = LSE(event_weights, axis=1); sq = LSE(2 * event_weights, axis=1)
        event_neff = 1. / (jnp.exp(sq - 2 * num) - 1. / cnt)
        selection_neff = jnp.exp(2 * LSE(inj_weights) - LSE(2 * inj_weights))
        numpyro.deterministic('event_neff', event_neff)
        numpyro.deterministic('selection_neff', selection_neff)
        if pp.EventNeffCut > 0.:
            taper = taper + jnp.sum(smooth(-jnp.log(event_neff), -jnp.log(pp.EventNeffCut), 0.1))
        # selection-bias correction (arXiv:2509.07221): 1 / xi^Nobs is biased high by 1 + Nobs (Nobs + 1) sigma^2 / 2
        n_inj = injections['total_generated']; n_obs = len(pp.event_counts)
        sel_sigma2 = (n_inj * jnp.exp(LSE(2 * inj_weights) - 2 * LSE(inj_weights)) - 1.) / (n_inj - 1.)
        log_correction = n_obs * (n_obs + 1.) * sel_sigma2 / 2.
        numpyro.deterministic('log_likelihood_correction', log_correction)
        if likelihood_correction:
            taper = taper - log_correction
        numpyro.deterministic('log_likelihood', lnL)
        numpyro.deterministic('log_likelihood_variance', var)
        numpyro.deterministic('pe_variance', pe_v)
        numpyro.deterministic('vt_variance', vt_v)
        numpyro.deterministic('detection_efficiency', jnp.exp(LSE(inj_weights) - jnp.log(injections['total_generated'])))
        numpyro.factor('log_likelihood_plus_taper', lnL + taper)

    log_bins = lambda shape: jnp.clip(jnp.log(jnp.asarray(shape, float)) - jnp.log(4.), 0.3, None)
    rng = np.random.default_rng(0)
    initial_value = {
        '_eigenbasis_sites': jnp.asarray(rng.normal(size=(Bm, Bq))),
        'lnsigma': jnp.asarray(0.), 'log_nu_spde': jnp.asarray(0.), 'log_ranges': log_bins([Bm, Bq]),
        '_spin_sites': jnp.asarray(0.1 * rng.normal(size=(km, km, Be, Bp) if km else (Be, Bp))),
        'lnsigma_spin': jnp.asarray(0.), 'log_nu_spin': jnp.asarray(0.),
        'log_ranges_spin': log_bins([Bm, Bq, Be, Bp] if km else [Be, Bp]),
    }
    for key, (args, distribution) in pp.priors.items():      # parametric hyperparameters (redshift) at their prior means
        if distribution.__name__ != 'Delta':
            initial_value[key] = jnp.asarray(distribution(*args).mean)
    return probabilistic_model, initial_value


def mass_matrix_blocks():
    """Dense mass-matrix blocks for NUTS: each field's hyperparameters together (latent white sites stay diagonal)."""
    return [L_SITES, SPIN_SITES]
