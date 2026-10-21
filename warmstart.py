"""Warm start: initial values for run.py --init-from, copied from one sample of an earlier fit with the same fields.

The mass field (_eigenbasis_sites, lnsigma, log_nu_spde, log_ranges) is copied as is (same 100 x 100 (ln m1, q) grid).
The spin field's white sites (k_src x k_src mass modes) are embedded in the lowest modes of this model's k x k block,
zero elsewhere; its hyperparameters are copied. The source's site names may be mass-gap-investigation's hierarchical
model (_spin_base_sites, lnsigma_base, ...) or this repository's (_spin_sites, lnsigma_spin, ...). The source sample is
the one with the highest corrected ln L. Prints ln L at the warm-start point.

For --spin-model split / gauss the source's spin field is first mapped onto this model's mass modes as a 'field'
start (above), then converted (convert_split, convert_gauss):

* split: nearly exactly. The mass-mean mode of the field becomes G0, the rest G1. Each part's hyperparameters (nu,
  ranges; lnsigma profiled) are fitted to its mode coefficients by maximum likelihood (fit_spectrum), so that its white
  sites are as close to unit normal as its spectrum allows; white sites beyond +-4 (high spin modes with negligible
  field amplitude) are clipped and the fit repeated (robust_fit). The warm start prints its ln L next to the field
  start's.
* gauss: approximately. The Gaussian's parameter fields are set from the field start's per-cell moments of the spin
  conditional (mean, sd of chi_eff and chi_p; rho = 0), each projected onto the k x k mass modes, with its ranges
  fitted the same way.

    python warmstart.py <run.py arguments> --source <chain_*_samples.h5> --out <npz>
"""
import os
import sys

import run

ap = run.make_parser()
ap.add_argument('--source', required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()
run.setup_environment(args)

import h5py  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402

import model  # noqa: E402

prob_model, target_init, posteriors, injections, pp = run.build(args, 'warmstart')
if args.spin_model == 'field':
    field_model, init = prob_model, target_init
else:
    # the field start uses this model's k, except for gauss, whose k may be the full grid (100): a field model with
    # 100 x 100 mass modes has 1e8 sites, so gauss starts from a field with the source's own k (its spin sites' shape)
    k_field = args.spin_mass_modes
    if args.spin_model == 'gauss':
        with h5py.File(args.source) as f:
            key = '_spin_sites' if '_spin_sites' in f else '_spin_base_sites'
            shape = f[key].shape[1:]
        k_field = min(k_field, shape[0]) if len(shape) == 4 else k_field
        print(f'gauss: field start with k = {k_field} (source spin sites {shape})', flush=True)
    field_model, init = model.setup_model(pp, spin_mass_modes=k_field, n_spin_bins=tuple(args.spin_bins),
                                          likelihood_correction=not args.no_likelihood_correction)
with h5py.File(args.source) as f:
    Lc = f['log_likelihood'][()] - f['log_likelihood_correction'][()]
    i = int(np.argmax(Lc))
    src = {k: f[k][i] for k in f if f[k].shape[:1] == Lc.shape}
print(f'source {args.source}: sample {i}, corrected ln L {Lc[i]:.2f}', flush=True)

val = dict(init)
for k in ('_eigenbasis_sites', 'lnsigma', 'log_nu_spde', 'log_ranges'):
    val[k] = jnp.asarray(src[k])
sfx = '_base' if '_spin_base_sites' in src else '_spin'
w_src = np.asarray(src[f'_spin{sfx}_sites' if sfx == '_base' else '_spin_sites'])
w = np.zeros(np.shape(init['_spin_sites']))
if w.ndim == 4 and w_src.ndim == 4:
    k = min(w.shape[0], w_src.shape[0])
    w[:k, :k] = w_src[:k, :k]
elif w.ndim == w_src.ndim:
    w = w_src
else:
    sys.exit(f'cannot map spin sites of shape {w_src.shape} onto {w.shape}')
for a, b in (('lnsigma', 'lnsigma_spin'), ('log_nu', 'log_nu_spin'), ('log_ranges', 'log_ranges_spin')):
    val[b] = jnp.asarray(src[f'{a}{sfx}' if sfx == '_base' else b])
if w.ndim == 4 and w_src.ndim == 4:
    # the per-mode amplitude depends on how many modes share the variance: rescale the copied white sites so the spin
    # field F = sum a_kl w_kl ... equals the source's on its modes
    import pixelpop
    from model import axis_eigs, matern_spectrum
    Bm, Bq = (int(x) for x in pp.bins); Be, Bp = args.spin_bins
    spin_adj = [pixelpop.utils.create_CAR_coupling_matrix(n, 1, isVisible=False) for n in (Be, Bp)]
    (wm, _), (wq, _), (we, _), (wp, _) = axis_eigs(pp.adj_matrices) + axis_eigs(spin_adj)

    def amp(kk):
        pin = np.zeros((kk, kk, Be, Bp), bool); pin[:, :, 0, 0] = True
        return np.asarray(matern_spectrum([wm[:kk], wq[:kk], we, wp], val['lnsigma_spin'], val['log_ranges_spin'],
                                          jnp.exp(val['log_nu_spin']), Bm * Bq * Be * Bp, pin=jnp.asarray(pin)))
    a_src, a_new = amp(w_src.shape[0]), amp(w.shape[0])
    k = min(w.shape[0], w_src.shape[0])
    ok = a_new[:k, :k] > 0
    w[:k, :k] = np.where(ok, w_src[:k, :k] * a_src[:k, :k] / np.where(ok, a_new[:k, :k], 1.), 0.)
val['_spin_sites'] = jnp.asarray(w)
for k in init:
    if k in src and k not in val and np.shape(src[k]) == np.shape(init[k]):
        val[k] = jnp.asarray(src[k])
if 'lamb' in src:
    val['lamb'] = jnp.asarray(src['lamb'])



def trace(m, values):
    return numpyro.handlers.trace(numpyro.handlers.substitute(numpyro.handlers.seed(m, 0), data=values)).get_trace(
        posteriors=posteriors, injections=injections)


def white_for(coeff, amp):
    """White sites reproducing ``coeff`` = amp * white with amplitudes ``amp`` (at lnsigma = 0), and the lnsigma that
    gives them unit mean square over the modes with amp > 0."""
    ok = amp > 0
    w = np.where(ok, coeff / np.where(ok, amp, 1.), 0.)
    lnsig = 0.5 * np.log(np.mean(w[ok] ** 2))
    return w / np.exp(lnsig), lnsig


def fit_spectrum(coeff, amp_fn, x0, bounds):
    """Hyperparameters x maximizing the likelihood of mode coefficients ``coeff`` ~ N(0, sigma^2 amp_fn(x)^2), sigma
    profiled out; modes with zero amplitude (pinned) are skipped."""
    from scipy.optimize import minimize

    def nll(x):
        a = amp_fn(x); ok = a > 0
        return 0.5 * ok.sum() * np.log(np.mean(coeff[ok] ** 2 / a[ok] ** 2)) + np.sum(np.log(a[ok]))
    # strictly inside the prior bounds: a value on a Uniform's edge is infinite in NUTS's unconstrained space
    bounds = [(lo + 1e-3, hi - 1e-3) for lo, hi in bounds]
    x0 = np.clip(np.asarray(x0, dtype=float), [b[0] for b in bounds], [b[1] for b in bounds])
    # derivative-free: the spectra are float32, too coarse for finite-difference gradients
    res = minimize(lambda x: float(nll(x)), x0, method='Nelder-Mead', bounds=bounds,
                   options=dict(maxiter=600, xatol=1e-3, fatol=1e-3))
    print(f'  spectrum fit: {np.round(x0, 2)} -> {np.round(res.x, 2)}, nll {nll(x0):.1f} -> {res.fun:.1f}', flush=True)
    return res.x


WHITE_CLIP = 4.


def robust_fit(coeff, amp_fn, x0, bounds, rounds=1):
    """fit_spectrum, then clip white sites beyond +-WHITE_CLIP (modes whose coefficient is tiny but the new spectrum
    tinier: they barely touch the field, but would dominate the prior), so |white| <= WHITE_CLIP. Refitting after the
    clip (rounds > 1) shortens the spin ranges and clips ever more sites: on the gwtc5 data from the mass-gap bm3
    source, 5 rounds moved ln L from -7335 to -7552, so the default is 1. Returns (white, lnsigma, x)."""
    x, total = np.asarray(x0, dtype=float), 0
    for _ in range(rounds):
        x = fit_spectrum(coeff, amp_fn, x, bounds)
        w, lnsig = white_for(coeff, amp_fn(x))
        n_clip = int(np.sum(np.abs(w) > WHITE_CLIP))
        if not n_clip:
            break
        total += n_clip
        coeff = np.clip(w, -WHITE_CLIP, WHITE_CLIP) * np.exp(lnsig) * amp_fn(x)
    w, lnsig = white_for(coeff, amp_fn(x))
    w = np.clip(w, -WHITE_CLIP, WHITE_CLIP)
    print(f'  clipped {total} white-site values over the rounds (of {w.size} sites); max |white| {np.abs(w).max():.2f}',
          flush=True)
    return w, lnsig, x


def convert_split(val):
    """Field start -> spin_model='split' latents, with the same spin field."""
    import pixelpop
    from model import axis_eigs, matern_spectrum
    Bm, Bq = (int(x) for x in pp.bins); Be, Bp = args.spin_bins; km = args.spin_mass_modes
    spin_adj = [pixelpop.utils.create_CAR_coupling_matrix(n, 1, isVisible=False) for n in (Be, Bp)]
    (wm, vm), (wq, vq), (we, _), (wp, _) = axis_eigs(pp.adj_matrices) + axis_eigs(spin_adj)
    pin = np.zeros((km, km, Be, Bp), bool); pin[:, :, 0, 0] = True
    nu, r = jnp.exp(val['log_nu_spin']), np.asarray(val['log_ranges_spin'])
    coeff = np.asarray(matern_spectrum([wm[:km], wq[:km], we, wp], val['lnsigma_spin'], r, nu, Bm * Bq * Be * Bp,
                                       pin=jnp.asarray(pin))) * np.asarray(val['_spin_sites'])   # (km, km, Be, Bp)
    out = {k: v for k, v in val.items() if k not in ('_spin_sites', 'lnsigma_spin', 'log_nu_spin', 'log_ranges_spin')}
    spin_lo = [float(np.log(max(0.05 / d, 1.))) for d in (2. / Be, 1. / Bp)]          # model.setup_model's floor
    nu_b, hi = (-1.5, 1.5), [float(np.log(n)) for n in (Bm, Bq, Be, Bp)]
    # G0's spin-mode coefficients: the field's mass-mean mode times the (constant) lowest mass eigenvectors
    c0 = coeff[0, 0] * float(vm[0, 0]) * float(vq[0, 0])
    amp0 = lambda x: np.asarray(matern_spectrum([we, wp], 0., x[1:], np.exp(x[0]), Be * Bp))
    out['_spin0_sites'], lnsig0, x0 = robust_fit(c0, amp0, [float(val['log_nu_spin']), *r[2:]],
                                                 [nu_b, (spin_lo[0], hi[2]), (spin_lo[1], hi[3])])
    out.update(lnsigma_spin0=jnp.asarray(lnsig0), log_nu_spin0=jnp.asarray(x0[0]), log_ranges_spin0=jnp.asarray(x0[1:]))
    c1 = coeff.copy(); c1[0, 0] = 0.

    def amp1(x):
        s_mass = np.asarray(matern_spectrum([wm[:km], wq[:km]], 0., x[1:3], np.exp(x[0]), 1))
        s_spin = np.asarray(matern_spectrum([we, wp], 0., x[3:], np.exp(x[0]), 1))
        spec = s_mass[:, :, None, None] * s_spin[None, None]
        return np.sqrt(Bm * Bq * Be * Bp / np.sum(spec ** 2)) * spec
    out['_spin1_sites'], lnsig1, x1 = robust_fit(c1, amp1, [float(val['log_nu_spin']), *r],
                                                 [nu_b, (0., hi[0]), (0., hi[1]), (spin_lo[0], hi[2]), (spin_lo[1], hi[3])])
    out.update(lnsigma_spin1=jnp.asarray(lnsig1), log_nu_spin1=jnp.asarray(x1[0]), log_ranges_spin1=jnp.asarray(x1[1:]))
    for k in ('_spin0_sites', '_spin1_sites'):
        print(f'{k}: max |white| {np.abs(out[k]).max():.2f}, rms {np.sqrt(np.mean(out[k] ** 2)):.2f}', flush=True)
        out[k] = jnp.asarray(out[k])
    return out


def convert_gauss(val):
    """Field start -> spin_model='gauss' latents, from the field's per-cell spin moments."""
    import pixelpop
    from model import GAUSS_OFFSET_BOUNDS, GAUSS_SD_FLOOR, axis_eigs, matern_spectrum
    Bm, Bq = (int(x) for x in pp.bins); km = args.spin_mass_modes
    mom = np.asarray(trace(field_model, val)['spin_moments']['value'], dtype=float)         # (Bm, Bq, 4)
    ln_sd = lambda sd: 0.5 * np.log(np.maximum(sd ** 2 - GAUSS_SD_FLOOR ** 2, 1e-4))
    target = np.stack([mom[..., 0], mom[..., 2], ln_sd(mom[..., 1]), ln_sd(mom[..., 3]), np.zeros((Bm, Bq))])
    (wm, vm), (wq, vq) = axis_eigs(pp.adj_matrices)
    Vm, Vq = np.asarray(vm)[:, :km], np.asarray(vq)[:, :km]
    lo, hi = (np.asarray(b) for b in GAUSS_OFFSET_BOUNDS)
    offsets = np.clip(target.mean((1, 2)), lo + 1e-3, hi - 1e-3)                    # grid mean: no mean-mode residual
    r = np.tile(np.log([Bm / 4., Bq / 4.]), (len(target), 1))
    nu = 1.                                                                         # log_nu_gauss = 0
    white, lnsig = [], []
    for i, t in enumerate(target):
        coeff = Vm.T @ (t - offsets[i]) @ Vq                                           # (km, km)
        amp = lambda x: np.asarray(matern_spectrum([wm[:km], wq[:km]], 0., x, nu, Bm * Bq))
        if np.any(np.abs(coeff[amp(r[i]) > 0]) > 1e-8):
            w, ls, r[i] = robust_fit(coeff, amp, r[i], [(0., float(np.log(Bm))), (0., float(np.log(Bq)))])
        else:
            w, ls = np.zeros_like(coeff), float(model.GAUSS_LNSIGMA_CENTRES[i])
        white.append(w); lnsig.append(ls)
    out = {k: v for k, v in val.items() if k not in ('_spin_sites', 'lnsigma_spin', 'log_nu_spin', 'log_ranges_spin')}
    out.update(gauss_offsets=jnp.asarray(offsets), lnsigma_gauss=jnp.asarray(lnsig), log_nu_gauss=jnp.asarray(0.),
               log_ranges_gauss=jnp.asarray(r), _gauss_sites=jnp.asarray(np.stack(white)))
    print(f'gauss offsets {np.round(offsets, 3)}, lnsigma {np.round(lnsig, 2)}, '
          f'max |white| {np.abs(np.stack(white)).max():.2f}', flush=True)
    return out


if args.spin_model != 'field':
    g0 = trace(field_model, val)
    print(f'field start: ln L {float(g0["log_likelihood"]["value"]):.2f}', flush=True)
    val = (convert_split if args.spin_model == 'split' else convert_gauss)(val)
    missing = sorted(set(target_init) - set(val))
    if missing:
        raise SystemExit(f'conversion left sites unset: {missing}')


def into_support(model_, values):
    """Move values on or outside a bounded support just inside it, as NUTS will need them (finite in unconstrained
    space), and check the potential energy and its gradient there. Exits if either is not finite."""
    import jax
    from numpyro.distributions import constraints
    from numpyro.distributions.transforms import biject_to
    from numpyro.infer.util import potential_energy, unconstrain_fn
    sites = {k: v for k, v in trace(model_, values).items() if v['type'] == 'sample' and not v.get('is_observed')}
    out = dict(values)
    for k, site in sites.items():
        support = site['fn'].support
        x = jnp.asarray(out[k])
        u = biject_to(support).inv(x)
        if bool(jnp.all(jnp.isfinite(u))):
            continue
        if isinstance(support, (constraints.interval, constraints.open_interval)) or hasattr(support, 'lower_bound'):
            lo, hi = jnp.broadcast_to(support.lower_bound, x.shape), jnp.broadcast_to(support.upper_bound, x.shape)
            eps = 1e-4 * (hi - lo)
            new = jnp.clip(x, lo + eps, hi - eps)
            print(f'moved {k} inside its support: {np.round(np.asarray(x), 6).tolist()} -> '
                  f'{np.round(np.asarray(new), 6).tolist()}', flush=True)
            out[k] = new
        else:
            sys.exit(f'{k}: value outside its support {support}, cannot fix')
    kw = dict(posteriors=posteriors, injections=injections)
    u = unconstrain_fn(model_, (), kw, {k: out[k] for k in sites})
    pe, grad = jax.value_and_grad(lambda z: potential_energy(model_, (), kw, z))(u)
    bad = [k for k, g_ in grad.items() if not bool(jnp.all(jnp.isfinite(g_)))]
    if not np.isfinite(float(pe)) or bad:
        sys.exit(f'potential energy {float(pe)} or gradient not finite at the warm start (sites {bad})')
    print(f'NUTS check: potential energy {float(pe):.2f}, gradient finite', flush=True)
    return out


val = into_support(prob_model, val)
tr = trace(prob_model, val)
g = lambda k: np.asarray(tr[k]['value'])
print(f'warm start: ln L {float(g("log_likelihood")):.2f}, corrected {float(g("log_likelihood") - g("log_likelihood_correction")):.2f}, '
      f'Var {float(g("log_likelihood_variance")):.2f}, min event Neff {g("event_neff").min():.0f}', flush=True)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
np.savez(args.out, **{k: np.asarray(v) for k, v in val.items()})
print(f'saved {args.out}')
