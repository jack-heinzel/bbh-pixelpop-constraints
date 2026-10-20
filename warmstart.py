"""Warm start: initial values for run.py --init-from, copied from one sample of an earlier fit with the same fields.

The mass field (_eigenbasis_sites, lnsigma, log_nu_spde, log_ranges) is copied as is (same 100 x 100 (ln m1, q) grid).
The spin field's white sites (k_src x k_src mass modes) are embedded in the lowest modes of this model's k x k block,
zero elsewhere; its hyperparameters are copied. The source's site names may be mass-gap-investigation's hierarchical
model (_spin_base_sites, lnsigma_base, ...) or this repository's (_spin_sites, lnsigma_spin, ...). The source sample is
the one with the highest corrected ln L. Prints ln L at the warm-start point.

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

prob_model, init, posteriors, injections, pp = run.build(args, 'warmstart')
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

tr = numpyro.handlers.trace(numpyro.handlers.substitute(numpyro.handlers.seed(prob_model, 0), data=val)).get_trace(
    posteriors=posteriors, injections=injections)
g = lambda k: np.asarray(tr[k]['value'])
print(f'warm start: ln L {float(g("log_likelihood")):.2f}, corrected {float(g("log_likelihood") - g("log_likelihood_correction")):.2f}, '
      f'Var {float(g("log_likelihood_variance")):.2f}, min event Neff {g("event_neff").min():.0f}', flush=True)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
np.savez(args.out, **{k: np.asarray(v) for k, v in val.items()})
print(f'saved {args.out}')
