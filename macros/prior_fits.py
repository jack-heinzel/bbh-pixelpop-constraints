"""The window fits of macros/peak_fits.py and macros/joint_fits.py on draws from the mass field's prior, as the reference
for how significant the posterior's peaks are (user, 2026-10-26).

    python macros/prior_fits.py [--n N] [--seed S] [<run> ...]   (default: ppc_gwtc5_k10 ppc_gwtc5_k100_gauss)
        -> results/post_processing/<run>_prior_fits.npz

Empirical-Bayes prior: for each posterior draw of the mass field's kernel parameters (lnsigma, log_nu_spde, log_ranges),
one draw of the white sites from N(0, 1), turned into a field exactly as model.py does (the Matern SPDE spectrum on the
grid's Laplacian eigenbasis, DC mode dropped), so the data enter only through the kernel parameters. Each field is
normalized, windowed (m2 > 3 Msun) and renormalized as analysis.load_light does with the posterior, then fitted
in every window: the 1D Gaussian + power law on p(m1 | q in Q) (peak_fits.fit_peak) and the joint (m1, q) fit
(joint_fits.fit_joint). Also saved, for prior and posterior: the fit-free mean and sd of q over each window's m1 range
(Q_WINDOWS). Windows run in parallel (one process each; set --cpus-per-task to match).
"""
import os
import sys
import time
from multiprocessing import get_context

import h5py
import numpy as np

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'notebooks')); sys.path.insert(0, HERE)
import analysis as A
from peak_fits import PEAKS, RUN_NAMES, fit_peak, fmt
from joint_fits import fit_joint, npz_path, q_moments

HYPER = ('lnsigma', 'log_nu_spde', 'log_ranges')
Q_WINDOWS = {'10': (8., 11.5), '35': (28., 37.), '20': (18., 24.), '60': (50., 70.)}   # tight around each peak


def prior_path(run):
    return os.path.join(REPO, 'results', 'post_processing', f'{run}_prior_fits.npz')


def load_hyper(run):
    """The mass field's kernel parameters, per posterior draw, in load_light's draw order."""
    import glob
    files = sorted(glob.glob(os.path.join(REPO, 'results', run, 'chain_*_samples.h5')))
    n_draws = min(len(h5py.File(f, 'r')['log_likelihood']) for f in files)
    return {k: np.concatenate([h5py.File(f, 'r')[k][:n_draws] for f in files]).astype(float) for k in HYPER}


def prior_p_mass(hyper, seed=0):
    """Windowed prior p(ln m1, q) per kernel draw, (draws, Bm, Bq), density per d ln m1 dq (as load_light's p_mass)."""
    import jax
    import jax.numpy as jnp
    from pixelpop.utils import create_CAR_coupling_matrix
    import model
    (wm, vm), (wq, vq) = model.axis_eigs([create_CAR_coupling_matrix(n, 1, isVisible=False) for n in (A.B, A.B)])

    @jax.jit
    def log_field(white, lnsigma, log_nu, log_ranges):
        amp = model.matern_spectrum([wm, wq], lnsigma, log_ranges, jnp.exp(log_nu), A.B * A.B)
        return model.to_grid(amp, white, [vm, vq])

    white = np.random.default_rng(seed).normal(size=(len(hyper['lnsigma']), A.B, A.B))
    lp = np.stack([np.asarray(log_field(w, s, n, r)) for w, s, n, r in
                   zip(white, hyper['lnsigma'], hyper['log_nu_spde'], hyper['log_ranges'])]) + A.mass_window()[None]
    mx = lp.max((1, 2), keepdims=True)
    lp -= np.log(np.exp(lp - mx).sum((1, 2), keepdims=True) * A.dlnm * A.dq) + mx
    return np.exp(lp).astype(np.float32)


def q_stats(p_mass, window):
    """Fit-free mean and sd of q over the m1 range ``window``, per draw."""
    s = (A.m_c >= window[0]) & (A.m_c <= window[1])
    pq = p_mass[:, s].astype(float).sum(1); pq /= pq.sum(1, keepdims=True)
    mean = pq @ A.q_c
    return mean, np.sqrt(np.maximum(pq @ A.q_c ** 2 - mean ** 2, 0.))


def _fit_window(job):
    name, p_mass = job
    qr, win, _ = PEAKS[name]
    p_m1 = A.p_m1_given_q_range(type('R', (), {'p_mass': p_mass})(), qr)
    return name, fit_peak(p_m1, win), fit_joint(p_mass / A.m_c[None, :, None], win)


def summarize(run, out):
    post_joint = np.load(npz_path(run)) if os.path.exists(npz_path(run)) else None
    print(f'{run}: prior vs posterior, median and 90% over draws with a peak')
    for name in PEAKS:
        J, P = out[f'joint_{name}'], out[f'peak_{name}']
        pf = lambda F: 1 - np.isnan(F[:, 1]).mean()
        print(f'  peak {name}: 1D peak frac prior {pf(P):.3f};  joint peak frac prior {pf(J):.3f}'
              + (f', posterior {pf(post_joint[name]):.3f}' if post_joint is not None else ''))
        print(f'    fit-free q over {Q_WINDOWS[name]}: mean prior {fmt(out[f"qmean_prior_{name}"])}, '
              f'posterior {fmt(out[f"qmean_post_{name}"])};  sd prior {fmt(out[f"qsd_prior_{name}"])}, '
              f'posterior {fmt(out[f"qsd_post_{name}"])}')
        if pf(J) > 0.05:
            for label, F in (('prior', J), ('posterior', None if post_joint is None else post_joint[name])):
                if F is None:
                    continue
                ok = np.all(np.isfinite(F), 1); qm, qs = q_moments(F[ok])
                print(f'    joint {label}: mu {fmt(F[:, 1])}, sigma {fmt(F[:, 2])}, QMean {fmt(qm)}, '
                      f'QSpread {fmt(qs)}, f_G {fmt(F[:, 9])}')


def main(runs, n=None, seed=0):
    for run in runs:
        t0 = time.time()
        hyper = load_hyper(run)
        if n:
            hyper = {k: v[:: max(1, len(v) // n)][:n] for k, v in hyper.items()}
        p_prior = prior_p_mass(hyper, seed)
        print(f'{run}: {len(p_prior)} prior draws in {time.time() - t0:.0f} s', flush=True)
        out = {k: v for k, v in hyper.items()}
        p_post = A.load_light(run).p_mass
        for name, win in Q_WINDOWS.items():
            out[f'qmean_prior_{name}'], out[f'qsd_prior_{name}'] = q_stats(p_prior, win)
            out[f'qmean_post_{name}'], out[f'qsd_post_{name}'] = q_stats(p_post, win)
        del p_post
        with get_context('spawn').Pool(len(PEAKS)) as pool:     # spawn: jax is loaded, fork can deadlock
            for name, peak, joint in pool.imap_unordered(_fit_window, [(name, p_prior) for name in PEAKS]):
                out[f'peak_{name}'], out[f'joint_{name}'] = peak, joint
                print(f'  window {name} done, {time.time() - t0:.0f} s', flush=True)
        np.savez(prior_path(run), **out)
        print('saved', prior_path(run))
        summarize(run, out)


if __name__ == '__main__':
    args = sys.argv[1:]
    kw = {}
    for flag, cast in (('--n', int), ('--seed', int)):
        if flag in args:
            i = args.index(flag); kw[flag[2:]] = cast(args[i + 1]); del args[i:i + 2]
    main(args or list(RUN_NAMES), **kw)
