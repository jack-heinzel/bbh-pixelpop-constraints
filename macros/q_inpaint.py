"""Mass-ratio contrasts against the conditional ("inpainting") prior (user, 2026-10-30): is <q> in one peak's window, on
the posterior, above (or below) what the field prior predicts for the other peak's window from the posterior field
outside it?

    python macros/q_inpaint.py [--buffer NB] [--k K] [<run> ...]   (default: both runs, buffer 0, K = 50)
        -> results/post_processing/<run>_q_inpaint_b<NB>.npz

<q>_W is q_contrast.py's: the mean of q over q >= Q0 and the m1 bins (by centre) of window W, on the windowed field.
For each posterior draw (kernel parameters and log_p_mass) and each ordered pair (A, B) of PAIRS, the m1 rows of B,
widened by NB bins on each side, at every q, are redrawn K times from the field prior given the kernel parameters and
the field outside them (inpaint.conditional_strip). A buffer may reach into (or past) A's window: <q>_A is always the
posterior's, and the redrawn strip then no longer sees A's field either. Then

    P(<q>_A[posterior] > <q>_B[inpainted])    and    P(<q>_B[posterior] < <q>_A[inpainted]),

pooled over draws and realizations, with their one-sided Gaussian equivalents. The same with a fresh prior field (same
kernel parameters) in place of the posterior's ('null'), for calibration.
"""
import glob
import os
import sys

import h5py
import numpy as np

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'notebooks')); sys.path.insert(0, HERE)
import analysis as A
from inpaint import basis, conditional_strip, spectrum
from peak_fits import RUN_NAMES
from prior_fits import load_hyper
from q_contrast import Q0, WINDOWS

PAIRS = (('Ten', 'Twenty'), ('ThirtyFive', 'Sixty'))


def rows_of(window):
    return (A.m_c >= window[0]) & (A.m_c <= window[1])


def mean_q(lp, t):
    """<q> over the q bins t of log fields lp (..., n_m1, Bq) (windowed, any normalization)."""
    lp = lp[..., t]
    w = np.exp(lp - lp.max(axis=(-2, -1), keepdims=True)).sum(-2)                      # (..., n_t)
    return (w @ A.q_c[t]) / w.sum(-1)


def sigma(p, n):
    """One-sided Gaussian equivalent of P(event) = p, bounded at 1 - 1/n."""
    from scipy.stats import norm
    return norm.ppf(min(p, 1. - 1. / n))


def run(run, buffer=0, K=50, seed=0):
    files = sorted(glob.glob(os.path.join(REPO, 'results', run, 'chain_*_samples.h5')))
    n_draws = min(len(h5py.File(f, 'r')['log_likelihood']) for f in files)
    X = np.concatenate([h5py.File(f, 'r')['log_p_mass'][:n_draws] for f in files]).astype(float)
    hyper = load_hyper(run)
    Bas, logW = basis(), A.mass_window()
    t = A.q_c >= Q0
    rng = np.random.default_rng(seed)
    out = {}
    for a_name, b_name in PAIRS:
        for post_w, inp_w in ((a_name, b_name), (b_name, a_name)):
            rows_p, rows_i = rows_of(WINDOWS[post_w]), rows_of(WINDOWS[inp_w])
            i0, i1 = np.where(rows_i)[0][[0, -1]]
            R = np.zeros(A.B, bool); R[max(0, i0 - buffer):i1 + buffer + 1] = True
            key = f'{post_w}_vs_{inp_w}'
            res = {k: np.empty((len(X),) + s) for k, s in (('post', ()), ('post_inp_w', ()), ('inp', (K,)),
                                                          ('null', ()), ('null_inp', (K,)))}
            for i, x in enumerate(X):
                a = spectrum(Bas, hyper['lnsigma'][i], hyper['log_nu_spde'][i], hyper['log_ranges'][i])
                for tag, f in (('', x), ('null', Bas[1] @ (a * rng.standard_normal(a.shape)) @ Bas[3].T)):
                    xR, _ = conditional_strip(f, a, Bas, R, rng, K)
                    q_inp = mean_q(xR[:, rows_i[R]] + logW[rows_i][None], t)
                    q_post = mean_q(f[rows_p] + logW[rows_p], t)
                    if tag:
                        res['null'][i], res['null_inp'][i] = q_post, q_inp
                    else:
                        res['post'][i], res['inp'][i] = q_post, q_inp
                        res['post_inp_w'][i] = mean_q(f[rows_i] + logW[rows_i], t)
            for k, v in res.items():
                out[f'{key}_{k}'] = v
            print(f'  {run}: {key} done', flush=True)
    return out


def summarize(run, out, buffer, K):
    print(f'\n{run}, buffer {buffer} bins, K = {K}; <q> over q >= {Q0:g}; windows '
          + ', '.join(f'{k} [{a:g}, {b:g}]' for k, (a, b) in WINDOWS.items()))
    for a_name, b_name in PAIRS:
        for post_w, inp_w, sign in ((a_name, b_name, 1), (b_name, a_name, -1)):
            k = f'{post_w}_vs_{inp_w}'
            post, inp, own = out[f'{k}_post'], out[f'{k}_inp'], out[f'{k}_post_inp_w']
            d = sign * (post[:, None] - inp)
            p = (d > 0).mean(); pn = (sign * (out[f'{k}_null'][:, None] - out[f'{k}_null_inp']) > 0).mean()
            per = (d > 0).mean(1)
            rel = '>' if sign > 0 else '<'
            lo, md, hi = np.percentile(inp, [5, 50, 95])
            print(f'  P(<q>_{post_w}[post] {rel} <q>_{inp_w}[inpaint]) = {p:.4f} ({sigma(p, d.size):.1f} sigma); '
                  f'draws with P < 0.95: {(per < 0.95).mean():.2f}; null {pn:.2f}\n'
                  f'      <q>_{post_w} post {np.median(post):.3f}; <q>_{inp_w} inpainted {md:.3f} ({lo:.3f}-{hi:.3f}), '
                  f'posterior {np.median(own):.3f}')


def main(runs, buffer=0, k=50):
    for r in runs:
        out = run(r, buffer, k)
        path = os.path.join(REPO, 'results', 'post_processing', f'{r}_q_inpaint_b{buffer}.npz')
        np.savez(path, **out)
        summarize(r, out, buffer, k)
        print('saved', path)


if __name__ == '__main__':
    args = sys.argv[1:]
    kw = {}
    for flag in ('--buffer', '--k'):
        if flag in args:
            i = args.index(flag); kw[flag[2:]] = int(args[i + 1]); del args[i:i + 2]
    main(args or list(RUN_NAMES), **kw)
