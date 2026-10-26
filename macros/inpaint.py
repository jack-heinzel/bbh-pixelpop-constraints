"""How significant is each window's peak? The posterior's field in the window against the prior's prediction for it
from the field outside (conditional simulation / "inpainting", user 2026-10-26; macros/conditional_prior.md).

    python macros/inpaint.py [--buffer NB] [--k K] [--n N] [<run> ...]   (default: both runs, buffer 0, K = 50, all draws)
        -> results/post_processing/<run>_inpaint_b<NB>.npz   (statistics)
           results/post_processing/<run>_inpaint_b<NB>.h5    (the conditional field realizations; see load_realizations)

For each posterior draw (kernel parameters theta and the unwindowed log field x = log_p_mass) and each window of
peak_fits.PEAKS: the m1 bins of the window, widened by NB bins on each side, at every q (the strip R) are redrawn K times
from the field's prior given theta and x outside R. The prior is model.py's Matern field with the DC mode given a flat
prior (precision 0) in place of zero variance, so the constant in log_p_mass drops out. With R a full-q strip, the q
eigenvectors stay a complete basis inside it, so the conditional decouples into one Gaussian per q-mode l over the m1
bins:

    y_R | y_O ~ N(-Q_RR^-1 Q_RO y_O, Q_RR^-1),    Q = v_m diag(1 / a_l^2) v_m^T,    y = x v_q (the field in q-modes).

Statistics, on the windowed, normalized field, over the window's m1 range and q band (PEAKS):
    mass: the probability in the window's (m1, q band) box;
    bump: max over the window's m1 bins of ln p(m1, q in band) minus its straight line in ln m1 between the window's end
          bins (normalization-free; ~0 for a field the outside predicts, > 0 for a peak).
Saved per draw: the posterior's value and the K conditional values. Summaries: the tail probability P(conditional >=
posterior), pooled over draws and conditional samples, and the excess 1 - E_cond[mass] / mass. The same is done with a
fresh prior field (same theta) in place of the posterior's ('null'): its tail probabilities must come out uniform.

The conditional realizations are saved for plots (user, 2026-10-26), per window: ``cond_<peak>`` (draws, K, nR, Bq),
the K redrawn log fields on the strip; ``mean_<peak>`` (draws, nR, Bq), the conditional mean (the outside's smooth
continuation into the strip); ``rows_<peak>``, the strip's m1 bin indices; ``draws``, the posterior draws used (indices
into load_light's order). They are unwindowed log fields with the posterior's normalization constant (log_p_mass's),
so a full field is log_p_mass with rows R replaced. ~3 GB per run at buffer 0 (float32). The null fields are not saved.
"""
import os
import sys
import time

import h5py
import numpy as np

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'notebooks')); sys.path.insert(0, HERE)
import analysis as A
from peak_fits import PEAKS, RUN_NAMES
from prior_fits import load_hyper


def basis():
    """Path-graph Laplacian eigenpairs of the mass grid's two axes (as model.py), as float64 numpy."""
    from pixelpop.utils import create_CAR_coupling_matrix
    import model
    (wm, vm), (wq, vq) = model.axis_eigs([create_CAR_coupling_matrix(A.B, 1, isVisible=False)] * 2)
    return tuple(np.asarray(x, float) for x in (wm, vm, wq, vq))


def spectrum(B, lnsigma, log_nu, log_ranges):
    """model.matern_spectrum's per-mode amplitudes a (Bm, Bq), DC zero."""
    import model
    wm, _, wq, _ = B
    return np.asarray(model.matern_spectrum([wm, wq], lnsigma, log_ranges, np.exp(log_nu), A.B * A.B), float)


def conditional_strip(x, a, B, R, rng, K):
    """K draws of the field on the m1 rows R (bool, Bm), every q, given x on the other rows. x, a: (Bm, Bq).
    Returns the draws (K, nR, Bq) and the conditional mean (nR, Bq)."""
    _, vm, _, vq = B
    with np.errstate(over='ignore', divide='ignore'):
        prec = np.where(a > 0, 1. / a ** 2, 0.)                             # (k, l); the DC mode flat
    y = x @ vq                                                              # (Bm, l): q-modes
    VR, VO = vm[R], vm[~R]
    QRR = np.einsum('ik,kl,jk->lij', VR, prec, VR)                          # (l, nR, nR)
    QRO = np.einsum('ik,kl,jk->lij', VR, prec, VO)
    mean = np.linalg.solve(QRR, -np.einsum('lij,jl->li', QRO, y[~R])[..., None])[..., 0]          # (l, nR)
    Lt = np.swapaxes(np.linalg.cholesky(QRR), -1, -2)                       # Q_RR = L L^T; draws: mean + L^-T z
    z = rng.standard_normal((K, *mean.shape, 1))
    yR = mean[None] + np.linalg.solve(Lt[None], z)[..., 0]                   # (K, l, nR)
    return np.swapaxes(yR, 1, 2) @ vq.T, mean.T @ vq.T                      # (K, nR, Bq), (nR, Bq)


def stats(xR, x, R, win_rows, band, logW):
    """mass and bump (see the module docstring) for fields equal to x outside the rows R and xR (K, nR, Bq) inside."""
    lw = logW
    out_lse = np.logaddexp.reduce((x + lw)[~R].ravel())
    lpR = xR + lw[R][None]                                                  # (K, nR, Bq)
    lse = np.logaddexp(out_lse, np.logaddexp.reduce(lpR.reshape(len(xR), -1), axis=1))
    rows = win_rows[R]                                                      # window rows within R
    lp_band = np.logaddexp.reduce(lpR[:, rows][:, :, band], axis=2)         # (K, n_win): ln p(m1 bin, band), unnormalized
    mass = np.exp(np.logaddexp.reduce(lp_band, axis=1) - lse)
    u = A.lnm_c[win_rows]
    t = (u - u[0]) / (u[-1] - u[0])
    chord = lp_band[:, :1] * (1 - t) + lp_band[:, -1:] * t
    return mass, (lp_band - chord).max(1)


def h5_path(run, buffer):
    return os.path.join(REPO, 'results', 'post_processing', f'{run}_inpaint_b{buffer}.h5')


def load_realizations(run, peak, buffer=0, draws=slice(None)):
    """Full unwindowed log fields (n, K, Bm, Bq) for the saved posterior draws ``draws`` (positions in the file's
    ``draws``): log_p_mass with the strip replaced by each conditional realization; also returns the posterior's own
    fields (n, Bm, Bq), the conditional-mean fields (n, Bm, Bq) and the posterior draw indices."""
    import glob
    with h5py.File(h5_path(run, buffer), 'r') as h:
        idx, rows = h['draws'][()][draws], h[f'rows_{peak}'][()]
        cond, mean = h[f'cond_{peak}'][draws], h[f'mean_{peak}'][draws]
    files = sorted(glob.glob(os.path.join(REPO, 'results', run, 'chain_*_samples.h5')))
    n_draws = min(len(h5py.File(f, 'r')['log_likelihood']) for f in files)
    per = [(f, j) for f in files for j in range(n_draws)]
    X = np.stack([h5py.File(per[i][0], 'r')['log_p_mass'][per[i][1]] for i in np.atleast_1d(idx)]).astype(float)
    full = np.repeat(X[:, None], cond.shape[1], axis=1)
    full[:, :, rows] = cond
    smooth = X.copy(); smooth[:, rows] = mean
    return full, X, smooth, idx


def run_windows(run, buffer=0, K=50, n=None, seed=0):
    import glob
    files = sorted(glob.glob(os.path.join(REPO, 'results', run, 'chain_*_samples.h5')))
    n_draws = min(len(h5py.File(f, 'r')['log_likelihood']) for f in files)
    X = np.concatenate([h5py.File(f, 'r')['log_p_mass'][:n_draws] for f in files]).astype(float)
    hyper = load_hyper(run)
    idx = np.arange(len(X)) if not n else np.linspace(0, len(X) - 1, n).astype(int)
    B, logW = basis(), A.mass_window()
    rng = np.random.default_rng(seed)
    out = {'draws': idx}
    h5 = None
    if n is None:                                       # full runs save the realizations
        h5 = h5py.File(h5_path(run, buffer), 'w')
        h5['draws'] = idx
        h5.attrs.update(buffer=buffer, K=K, seed=seed)
    t0 = time.time()
    for name, (qr, win, _) in PEAKS.items():
        win_rows = (A.m_c >= win[0]) & (A.m_c <= win[1])
        i0, i1 = np.where(win_rows)[0][[0, -1]]
        R = np.zeros(A.B, bool); R[max(0, i0 - buffer):i1 + buffer + 1] = True
        band = (A.q_c >= qr[0]) & (A.q_c <= qr[1])
        if h5 is not None:
            h5[f'rows_{name}'] = np.where(R)[0]
            d_cond = h5.create_dataset(f'cond_{name}', (len(idx), K, R.sum(), A.B), 'f4', chunks=(1, K, R.sum(), A.B))
            d_mean = h5.create_dataset(f'mean_{name}', (len(idx), R.sum(), A.B), 'f4', chunks=(1, R.sum(), A.B))
        res = {k: [] for k in ('mass_post', 'bump_post', 'mass_cond', 'bump_cond',
                               'mass_null', 'bump_null', 'mass_null_cond', 'bump_null_cond')}
        for j, i in enumerate(idx):
            a = spectrum(B, hyper['lnsigma'][i], hyper['log_nu_spde'][i], hyper['log_ranges'][i])
            for tag, x in (('post', X[i]),
                           ('null', B[1] @ (a * rng.standard_normal(a.shape)) @ B[3].T)):   # a fresh prior field
                m0, b0 = stats(x[R][None], x, R, win_rows, band, logW)
                xR, mean = conditional_strip(x, a, B, R, rng, K)
                m, b = stats(xR, x, R, win_rows, band, logW)
                if tag == 'post' and h5 is not None:
                    d_cond[j], d_mean[j] = xR, mean
                key = '' if tag == 'post' else 'null_'
                res[f'mass_{tag}'].append(m0[0]); res[f'bump_{tag}'].append(b0[0])
                res[f'mass_{key}cond'].append(m); res[f'bump_{key}cond'].append(b)
        for k, v in res.items():
            out[f'{k}_{name}'] = np.array(v)
        print(f'  {run} window {name} done, {time.time() - t0:.0f} s', flush=True)
    if h5 is not None:
        h5.close()
        print('saved', h5_path(run, buffer))
    return out


def tail(post, cond):
    """Per-draw P(cond >= post) and its pooled mean; the pooled value's one-sided Gaussian equivalent in sigma."""
    from scipy.stats import norm
    per = (cond >= post[:, None]).mean(1)
    pooled = per.mean()
    floor = 1. / cond.size
    return per, pooled, norm.isf(max(pooled, floor)), pooled < floor


def summarize(run, out, buffer):
    print(f'{run}, buffer {buffer} bins: tail probability P(conditional >= posterior), pooled over draws '
          f'(null: a prior field in place of the posterior; should be ~0.5 with ~5% of draws below 0.05)')
    for name in PEAKS:
        line = [f'  peak {name}:']
        for stat in ('mass', 'bump'):
            per, pooled, sig, bound = tail(out[f'{stat}_post_{name}'], out[f'{stat}_cond_{name}'])
            nper, npooled, _, _ = tail(out[f'{stat}_null_{name}'], out[f'{stat}_null_cond_{name}'])
            line.append(f'{stat} p {"<" if bound else ""}{max(pooled, 1. / out[f"{stat}_cond_{name}"].size):.2g} '
                        f'({"> " if bound else ""}{sig:.1f} sigma; draws p < 0.05: {(per < 0.05).mean():.2f}) '
                        f'[null p {npooled:.2f}, {(nper < 0.05).mean():.2f}]')
        exc = 1 - out[f'mass_cond_{name}'].mean(1) / out[f'mass_post_{name}']
        line.append(f'excess {np.median(exc):.2f} ({np.percentile(exc, 5):.2f}-{np.percentile(exc, 95):.2f})')
        print(' '.join(line[:1]) + '\n    ' + '\n    '.join(line[1:]))


def main(runs, buffer=0, k=50, n=None):
    for run in runs:
        out = run_windows(run, buffer, k, n)
        path = os.path.join(REPO, 'results', 'post_processing', f'{run}_inpaint_b{buffer}.npz')
        if n is None:
            np.savez(path, **out)
            print('saved', path)
        summarize(run, out, buffer)


if __name__ == '__main__':
    args = sys.argv[1:]
    kw = {}
    for flag in ('--buffer', '--k', '--n'):
        if flag in args:
            i = args.index(flag); kw[flag[2:]] = int(args[i + 1]); del args[i:i + 2]
    main(args or list(RUN_NAMES), **kw)
