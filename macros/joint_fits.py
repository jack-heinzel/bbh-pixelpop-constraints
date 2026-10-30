"""Joint (m1, q) fits to the peaks of the mass distribution, per hyperparameter draw, written as LaTeX macros.

    python macros/joint_fits.py [<run> ...]     (default: ppc_gwtc5_k10 ppc_gwtc5_k100_gauss)
        -> macros/joint_fits.tex, plots/<run>_joint_fits.pdf

Unlike macros/peak_fits.py, no mass-ratio range is imposed (user, 2026-10-24): each peak's m1 window (peak_fits.PEAKS)
is fitted over every q cell allowed by m2 > 3 Msun, using the windowed p(m1, q) as a density per Msun per unit q, with
two components,

    peak:        A  N(m1; mu, sigma)  TN(q; mu_q, sigma_q)
    background:  B  (m1 / m_ref)^(-alpha)  TN(q; mu_qb, sigma_qb)

where TN is a Gaussian truncated to [3 Msun / m1, 1], normalized at each m1 (so A N and B PL are the components'
m1 distributions) and m_ref is the window's centre. The fit minimizes the generalized KL divergence of the model f
from the posterior density p over the supported cells (user, 2026-10-26: cross entropy, the likelihood-based fit, in
place of least squares in log density, which let the many low-density cells at low q swamp the peaks),

    D(p || f) = sum_cells a [p ln(p / f) - p + f],      a = the cell's area in (m1, q),

i.e. the cross entropy -sum a p ln f with f's normalization free (a Poisson likelihood). least_squares minimizes it
through the deviance residuals sign(f - p) sqrt(2 a [p ln(p / f) - p + f]). It starts from the window's m1 marginal and
q moments, from 6 starts (the Gaussian at the window's maximum or centre; the peak's q distribution from the moments
at that mass, or at mu_q = 0.9 or 0.5), keeping the lowest divergence: the objective has local minima with the Gaussian
at a window edge. Bounds: mu in the window, sigma in
(0.03, 1) x its width, alpha in (-20, 20), mu_q and mu_qb in (0, 1) (a centre outside the truncation range only
measures mu_q / sigma_q^2), sigma_q in (0.02, 3). Draws whose fit fails or ends with mu
or sigma on a bound have no peak in the window: they are counted (PeakFrac) and left out of the summaries. Reported,
conditional on a peak: mu, sigma, mu_q, sigma_q, f_G (the peak's share of the probability in the window), alpha,
mu_qb, sigma_qb, and the mean and spread in q of the peak's truncated Gaussian (QMean, QSpread). Run with macros/fits.sbatch (~15 min).
"""
import os
import sys
import time
import warnings

import numpy as np
from scipy.optimize import least_squares
from scipy.special import log_ndtr, ndtr

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'notebooks')); sys.path.insert(0, HERE)
import analysis as A
from peak_fits import PEAKS, RUN_NAMES, deviance, fmt

NAMES = ('lnA', 'mu', 'sigma', 'mu_q', 'sigma_q', 'lnB', 'alpha', 'mu_qb', 'sigma_qb')
LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)


def log_tn(q, qmin, mu, sd):
    """log of the Gaussian N(mu, sd) truncated to [qmin, 1], at q (arrays broadcast)."""
    z = ndtr((1 - mu) / sd) - ndtr((qmin - mu) / sd)
    return -0.5 * ((q - mu) / sd) ** 2 - LOG_SQRT_2PI - np.log(sd) - np.log(np.maximum(z, 1e-300))


def log_components(x, m, q, m_ref):
    """log of the peak and background densities at cells (m, q)."""
    lnA, mu, sigma, mu_q, sd_q, lnB, alpha, mu_b, sd_b = x
    qmin = A.MIN_M / m
    lp = lnA - 0.5 * ((m - mu) / sigma) ** 2 + log_tn(q, qmin, mu_q, sd_q)
    lb = lnB - alpha * np.log(m / m_ref) + log_tn(q, qmin, mu_b, sd_b)
    return lp, lb


def bounds(window):
    w = window[1] - window[0]
    # mu_q, mu_qb in [0, 1] (user, 2026-10-24): a centre outside the truncation range makes the truncated Gaussian an
    # exponential in q, which only measures mu_q / sigma_q^2
    lo = [-60., window[0], 0.03 * w, 0., 0.02, -60., -20., 0., 0.02]
    hi = [20., window[1], 1.0 * w, 1., 3., 20., 20., 1., 3.]
    return lo, hi


def fit_joint(dens, window):
    """dens: (draws, Bm, Bq) density per Msun per unit q. Returns (draws, 10): NAMES + f_G (NaN: no peak)."""
    msel = (A.m_c >= window[0]) & (A.m_c <= window[1])
    M, Qg = np.meshgrid(A.m_c[msel], A.q_c, indexing='ij')
    support = Qg * M >= A.MIN_M
    m_ref = 0.5 * (window[0] + window[1])
    lo, hi = bounds(window)
    area = (M * A.dlnm * A.dq)                                   # cell areas in (m1, q)
    out = np.full((len(dens), len(NAMES) + 1), np.nan)
    t0 = time.time()
    for i, d in enumerate(dens[:, msel]):
        ok = support & (d > 0) & np.isfinite(d)
        m, q, y, a_ok = M[ok], Qg[ok], d[ok].astype(float), area[ok]
        ly = np.log(y)
        # start: power law + Gaussian through the window's m1 marginal; q moments at the peak and over the window
        marg = (d * support).sum(1) * A.dq
        mm = A.m_c[msel]
        slope, icpt = np.polyfit(np.log(mm / m_ref), np.log(np.maximum(marg, 1e-300)), 1)
        resid = marg - np.exp(icpt + slope * np.log(mm / m_ref))
        k = int(np.argmax(marg))
        pq = d[k] * support[k]; pq = pq / pq.sum()
        mq, sq = (pq * A.q_c).sum(), np.sqrt(max((pq * A.q_c ** 2).sum() - (pq * A.q_c).sum() ** 2, 0.05 ** 2))
        pw = (d * support * area).sum(0); pw = pw / pw.sum()
        mb, sb = (pw * A.q_c).sum(), np.sqrt(max((pw * A.q_c ** 2).sum() - (pw * A.q_c).sum() ** 2, 0.05 ** 2))
        a0 = np.log(max(resid.max(), 0.1 * marg.max()))
        best = None
        # several starts (the objective has local minima with the Gaussian at an edge); keep the lowest divergence
        for mu0 in (mm[k], m_ref):
            for mq0, sq0 in ((mq, sq), (0.9, 0.1), (0.5, 0.15)):
                x0 = np.clip([a0, mu0, 0.2 * (window[1] - window[0]), mq0, sq0, icpt, -slope, mb, sb],
                             np.add(lo, 1e-6), np.subtract(hi, 1e-6))
                f = lambda x: deviance(np.logaddexp(*log_components(x, m, q, m_ref)), y, ly, a_ok)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        res = least_squares(f, x0, bounds=(lo, hi), method='trf', x_scale='jac')
                except (ValueError, FloatingPointError):
                    continue
                if res.success and (best is None or res.cost < best.cost):
                    best = res
        if best is None or _on_bound(best.x, lo, hi):
            continue
        lp, lb = log_components(best.x, M[support], Qg[support], m_ref)
        a = area[support]
        out[i] = [*best.x, (np.exp(lp) * a).sum() / ((np.exp(lp) + np.exp(lb)) * a).sum()]
        if (i + 1) % 250 == 0:
            print(f'    {i + 1}/{len(dens)} draws, {(time.time() - t0) / (i + 1):.2f} s each', flush=True)
    return out


def _on_bound(x, lo, hi):
    return bool(np.any(np.isclose(x[1:3], lo[1:3], rtol=1e-4) | np.isclose(x[1:3], hi[1:3], rtol=1e-4)))


def diagnostic(run, dens, fits, path, n_model=200):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import plot_style
    plot_style.apply()
    fig, axes = plt.subplots(1, len(PEAKS), figsize=(4.6 * len(PEAKS), 4.2))
    for ax, (name, (_, win, _)) in zip(axes, PEAKS.items()):
        msel = (A.m_c >= win[0]) & (A.m_c <= win[1])
        M, Qg = np.meshgrid(A.m_c[msel], A.q_c, indexing='ij')
        support = Qg * M >= A.MIN_M
        med = np.where(support, np.median(dens[:, msel], 0), np.nan)
        vmax = np.nanmax(med)
        medges = np.exp(A.lnm_edges[np.r_[np.where(msel)[0], np.where(msel)[0][-1] + 1]])
        pc = ax.pcolormesh(medges, A.q_edges, med.T, norm=LogNorm(vmin=vmax * 1e-3, vmax=vmax), cmap='viridis',
                           shading='flat', rasterized=True)
        F = fits[name]; okd = np.where(np.all(np.isfinite(F), 1))[0][:n_model]
        m_ref = 0.5 * (win[0] + win[1])
        model = np.median([np.exp(np.logaddexp(*log_components(F[j, :9], M, Qg, m_ref))) for j in okd], 0)
        model = np.where(support, model, np.nan)
        ax.contour(M, Qg, np.log10(model), levels=np.log10(vmax) - np.array([2., 1., 0.5]), colors='w',
                   linewidths=1)
        ax.set_xscale('log'); ax.set_xlim(win); ax.set_ylim(0, 1)
        ax.set_xlabel(r'$m_1\,[\textrm{M}_\odot]$'); ax.tick_params(which='both', direction='out')
        ax.text(0.04, 0.05, rf'$\mu = {fmt(F[:, 1])}$' '\n' rf'$\sigma = {fmt(F[:, 2])}$' '\n'
                rf'$\mu_q = {fmt(F[:, 3])}$' '\n' rf'$\sigma_q = {fmt(F[:, 4])}$' '\n' rf'$f_G = {fmt(F[:, 9])}$',
                transform=ax.transAxes, va='bottom', fontsize=10, color='w')
    axes[0].set_ylabel(r'$q$')
    cb = fig.colorbar(pc, ax=axes, fraction=0.015, pad=0.01); cb.set_label(r'median $p(m_1, q)$ (per panel scale)')
    fig.savefig(path, bbox_inches='tight')
    print('saved', path)


def npz_path(run):
    return os.path.join(REPO, 'results', 'post_processing', f'{run}_joint_fits.npz')


def q_moments(F):
    """Mean and standard deviation in q of each draw's peak, TN(q; mu_q, sigma_q) on [3 Msun / mu, 1]: unlike mu_q,
    which can sit at the q = 1 bound, these are the peak's typical mass ratio and its spread."""
    from scipy.stats import truncnorm
    mu, mq, sq = F[:, 1], F[:, 3], F[:, 4]
    a, b = (3. / mu - mq) / sq, (1. - mq) / sq
    return truncnorm.mean(a, b, loc=mq, scale=sq), truncnorm.std(a, b, loc=mq, scale=sq)


def write_tex():
    """macros/joint_fits.tex from every run in RUN_NAMES whose per-draw fits are saved (npz_path)."""
    lines = ['% Joint (m1, q) fits: Gaussian x truncated Gaussian peak + power law x truncated Gaussian background,'
             ' minimizing the KL divergence (cross entropy) from p(m1, q), no q range imposed; median and 90% over draws'
             ' with a peak (macros/joint_fits.py)']
    for run, suffix in RUN_NAMES.items():
        if not os.path.exists(npz_path(run)):
            print(f'no fits saved for {run}: left out of the macros')
            continue
        fits = np.load(npz_path(run))
        lines.append(f'% {run}')
        for name, (_, win, tex) in PEAKS.items():
            F = fits[name]; bad = int(np.isnan(F[:, 1]).sum())
            lines.append(f'% peak {name}: m1 window {win} Msun, all q; {bad} of {len(F)} draws without a peak')
            for par, col in (('Mu', 1), ('Sigma', 2), ('QMu', 3), ('QSigma', 4), ('FracGauss', 9), ('Alpha', 6),
                             ('BkgQMu', 7), ('BkgQSigma', 8)):
                lines.append(rf'\newcommand{{\joint{tex}{par}{suffix}}}{{{fmt(F[:, col])}}}')
            for par, x in zip(('QMean', 'QSpread'), q_moments(F)):
                lines.append(rf'\newcommand{{\joint{tex}{par}{suffix}}}{{{fmt(x)}}}')
            lines.append(rf'\newcommand{{\joint{tex}PeakFrac{suffix}}}{{{1 - bad / len(F):.3f}}}')
    path = os.path.join(HERE, 'joint_fits.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('wrote', path)


def main(runs):
    """Fit each run in ``runs`` (saving its per-draw fits as soon as they are done), then write the macros from every
    run with saved fits. ``--tex-only``: just rewrite the macros."""
    for run in runs:
        r = A.load_light(run)
        dens = (r.p_mass / A.m_c[None, :, None]).astype(np.float32)        # per Msun per unit q, windowed
        del r
        fits = {}
        for name, (_, win, tex) in PEAKS.items():
            print(f'{run} peak {name}: m1 window {win}', flush=True)
            fits[name] = F = fit_joint(dens, win)
            bad = int(np.isnan(F[:, 1]).sum())
            print(f'  mu {fmt(F[:, 1])}, sigma {fmt(F[:, 2])}, mu_q {fmt(F[:, 3])}, sigma_q {fmt(F[:, 4])}, '
                  f'f_G {fmt(F[:, 9])}, alpha {fmt(F[:, 6])}, mu_qb {fmt(F[:, 7])}, sigma_qb {fmt(F[:, 8])}'
                  f'  [{bad} of {len(F)} draws without a peak]', flush=True)
        np.savez(npz_path(run), **fits, columns=np.array(NAMES + ('f_G',)))
        diagnostic(run, dens, fits, os.path.join(REPO, 'plots', f'{run}_joint_fits.pdf'))
        del dens
    write_tex()


if __name__ == '__main__':
    args = sys.argv[1:]
    if '--tex-only' in args:
        write_tex()
    else:
        main(args or list(RUN_NAMES))
