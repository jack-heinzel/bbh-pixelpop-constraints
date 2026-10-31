"""Gaussian + power-law fits to the peaks of p(m1 | q in Q), per hyperparameter draw, written as LaTeX macros.

    python macros/peak_fits.py [<run> ...]      (default: ppc_gwtc5_k10 ppc_gwtc5_k100_gauss)
        -> macros/peak_fits.tex, plots/<run>_peak_fits.pdf

For each draw, p(m1 | q in Q) is the windowed mass field summed over the q bins in Q (analysis.p_m1_given_q_range), a
density per Msun on the m1 bin centres. On the bins inside each peak's window (PEAKS: by eye, user 2026-10-24) it is
fitted with a mixture (user, 2026-10-24)

    f(m1) = A exp(-(m1 - mu)^2 / 2 sigma^2) + B (m1 / m_ref)^(-alpha),     m_ref = the window's centre,

by minimizing the generalized KL divergence of f from p over the window's bins (user, 2026-10-26: cross entropy, the
likelihood-based fit, for both these and the joint fits),

    D(p || f) = sum_bins a [p ln(p / f) - p + f],      a = the bin's width in m1 (m1 dln m1),

i.e. the cross entropy -sum a p ln f with f's normalization free (a Poisson likelihood), through least_squares on the
deviance residuals sign(f - p) sqrt(2 a [p ln(p / f) - p + f]); ln A and ln B are the parameters (both terms positive). mu
is bounded to the window, sigma to (0.03, 1) x its width, alpha to (-20, 20); each draw starts from two initial
points (the Gaussian at the window's maximum and at its centre) and keeps the better fit. Draws whose fit fails or
ends with mu or sigma on a bound are counted and left out of the summaries (in practice: draws with no peak inside the window, the
Gaussian pushed to an edge). Reported: mu, sigma, alpha, f_G (the Gaussian's share of the fitted probability in the
window), all conditional on a peak, and PeakFrac, the fraction of draws with one. Runs on a login node (pinned; < 1.5 GB).
"""
import os
import sys
import warnings

import numpy as np
from scipy.optimize import least_squares

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'notebooks'))
import analysis as A

# name: (q range, m1 window [Msun], LaTeX name)
PEAKS = {'10': ((0.8, 1.0), (5., 15.), 'Ten'),          # widened from 8-12 (user, 2026-10-24)
         '35': ((0.8, 1.0), (25., 40.), 'ThirtyFive'),
         '20': ((0.3, 0.65), (15., 30.), 'Twenty'),
         '60': ((0.3, 0.65), (40., 100.), 'Sixty')}
RUN_NAMES = {'ppc_gwtc5_k10': 'Flex', 'ppc_gwtc5_k100_gauss': 'Gauss'}     # macro suffixes
PARAMS = ('lnA', 'mu', 'sigma', 'lnB', 'alpha')


def components(m, lnA, mu, sigma, lnB, alpha, m_ref):
    """The Gaussian and power-law terms of the fitted mixture."""
    return np.exp(lnA - 0.5 * ((m - mu) / sigma) ** 2), np.exp(lnB - alpha * np.log(m / m_ref))


def deviance(lf, p, lp, a):
    """Deviance residuals of model log density lf against density p (log lp) on bins / cells of size a: their squares
    sum to 2 D(p || f), the generalized KL divergence (cross entropy with free normalization)."""
    f = np.exp(lf)
    term = np.maximum(p * (lp - lf) - p + f, 0.)                  # >= 0 analytically; clip rounding
    return np.sign(f - p) * np.sqrt(2 * a * term)


def fit_peak(p_m1, window):
    """Fit the mixture to each draw of p_m1 (draws, Bm) on the bins whose centres lie in ``window``, by minimizing
    D(p || f) (deviance). Returns (draws, 6): lnA, mu, sigma, lnB, alpha, f_G (NaN where the fit failed or ended on a bound)."""
    sel = (A.m_c >= window[0]) & (A.m_c <= window[1])
    m, width = A.m_c[sel], window[1] - window[0]
    a = m * A.dlnm                                                          # bin widths in m1
    m_ref = 0.5 * (window[0] + window[1])
    lo = [-60., window[0], 0.03 * width, -60., -20.]
    hi = [20., window[1], 1.0 * width, 20., 20.]
    mm = np.linspace(*window, 400)
    out = np.full((len(p_m1), 6), np.nan)
    for i, y in enumerate(p_m1[:, sel]):
        if not np.all(np.isfinite(y)) or np.any(y <= 0):
            continue
        ly = np.log(y)
        slope, icpt = np.polyfit(np.log(m / m_ref), ly, 1)                 # power law through the window
        resid = y - np.exp(icpt + slope * np.log(m / m_ref))
        a0 = np.log(max(resid.max(), 0.1 * y.max()))
        best = None
        for mu0 in (m[np.argmax(y)], m_ref):
            x0 = np.clip([a0, mu0, 0.2 * width, icpt, -slope], np.add(lo, 1e-6), np.subtract(hi, 1e-6))
            f = lambda x: deviance(np.log(sum(components(m, *x, m_ref))), y, ly, a)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    res = least_squares(f, x0, bounds=(lo, hi), method='trf')
            except (ValueError, FloatingPointError):
                continue
            if res.success and (best is None or res.cost < best.cost):
                best = res
        if best is None:
            continue
        x = best.x
        on_bound = np.isclose(x[1:3], lo[1:3], rtol=1e-4) | np.isclose(x[1:3], hi[1:3], rtol=1e-4)
        if on_bound.any():
            continue
        g, p = components(mm, *x, m_ref)
        out[i] = [*x, g.sum() / (g + p).sum()]                              # f_G: Gaussian's share in the window
    return out


def error_decimals(err):
    """Decimals that show ``err`` to one significant figure, or two when that figure is 1 (+0.14, not +0.1, which could
    be a rounded +0.08; user, 2026-10-30); at most 3, at least 0."""
    if not err > 0:
        return 1
    e = int(np.floor(np.log10(err)))
    if round(err, -e) >= 10 ** (e + 1):                       # rounds up to the next decade (0.096 -> 0.1)
        e += 1
    lead = int(round(round(err, -e) / 10 ** e))
    return min(3, max(0, -e + (1 if lead == 1 else 0)))


MASS_DEC, Q_DEC = 1, 2   # masses (Msun) to one decimal, mass ratios to two (user, 2026-10-30)


def fmt(x, decimals=None):
    """median^{+hi}_{-lo} (90%), all to the decimals of the smaller error (error_decimals), or to ``decimals`` (MASS_DEC,
    Q_DEC: a quantity kept uniform across macros), with at least two significant figures in the larger of |median| and
    the smaller error (user, 2026-10-30; a median near 0, e.g. a contrast, then follows its error)."""
    lo, md, hi = np.nanpercentile(x, [5, 50, 95])
    err = min(hi - md, md - lo)
    d = error_decimals(err) if decimals is None else decimals
    scale = max(abs(md), err)
    if scale > 0 and np.isfinite(scale):
        d = max(d, 1 - int(np.floor(np.log10(scale))))
    return f'{md:.{d}f}^{{+{hi - md:.{d}f}}}_{{-{md - lo:.{d}f}}}'


def diagnostic(run, p_m1_by_q, fits, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import plot_style
    plot_style.apply()
    fig, axes = plt.subplots(1, len(PEAKS), figsize=(4.5 * len(PEAKS), 3.8))
    for ax, (name, (qr, win, _)) in zip(axes, PEAKS.items()):
        sel = (A.m_c >= win[0] / 1.4) & (A.m_c <= win[1] * 1.4)
        lo, md, hi = np.nanpercentile(p_m1_by_q[qr][:, sel], [5, 50, 95], axis=0)
        ax.fill_between(A.m_c[sel], lo, hi, color='C0', alpha=0.3, lw=0)
        ax.plot(A.m_c[sel], md, color='C0', label=rf'$p(m_1 \mid {qr[0]:g} < q < {qr[1]:g})$')
        mm = np.linspace(*win, 200); m_ref = 0.5 * (win[0] + win[1])
        ok = np.all(np.isfinite(fits[name]), 1)
        comps = np.array([components(mm, *p[:5], m_ref) for p in fits[name][ok]])     # (draws, 2, mm)
        total = comps.sum(1)
        ax.fill_between(mm, *np.percentile(total, [5, 95], 0), color='C3', alpha=0.25, lw=0)
        ax.plot(mm, np.median(total, 0), color='C3', label='Gaussian + power law')
        ax.plot(mm, np.median(comps[:, 0], 0), color='C3', ls='--', lw=1, label='Gaussian part')
        ax.plot(mm, np.median(comps[:, 1], 0), color='C3', ls=':', lw=1, label='power-law part')
        ax.axvspan(*win, color='0.92', zorder=0)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_ylim(np.nanmin(lo[lo > 0]) / 2, hi.max() * 2)
        ax.set_xlabel(r'$m_1\,[\textrm{M}_\odot]$')
        ax.text(0.04, 0.05, rf'$\mu = {fmt(fits[name][:, 1])}$' '\n' rf'$\sigma = {fmt(fits[name][:, 2])}$' '\n'
                rf'$\alpha = {fmt(fits[name][:, 4])}$' '\n' rf'$f_G = {fmt(fits[name][:, 5])}$',
                transform=ax.transAxes, va='bottom', fontsize=11)
    axes[0].set_ylabel(r'$p(m_1 \mid q)\,[\textrm{M}_\odot^{-1}]$')
    axes[-1].legend(loc='upper right', **dict(plot_style.LEG_KW, fontsize=11))
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    print('saved', path)


def main(runs):
    lines = ['% Gaussian + power-law fits to the peaks of p(m1 | q in Q), minimizing the KL divergence (cross entropy),'
             ' median and 90% over draws with a peak (macros/peak_fits.py)']
    for run in runs:
        r = A.load_light(run)
        p_m1_by_q = {qr: A.p_m1_given_q_range(r, qr) for qr in {v[0] for v in PEAKS.values()}}
        del r
        fits = {name: fit_peak(p_m1_by_q[qr], win) for name, (qr, win, _) in PEAKS.items()}
        suffix = RUN_NAMES.get(run, run.replace('_', ''))
        lines.append(f'% {run}')
        for name, (qr, win, tex) in PEAKS.items():
            F = fits[name]
            bad = int(np.isnan(F[:, 1]).sum())
            print(f'{run} peak {name} (q {qr}, m1 {win}): mu {fmt(F[:, 1])}, sigma {fmt(F[:, 2])}, '
                  f'alpha {fmt(F[:, 4])}, f_G {fmt(F[:, 5])}  [{bad} of {len(F)} draws failed or on a bound]')
            lines.append(f'% peak {name}: q in {qr}, m1 window {win} Msun; {bad} of {len(F)} draws dropped')
            for par, col in (('Mu', 1), ('Sigma', 2), ('Alpha', 4), ('FracGauss', 5)):
                dec = MASS_DEC if par in ('Mu', 'Sigma') else None
                lines.append(rf'\newcommand{{\peak{tex}{par}{suffix}}}{{{fmt(F[:, col], dec)}}}')
            # the fraction of draws with a peak inside the window (the summaries above are conditional on it)
            lines.append(rf'\newcommand{{\peak{tex}PeakFrac{suffix}}}{{{1 - bad / len(F):.3f}}}')
        diagnostic(run, p_m1_by_q, fits, os.path.join(REPO, 'plots', f'{run}_peak_fits.pdf'))
    path = os.path.join(HERE, 'peak_fits.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('wrote', path)


if __name__ == '__main__':
    main(sys.argv[1:] or list(RUN_NAMES))
