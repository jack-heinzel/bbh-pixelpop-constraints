"""Gaussian + power-law fits to the peaks of p(m1 | q in Q), per hyperparameter draw, written as LaTeX macros.

    python macros/peak_fits.py [--wide] [<run> ...]      (default: ppc_gwtc5_k10 ppc_gwtc5_k100_gauss)
        -> macros/peak_fits.tex, plots/<run>_peak_fits.pdf
           (--wide: WIDE_PEAKS windows -> macros/peak_fits_wide.tex (\\peakWideTen...), plots/<run>_peak_fits_wide.pdf)

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
# --wide: the same peaks in windows ~1.25x wider on each side (user, 2026-11-09), a check of the window choice
WIDE_PEAKS = {'10': ((0.8, 1.0), (4., 19.), 'WideTen'),
              '35': ((0.8, 1.0), (20., 50.), 'WideThirtyFive'),
              '20': ((0.3, 0.65), (12., 36.), 'WideTwenty'),
              '60': ((0.3, 0.65), (32., 130.), 'WideSixty')}
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


def pct(p, n=None):
    """A probability or fraction in percent, the number only (user, 2026-10-30: 0.998 -> 99.8; the text writes the \\%):
    whole percent in [10, 90], one decimal outside (more if needed to stay off 0 and 100; 100 and 0 exactly as such);
    with ``n`` draws, p = 1 is '> 100 (1 - 1 / n)' and p = 0 '< 100 / n' (not resolvable)."""
    if n is not None and p >= 1:
        return f'> {100 * (1 - 1 / n):g}'
    if n is not None and p <= 0:
        return f'< {100 / n:g}'
    x = 100 * p
    if x in (0., 100.):
        return f'{x:.0f}'
    d = 0 if 10 <= x <= 90 else 1
    while d < 4 and float(f'{x:.{d}f}') in (0., 100.):
        d += 1
    return f'{x:.{d}f}'


def fmt(x, decimals=None, strict=False):
    """median^{+hi}_{-lo} (90%), all to the decimals of the smaller error (error_decimals), or to ``decimals`` (MASS_DEC,
    Q_DEC: a quantity kept uniform across macros), with at least two significant figures in the larger of |median| and
    the smaller error (user, 2026-10-30; a median near 0, e.g. a contrast, then follows its error). ``strict``: exactly
    ``decimals``, no significant-figure rule (the Table 1 masses, e.g. the ~0.8 Msun width of the 10 Msun peak; user,
    2026-11-08)."""
    lo, md, hi = np.nanpercentile(x, [5, 50, 95])
    err = min(hi - md, md - lo)
    d = error_decimals(err) if decimals is None else decimals
    scale = max(abs(md), err)
    if scale > 0 and np.isfinite(scale) and not (strict and decimals is not None):
        d = max(d, 1 - int(np.floor(np.log10(scale))))
    return f'{md:.{d}f}^{{+{hi - md:.{d}f}}}_{{-{md - lo:.{d}f}}}'


def diagnostic(run, p_m1_by_q, fits, path, peaks=PEAKS):
    """2 x 2 blocks, one per peak (near-equal-mass top row, q ~ 0.5 bottom row), the legend above. Top of each: the PixelPop p(m1 | q band) (90% band, median) against the Gaussian + power-law fit
    (90% band, median; median Gaussian and power-law parts), the fit window shaded, the q band in each panel's label.
    Below it: the residual ln[p_PixelPop / p_fit] per draw, each draw's fit against its own density (median, 90%), over
    the window (user, 2026-11-09)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import plot_style
    plot_style.apply()
    from matplotlib.ticker import FixedLocator, NullFormatter
    fig = plt.figure(figsize=(9.5, 9.))
    outer = fig.add_gridspec(2, 2, hspace=0.22, wspace=0.2)      # near-equal-mass peaks on top, q ~ 0.5 below
    left = []
    for k, (name, (qr, win, _)) in enumerate(peaks.items()):
        sub = outer[k // 2, k % 2].subgridspec(2, 1, height_ratios=(2.6, 1), hspace=0.05)
        ax = fig.add_subplot(sub[0]); ar = fig.add_subplot(sub[1], sharex=ax)
        if k % 2 == 0:
            left.append((ax, ar))
        sel = (A.m_c >= win[0] / 1.4) & (A.m_c <= win[1] * 1.4)
        lo, md, hi = np.nanpercentile(p_m1_by_q[qr][:, sel], [5, 50, 95], axis=0)
        ax.fill_between(A.m_c[sel], lo, hi, color='C0', alpha=0.3, lw=0)
        ax.plot(A.m_c[sel], md, color='C0')
        mm = np.linspace(*win, 200); m_ref = 0.5 * (win[0] + win[1])
        ok = np.all(np.isfinite(fits[name]), 1)
        comps = np.array([components(mm, *p[:5], m_ref) for p in fits[name][ok]])     # (draws, 2, mm)
        total = comps.sum(1)
        ax.fill_between(mm, *np.percentile(total, [5, 95], 0), color='C3', alpha=0.25, lw=0)
        ax.plot(mm, np.median(total, 0), color='C3')
        ax.plot(mm, np.median(comps[:, 0], 0), color='C3', ls='--', lw=1)
        ax.plot(mm, np.median(comps[:, 1], 0), color='C3', ls=':', lw=1)
        # residuals: each draw's fit against its own PixelPop density, on the bins inside the window
        w = (A.m_c >= win[0]) & (A.m_c <= win[1])
        fit_c = np.array([components(A.m_c[w], *p[:5], m_ref) for p in fits[name][ok]]).sum(1)
        with np.errstate(divide='ignore', invalid='ignore'):
            res = np.log(p_m1_by_q[qr][ok][:, w] / fit_c)
        rlo, rmd, rhi = np.nanpercentile(np.where(np.isfinite(res), res, np.nan), [5, 50, 95], axis=0)
        ar.fill_between(A.m_c[w], rlo, rhi, color='0.4', alpha=0.3, lw=0)
        ar.plot(A.m_c[w], rmd, color='k', lw=1.2)
        ar.axhline(0, color='C3', lw=0.8)
        for a in (ax, ar):
            a.axvspan(*win, color='0.92', zorder=0)
            a.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(np.nanmin(lo[lo > 0]) / 2, hi.max() * 2)
        lim = 1.1 * np.nanmax(np.abs([rlo, rhi]))
        ar.set_ylim(-lim, lim)
        ar.set_xlabel(r'$m_1\,[\textrm{M}_\odot]$')
        ax.set_xlim(A.m_c[sel][0], A.m_c[sel][-1])
        ticks = [t for t in (3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200)
                 if A.m_c[sel][0] <= t <= A.m_c[sel][-1]]
        for a in (ax, ar):                                  # plain labels on the log axis, no 4x10^0
            a.xaxis.set_major_locator(FixedLocator(ticks)); a.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticklabels([]); ar.set_xticklabels([f'{t:g}' for t in ticks])
        ax.tick_params(labelbottom=False)
        ax.text(0.04, 0.96, rf'${qr[0]:g} < q < {qr[1]:g}$', transform=ax.transAxes, va='top', fontsize=11)
        ax.text(0.04, 0.05, rf'$\mu = {fmt(fits[name][:, 1])}$' '\n' rf'$\sigma = {fmt(fits[name][:, 2])}$' '\n'
                rf'$\alpha = {fmt(fits[name][:, 4])}$' '\n' rf'$f_G = {fmt(fits[name][:, 5])}$',
                transform=ax.transAxes, va='bottom', fontsize=11, linespacing=1.6,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.85))
    for ax, ar in left:
        ax.set_ylabel(r'$p(m_1 \mid q\ \textrm{band})\,[\textrm{M}_\odot^{-1}]$')
        ar.set_ylabel(r'$\ln (p_\mathrm{PP} / p_\mathrm{fit})$', fontsize=10)
    fig.align_ylabels([a for pair in left for a in pair])
    handles = [(Patch(color='C0', alpha=0.3, lw=0), Line2D([], [], color='C0')),
               (Patch(color='C3', alpha=0.25, lw=0), Line2D([], [], color='C3')),
               Line2D([], [], color='C3', ls='--', lw=1), Line2D([], [], color='C3', ls=':', lw=1),
               (Patch(color='0.4', alpha=0.3, lw=0), Line2D([], [], color='k', lw=1.2))]
    labels = ['PixelPop', 'Gaussian + power law', 'Gaussian part', 'power-law part', 'residual per draw']
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.905), ncol=len(labels),
               **dict(plot_style.LEG_KW, fontsize=10))
    fig.savefig(path, bbox_inches='tight')
    print('saved', path)


def main(runs, peaks=PEAKS, tag=''):
    lines = ['% Gaussian + power-law fits to the peaks of p(m1 | q in Q), minimizing the KL divergence (cross entropy),'
             ' median and 90% over draws with a peak (macros/peak_fits.py)']
    for run in runs:
        r = A.load_light(run)
        p_m1_by_q = {qr: A.p_m1_given_q_range(r, qr) for qr in {v[0] for v in peaks.values()}}
        del r
        fits = {name: fit_peak(p_m1_by_q[qr], win) for name, (qr, win, _) in peaks.items()}
        suffix = RUN_NAMES.get(run, run.replace('_', ''))
        lines.append(f'% {run}')
        for name, (qr, win, tex) in peaks.items():
            F = fits[name]
            bad = int(np.isnan(F[:, 1]).sum())
            print(f'{run} peak {name} (q {qr}, m1 {win}): mu {fmt(F[:, 1])}, sigma {fmt(F[:, 2])}, '
                  f'alpha {fmt(F[:, 4])}, f_G {fmt(F[:, 5])}  [{bad} of {len(F)} draws failed or on a bound]')
            lines.append(f'% peak {name}: q in {qr}, m1 window {win} Msun; {bad} of {len(F)} draws dropped')
            for par, col in (('Mu', 1), ('Sigma', 2), ('Alpha', 4), ('FracGauss', 5)):
                dec = MASS_DEC if par in ('Mu', 'Sigma') else None
                lines.append(rf'\newcommand{{\peak{tex}{par}{suffix}}}{{{fmt(F[:, col], dec, strict=dec is not None)}}}')
            # the fraction of draws with a peak inside the window (the summaries above are conditional on it)
            lines.append(rf'\newcommand{{\peak{tex}PeakFrac{suffix}}}{{{pct(1 - bad / len(F))}}}')
        diagnostic(run, p_m1_by_q, fits, os.path.join(REPO, 'plots', f'{run}_peak_fits{tag}.pdf'), peaks)
    path = os.path.join(HERE, f'peak_fits{tag}.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('wrote', path)


if __name__ == '__main__':
    wide = '--wide' in sys.argv[1:]
    runs = [a for a in sys.argv[1:] if not a.startswith('--')] or list(RUN_NAMES)
    main(runs, WIDE_PEAKS, '_wide') if wide else main(runs)
