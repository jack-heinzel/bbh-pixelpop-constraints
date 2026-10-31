"""Peak significance as the excess over a local power law, per posterior draw, in the fit windows (PEAKS).

    python macros/peak_excess.py [<run> ...]      (default: paper_figs.RUNS, ppc_gwtc5_k10 ppc_gwtc5_k100_gauss)
        -> macros/peak_excess.tex, results/post_processing/peak_excess.csv,
           plots/peak_excess.pdf (every peak x every q slice), plots/peak_excess_own_q.pdf (each peak in its own q range),
           plots/peak_excess_fits_<model>[_own_q].pdf (one spin model each, with the joint fits as dashed lines)

The method of ~/public_html/o4c_population/pixelpop-o4c/post_processing/sept_reruns/bbh_m1m2.ipynb (`peak_locations`,
`peak_significance`), on the (ln m1, q) plane: for each draw, take p(ln m1, q in Q), the windowed mass field
(m2 > 3 Msun) summed over the q bins whose centres lie in Q. In a peak's m1 window, draw the chord in (ln m1, ln p)
from the window's first bin to its last, a local power law, and take the excess

    d(m1) = ln p(ln m1, q in Q) - chord(ln m1).

d is the same for p(ln m1) and p(m1) (they differ by -ln m1, linear in ln m1, which the chord absorbs), and for any
normalization of p within Q. Reported per window and q slice (macros/peak_significance.md, section 1):

* the peak location m_hat (the interior bin of largest d) and height d_max = max d, per draw;
* P_loc, the notebook's local significance: the fraction of draws with d > 0 at m_star, one fixed bin, the interior bin
  where the median of d over draws is largest (the notebook fixes a peak mass by hand instead);
* F_int, the signed integrated excess, sum over the window's bins of (p - exp(chord)) / sum of p, and P(F_int > 0).

These are posterior probabilities, not p-values: a featureless density with noise gives d > 0 about half the time,
and the windows were chosen after seeing the field (see macros/peak_significance.md, section 3, for the null).

Two departures from the notebook:

* The chord joins bin centres. The notebook pairs bin edges with the column of the bin to their left, so its excess
  is exactly 0 in the window's first bin; here both end bins sit on the chord and the interior bins carry the excess.
* m2 > 3 Msun empties part of the low-m1 end at low q (for 0.3 < q < 0.65, every q in the range is allowed only for
  m1 > 10 Msun). In the q bands, a bin is kept only if the cut leaves all its q bins in Q, i.e. m1 q_min >= 3 Msun, so
  the cut cannot make a rising edge; the window used is printed and written to the CSV, and flagged where it was
  shortened. Over all q the cut is part of the population at every m1 (q < 3 / m1 is always removed), so all bins are
  kept there, as in the notebook's marginal; its excess at the low-mass end includes the cut's shape.

The fit figures (user, 2026-10-26) put the joint (m1, q) Gaussian + power-law fits of macros/joint_fits.py (per draw,
results/post_processing/<run>_joint_fits.npz) through the same chord subtraction: each draw's fitted density is
integrated over the panel's q slice (truncated Gaussians in q, analytic), evaluated on the same m1 bins, and its own
chord subtracted; draws without a fitted peak are left out.

Also (user, 2026-10-30, draft TODOs): PLocPrior, P_loc under the mass field's empirical-Bayes prior
(prior_fits.prior_p_mass: each posterior draw's kernel parameters, fresh white noise) at the posterior's m_star bin;
PLocAllQ, P_loc over all q (each peak's window in the 'all' slice, its own m_star); and the events in each peak's
count window (EVENT_WINDOWS, with q in its own range): NEvents, those whose PE medians of m1 and q lie in it, and
NEventsPE, the sum over events of the fraction of their PE samples in it (the PE prior, not population-reweighted).

Needs only the chains' log_p_mass (no spin cache). Runs on a Rusty login node, pinned (taskset; < 1 GB per run).
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
from peak_fits import MASS_DEC, PEAKS, RUN_NAMES, fmt, pct
from prior_fits import load_hyper, prior_p_mass

# q slices of the figure: all q, then the two fit q ranges (PEAKS)
Q_SLICES = {'all': (0., 1.), 'low': (0.3, 0.65), 'high': (0.8, 1.0)}
Q_LABELS = {'all': r'all $q$', 'low': r'$0.3 < q < 0.65$', 'high': r'$0.8 < q < 1$'}
PEAK_LABELS = {'10': r'$10\,\textrm{M}_\odot$', '35': r'$35\,\textrm{M}_\odot$', '20': r'$20\,\textrm{M}_\odot$', '60': r'$60\,\textrm{M}_\odot$'}
RUN_STYLE = ('C0', 'C3')                     # first run: filled bands; second: lines (as the notebook's var 1 / var 4)
CSV = os.path.join(REPO, 'results', 'post_processing', 'peak_excess.csv')


def load_mass(run):
    """Windowed mass field per draw, (draws, Bm, Bq), density per d ln m1 dq, from every chain of results/<run>/
    (as analysis.load_light, without the spin cache)."""
    files = sorted(glob.glob(os.path.join(REPO, 'results', run, 'chain_*_samples.h5')))
    if not files:
        raise FileNotFoundError(f'no chains in results/{run}')
    n_draws = min(len(h5py.File(f, 'r')['log_likelihood']) for f in files)
    lp = np.concatenate([h5py.File(f, 'r')['log_p_mass'][:n_draws] for f in files]).astype(float)
    lp += A.mass_window()[None]
    mx = lp.max((1, 2), keepdims=True)
    lp -= np.log(np.exp(lp - mx).sum((1, 2), keepdims=True) * A.dlnm * A.dq) + mx
    print(f'{run}: {len(files)} chain(s) x {n_draws} draws')
    return np.exp(lp).astype(np.float32)


def slice_density(p_mass, q_range):
    """p(ln m1, q in Q) per draw, (draws, Bm), and the m1 bins where m2 > 3 Msun leaves every q bin of Q."""
    sel = (A.q_c >= q_range[0] - 1e-9) & (A.q_c <= q_range[1] + 1e-9)
    p = p_mass[:, :, sel].sum(2).astype(float) * A.dq
    full = A.m_c * A.q_c[sel].min() >= A.MIN_M if q_range[0] > 0 else np.ones(A.B, bool)   # see the docstring
    return p, full


def excess(p, full, window):
    """Chord excess in one m1 window: dict with the bins used and the per-draw statistics (see the module docstring)."""
    idx = np.flatnonzero((A.m_c >= window[0]) & (A.m_c <= window[1]) & full)
    if len(idx) < 3:
        return None
    x = A.lnm_c[idx]
    with np.errstate(divide='ignore'):
        lp = np.log(p[:, idx])
    t = (x - x[0]) / (x[-1] - x[0])
    chord = lp[:, :1] * (1 - t) + lp[:, -1:] * t
    d = lp - chord                                                    # (draws, n_bins); 0 at both ends
    inner = d[:, 1:-1]
    j = 1 + np.argmax(inner, axis=1)
    j_star = 1 + int(np.argmax(np.median(inner, axis=0)))
    pw = p[:, idx]
    F = (pw - np.exp(chord)).sum(1) / pw.sum(1)
    return dict(idx=idx, m=A.m_c[idx], d=d, m_hat=A.m_c[idx][j], height=inner.max(1), m_star=A.m_c[idx][j_star],
                P_loc=float(np.mean(d[:, j_star] > 0)), F=F, P_F=float(np.mean(F > 0)),
                window_used=(float(A.m_c[idx[0]]), float(A.m_c[idx[-1]])),
                shortened=bool(A.m_c[idx[0]] > A.m_c[(A.m_c >= window[0]) & (A.m_c <= window[1])][0]))


def compute(runs):
    """{run: {(peak, slice): excess dict}} for every peak window and q slice."""
    out = {}
    for run in runs:
        p_mass = load_mass(run)
        out[run] = {}
        for qs, qr in Q_SLICES.items():
            p, full = slice_density(p_mass, qr)
            for name, (_, win, _) in PEAKS.items():
                out[run][(name, qs)] = excess(p, full, win)
        del p_mass
        prior = prior_p_mass(load_hyper(run))
        for name, (qr, win, _) in PEAKS.items():
            e = out[run][(name, own_slice(name))]
            if e is not None:
                p, _ = slice_density(prior, qr)
                d = excess_at(p, e['idx'])
                e['P_loc_prior'] = float(np.mean(d[:, list(e['idx']).index(int(np.flatnonzero(A.m_c == e['m_star'])[0]))] > 0))
        del prior
    return out


def excess_at(p, idx):
    """The chord excess d (draws, len(idx)) on the bins idx (as excess())."""
    x = A.lnm_c[idx]
    with np.errstate(divide='ignore'):
        lp = np.log(p[:, idx])
    t = (x - x[0]) / (x[-1] - x[0])
    return lp - (lp[:, :1] * (1 - t) + lp[:, -1:] * t)


# events counted in each peak's m1 window (the text's windows, user 2026-10-30), q in the peak's own range
EVENT_WINDOWS = {'10': (8., 12.), '35': (30., 40.), '20': (15., 25.), '60': (50., 70.)}


def event_counts():
    """{peak: (n by PE medians, sum of PE fractions)} in EVENT_WINDOWS x the peak's q range."""
    import pickle
    with open(os.path.join(A.DATA_DIRS['gwtc5'], 'posteriors.pkl'), 'rb') as f:
        d = pickle.load(f)
    m1, q, n = np.asarray(d['posteriors']['mass_1']), np.asarray(d['posteriors']['mass_ratio']), np.asarray(d['event_counts'])
    out = {}
    for name, (qr, _, _) in PEAKS.items():
        lo, hi = EVENT_WINDOWS[name]
        med, frac = 0, 0.
        for i in range(len(n)):
            a, b = m1[i, :n[i]], q[i, :n[i]]
            inside = lambda a, b: (a >= lo) & (a <= hi) & (b >= qr[0]) & (b <= qr[1])
            med += bool(inside(np.median(a), np.median(b)))
            frac += float(np.mean(inside(a, b)))
        out[name] = (med, frac)
    return out


def own_slice(name):
    """The q slice that is the peak's own fit q range (PEAKS)."""
    qr = PEAKS[name][0]
    return next(k for k, v in Q_SLICES.items() if tuple(v) == tuple(qr))


def table(res):
    import pandas as pd
    rows = []
    for run, R in res.items():
        for (name, qs), e in R.items():
            row = dict(run=run, peak=name, q_slice=qs, q_range=Q_SLICES[qs], window=PEAKS[name][1],
                       own_q=qs == own_slice(name))
            if e is None:
                rows.append(dict(row, note='fewer than 3 bins left by m2 > 3 Msun'))
                continue
            q = lambda v: np.percentile(v, [50, 5, 95])
            mh, ht, F = q(e['m_hat']), q(e['height']), q(e['F'])
            rows.append(dict(row, window_used=tuple(round(v, 2) for v in e['window_used']), shortened=e['shortened'],
                             m_hat=mh[0], m_hat_lo=mh[1], m_hat_hi=mh[2], height=ht[0], height_lo=ht[1], height_hi=ht[2],
                             m_star=round(float(e['m_star']), 2), P_loc=e['P_loc'], F_int=F[0], F_int_lo=F[1],
                             F_int_hi=F[2], P_F_int=e['P_F'], note=''))
    return pd.DataFrame(rows)


def macros(res):
    lines = ['% Excess over a local power law (chord in ln m1, ln p) in the fit windows, each peak in its own q range '
             '(macros/peak_excess.py)',
             '% PLoc: P(excess > 0) at the bin of largest median excess; Height: max excess (natural log), median and '
             '90%; Loc: its m1; FInt: signed integrated excess fraction; PFInt: P(FInt > 0)']
    for run, R in res.items():
        suffix = RUN_NAMES.get(run, run.replace('_', ''))
        lines.append(f'% {run}')
        for name, (qr, win, tex) in PEAKS.items():
            e = R[(name, own_slice(name))]
            if e is None:
                continue
            lines.append(f'% peak {name}: q in {qr}, m1 window {win} Msun, bins used {e["window_used"]}'
                         + (' (shortened by m2 > 3 Msun)' if e['shortened'] else ''))
            lines += [rf'\newcommand{{\peak{tex}ExcessPLoc{suffix}}}{{{pct(e["P_loc"])}}}',
                      rf'\newcommand{{\peak{tex}ExcessHeight{suffix}}}{{{fmt(e["height"])}}}',
                      rf'\newcommand{{\peak{tex}ExcessLoc{suffix}}}{{{fmt(e["m_hat"], MASS_DEC)}}}',
                      rf'\newcommand{{\peak{tex}ExcessFInt{suffix}}}{{{fmt(e["F"])}}}',
                      rf'\newcommand{{\peak{tex}ExcessPFInt{suffix}}}{{{pct(e["P_F"])}}}']
            if 'P_loc_prior' in e:
                lines.append(rf'\newcommand{{\peak{tex}ExcessPLocPrior{suffix}}}{{{pct(e["P_loc_prior"])}}}')
            a = R[(name, 'all')]
            if a is not None:
                lines.append(rf'\newcommand{{\peak{tex}ExcessPLocAllQ{suffix}}}{{{pct(a["P_loc"])}}}')
    lines.append(f'% events in m1 windows {EVENT_WINDOWS} Msun, q in each peak\'s own range: NEvents by PE medians, '
                 'NEventsPE the summed PE fractions')
    for name, (n_med, n_pe) in event_counts().items():
        tex = PEAKS[name][2]
        print(f'peak {name}: {n_med} events by PE medians, {n_pe:.1f} by PE fractions', flush=True)
        lines += [rf'\newcommand{{\peak{tex}NEvents}}{{{n_med}}}', rf'\newcommand{{\peak{tex}NEventsPE}}{{{n_pe:.0f}}}']
    path = os.path.join(HERE, 'peak_excess.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'wrote {path}')


def fit_slice_density(F, m, q_range, window):
    """ln of each fitted joint model (rows of F, joint_fits columns) integrated over q in q_range, at m1 values m,
    per unit ln m1 up to a constant (draws, len(m)); NaN rows where the fit has no peak."""
    from scipy.special import ndtr
    lnA, mu, sigma, mu_q, sd_q, lnB, alpha, mu_b, sd_b = (F[:, k][:, None] for k in range(9))
    qmin = A.MIN_M / m[None]
    lo, hi = np.maximum(q_range[0], qmin), np.minimum(q_range[1], 1.)

    def frac(mq, sq):                                                  # P(q in [lo, hi]) under TN on [qmin, 1]
        z = ndtr((1 - mq) / sq) - ndtr((qmin - mq) / sq)
        return np.clip(ndtr((hi - mq) / sq) - ndtr((lo - mq) / sq), 0, None) / np.maximum(z, 1e-300)
    m_ref = 0.5 * (window[0] + window[1])
    peak = np.exp(lnA - 0.5 * ((m[None] - mu) / sigma) ** 2) * frac(mu_q, sd_q)
    bkg = np.exp(lnB - alpha * np.log(m[None] / m_ref)) * frac(mu_b, sd_b)
    with np.errstate(divide='ignore'):
        return np.log(m[None] * (peak + bkg))


def fit_excess(run, name, q_range, e):
    """The chord excess of the joint fits for one window and q slice, on the bins of ``e`` (draws with a peak)."""
    from joint_fits import npz_path
    F = np.load(npz_path(run))[name]
    F = F[np.all(np.isfinite(F), 1)]
    lf = fit_slice_density(F, e['m'], q_range, PEAKS[name][1])
    t = (A.lnm_c[e['idx']] - A.lnm_c[e['idx'][0]]) / (A.lnm_c[e['idx'][-1]] - A.lnm_c[e['idx'][0]])
    return lf - (lf[:, :1] * (1 - t) + lf[:, -1:] * t)


def draw_panel(ax, entries, labels, colours=RUN_STYLE, fit=None, window=None):
    """One window x q slice: the excess d(m1) per run (first run filled 90% / 99% bands and median, second run lines),
    the median peak location (dotted), and P_loc per run in the corner."""
    texts = []
    for i, (e, col) in enumerate(zip(entries, colours)):
        if e is None:
            continue
        m, d = e['m'], e['d']
        p05, p5, p50, p95, p995 = np.percentile(d, [0.5, 5, 50, 95, 99.5], axis=0)
        if i == 0:
            ax.fill_between(m, p05, p995, step='mid', color=col, alpha=0.15, lw=0)
            ax.fill_between(m, p5, p95, step='mid', color=col, alpha=0.3, lw=0)
            ax.step(m, p50, where='mid', color=col)
        else:
            for y, ls, lw in ((p05, '--', 0.7), (p995, '--', 0.7), (p5, '-', 0.9), (p95, '-', 0.9)):
                ax.step(m, y, where='mid', color=col, ls=ls, lw=lw)
        ax.axvline(np.median(e['m_hat']), color=col, ls=':', lw=0.8, zorder=1)
        texts.append((rf'$P_{{\rm loc}} = {e["P_loc"]:.2f}$', col))
    if fit is not None and entries[0] is not None:                 # the joint fits through the same subtraction
        f5, f50, f95 = np.nanpercentile(fit, [5, 50, 95], axis=0)
        m = entries[0]['m']
        ax.step(m, f50, where='mid', color='k', ls='--', lw=0.9, zorder=4)
        for y in (f5, f95):
            ax.step(m, y, where='mid', color='k', ls='--', lw=0.5, zorder=4)
    ax.axhline(0, color='0.5', ls=':', lw=0.7)
    for k, (s, col) in enumerate(texts):
        ax.text(0.04, 0.93 - 0.13 * k, s, transform=ax.transAxes, va='top', ha='left', color=col, fontsize=6.5,
                zorder=5, bbox=dict(boxstyle='round,pad=0.2', facecolor='w', alpha=0.9, edgecolor='none'))
    e0 = next((e for e in entries if e is not None), None)
    if e0 is not None:
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
        # the window's nominal edges as the axis limits and labelled ticks, with one tick between (user, 2026-10-26)
        a, b = window if window is not None else (e0['m'][0], e0['m'][-1])
        ax.set_xscale('log'); ax.set_xlim(a, b)
        ax.xaxis.set_major_locator(FixedLocator([a, float(np.round(np.sqrt(a * b))), b]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'${v:g}$'))
    else:
        ax.text(0.5, 0.5, r'excluded by $m_2 > 3\,\textrm{M}_\odot$', transform=ax.transAxes, ha='center', va='center',
                fontsize=6.5, color='0.4')
        ax.set_xticks([])


def legend(fig, labels, top):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import paper_figs as P
    c0, c1 = RUN_STYLE
    handles = [(Patch(color=c0, alpha=0.3, lw=0), Line2D([], [], color=c0)), Patch(color=c0, alpha=0.15, lw=0),
               Line2D([], [], color=c1, lw=0.9), Line2D([], [], color=c1, ls='--', lw=0.7)][:2 * len(labels)]
    names = [f'{labels[0]}, 90\\%', f'{labels[0]}, 99\\%', f'{labels[1]}, 90\\%', f'{labels[1]}, 99\\%'][:2 * len(labels)]
    fig.legend(handles, names, loc='upper center', bbox_to_anchor=(0.5, top), ncol=len(names), **P.LEG)


def figures(res):
    import pixelpop  # noqa: F401  (its import resets matplotlib's rcParams; import it before the style)
    import matplotlib.pyplot as plt
    import paper_figs as P
    import plot_style
    plot_style.apply(); plt.rcParams.update(P.SMALL_RC)
    runs = list(res)[:2]
    labels = [P.RUNS.get(r, r.replace('_', r'\_')) for r in runs]
    ylab = r'$\ln p - \ln p_{\rm PL}$'

    # every peak window x every q slice
    fig, axes = plt.subplots(len(PEAKS), len(Q_SLICES), figsize=(P.TWO_COL, 1.45 * len(PEAKS) + 0.5), sharey='row')
    for i, (name, (_, win, _)) in enumerate(PEAKS.items()):
        for j, qs in enumerate(Q_SLICES):
            ax = axes[i, j]
            draw_panel(ax, [res[r][(name, qs)] for r in runs], labels, window=win)
            if qs == own_slice(name):
                for s in ax.spines.values():
                    s.set_linewidth(1.2)
            if i == 0:
                ax.text(0.5, 1.04, Q_LABELS[qs], transform=ax.transAxes, ha='center', va='bottom')
            if j == 0:
                ax.set_ylabel(f'{PEAK_LABELS[name]}, ${win[0]:g}$--${win[1]:g}$\n{ylab}')
            if i == len(PEAKS) - 1:
                ax.set_xlabel(r'$m_1\,[\textrm{M}_\odot]$')
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight())); fig.subplots_adjust(hspace=0.35, wspace=0.08)
    legend(fig, labels, 1.0)
    plot_style.save(fig, 'peak_excess')
    plt.close(fig)

    # each peak in its own q range (column width)
    fig, axes = plt.subplots(len(PEAKS), 1, figsize=(P.ONE_COL, 1.25 * len(PEAKS) + 0.5))
    for ax, (name, (qr, win, _)) in zip(axes, PEAKS.items()):
        draw_panel(ax, [res[r][(name, own_slice(name))] for r in runs], labels, window=win)
        ax.text(0.97, 0.93, f'{PEAK_LABELS[name]}, {Q_LABELS[own_slice(name)]}', transform=ax.transAxes, va='top',
                ha='right', bbox=dict(facecolor='w', alpha=0.8, edgecolor='none', pad=1.5))
        ax.set_ylabel(ylab)
    axes[-1].set_xlabel(r'$m_1\,[\textrm{M}_\odot]$')
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight())); fig.subplots_adjust(hspace=0.4)
    legend(fig, labels, 1.0)
    plot_style.save(fig, 'peak_excess_own_q')
    plt.close(fig)


def fit_figures(res):
    """Per spin model: the excess grid and the own-q column, with the joint fits through the same subtraction."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import paper_figs as P
    import plot_style
    ylab = r'$\ln p - \ln p_{\rm PL}$'
    for run, col in zip(list(res)[:2], RUN_STYLE):
        label = P.RUNS.get(run, run)
        fits = {(name, qs): (fit_excess(run, name, Q_SLICES[qs], res[run][(name, qs)])
                             if res[run][(name, qs)] is not None else None)
                for name in PEAKS for qs in Q_SLICES}
        handles = [(Patch(color=col, alpha=0.3, lw=0), Line2D([], [], color=col)), Patch(color=col, alpha=0.15, lw=0),
                   Line2D([], [], color='k', ls='--', lw=0.9), Line2D([], [], color='k', ls='--', lw=0.5)]
        names = [f'{label}, 90\\%', f'{label}, 99\\%', 'window fits (peak + PL), median', 'window fits, 90\\%']
        for grid in (True, False):
            if grid:
                fig, axes = plt.subplots(len(PEAKS), len(Q_SLICES), figsize=(P.TWO_COL, 1.45 * len(PEAKS) + 0.5),
                                         sharey='row')
                cells = [(axes[i, j], name, qs) for i, name in enumerate(PEAKS) for j, qs in enumerate(Q_SLICES)]
            else:
                fig, axes = plt.subplots(len(PEAKS), 1, figsize=(P.ONE_COL, 1.25 * len(PEAKS) + 0.6))
                cells = [(ax, name, own_slice(name)) for ax, name in zip(axes, PEAKS)]
            for ax, name, qs in cells:
                draw_panel(ax, [res[run][(name, qs)]], [label], colours=(col,), fit=fits[(name, qs)],
                           window=PEAKS[name][1])
                win = PEAKS[name][1]
                if grid:
                    i, j = list(PEAKS).index(name), list(Q_SLICES).index(qs)
                    if qs == own_slice(name):
                        for sp in ax.spines.values():
                            sp.set_linewidth(1.2)
                    if i == 0:
                        ax.text(0.5, 1.04, Q_LABELS[qs], transform=ax.transAxes, ha='center', va='bottom')
                    if j == 0:
                        ax.set_ylabel(f'{PEAK_LABELS[name]}, ${win[0]:g}$--${win[1]:g}$\n{ylab}')
                    if i == len(PEAKS) - 1:
                        ax.set_xlabel(r'$m_1\,[\textrm{M}_\odot]$')
                else:
                    ax.text(0.97, 0.93, f'{PEAK_LABELS[name]}, {Q_LABELS[qs]}', transform=ax.transAxes, va='top',
                            ha='right', bbox=dict(facecolor='w', alpha=0.8, edgecolor='none', pad=1.5))
                    ax.set_ylabel(ylab)
            if not grid:
                axes[-1].set_xlabel(r'$m_1\,[\textrm{M}_\odot]$')
            top = 1 - (0.35 if grid else 0.5) / fig.get_figheight()
            fig.tight_layout(rect=(0, 0, 1, top)); fig.subplots_adjust(hspace=0.35 if grid else 0.4, wspace=0.08)
            fig.legend(handles, names, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4 if grid else 2, **P.LEG)
            plot_style.save(fig, f'peak_excess_fits_{label.lower()}' + ('' if grid else '_own_q'))
            plt.close(fig)


def main(runs):
    import paper_figs as P
    runs = runs or list(P.RUNS)
    res = compute(runs)
    df = table(res)
    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    df.to_csv(CSV, index=False)
    print(f'wrote {CSV}')
    cols = ['run', 'peak', 'q_slice', 'window_used', 'shortened', 'm_star', 'P_loc', 'height', 'm_hat', 'F_int',
            'P_F_int', 'note']
    with np.printoptions(precision=3):
        print(df[[c for c in cols if c in df]].to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    macros(res)
    figures(res)
    fit_figures(res)


if __name__ == '__main__':
    main(sys.argv[1:])
