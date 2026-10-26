"""Paper figures at journal column widths (user, 2026-10-26), drawn at their final size with small fonts instead of shrinking the summary
figure:

    python notebooks/paper_figs.py [mass|chi|m2|spin|book|all] [--core] [--mean-asinh W]
        -> plots/mass_ratio_m1.pdf         Fig. 1: p(m1 | q range) above p(q | m1), the flexible run, one column
           plots/chi_eff_mean_m1.pdf       mu_chi_eff(m1), one row per spin model, one column
           plots/chi_eff_sd_m1.pdf         sigma_chi_eff(m1), one row per spin model, one column
           plots/chi_eff_mean_sd_m1.pdf    both: rows = spin models, columns = mean, sd; two columns
           plots/m2_q_ranges.pdf           p(m2 | q range) for the two q ranges, one row per spin model (user,
                                           2026-10-26); one column
           plots/m2_m1_windows.pdf         p(m2 | m1 in a window around each peak), one row per spin model; one
                                           column
           plots/spin_peak_regions.pdf     Fig. 2: p(chi_eff, chi_p) in the four peak regions, the flexible model's
                                           median (heatmap) and the Gaussian model's (contours); two columns
           plots/peak_bookkeeping.pdf      Fig. 3: (a) peak location against mean q (joint fits, bounded KDE);
                                           (b) fit-free retained fraction for both pairs; one column

The panels are drawn as in analysis.summary_figure (90% bands and medians per q range, black lines for all q; p(q | m1)
median with the 90% / 99% intervals of <q>), on m1 from M_MIN to 200 Msun. --core: the chi_eff rows use
|chi_eff| < analysis.CHI_CORE. The mean rows use a shared asinh(mu / W) scale (W = 0.03 by default, as summary_mean).
Login node: about 1.5 GB for both runs; pin it (taskset) with the texlive module loaded.
"""
import argparse
import os
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'macros'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import plot_style
import analysis as A

RUNS = {'ppc_gwtc5_k10': 'Flexible', 'ppc_gwtc5_k100_gauss': 'Gaussian'}
M_MIN = 5.
ONE_COL, TWO_COL = 3.375, 7.0                   # APS column widths [in]
SMALL_RC = {'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5,
          'lines.linewidth': 1.0, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
          'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4, 'xtick.major.size': 3, 'ytick.major.size': 3,
          'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5}
LEG = dict(handlelength=1.2, handletextpad=0.4, borderaxespad=0.4, labelspacing=0.3, frameon=False)
COLS = [f'C{i}' for i in range(len(A.SUMMARY_Q_RANGES))]
Q_NAMES = [rf'${a:g} < q < {b:g}$' for a, b in A.SUMMARY_Q_RANGES]

m_edges = np.exp(A.lnm_edges)
mx = np.concatenate([[m_edges[0]], A.m_c, [m_edges[-1]]])        # bin centres, extended to the outer edges
ext = lambda y: np.concatenate([y[..., :1], y, y[..., -1:]], -1)
q5 = lambda x: np.nanpercentile(x, [5, 50, 95], axis=0)
vis = A.m_c >= M_MIN


def bands(ax, allq, byq):
    for y, col in zip(byq, COLS):
        lo, med, hi = ext(q5(y))
        ax.fill_between(mx, lo, hi, color=col, alpha=0.3, lw=0)
        ax.plot(mx, med, color=col, lw=1.0)
    lo, _, hi = ext(q5(allq))
    ax.plot(mx, lo, color='k', lw=0.7); ax.plot(mx, hi, color='k', lw=0.7)


def m1_axis(ax, label=True):
    for m in A.SUMMARY_M_GRID:
        ax.axvline(m, color='0.6', lw=0.5, alpha=0.7, zorder=3)
    ax.set_xscale('log'); ax.set_xlim(M_MIN, A.MAX_M)
    ax.set_xticks(A.SUMMARY_M_GRID, [f'${m:g}$' for m in A.SUMMARY_M_GRID])
    if label:
        ax.set_xlabel(r'$m_1\,[M_\odot]$')


def band_legend():
    handles = [Line2D([], [], color='k', lw=0.7)] + [(Patch(color=c, alpha=0.3, lw=0), Line2D([], [], color=c, lw=1.0))
                                                    for c in COLS]
    return handles, [r'all $q$ (90\%)'] + Q_NAMES


def mass_figure(r):
    """Fig. 1: p(m1) for all q and each q range (top) and p(q | m1) (bottom), one column."""
    ok = A.q_c[None, :] * A.m_c[:, None] >= A.MIN_M
    pq = r.p_mass / r.p_mass.sum(2, keepdims=True)
    mq = (pq * A.q_c).sum(2)
    img = np.where(ok, np.median(pq, 0) / A.dq, np.nan)
    del pq
    p_m, p_mq = r.p_lnm / A.m_c[None], [A.p_m1_given_q_range(r, qr) for qr in A.SUMMARY_Q_RANGES]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(ONE_COL, 3.9), sharex=True)
    bands(a1, p_m, p_mq)
    a1.set_yscale('log')
    lo = np.nanpercentile(np.nanpercentile(p_m, 5, 0)[vis], 2)
    hi = max(np.nanmax(np.nanpercentile(y, 95, 0)[vis]) for y in [p_m] + p_mq)
    a1.set_ylim(lo / 3, hi * 3)
    a1.set_ylabel(r'$p(m_1)\,[M_\odot^{-1}]$')
    fig.legend(*band_legend(), loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0), **LEG)
    pc = a2.pcolormesh(m_edges, A.q_edges, img.T, cmap='viridis', shading='flat', rasterized=True, vmin=0,
                       vmax=np.nanpercentile(img[vis], 99.5))
    for (a, b), ls in (((5, 95), '-'), ((0.5, 99.5), '--')):
        for f in (a, b):
            a2.plot(mx, ext(np.percentile(mq, f, 0)), color='w', lw=0.7, ls=ls)
    a2.set_ylim(0, 1); a2.set_ylabel(r'$q$')
    a2.tick_params(which='both', direction='out')
    dark = lambda ls: Line2D([], [], color='k', lw=0.7, ls=ls)
    a2.legend([dark('-'), dark('--')], [r'$\langle q \rangle$, 90\%', r'$\langle q \rangle$, 99\%'], loc='lower left',
              **dict(LEG, frameon=True, framealpha=0.85, edgecolor='none'))
    for ax in (a1, a2):
        m1_axis(ax, ax is a2)
    cb = fig.colorbar(pc, cax=a2.inset_axes([1.02, 0, 0.03, 1])); cb.set_label(r'$p(q \mid m_1)$')
    cb.ax.tick_params(direction='out')
    fig.tight_layout(rect=(0, 0, 1, 0.955)); fig.subplots_adjust(hspace=0.05)
    return fig


NS = 4                                                          # sub-cells per axis for the m2 transform
lm2_edges = np.linspace(np.log(A.MIN_M), np.log(A.MAX_M), 121)
m2_c = np.exp(0.5 * (lm2_edges[1:] + lm2_edges[:-1]))


def p_m2_given_q_range(r, q_range, m1_range=None):
    """p(m2 | q in q_range[, m1 in m1_range]) per draw, density per Msun on m2_c: each (ln m1, q) cell split into
    NS x NS sub-cells, binned in ln m2 = ln m1 + ln q."""
    du = (np.arange(NS) + 0.5) / NS - 0.5
    lm2 = (A.lnm_c[:, None, None, None] + du[None, :, None, None] * A.dlnm
           + np.log(np.maximum(A.q_c[None, None, :, None] + du[None, None, None, :] * A.dq, 1e-6)))
    idx = np.moveaxis(np.searchsorted(lm2_edges, lm2) - 1, 2, 1).reshape(A.B * A.B, NS * NS)
    keep = ((A.q_c >= q_range[0]) & (A.q_c <= q_range[1]))[None] & np.ones((A.B, 1), bool)
    if m1_range is not None:
        keep = keep & ((A.m_c >= m1_range[0]) & (A.m_c <= m1_range[1]))[:, None]
    keep = keep.ravel()
    W = np.zeros((A.B * A.B, len(m2_c)))
    for k in range(NS * NS):
        good = (idx[:, k] >= 0) & (idx[:, k] < len(m2_c))
        np.add.at(W, (np.where(good)[0], idx[good, k]), keep[good] / NS ** 2)
    out = r.p_mass.reshape(len(r.p_mass), -1).astype(float) @ W
    return out / (out.sum(1, keepdims=True) * np.diff(lm2_edges)[0]) / m2_c[None]


def m2_figure(R):
    """p(m2 | q range) for each of SUMMARY_Q_RANGES (90% bands, medians; colours as in mass_figure), one row per spin
    model, shared axes."""
    fig, axes = plt.subplots(len(R), 1, figsize=(ONE_COL, 1.65 * len(R) + 0.45), sharex=True, sharey=True)
    hi = 0
    for ax, (run, r) in zip(axes, R.items()):
        for qr, col in zip(A.SUMMARY_Q_RANGES, COLS):
            p2 = p_m2_given_q_range(r, qr)
            lo_, med, hi_ = np.nanpercentile(p2, [5, 50, 95], axis=0)
            ax.fill_between(m2_c, lo_, hi_, color=col, alpha=0.3, lw=0)
            ax.plot(m2_c, med, color=col, lw=1.0)
            hi = max(hi, np.nanmax(hi_[m2_c >= M_MIN]))
            s = (m2_c >= 5) & (m2_c <= 20)
            modes = m2_c[s][np.argmax(p2[:, s], 1)]
            print(f'{RUNS[run]}: mode of p(m2 | {qr[0]} < q < {qr[1]}) in 5-20 Msun: {np.median(modes):.1f} '
                  f'({np.percentile(modes, 5):.1f}-{np.percentile(modes, 95):.1f}) Msun')
        m1_axis(ax, ax is axes[-1])
        ax.set_ylabel(r'$p(m_2)\,[M_\odot^{-1}]$')
        ax.text(0.03, 0.07, RUNS[run], transform=ax.transAxes, va='bottom', ha='left',
                bbox=dict(facecolor='w', alpha=0.8, edgecolor='none', pad=1.5))
    axes[-1].set_xlabel(r'$m_2\,[M_\odot]$')
    axes[0].set_yscale('log'); axes[0].set_ylim(hi * 1e-5, hi * 3)
    handles = [(Patch(color=c, alpha=0.3, lw=0), Line2D([], [], color=c, lw=1.0)) for c in COLS]
    fig.legend(handles, Q_NAMES, loc='upper center', ncol=len(COLS), bbox_to_anchor=(0.5, 1.0), **LEG)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.17 / fig.get_figheight())); fig.subplots_adjust(hspace=0.06)
    return fig


# m1 windows around each peak for p(m2 | m1 in window) (user, 2026-10-26: e.g. [15, 25] for the ~20 Msun peak)
M1_WINDOWS = {'10': (8., 12.), '20': (15., 25.), '35': (28., 40.), '60': (50., 70.)}
M1_WINDOW_COLOURS = {'10': 'C0', '20': 'C2', '35': 'C3', '60': 'C1'}


def m2_given_m1_figure(R):
    """p(m2 | m1 in each of M1_WINDOWS), all q (90% bands, medians), one row per spin model, shared axes."""
    fig, axes = plt.subplots(len(R), 1, figsize=(ONE_COL, 1.65 * len(R) + 0.45), sharex=True, sharey=True)
    hi = 0
    for ax, (run, r) in zip(axes, R.items()):
        for name, win in M1_WINDOWS.items():
            p2 = p_m2_given_q_range(r, (0., 1.), win)
            lo_, med, hi_ = np.nanpercentile(p2, [5, 50, 95], axis=0)
            s = m2_c <= win[1]
            col = M1_WINDOW_COLOURS[name]
            ax.fill_between(m2_c[s], lo_[s], hi_[s], color=col, alpha=0.3, lw=0)
            ax.plot(m2_c[s], med[s], color=col, lw=1.0)
            hi = max(hi, np.nanmax(hi_[s & (m2_c >= M_MIN)]))
            modes = m2_c[s][np.argmax(p2[:, s], 1)]
            print(f'{RUNS[run]}: mode of p(m2 | {win[0]:g} < m1 < {win[1]:g}): {np.median(modes):.1f} '
                  f'({np.percentile(modes, 5):.1f}-{np.percentile(modes, 95):.1f}) Msun')
        m1_axis(ax, ax is axes[-1])
        ax.set_ylabel(r'$p(m_2 \mid m_1)\,[M_\odot^{-1}]$')
        ax.text(0.97, 0.93, RUNS[run], transform=ax.transAxes, va='top', ha='right',
                bbox=dict(facecolor='w', alpha=0.8, edgecolor='none', pad=1.5))
    axes[-1].set_xlabel(r'$m_2\,[M_\odot]$')
    axes[0].set_yscale('log'); axes[0].set_ylim(hi * 1e-4, hi * 3)
    handles = [(Patch(color=c, alpha=0.3, lw=0), Line2D([], [], color=c, lw=1.0)) for c in M1_WINDOW_COLOURS.values()]
    labels = [rf'${a:g} < m_1 < {b:g}$' for a, b in M1_WINDOWS.values()]
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0), **LEG)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / fig.get_figheight())); fig.subplots_adjust(hspace=0.06)
    return fig


# Fig. 2: the four peak regions (m1 range, q range, label), in the plan's order (draft/main.tex)
SPIN_PEAK_REGIONS = (((8., 12.), (0.8, 1.0)), ((30., 40.), (0.8, 1.0)), ((16., 22.), (0.3, 0.65)),
                     ((55., 65.), (0.3, 0.65)))


def region_medians(run, regions):
    """Median over draws of p(chi_eff, chi_p | region) (density) for each of ``regions`` (members of
    analysis.REGION_LIST), reading only those regions from the spin cache."""
    import glob
    import h5py
    files = sorted(glob.glob(os.path.join(A.REPO, 'results', run, 'chain_*_samples.h5')))
    n = min(len(h5py.File(f, 'r')['log_likelihood']) for f in files)
    idx = [A.REGION_LIST.index(reg) for reg in regions]
    order = np.argsort(idx)
    P = np.concatenate([h5py.File(A.post_path(run, f), 'r')['REGION'][:n, sorted(idx)] for f in files])
    P = P[:, np.argsort(order)].astype(float)                              # back to the order of ``regions``
    P /= P.sum((2, 3), keepdims=True)
    return np.median(P, 0) / (A.de * A.dp)


def hpd_levels_2d(img, fracs):
    """Density levels of the 2D density ``img`` (on equal cells) enclosing ``fracs`` of its probability."""
    d = np.sort(np.nan_to_num(img).ravel())[::-1]
    c = np.cumsum(d) / d.sum()
    return sorted(d[np.searchsorted(c, f)] for f in fracs)


def spin_peaks_figure():
    """Fig. 2: one panel per peak region: the flexible model's median p(chi_eff, chi_p | region) (log colour scale,
    shared) and the Gaussian model's 50% / 90% regions (contours)."""
    from matplotlib.colors import LogNorm
    Mf = region_medians('ppc_gwtc5_k10', SPIN_PEAK_REGIONS)
    Mg = region_medians('ppc_gwtc5_k100_gauss', SPIN_PEAK_REGIONS)
    vmax, vmin = np.nanmax(Mf), A.REGION_DENSITY_MIN
    fig, axes = plt.subplots(1, len(SPIN_PEAK_REGIONS), figsize=(TWO_COL, 2.05), sharey=True)
    for ax, (mr, qr), f, g in zip(axes, SPIN_PEAK_REGIONS, Mf, Mg):
        pcm = ax.pcolormesh(A.e_edges, A.p_edges, np.where(f > 0, f, np.nan).T, norm=LogNorm(vmin=vmin, vmax=vmax),
                            cmap='inferno', shading='flat', rasterized=True)
        ax.contour(A.e_c, A.p_c, g.T, levels=hpd_levels_2d(g, (0.9, 0.5)), colors='c', linewidths=(0.7, 1.0),
                   linestyles=('--', '-'))
        ax.set_xlim(*A.REGION_CHI_EFF_RANGE); ax.set_ylim(0, 1)
        ax.tick_params(which='both', direction='out')
        ax.set_xlabel(r'$\chi_\mathrm{eff}$')
        ax.text(0.04, 0.96, rf'$m_1 \in [{mr[0]:g}, {mr[1]:g}]\,M_\odot$' '\n' rf'$q \in [{qr[0]:g}, {qr[1]:g}]$',
                transform=ax.transAxes, va='top', ha='left', color='w', fontsize=7)
    axes[0].set_ylabel(r'$\chi_p$')
    handles = [Line2D([], [], color='c', lw=1.0), Line2D([], [], color='c', lw=0.7, ls='--')]
    axes[-1].legend(handles, ['Gaussian, 50\\%', 'Gaussian, 90\\%'], loc='lower right', labelcolor='w', **LEG)
    fig.tight_layout(rect=(0, 0, 0.93, 1)); fig.subplots_adjust(wspace=0.06)
    cax = fig.add_axes([0.935, axes[0].get_position().y0, 0.012, axes[0].get_position().height])
    cb = fig.colorbar(pcm, cax=cax); cb.set_label(r'$p(\chi_\mathrm{eff}, \chi_p \mid m_1, q)$', fontsize=7)
    cb.ax.tick_params(direction='out')
    return fig


def bookkeeping_figure():
    """Fig. 3: (a) each peak's location against the mean q of its peak (joint fits; reflection-bounded KDE, 90% solid
    and 50% dashed; both spin models); (b) the fit-free retained fraction mu_hi / M_lo for both pairs."""
    import window_fits_fig as W
    import peak_pairs as PP
    from peak_fits import PEAKS
    post, _ = W.load()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(ONE_COL, 4.5), gridspec_kw=dict(height_ratios=(1.3, 1)))
    colours = {'Flexible': 'C0', 'Gaussian': 'C3'}
    for name, (_, win, _) in PEAKS.items():
        lo, hi = (np.log(win[0]), 0.), (np.log(win[1]), 1.)
        X, Y = np.meshgrid(np.linspace(lo[0], hi[0], 120), np.linspace(0, 1, 120))
        grid = np.vstack([X.ravel(), Y.ravel()])
        for run, (label, _) in W.RUNS.items():
            st = post[run][name]
            xy = np.vstack([np.log(st['mu']), st['qmean']])
            dens = W.bounded_kde(xy, lo, hi)
            a1.contour(np.exp(X), Y, dens(grid).reshape(X.shape), levels=sorted(W.hpd_levels(dens, xy)),
                       colors=colours[label], linewidths=(1.0, 0.7), linestyles=('-', '--'))
    m1_axis(a1, True); a1.set_xlim(4, 110)
    a1.set_xlabel(r'peak location $\mu\,[M_\odot]$'); a1.set_ylabel(r'mean $q$ of the peak'); a1.set_ylim(0, 1)
    a1.legend([Line2D([], [], color=c, lw=1.0) for c in colours.values()], list(colours), loc='lower left', **LEG)
    bins = np.linspace(0.6, 1.5, 46)
    pair_col = {('10', '20'): 'C2', ('35', '60'): 'C1'}
    for run, (label, _) in PP.RUN_LABELS.items():
        for pair, col in pair_col.items():
            st = PP.pair_stats(run, *pair)
            a2.hist(st['retained_mode'], bins=bins, density=True, histtype='step', color=col, lw=1.0,
                    ls='-' if label == 'Flexible' else '--')
    a2.axvline(PP.RETAINED, color='k', lw=0.8, ls=':')
    a2.axvspan(1., bins[-1], color='0.92', zorder=0, lw=0)
    a2.set_xlim(bins[0], bins[-1]); a2.set_yticks([])
    a2.set_xlabel(r'retained fraction $m_{1,\mathrm{hi}} / M_\mathrm{lo}$'); a2.set_ylabel('density')
    handles = ([Line2D([], [], color=c, lw=1.0) for c in pair_col.values()] +
               [Line2D([], [], color='0.3', lw=1.0), Line2D([], [], color='0.3', lw=1.0, ls='--')])
    labels = [r'$\sim 10 \to \sim 20\,M_\odot$', r'$\sim 35 \to \sim 60\,M_\odot$', 'Flexible', 'Gaussian']
    a2.legend(handles, labels, loc='upper right', **LEG)
    for ax, t in ((a1, '(a)'), (a2, '(b)')):
        ax.text(0.97 if ax is a1 else 0.03, 0.96, t, transform=ax.transAxes, va='top',
                ha='right' if ax is a1 else 'left')
    fig.tight_layout(); fig.subplots_adjust(hspace=0.38)
    return fig


def chi_data(r, core):
    chi = lambda qr=None: A.q_averaged_moments(r, qr, names=('chi_eff',), core=core)['chi_eff']
    allq, byq = chi(), [chi(qr) for qr in A.SUMMARY_Q_RANGES]
    return {'mean': (allq['mean'], [x['mean'] for x in byq]), 'sd': (allq['sd'], [x['sd'] for x in byq])}


def chi_figure(D, stats, core, asinh):
    """Rows: spin models; columns: ``stats`` ('mean', 'sd'). Shared y per column."""
    cut = rf', $|\chi_\mathrm{{eff}}| < {A.CHI_CORE:g}$' if core else ''
    labels = {'mean': r'$\mu_{\chi_\mathrm{eff}}(m_1)$' + cut, 'sd': r'$\sigma_{\chi_\mathrm{eff}}(m_1)$' + cut}
    width = ONE_COL if len(stats) == 1 else TWO_COL
    fig, axes = plt.subplots(len(D), len(stats), figsize=(width, 1.65 * len(D) + 0.45), sharex=True, squeeze=False)
    for j, st in enumerate(stats):
        ys = [y for d in D.values() for y in [d[st][0]] + d[st][1]]
        lo = min(np.nanmin(np.nanpercentile(y, 5, 0)[vis]) for y in ys)
        hi = max(np.nanmax(np.nanpercentile(y, 95, 0)[vis]) for y in ys)
        for i, (name, d) in enumerate(D.items()):
            ax = axes[i, j]
            bands(ax, *d[st])
            m1_axis(ax, i == len(D) - 1)
            if st == 'mean':
                ax.axhline(0, color='0.5', lw=0.6, ls=':')
                ax.set_yscale('asinh', linear_width=asinh)
                ticks = [t for t in (-0.5, -0.1, -0.03, 0, 0.03, 0.1, 0.5)
                         if lo * 1.1 <= t <= hi * 1.1 or t == 0]
                ax.set_yticks(ticks, [f'${t:g}$' for t in ticks])
                ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
                ax.set_ylim(lo * 1.1, hi * 1.1)
            else:
                ax.set_ylim(0, 1.05 * hi)
            ax.set_ylabel(labels[st])
            ax.text(0.03, 0.93, name, transform=ax.transAxes, va='top', ha='left',
                    bbox=dict(facecolor='w', alpha=0.8, edgecolor='none', pad=1.5))
    fig.legend(*band_legend(), loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0), **LEG)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.17 / fig.get_figheight())); fig.subplots_adjust(hspace=0.06)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', nargs='?', default='all', choices=('mass', 'chi', 'm2', 'spin', 'book', 'all'))
    ap.add_argument('--core', action='store_true')
    ap.add_argument('--mean-asinh', type=float, default=0.03)
    args = ap.parse_args()
    need = {'mass': list(RUNS)[:1], 'chi': list(RUNS), 'm2': list(RUNS), 'all': list(RUNS)}.get(args.what, [])
    R = {run: A.load_light(run) for run in need}
    import pixelpop  # noqa: F401  (its import resets matplotlib's rcParams; import it before the style)
    plot_style.apply(); plt.rcParams.update(SMALL_RC)
    core = '_core' if args.core else ''
    if args.what in ('mass', 'all'):
        plot_style.save(mass_figure(R['ppc_gwtc5_k10']), 'mass_ratio_m1')
    if args.what in ('spin', 'all'):
        plot_style.save(spin_peaks_figure(), 'spin_peak_regions')
    if args.what in ('book', 'all'):
        plot_style.save(bookkeeping_figure(), 'peak_bookkeeping')
    if args.what in ('m2', 'all'):
        plot_style.save(m2_figure(R), 'm2_q_ranges')
        plot_style.save(m2_given_m1_figure(R), 'm2_m1_windows')
    if args.what in ('chi', 'all'):
        D = {RUNS[run]: chi_data(r, args.core) for run, r in R.items()}
        for stats, name in ((('mean',), 'mean'), (('sd',), 'sd'), (('mean', 'sd'), 'mean_sd')):
            plot_style.save(chi_figure(D, stats, args.core, args.mean_asinh), f'chi_eff_{name}_m1{core}')


if __name__ == '__main__':
    main()
