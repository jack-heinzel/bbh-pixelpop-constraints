"""Paper figures at journal column widths (user, 2026-10-26), drawn at their final size with small fonts instead of shrinking the summary
figure:

    python notebooks/paper_figs.py [mass|chi|m2|spin|book|talk|all] [--core] [--mean-asinh W]
        -> plots/mass_ratio_m1.pdf         Fig. 1: p(m1 | q range) above p(q | m1), the flexible run, one column
           plots/mass_ratio_m1_mode.pdf    Fig. 1 with intervals of the mode of p(q | m1) instead of <q>
           plots/mass_ratio_m1_hpd50.pdf   Fig. 1 with the 50% highest-density set of p(q | m1) instead of <q>
           plots/chi_eff_mean_m1.pdf       mu_chi_eff(m1), one row per spin model, one column
           plots/chi_eff_sd_m1.pdf         sigma_chi_eff(m1), one row per spin model, one column
           plots/chi_eff_mean_sd_m1.pdf    both: rows = spin models, columns = mean, sd; two columns
           plots/chi_p_{mean,sd,mean_sd}_m1.pdf   the same for chi_p (rows = spin models)
           --prior: the chi / chip figures with the empirical-Bayes spin prior (macros/spin_prior.py) overlaid,
           suffix _prior
           plots/m2_q_ranges.pdf           p(m2 | q range) for the two q ranges, one row per spin model (user,
                                           2026-10-26); one column
           plots/m2_m1_windows.pdf         p(m2 | m1 in a window around each peak), one row per spin model; one
                                           column
           plots/spin_peak_regions.pdf     Fig. 2: p(chi_eff, chi_p) in the four peak regions, the flexible model's
                                           median (heatmap) and the Gaussian model's (contours); two columns
           plots/spin_peak_regions_slices.pdf   Fig. 2 with slice panels (median and 90% over draws)
           plots/spin_peak_regions_marginals.pdf   Fig. 2 with 1D marginal panels of both models (median and 90%)
           plots/mass_spin_zoom_talk.pdf   for talks: median p(m1, q), zooms on the peak regions, their spin blocks
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
          'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
          'savefig.dpi': 600}          # rasterized heatmaps: at 100 dpi their bins resample unevenly (aliasing; user, 2026-10-30)
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
        ax.set_xlabel(r'$m_1\,[\textrm{M}_\odot]$')


def band_legend():
    handles = [Line2D([], [], color='k', lw=0.7)] + [(Patch(color=c, alpha=0.3, lw=0), Line2D([], [], color=c, lw=1.0))
                                                    for c in COLS]
    return handles, [r'all $q$ (90\%)'] + Q_NAMES


def hpd_membership(pq, frac):
    """(draws, Bm, Bq) bool: the q bins in each draw's and m1 column's highest-density set holding ``frac`` of
    p(q | m1) (``pq``: per-bin probabilities summing to 1 over q; the bin that crosses ``frac`` is included)."""
    order = np.argsort(-pq, axis=2)
    c = np.cumsum(np.take_along_axis(pq, order, 2), 2)
    keep = (c - np.take_along_axis(pq, order, 2)) < frac            # cumulative mass before the bin is below frac
    out = np.zeros(pq.shape, bool)
    np.put_along_axis(out, order, keep, 2)
    return out


def mass_figure(r, mode=False, hpd=None):
    """Fig. 1: p(m1) for all q and each q range (top) and p(q | m1) (bottom), one column. ``mode`` (user,
    2026-10-30): the intervals over draws of the mode of p(q | m1) (dashed: 90%, dotted: 99%) instead of <q>; the mode
    ignores the tails that broaden where the data run out, which pull <q> towards the middle of the window.
    ``hpd`` = f: instead, the per-draw highest-density set of p(q | m1) holding f of it at each m1, shown by the
    fraction of draws containing each (m1, q) cell: contours at 0.5 (solid) and 0.05 (dashed). Unlike <q> it
    reaches the q = 1 edge, and splits where p(q | m1) has two peaks.
    The excluded zone m2 < MIN_M is grey below the smooth curve q = MIN_M / m1 (as Fig. 2's chi_p_max): every cell
    with any part above it is coloured (zero density if its centre is below, as the model's window), so no cell left
    blank (NaN) reaches above the curve."""
    ok = A.q_edges[None, 1:] * m_edges[1:, None] >= A.MIN_M                 # any part of the cell has m2 >= MIN_M
    pq = r.p_mass / r.p_mass.sum(2, keepdims=True)
    mq = A.q_c[np.argmax(pq, 2)] if mode else (pq * A.q_c).sum(2)
    img = np.where(ok, np.median(pq, 0) / A.dq, np.nan)
    if hpd:
        P_in = hpd_membership(pq, hpd).mean(0)
    del pq
    p_m, p_mq = r.p_lnm / A.m_c[None], [A.p_m1_given_q_range(r, qr) for qr in A.SUMMARY_Q_RANGES]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(ONE_COL, 3.9), sharex=True)
    bands(a1, p_m, p_mq)
    a1.set_yscale('log')
    lo = np.nanpercentile(np.nanpercentile(p_m, 5, 0)[vis], 2)
    hi = max(np.nanmax(np.nanpercentile(y, 95, 0)[vis]) for y in [p_m] + p_mq)
    a1.set_ylim(lo / 3, hi * 3)
    a1.set_ylabel(r'$p(m_1)\,[\textrm{M}_\odot^{-1}]$')
    fig.legend(*band_legend(), loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0), **LEG)
    pc = a2.pcolormesh(m_edges, A.q_edges, img.T, cmap='viridis', shading='flat', rasterized=True, vmin=0,
                       vmax=np.nanpercentile(img[vis], 99.5))
    m = np.geomspace(m_edges[0], m_edges[-1], 2000)
    a2.fill_between(m, 0., np.minimum(A.MIN_M / m, 1.), color=EXCLUDED_GREY, lw=0, zorder=1.5)
    styles = (('--', ':') if mode else ('-', '--'))
    if hpd:
        qq = np.concatenate([[0.], A.q_c, [1.]])                     # to the q = 0, 1 edges: the set can reach them
        P = np.concatenate([P_in[:, :1], P_in, P_in[:, -1:]], 1)
        a2.contour(A.m_c, qq, P.T, levels=[0.05, 0.5], colors='w', linewidths=(0.6, 0.9), linestyles=('--', '-'))
    else:
        for (a, b), ls in zip(((5, 95), (0.5, 99.5)), styles):
            for f in (a, b):
                a2.plot(mx, ext(np.percentile(mq, f, 0)), color='w', lw=0.7, ls=ls)
    a2.set_ylim(0, 1); a2.set_ylabel(r'$q$')
    a2.tick_params(which='both', direction='out')
    dark = lambda ls: Line2D([], [], color='k', lw=0.7, ls=ls)
    if hpd:
        h = rf'{100 * hpd:g}\% HPD of $p(q \mid m_1)$'
        a2.legend([dark('-'), dark('--')], [h + r', in $\geq 50\%$ of draws', h + r', in $\geq 5\%$'], loc='lower left',
                  **dict(LEG, frameon=True, framealpha=0.85, edgecolor='none'))
    else:
        stat = r'mode of $q$' if mode else r'$\langle q \rangle$'
        a2.legend([dark(ls) for ls in styles], [stat + r', 90\%', stat + r', 99\%'], loc='lower left',
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
        ax.set_ylabel(r'$p(m_2)\,[\textrm{M}_\odot^{-1}]$')
        ax.text(0.03, 0.07, RUNS[run], transform=ax.transAxes, va='bottom', ha='left',
                bbox=dict(facecolor='w', alpha=0.8, edgecolor='none', pad=1.5))
    axes[-1].set_xlabel(r'$m_2\,[\textrm{M}_\odot]$')
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
        ax.set_ylabel(r'$p(m_2 \mid m_1)\,[\textrm{M}_\odot^{-1}]$')
        ax.text(0.97, 0.93, RUNS[run], transform=ax.transAxes, va='top', ha='right',
                bbox=dict(facecolor='w', alpha=0.8, edgecolor='none', pad=1.5))
    axes[-1].set_xlabel(r'$m_2\,[\textrm{M}_\odot]$')
    axes[0].set_yscale('log'); axes[0].set_ylim(hi * 1e-4, hi * 3)
    handles = [(Patch(color=c, alpha=0.3, lw=0), Line2D([], [], color=c, lw=1.0)) for c in M1_WINDOW_COLOURS.values()]
    labels = [rf'${a:g} < m_1 < {b:g}$' for a, b in M1_WINDOWS.values()]
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0), **LEG)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / fig.get_figheight())); fig.subplots_adjust(hspace=0.06)
    return fig


# Fig. 2: the four peak regions (m1 range, q range, label), in the plan's order (draft/main.tex)
SPIN_PEAK_REGIONS = (((8., 12.), (0.8, 1.0)), ((30., 40.), (0.8, 1.0)), ((16., 22.), (0.3, 0.65)),
                     ((55., 65.), (0.3, 0.65)))


def region_draws(run, regions):
    """p(chi_eff, chi_p | region) per draw (density, float32, (draws, n_regions, Be, Bp)) for each of ``regions``
    (members of analysis.REGION_LIST), reading only those regions from the spin cache."""
    import glob
    import h5py
    files = sorted(glob.glob(os.path.join(A.REPO, 'results', run, 'chain_*_samples.h5')))
    n = min(len(h5py.File(f, 'r')['log_likelihood']) for f in files)
    idx = [A.REGION_LIST.index(reg) for reg in regions]
    P = np.concatenate([h5py.File(A.post_path(run, f), 'r')['REGION'][:n, sorted(idx)] for f in files])
    P = P[:, np.argsort(np.argsort(idx))].astype(np.float32)               # back to the order of ``regions``
    P /= P.sum((2, 3), keepdims=True)
    return P / np.float32(A.de * A.dp)


def region_medians(run, regions):
    """Median over draws of region_draws."""
    return np.median(region_draws(run, regions), 0)


def hpd_levels_2d(img, fracs):
    """Density levels of the 2D density ``img`` (on equal cells) enclosing ``fracs`` of its probability."""
    d = np.sort(np.nan_to_num(img).ravel())[::-1]
    c = np.cumsum(d) / d.sum()
    return sorted(d[np.searchsorted(c, f)] for f in fracs)


LABEL_BOX = dict(boxstyle='round,pad=0.25', facecolor='w', alpha=0.85, edgecolor='none')
SLICE_COLOURS = ('C0', 'C3')                     # as analysis.spin_region_figure (blue, red)


def region_label(mr, qr):
    return rf'$m_1 \in [{mr[0]:g}, {mr[1]:g}]\,\textrm{{M}}_\odot$' '\n' rf'$q \in [{qr[0]:g}, {qr[1]:g}]$'


def spin_heatmap(ax, med_f, med_g, norm, slices=False, contour=None):
    """The flexible model's median p(chi_eff, chi_p) (heatmap) and the Gaussian model's 50% / 90% regions; with
    ``slices``, dashed lines at the slice values (analysis.REGION_CHI_P_SLICES, REGION_CHI_EFF_SLICES)."""
    pcm = ax.pcolormesh(A.e_edges, A.p_edges, np.where(med_f > 0, med_f, np.nan).T, norm=norm, cmap='inferno',
                        shading='flat', rasterized=True)
    if contour is None:                                    # the Gaussian's 50% / 90% regions
        ax.contour(A.e_c, A.p_c, med_g.T, levels=hpd_levels_2d(med_g, (0.9, 0.5)), colors='c', linewidths=(0.7, 1.0),
                   linestyles=('--', '-'))
    else:                                                  # its 90% region only, in the colour ``contour``
        ax.contour(A.e_c, A.p_c, med_g.T, levels=hpd_levels_2d(med_g, (0.9,)), colors=contour, linewidths=0.9)
    if slices:
        for v, c in zip(A.REGION_CHI_P_SLICES, SLICE_COLOURS):
            ax.axhline(A.p_c[_bin(A.p_edges, v)], color=c, lw=0.8, ls='--')
        for v, c in zip(A.REGION_CHI_EFF_SLICES, SLICE_COLOURS):
            ax.axvline(A.e_c[_bin(A.e_edges, v)], color=c, lw=0.8, ls='--')
    ax.set_xlim(*A.REGION_CHI_EFF_RANGE); ax.set_ylim(0, 1)
    ax.tick_params(which='both', direction='out')
    return pcm


def _bin(edges, v):
    return int(np.clip(np.searchsorted(edges, v, side='right') - 1, 0, len(edges) - 2))


def spin_block(fig, spec, Df, med_g, mr, qr, norm, label=True, ylabel=True, xlabel=True, fontsize=7):
    """A heatmap with slice panels above (along chi_eff at fixed chi_p) and to the right (along chi_p at fixed
    chi_eff): the flexible model's median and 90% band over draws (Df: draws, Be, Bp) in the slice colours."""
    sub = spec.subgridspec(2, 2, width_ratios=(3, 1.1), height_ratios=(1.1, 3), hspace=0.05, wspace=0.05)
    ax_m, ax_t, ax_r = fig.add_subplot(sub[1, 0]), fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[1, 1])
    pcm = spin_heatmap(ax_m, np.median(Df, 0), med_g, norm, slices=True)
    lo_d, hi_d = norm.vmin, norm.vmax * 2
    for v, c in zip(A.REGION_CHI_P_SLICES, SLICE_COLOURS):
        lo, md, hi = np.percentile(Df[:, :, _bin(A.p_edges, v)], [5, 50, 95], axis=0)
        ax_t.fill_between(A.e_c, np.maximum(lo, lo_d), np.maximum(hi, lo_d), color=c, alpha=0.3, lw=0)
        ax_t.plot(A.e_c, np.maximum(md, lo_d), color=c, lw=0.8)
    for v, c in zip(A.REGION_CHI_EFF_SLICES, SLICE_COLOURS):
        lo, md, hi = np.percentile(Df[:, _bin(A.e_edges, v), :], [5, 50, 95], axis=0)
        ax_r.fill_betweenx(A.p_c, np.maximum(lo, lo_d), np.maximum(hi, lo_d), color=c, alpha=0.3, lw=0)
        ax_r.plot(np.maximum(md, lo_d), A.p_c, color=c, lw=0.8)
    ax_t.set_yscale('log'); ax_t.set_ylim(lo_d, hi_d); ax_t.set_xlim(ax_m.get_xlim())
    ax_r.set_xscale('log'); ax_r.set_xlim(lo_d, hi_d); ax_r.set_ylim(0, 1)
    ax_t.tick_params(labelbottom=False, labelleft=False, which='both')
    ax_r.tick_params(labelleft=False, labelbottom=False, which='both')
    if xlabel:
        ax_m.set_xlabel(r'$\chi_\mathrm{eff}$')
    else:
        ax_m.tick_params(labelbottom=False)
    if ylabel:
        ax_m.set_ylabel(r'$\chi_p$')
    else:
        ax_m.tick_params(labelleft=False)
    if label:
        ax_m.text(0.04, 0.96, region_label(mr, qr), transform=ax_m.transAxes, va='top', ha='left', fontsize=fontsize,
                  bbox=LABEL_BOX)
    return ax_m, pcm


EXCLUDED_GREY = '0.65'                           # outside the physical (chi_eff, chi_p) window
GRID = dict(color='0.6', lw=0.5, alpha=0.7, zorder=3)   # Fig. 1's reference lines (mass_figure)
FLEX_COLOUR, GAUSS_COLOUR = 'C1', 'k'             # marginal figure: flexible as Fig. 1's q ~ 1 colour; Gaussian 90% only


def marginal_block(fig, spec, Df, Dg, mr, qr, norm, ylabel=True, fontsize=6):
    """The flexible model's median heatmap and the Gaussian model's contours (spin_heatmap), with the 1D marginals
    p(chi_eff | region) above and p(chi_p | region) to the right (Df, Dg: draws, Be, Bp densities): the flexible
    model's median and 90% band, the Gaussian model's 90% interval only (lines; its 90% region on the heatmap). Each
    marginal axis fits its own block's 90% bands (user, 2026-10-30); the region outside the physical (chi_eff, chi_p)
    window is grey, bounded by the smooth curve model.chi_p_max(chi_eff, q) at the region's largest q (the upper edge of
    its last q bin; chi_p_max grows with q), which covers every empty heatmap bin; gridlines on the major ticks in
    Fig. 1's grey."""
    sub = spec.subgridspec(2, 2, width_ratios=(3, 1.1), height_ratios=(1.1, 3), hspace=0.05, wspace=0.05)
    ax_m, ax_t, ax_r = fig.add_subplot(sub[1, 0]), fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[1, 1])
    from matplotlib.ticker import MultipleLocator
    from model import chi_p_max
    pcm = spin_heatmap(ax_m, np.median(Df, 0), np.median(Dg, 0), norm, contour=GAUSS_COLOUR)
    q_top = A.q_edges[1:][(A.q_c >= qr[0]) & (A.q_c <= qr[1])].max()   # bins belong to a region by their centres
    e = np.linspace(-1., 1., 2001)
    ax_m.fill_between(e, chi_p_max(e, q_top), 1.01, color=EXCLUDED_GREY, lw=0, zorder=1.5)
    ex, px = np.repeat(A.e_edges, 2)[1:-1], np.repeat(A.p_edges, 2)[1:-1]
    step = lambda y: np.repeat(y, 2)
    lo, md, hi = np.percentile(Df.sum(2) * A.dp, [5, 50, 95], axis=0)              # p(chi_eff | region)
    e_in = (A.e_c >= A.REGION_CHI_EFF_RANGE[0]) & (A.e_c <= A.REGION_CHI_EFF_RANGE[1])
    g_e, g_p = (np.percentile(Dg.sum(a) * d, [5, 95], axis=0) for a, d in ((2, A.dp), (1, A.de)))
    top_e = 1.08 * max(hi[e_in].max(), g_e[1][e_in].max())
    ax_t.fill_between(ex, step(lo), step(hi), color=FLEX_COLOUR, alpha=0.3, lw=0)
    ax_t.plot(ex, step(md), color=FLEX_COLOUR, lw=0.8)
    lo, md, hi = np.percentile(Df.sum(1) * A.de, [5, 50, 95], axis=0)              # p(chi_p | region)
    ax_r.fill_betweenx(px, step(lo), step(hi), color=FLEX_COLOUR, alpha=0.3, lw=0)
    ax_r.plot(step(md), px, color=FLEX_COLOUR, lw=0.8)
    top_p = 1.08 * max(hi.max(), g_p[1].max())
    for y in g_e:                                           # the Gaussian is continuous: through the bin centres
        ax_t.plot(A.e_c, y, color=GAUSS_COLOUR, lw=0.6)
    for y in g_p:
        ax_r.plot(y, A.p_c, color=GAUSS_COLOUR, lw=0.6)
    ax_t.set_xlim(ax_m.get_xlim()); ax_t.set_ylim(0, top_e)
    ax_r.set_ylim(0, 1); ax_r.set_xlim(0, top_p)
    for ax in (ax_m, ax_r):                                 # chi_p ticks every 0.2
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
    for ax, axis in ((ax_m, 'both'), (ax_t, 'x'), (ax_r, 'y')):    # gridlines on the major ticks, as Fig. 1's
        ax.grid(True, which='major', axis=axis, **GRID)
    ax_t.tick_params(labelbottom=False, labelleft=False, which='both', direction='out')
    ax_r.tick_params(labelleft=False, labelbottom=False, which='both', direction='out')
    ax_m.set_xlabel(r'$\chi_\mathrm{eff}$')
    if ylabel:
        ax_m.set_ylabel(r'$\chi_p$')
    else:
        ax_m.tick_params(labelleft=False)
    ax_m.text(0.04, 0.96, region_label(mr, qr), transform=ax_m.transAxes, va='top', ha='left', fontsize=fontsize,
              bbox=LABEL_BOX)
    return ax_m, pcm


def slice_legend_handles():
    return ([Line2D([], [], color=c, lw=0.8, ls='--') for c in SLICE_COLOURS],
            [rf'$\chi_p = {a:g}$, $\chi_\mathrm{{eff}} = {b:g}$' for a, b in zip(A.REGION_CHI_P_SLICES,
                                                                               A.REGION_CHI_EFF_SLICES)])


def spin_peaks_figure(slices=False, marginals=False):
    """Fig. 2: one panel per peak region: the flexible model's median p(chi_eff, chi_p | region) (log colour scale,
    shared) and the Gaussian model's 50% / 90% regions (contours). ``slices``: each panel with slice panels (median
    and 90% over draws) above and to the right, as analysis.spin_region_figure. ``marginals``: the side panels show
    the 1D marginals p(chi_eff | region), p(chi_p | region) of both models instead (median and 90% band; user,
    2026-10-30), the legend above the panels."""
    from matplotlib.colors import LogNorm
    Df = region_draws('ppc_gwtc5_k10', SPIN_PEAK_REGIONS)
    if marginals:
        Dg = region_draws('ppc_gwtc5_k100_gauss', SPIN_PEAK_REGIONS)
        Mg = np.median(Dg, 0)
    else:
        Mg = region_medians('ppc_gwtc5_k100_gauss', SPIN_PEAK_REGIONS)
    Mf = np.median(Df, 0)
    norm = LogNorm(vmin=A.REGION_DENSITY_MIN, vmax=np.nanmax(Mf))
    handles = [Line2D([], [], color='c', lw=1.0), Line2D([], [], color='c', lw=0.7, ls='--')]
    names = ['Gaussian, 50\\%', 'Gaussian, 90\\%']
    if marginals:
        fig = plt.figure(figsize=(TWO_COL, 2.45))
        gs = fig.add_gridspec(1, len(SPIN_PEAK_REGIONS), left=0.065, right=0.915, bottom=0.17, top=0.88, wspace=0.12)
        for k, ((mr, qr), f, g) in enumerate(zip(SPIN_PEAK_REGIONS, np.moveaxis(Df, 1, 0), np.moveaxis(Dg, 1, 0))):
            ref, pcm = marginal_block(fig, gs[0, k], f, g, mr, qr, norm, ylabel=k == 0)
        from matplotlib.patches import Patch
        fig.legend([(Patch(color=FLEX_COLOUR, alpha=0.3, lw=0), Line2D([], [], color=FLEX_COLOUR, lw=0.8)),
                    Line2D([], [], color=GAUSS_COLOUR, lw=0.8)],
                   ['Flexible, median and 90\\%', 'Gaussian, 90\\%'], loc='upper center', ncol=2,
                   bbox_to_anchor=(0.5, 1.0), **LEG)
    elif not slices:
        fig, axes = plt.subplots(1, len(SPIN_PEAK_REGIONS), figsize=(TWO_COL, 2.05), sharey=True)
        for ax, (mr, qr), f, g in zip(axes, SPIN_PEAK_REGIONS, Mf, Mg):
            pcm = spin_heatmap(ax, f, g, norm)
            ax.set_xlabel(r'$\chi_\mathrm{eff}$')
            ax.text(0.04, 0.96, region_label(mr, qr), transform=ax.transAxes, va='top', ha='left', fontsize=7,
                    bbox=LABEL_BOX)
        axes[0].set_ylabel(r'$\chi_p$')
        axes[-1].legend(handles, names, loc='lower right', labelcolor='w', **LEG)
        fig.tight_layout(rect=(0, 0, 0.93, 1)); fig.subplots_adjust(wspace=0.06)
        ref = axes[0]
    else:
        fig = plt.figure(figsize=(TWO_COL, 2.45))
        gs = fig.add_gridspec(1, len(SPIN_PEAK_REGIONS), left=0.065, right=0.915, bottom=0.17, top=0.86, wspace=0.12)
        for k, ((mr, qr), D, g) in enumerate(zip(SPIN_PEAK_REGIONS, np.moveaxis(Df, 1, 0), Mg)):
            ref, pcm = spin_block(fig, gs[0, k], D, g, mr, qr, norm, ylabel=k == 0, fontsize=6)
        h2, n2 = slice_legend_handles()
        fig.legend(handles + h2, names + n2, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.0), **LEG)
    pos = ref.get_position()
    cax = fig.add_axes([0.935, pos.y0, 0.012, pos.height])
    cb = fig.colorbar(pcm, cax=cax); cb.set_label(r'$p(\chi_\mathrm{eff}, \chi_p \mid m_1, q)$', fontsize=7)
    cb.ax.tick_params(direction='out')
    return fig


# the talk figure's regions: Fig. 2's, but the ~60 Msun one widened to the cached 50-75 Msun (55-65 holds few events)
TALK_REGIONS = SPIN_PEAK_REGIONS[:3] + (((50., 75.), (0.3, 0.65)),)
# where each region's spin block sits on the mass map, in axes fraction (x0, y0, width, height), chosen to cover dark or
# empty parts of the map (user, 2026-10-26: insets over the mass panel)
TALK_INSETS = ((0.005, 0.03, 0.175, 0.40), (0.80, 0.57, 0.175, 0.40), (0.44, 0.25, 0.175, 0.40),
               (0.835, 0.03, 0.16, 0.40))


def talk_figure(r):
    """For talks: the flexible run's median p(ln m1, q) (wide, 2:1, mako colour map) with each peak region boxed and
    its spin block (flexible median heatmap, Gaussian contours, slices) as an inset over the map, joined to the box by
    two lines between their nearest corners."""
    import seaborn as sns
    from matplotlib.colors import LogNorm
    from matplotlib.patches import ConnectionPatch, FancyBboxPatch, Rectangle
    plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'legend.fontsize': 9})
    ok = A.q_c[None, :] * A.m_c[:, None] >= A.MIN_M
    med = np.where(ok, np.median(r.p_mass, 0), np.nan)                     # per d ln m1 dq
    m_edges = np.exp(A.lnm_edges)
    mnorm = LogNorm(vmin=np.nanmax(med) * 1e-3, vmax=np.nanmax(med))
    Df = region_draws('ppc_gwtc5_k10', TALK_REGIONS)
    Mg = region_medians('ppc_gwtc5_k100_gauss', TALK_REGIONS)
    snorm = LogNorm(vmin=A.REGION_DENSITY_MIN, vmax=np.nanmax(np.median(Df, 0)))
    W, H = 14., 8.2
    fig = plt.figure(figsize=(W, H))
    left, width = 0.055, 0.86
    height = width * W / 2 / H                                             # a 2:1 panel
    ax0 = fig.add_axes([left, 0.12, width, height])
    pc = ax0.pcolormesh(m_edges, A.q_edges, med.T, norm=mnorm, cmap=sns.color_palette('mako', as_cmap=True),
                        shading='flat', rasterized=True)
    ax0.set_xscale('log'); ax0.set_xlim(M_MIN, A.MAX_M); ax0.set_ylim(0, 1)
    ax0.set_xticks((5, 10, 20, 50, 100, 200), ['$5$', '$10$', '$20$', '$50$', '$100$', '$200$'])
    ax0.set_xlabel(r'$m_1\,[\textrm{M}_\odot]$'); ax0.set_ylabel(r'$q$')
    ax0.tick_params(which='both', direction='out')
    to_axes = lambda m, q: ax0.transAxes.inverted().transform(ax0.transData.transform((m, q)))
    for (mr, qr), D, g, (x0, y0, w, h) in zip(TALK_REGIONS, np.moveaxis(Df, 1, 0), Mg, TALK_INSETS):
        ax0.add_patch(Rectangle((mr[0], qr[0]), mr[1] - mr[0], qr[1] - qr[0], fill=False, ec='w', lw=1.3, zorder=3))
        # the inset: a white rounded backing, then the spin block inside it (in figure coordinates)
        fx0, fy0 = ax0.transAxes.transform((x0, y0)); fx1, fy1 = ax0.transAxes.transform((x0 + w, y0 + h))
        inv = fig.transFigure.inverted()
        (bx0, by0), (bx1, by1) = inv.transform((fx0, fy0)), inv.transform((fx1, fy1))
        fig.add_artist(FancyBboxPatch((bx0, by0), bx1 - bx0, by1 - by0, boxstyle='round,pad=0.004',
                                      transform=fig.transFigure, fc='w', ec='0.6', lw=0.6, alpha=0.95, zorder=2))
        n_before = len(fig.axes)
        gs = fig.add_gridspec(1, 1, left=bx0 + 0.038, right=bx1 - 0.006, bottom=by0 + 0.045, top=by1 - 0.006)
        ax_m, pcs = spin_block(fig, gs[0, 0], D, g, mr, qr, snorm, fontsize=7.5)
        for ax in fig.axes[n_before:]:
            ax.set_zorder(4)
            ax.tick_params(labelsize=7)
        ax_m.xaxis.label.set_size(9); ax_m.yaxis.label.set_size(9)
        # two lines from the region's box to the inset's nearest corners
        rx0, ry0 = to_axes(mr[0], qr[0]); rx1, ry1 = to_axes(mr[1], qr[1])
        if x0 >= rx1 or x0 + w <= rx0:                                     # wholly to one side: join facing edges
            xr, xi = (rx1, x0) if x0 >= rx1 else (rx0, x0 + w)
            pairs = (((xr, ry0), (xi, y0)), ((xr, ry1), (xi, y0 + h)))
        else:                                                               # above / below
            yr, yi = (ry1, y0) if y0 >= ry1 else (ry0, y0 + h)
            pairs = (((rx0, yr), (x0, yi)), ((rx1, yr), (x0 + w, yi)))
        for pa, pb in pairs:
            fig.add_artist(ConnectionPatch(pa, pb, coordsA='axes fraction', coordsB='axes fraction', axesA=ax0,
                                           axesB=ax0, color='w', lw=0.8, alpha=0.8, zorder=1.5))
    pos = ax0.get_position()
    cb = fig.colorbar(pc, cax=fig.add_axes([pos.x0, pos.y1 + 0.015, 0.3, 0.015]), orientation='horizontal')
    cb.set_label(r'median $p(\ln m_1, q)$', labelpad=2); cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('top'); cb.ax.tick_params(direction='out')
    cb2 = fig.colorbar(pcs, cax=fig.add_axes([pos.x1 + 0.012, pos.y0, 0.01, pos.height]))
    cb2.set_label(r'$p(\chi_\mathrm{eff}, \chi_p \mid m_1, q)$, flexible median'); cb2.ax.tick_params(direction='out')
    h2, n2 = slice_legend_handles()
    handles = [Line2D([], [], color='c', lw=1.0), Line2D([], [], color='c', lw=0.7, ls='--')] + h2
    fig.legend(handles, ['Gaussian, 50\\%', 'Gaussian, 90\\%'] + n2, loc='lower right', ncol=4,
               bbox_to_anchor=(pos.x1, pos.y1 + 0.008), **LEG)
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
    a1.set_xlabel(r'peak location $\mu\,[\textrm{M}_\odot]$'); a1.set_ylabel(r'mean $q$ of the peak'); a1.set_ylim(0, 1)
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
    labels = [r'$\sim 10 \to \sim 20\,\textrm{M}_\odot$', r'$\sim 35 \to \sim 60\,\textrm{M}_\odot$', 'Flexible', 'Gaussian']
    a2.legend(handles, labels, loc='upper right', **LEG)
    for ax, t in ((a1, '(a)'), (a2, '(b)')):
        ax.text(0.97 if ax is a1 else 0.03, 0.96, t, transform=ax.transAxes, va='top',
                ha='right' if ax is a1 else 'left')
    fig.tight_layout(); fig.subplots_adjust(hspace=0.38)
    return fig


def add_chi_p_moments(r):
    """Append the per-cell chi_p mean and sd (MOM[..., 2:4] of the spin cache) to load_light's MOM[..., :2]."""
    import glob
    import h5py
    files = sorted(glob.glob(os.path.join(A.REPO, 'results', r.run, 'chain_*_samples.h5')))
    mp = np.concatenate([h5py.File(A.post_path(r.run, f), 'r')['MOM'][:r.n_draws, ..., 2:4] for f in files])
    r.MOM = np.concatenate([r.MOM, mp.astype(r.MOM.dtype)], -1)


def chi_data(r, core, spin='chi_eff'):
    chi = lambda qr=None: A.q_averaged_moments(r, qr, names=(spin,), core=core)[spin]
    allq, byq = chi(), [chi(qr) for qr in A.SUMMARY_Q_RANGES]
    return {'mean': (allq['mean'], [x['mean'] for x in byq]), 'sd': (allq['sd'], [x['sd'] for x in byq])}


def chi_prior_data(r, core, spin='chi_eff'):
    """As chi_data, for the empirical-Bayes spin prior draws of macros/spin_prior.py (each with its posterior draw's
    mass field, so the same p(q | m1))."""
    from types import SimpleNamespace
    d = np.load(os.path.join(A.REPO, 'results', 'post_processing', f'{r.run}_spin_prior.npz'))
    rp = SimpleNamespace(p_mass=r.p_mass[d['draws']], MOM=d['MOM'], MOM_CORE=d['MOM_CORE'])
    return chi_data(rp, core, spin)


def prior_lines(ax, allq, byq):
    """The prior's 90% interval: dashed lines in each q range's colour, grey dashed for all q."""
    for y, col in zip(byq, COLS):
        for f in (5, 95):
            ax.plot(mx, ext(np.nanpercentile(y, f, 0)), color=col, lw=0.8, ls='--')
    for f in (5, 95):
        ax.plot(mx, ext(np.nanpercentile(allq, f, 0)), color='0.45', lw=0.7, ls='--')


SD_TICKS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)


def log_sd_axis(ax, lo, hi):
    """Log y axis for a spin sd, from lo to hi (the bands' 5% and 95% extremes), labelled at SD_TICKS."""
    lo, hi = max(lo, 5e-3) / 1.15, hi * 1.15
    ax.set_yscale('log'); ax.set_ylim(lo, hi)
    ticks = [t for t in SD_TICKS if lo <= t <= hi]
    ax.set_yticks(ticks, [f'${t:g}$' for t in ticks])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def chi_figure(D, stats, core, asinh, spin='chi_eff', prior=None):
    """Rows: spin models; columns: ``stats`` ('mean', 'sd') of ``spin`` ('chi_eff': the mean on an asinh scale;
    'chi_p': linear). Shared y per column."""
    cut = rf', $|\chi_\mathrm{{eff}}| < {A.CHI_CORE:g}$' if core else ''
    sym = {'chi_eff': r'\chi_\mathrm{eff}', 'chi_p': r'\chi_p'}[spin]
    labels = {'mean': rf'$\mu_{{{sym}}}(m_1)$' + cut, 'sd': rf'$\sigma_{{{sym}}}(m_1)$' + cut}
    width = ONE_COL if len(stats) == 1 else TWO_COL
    fig, axes = plt.subplots(len(D), len(stats), figsize=(width, 1.65 * len(D) + 0.45), sharex=True, squeeze=False)
    for j, st in enumerate(stats):
        ys = [y for d in D.values() for y in [d[st][0]] + d[st][1]]
        if prior is not None:                                        # the prior's 90% fits too
            ys += [y for d in prior.values() for y in [d[st][0]] + d[st][1]]
        lo = min(np.nanmin(np.nanpercentile(y, 5, 0)[vis]) for y in ys)
        hi = max(np.nanmax(np.nanpercentile(y, 95, 0)[vis]) for y in ys)
        for i, (name, d) in enumerate(D.items()):
            ax = axes[i, j]
            bands(ax, *d[st])
            if prior is not None:
                prior_lines(ax, *prior[name][st])
            m1_axis(ax, i == len(D) - 1)
            if st == 'mean' and spin == 'chi_p':
                ax.set_ylim(0, 1.05 * hi)
            elif st == 'mean':
                ax.axhline(0, color='0.5', lw=0.6, ls=':')
                ax.set_yscale('asinh', linear_width=asinh)
                ticks = [t for t in (-0.5, -0.1, -0.03, 0, 0.03, 0.1, 0.5)
                         if lo * 1.1 <= t <= hi * 1.1 or t == 0]
                ax.set_yticks(ticks, [f'${t:g}$' for t in ticks])
                ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
                ax.set_ylim(lo * 1.1, hi * 1.1)
            elif st == 'sd':
                log_sd_axis(ax, lo, hi)
            else:
                ax.set_ylim(0, 1.05 * hi)
            ax.set_ylabel(labels[st])
            ax.text(0.03, 0.93, name, transform=ax.transAxes, va='top', ha='left',
                    bbox=dict(facecolor='w', alpha=0.8, edgecolor='none', pad=1.5))
    handles, names = band_legend()
    if prior is not None:
        handles = handles + [Line2D([], [], color='0.45', lw=0.8, ls='--')]
        names = names + [r'prior (90\%, dashed)']
    fig.legend(handles, names, loc='upper center', ncol=len(names), bbox_to_anchor=(0.5, 1.0), **LEG)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.17 / fig.get_figheight())); fig.subplots_adjust(hspace=0.06)
    return fig


Q_MIN = 0.1                                      # the q axis of the moments-against-q figures


def m1_averaged_moments(r, core, spin='chi_eff', m1_range=None):
    """Spin moments given q alone: the conditional averaged over the windowed p(m1 | q) (as A.q_averaged_moments, with
    the axes swapped): 'mean' and 'sd' (draws, Bq). With ``m1_range`` = (lo, hi) [Msun], only the m1 bins with centres
    in it (NaN where the window leaves none)."""
    w = r.p_mass.astype(np.float32)
    if core:
        w = w * (1 - r.MOM_CORE[..., 2])
    if m1_range is not None:
        w = w * ((A.m_c >= m1_range[0]) & (A.m_c <= m1_range[1]))[None, :, None]
    tot = w.sum(1, keepdims=True)
    w = w / np.maximum(tot, 1e-30)
    i_mu, i_sd = {'chi_eff': (0, 1), 'chi_p': (2, 3)}[spin]
    mu, sd = (r.MOM_CORE[..., 0], r.MOM_CORE[..., 1]) if core else (r.MOM[..., i_mu], r.MOM[..., i_sd])
    mean = (w * mu).sum(1)
    var = (w * (sd ** 2 + (mu - mean[:, None]) ** 2)).sum(1)
    empty = tot[:, 0] <= 0
    return {'mean': np.where(empty, np.nan, mean), 'sd': np.where(empty, np.nan, np.sqrt(var))}


def m1_averaged_prior(r, core, spin='chi_eff'):
    """As m1_averaged_moments, for the empirical-Bayes spin prior draws (chi_prior_data)."""
    from types import SimpleNamespace
    d = np.load(os.path.join(A.REPO, 'results', 'post_processing', f'{r.run}_spin_prior.npz'))
    rp = SimpleNamespace(p_mass=r.p_mass[d['draws']], MOM=d['MOM'], MOM_CORE=d['MOM_CORE'])
    return m1_averaged_moments(rp, core, spin)


# GWTC-5 population release (zenodo.org/records/20292639; popsummary files): the chi_eff | q models
LVK_DIR = '/mnt/ceph/users/jheinzel/gwtc5_pop_release/popsummary/popsummary_files'
LVK_MODELS = {'Linear': ('linear_mass_ratio_popsummary.h5', '#CD5C5C', '-.'),
              'Spline': ('spline_mass_ratio_k4_popsummary.h5', '#2E8B57', '--')}    # not blue: ours is C0


def lvk_chi_q():
    """{label: (q, {'mean': draws, 'sd': draws}, colour, line style)} for the GWTC-5 chi_eff | q models (their
    Gaussian's mean and width against q) whose popsummary files exist under LVK_DIR."""
    import h5py
    out = {}
    for label, (fname, col, ls) in LVK_MODELS.items():
        path = os.path.join(LVK_DIR, fname)
        if not os.path.exists(path):
            print('missing', path)
            continue
        with h5py.File(path, 'r') as h:
            g = h['posterior/rates_on_grids']
            out[label] = (g['mu_chieff/positions'][0], {'mean': g['mu_chieff/rates'][:],
                                                        'sd': g['sigma_chieff/rates'][:]}, col, ls)
    return out


def chi_q_figure(D, stats, core, spin='chi_eff', prior=None, lvk=None):
    """Rows: spin models; columns: ``stats`` of ``spin`` against q, averaged over p(m1 | q): the posterior's median and
    90% band, the prior's 90% (dashed), and ``lvk`` (lvk_chi_q: the GWTC-5 chi_eff | q models' 90% bounds, as lines in
    each model's colour and style over a faint fill)."""
    cut = rf', $|\chi_\mathrm{{eff}}| < {A.CHI_CORE:g}$' if core else ''
    sym = {'chi_eff': r'\chi_\mathrm{eff}', 'chi_p': r'\chi_p'}[spin]
    labels = {'mean': rf'$\mu_{{{sym}}}(q)$' + cut, 'sd': rf'$\sigma_{{{sym}}}(q)$' + cut}
    qx = np.concatenate([[0.], A.q_c, [1.]])
    show = A.q_c >= Q_MIN
    width = ONE_COL if len(stats) == 1 else TWO_COL
    fig, axes = plt.subplots(len(D), len(stats), figsize=(width, 1.65 * len(D) + 0.45), sharex=True, squeeze=False)
    for j, st in enumerate(stats):
        ys = [d[st] for d in D.values()] + ([p[st] for p in prior.values()] if prior is not None else [])
        lo = min(np.nanmin(np.nanpercentile(y, 5, 0)[show]) for y in ys)
        hi = max(np.nanmax(np.nanpercentile(y, 95, 0)[show]) for y in ys)
        for i, (name, d) in enumerate(D.items()):
            ax = axes[i, j]
            l, m, h = ext(q5(d[st]))
            ax.fill_between(qx, l, h, color='C0', alpha=0.3, lw=0); ax.plot(qx, m, color='C0')
            if prior is not None:
                for f in (5, 95):
                    ax.plot(qx, ext(np.nanpercentile(prior[name][st], f, 0)), color='0.45', lw=0.8, ls='--')
            for qg, L, col, ls in (lvk or {}).values():
                l5, h5 = np.percentile(L[st], [5, 95], 0)
                ax.fill_between(qg, l5, h5, color=col, alpha=0.06, lw=0)
                ax.plot(qg, l5, color=col, lw=0.9, ls=ls); ax.plot(qg, h5, color=col, lw=0.9, ls=ls)
            if st == 'mean' and spin == 'chi_eff':
                ax.axhline(0, color='0.5', lw=0.6, ls=':')
                ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
            elif st == 'sd':
                log_sd_axis(ax, lo, hi)
            else:
                ax.set_ylim(0, 1.05 * hi)
            ax.set_xlim(Q_MIN, 1)
            if i == len(D) - 1:
                ax.set_xlabel('$q$')
            ax.set_ylabel(labels[st])
            ax.text(0.03, 0.93, name, transform=ax.transAxes, va='top', ha='left',
                    bbox=dict(facecolor='w', alpha=0.8, edgecolor='none', pad=1.5))
    handles = [(Patch(color='C0', alpha=0.3, lw=0), Line2D([], [], color='C0'))]
    names = [r'posterior (median, 90\%)']
    if prior is not None:
        handles.append(Line2D([], [], color='0.45', lw=0.8, ls='--')); names.append(r'prior (90\%)')
    for label, (_, _, col, ls) in (lvk or {}).items():
        handles.append(Line2D([], [], color=col, lw=0.9, ls=ls))
        names.append(rf'GWTC-5 {label} (90\%)')
    fig.legend(handles, names, loc='upper center', ncol=len(names), bbox_to_anchor=(0.5, 1.0), **LEG)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.17 / fig.get_figheight())); fig.subplots_adjust(hspace=0.06)
    return fig


M1_BANDS = ((8., 12.), (18., 24.), (30., 40.), (55., 65.))     # the four peaks (internal figure)
M1_BAND_COLS = ('C2', 'C0', 'C1', 'C3')


def chi_q_bands_figure(R, core, spin='chi_eff', bands=M1_BANDS):
    """As chi_q_figure, for p(chi | q, m1 in each of ``bands``) (90% bands and medians, M1_BAND_COLS) and over all m1
    (black lines, 90%): rows the runs of R, columns mean and sd. Internal figure."""
    cut = rf', $|\chi_\mathrm{{eff}}| < {A.CHI_CORE:g}$' if core else ''
    sym = {'chi_eff': r'\chi_\mathrm{eff}', 'chi_p': r'\chi_p'}[spin]
    labels = {'mean': rf'$\mu_{{{sym}}}(q)$' + cut, 'sd': rf'$\sigma_{{{sym}}}(q)$' + cut}
    qx = np.concatenate([[0.], A.q_c, [1.]])
    D = {RUNS[run]: [m1_averaged_moments(r, core, spin)] + [m1_averaged_moments(r, core, spin, b) for b in bands]
         for run, r in R.items()}
    fig, axes = plt.subplots(len(D), 2, figsize=(TWO_COL, 1.65 * len(D) + 0.45), sharex=True, squeeze=False)
    show = A.q_c >= Q_MIN
    for j, st in enumerate(('mean', 'sd')):
        ys = [x[st] for d in D.values() for x in d]
        with np.errstate(all='ignore'), __import__('warnings').catch_warnings():
            __import__('warnings').simplefilter('ignore')
            lo = min(np.nanmin(np.nanpercentile(y, 5, 0)[show]) for y in ys)
            hi = max(np.nanmax(np.nanpercentile(y, 95, 0)[show]) for y in ys)
        for i, (name, d) in enumerate(D.items()):
            ax = axes[i, j]
            for x, col in zip(d[1:], M1_BAND_COLS):
                with __import__('warnings').catch_warnings():
                    __import__('warnings').simplefilter('ignore')
                    l, m, h = ext(q5(x[st]))
                ax.fill_between(qx, l, h, color=col, alpha=0.25, lw=0); ax.plot(qx, m, color=col, lw=1.0)
            l, _, h = ext(q5(d[0][st]))
            ax.plot(qx, l, color='k', lw=0.7); ax.plot(qx, h, color='k', lw=0.7)
            if st == 'mean' and spin == 'chi_eff':
                ax.axhline(0, color='0.5', lw=0.6, ls=':')
                ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
            elif st == 'sd':
                log_sd_axis(ax, lo, hi)
            else:
                ax.set_ylim(0, 1.05 * hi)
            ax.set_xlim(Q_MIN, 1)
            if i == len(D) - 1:
                ax.set_xlabel('$q$')
            ax.set_ylabel(labels[st])
            ax.text(0.03, 0.93, name, transform=ax.transAxes, va='top', ha='left',
                    bbox=dict(facecolor='w', alpha=0.8, edgecolor='none', pad=1.5))
    handles = [Line2D([], [], color='k', lw=0.7)] + [(Patch(color=c, alpha=0.25, lw=0), Line2D([], [], color=c, lw=1.0))
                                                    for c in M1_BAND_COLS]
    names = [r'all $m_1$ (90\%)'] + [rf'${a:g} < m_1 < {b:g}\,\textrm{{M}}_\odot$' for a, b in bands]
    fig.legend(handles, names, loc='upper center', ncol=len(names), bbox_to_anchor=(0.5, 1.0), **LEG)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.17 / fig.get_figheight())); fig.subplots_adjust(hspace=0.06)
    return fig


Q_CURVE_COL = 'C4'                               # the given-q curve (C0, C1 are the q ranges of the m1 column)


def corr_label(core, of):
    cut = rf', $|\chi_\mathrm{{eff}}| < {A.CHI_CORE:g}$' if core else ''
    return rf'$\rho_{{\chi_\mathrm{{eff}}\chi_p}}({of})$' + cut


def corr_curves_figure(R, core):
    """Rows: spin models; columns: the chi_eff-chi_p correlation coefficient given m1 alone (over p(q | m1); all q as
    black 90% lines, each SUMMARY_Q_RANGES as bands) and given q alone (over p(m1 | q)), A.chi_correlations."""
    D = {RUNS[run]: (A.chi_correlations(r, 'm1', core=core),
                     [A.chi_correlations(r, 'm1', qr, core=core) for qr in A.SUMMARY_Q_RANGES],
                     A.chi_correlations(r, 'q', core=core)) for run, r in R.items()}
    qx, show = np.concatenate([[0.], A.q_c, [1.]]), A.q_c >= Q_MIN
    with np.errstate(all='ignore'), __import__('warnings').catch_warnings():
        __import__('warnings').simplefilter('ignore')
        ys = [(y, vis) for d in D.values() for y in [d[0]] + d[1]] + [(d[2], show) for d in D.values()]
        lo = min(np.nanmin(np.nanpercentile(y, 5, 0)[v]) for y, v in ys)
        hi = max(np.nanmax(np.nanpercentile(y, 95, 0)[v]) for y, v in ys)
    pad = 0.05 * (hi - lo)
    fig, axes = plt.subplots(len(D), 2, figsize=(TWO_COL, 1.65 * len(D) + 0.45), squeeze=False)
    for i, (name, (allq, byq, rq)) in enumerate(D.items()):
        a, b = axes[i]
        with __import__('warnings').catch_warnings():
            __import__('warnings').simplefilter('ignore')
            bands(a, allq, byq)
            l, m, h = ext(q5(rq))
        m1_axis(a, i == len(D) - 1)
        b.fill_between(qx, l, h, color=Q_CURVE_COL, alpha=0.3, lw=0); b.plot(qx, m, color=Q_CURVE_COL)
        b.set_xlim(Q_MIN, 1)
        if i == len(D) - 1:
            b.set_xlabel('$q$')
        for ax, of in ((a, 'm_1'), (b, 'q')):
            ax.axhline(0, color='0.5', lw=0.6, ls=':')
            ax.set_ylim(lo - pad, hi + pad); ax.set_ylabel(corr_label(core, of))
            ax.text(0.03, 0.93, name, transform=ax.transAxes, va='top', ha='left',
                    bbox=dict(facecolor='w', alpha=0.8, edgecolor='none', pad=1.5))
        if i < len(D) - 1:
            a.tick_params(labelbottom=False); b.tick_params(labelbottom=False)
    handles, names = band_legend()
    handles = handles + [(Patch(color=Q_CURVE_COL, alpha=0.3, lw=0), Line2D([], [], color=Q_CURVE_COL))]
    names = names + [r'given $q$ (median, 90\%)']
    fig.legend(handles, names, loc='upper center', ncol=len(names), bbox_to_anchor=(0.5, 1.0), **LEG)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.17 / fig.get_figheight())); fig.subplots_adjust(hspace=0.06)
    return fig


def corr_map_figure(R, core):
    """Per spin model: the median chi_eff-chi_p correlation coefficient in each (m1, q) cell (cells with m2 < 3 Msun
    blank), with the 50% and 90% highest-probability contours of the median windowed p(m1, q)."""
    M = {}
    for run, r in R.items():
        with np.errstate(all='ignore'), __import__('warnings').catch_warnings():
            __import__('warnings').simplefilter('ignore')
            M[RUNS[run]] = (np.nanmedian(A.chi_correlations(r, None, core=core), 0), np.median(r.p_mass, 0))
    ok = (A.q_c[None, :] * A.m_c[:, None] >= A.MIN_M) & vis[:, None]
    lim = max(np.nanmax(np.abs(np.where(ok, m, np.nan))) for m, _ in M.values())
    fig, axes = plt.subplots(1, len(M), figsize=(TWO_COL, 2.6), sharey=True, squeeze=False)
    for ax, (name, (rho, pm)) in zip(axes[0], M.items()):
        pc = ax.pcolormesh(m_edges, A.q_edges, np.where(ok, rho, np.nan).T, cmap='RdBu_r', vmin=-lim, vmax=lim,
                           shading='flat', rasterized=True)
        ax.contour(A.m_c, A.q_c, np.where(ok, pm, 0.).T, levels=hpd_levels_2d(np.where(ok, pm, 0.), (0.9, 0.5)),
                   colors='k', linewidths=(0.6, 0.9), linestyles=('--', '-'))
        m1_axis(ax); ax.set_ylim(0, 1)
        ax.tick_params(which='both', direction='out')
        ax.text(0.03, 0.06, name, transform=ax.transAxes, va='bottom', ha='left', bbox=LABEL_BOX)
    axes[0, 0].set_ylabel('$q$')
    cb = fig.colorbar(pc, cax=axes[0, -1].inset_axes([1.03, 0, 0.03, 1]))
    cb.set_label(corr_label(core, 'm_1, q')); cb.ax.tick_params(direction='out')
    dark = lambda ls, lw: Line2D([], [], color='k', lw=lw, ls=ls)
    axes[0, 0].legend([dark('-', 0.9), dark('--', 0.6)], [r'$p(m_1, q)$: 50\%', r'90\%'], loc='upper right',
                      **dict(LEG, frameon=True, framealpha=0.85, edgecolor='none'))
    fig.tight_layout(); fig.subplots_adjust(wspace=0.06, right=0.9)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', nargs='?', default='all', choices=('mass', 'chi', 'chip', 'chiq', 'chiqm', 'corr', 'm2', 'spin', 'book', 'talk', 'all'))
    ap.add_argument('--core', action='store_true')
    ap.add_argument('--prior', action='store_true', help='chi / chip: overlay the spin prior (macros/spin_prior.py)')
    ap.add_argument('--lvk', action='store_true', help='chiq: overlay the GWTC-5 chi_eff | q models (lvk_chi_q)')
    ap.add_argument('--mean-asinh', type=float, default=0.03)
    args = ap.parse_args()
    need = {'mass': list(RUNS)[:1], 'chi': list(RUNS), 'chip': list(RUNS), 'chiq': list(RUNS), 'chiqm': list(RUNS), 'corr': list(RUNS), 'm2': list(RUNS), 'talk': list(RUNS)[:1],
            'all': list(RUNS)}.get(args.what, [])
    R = {run: A.load_light(run) for run in need}
    import pixelpop  # noqa: F401  (its import resets matplotlib's rcParams; import it before the style)
    plot_style.apply(); plt.rcParams.update(SMALL_RC)
    core = '_core' if args.core else ''
    if args.what in ('mass', 'all'):
        plot_style.save(mass_figure(R['ppc_gwtc5_k10']), 'mass_ratio_m1')
        plot_style.save(mass_figure(R['ppc_gwtc5_k10'], mode=True), 'mass_ratio_m1_mode')
        plot_style.save(mass_figure(R['ppc_gwtc5_k10'], hpd=0.5), 'mass_ratio_m1_hpd50')
    if args.what in ('spin', 'all'):
        plot_style.save(spin_peaks_figure(), 'spin_peak_regions')
        plot_style.save(spin_peaks_figure(slices=True), 'spin_peak_regions_slices')
        plot_style.save(spin_peaks_figure(marginals=True), 'spin_peak_regions_marginals')
    if args.what in ('talk', 'all'):
        plot_style.save(talk_figure(R['ppc_gwtc5_k10']), 'mass_spin_zoom_talk')
        plt.rcParams.update(SMALL_RC)
    if args.what in ('book', 'all'):
        plot_style.save(bookkeeping_figure(), 'peak_bookkeeping')
    if args.what in ('chip', 'all'):
        for r in R.values():
            if r.MOM.shape[-1] == 2:
                add_chi_p_moments(r)
        D = {RUNS[run]: chi_data(r, False, 'chi_p') for run, r in R.items()}
        Pr = {RUNS[run]: chi_prior_data(r, False, 'chi_p') for run, r in R.items()} if args.prior else None
        tag = '_prior' if args.prior else ''
        for stats, name in ((('mean',), 'mean'), (('sd',), 'sd'), (('mean', 'sd'), 'mean_sd')):
            plot_style.save(chi_figure(D, stats, False, args.mean_asinh, 'chi_p', prior=Pr), f'chi_p_{name}_m1{tag}')
    if args.what in ('chiq', 'all'):
        for r in R.values():
            if r.MOM.shape[-1] == 2:
                add_chi_p_moments(r)
        tag = ('_prior' if args.prior else '') + ('_lvk' if args.lvk else '')
        lvk = lvk_chi_q() if args.lvk else None
        for spin, core_ in (('chi_eff', args.core),) + ((('chi_p', False),) if not args.core and not args.lvk else ()):
            D = {RUNS[run]: m1_averaged_moments(r, core_, spin) for run, r in R.items()}
            Pr = {RUNS[run]: m1_averaged_prior(r, core_, spin) for run, r in R.items()} if args.prior else None
            c = '_core' if core_ else ''
            for stats, name in ((('mean',), 'mean'), (('sd',), 'sd'), (('mean', 'sd'), 'mean_sd')):
                plot_style.save(chi_q_figure(D, stats, core_, spin, prior=Pr, lvk=lvk if spin == 'chi_eff' else None),
                                f'{spin}_{name}_q{c}{tag}')
    if args.what == 'corr':                     # chi_eff-chi_p correlation (needs the CROSS caches, post_cross.sbatch)
        for r in R.values():
            A.load_cross(r)
        plot_style.save(corr_curves_figure(R, args.core), f'chi_corr_m1_q{core}')
        plot_style.save(corr_map_figure(R, args.core), f'chi_corr_map{core}')
    if args.what == 'chiqm':                                          # internal: by m1 band
        plot_style.save(chi_q_bands_figure(R, args.core), f'chi_eff_mean_sd_q_m1bands{core}')
    if args.what in ('m2', 'all'):
        plot_style.save(m2_figure(R), 'm2_q_ranges')
        plot_style.save(m2_given_m1_figure(R), 'm2_m1_windows')
    if args.what in ('chi', 'all'):
        D = {RUNS[run]: chi_data(r, args.core) for run, r in R.items()}
        Pr = {RUNS[run]: chi_prior_data(r, args.core) for run, r in R.items()} if args.prior else None
        tag = core + ('_prior' if args.prior else '')
        for stats, name in ((('mean',), 'mean'), (('sd',), 'sd'), (('mean', 'sd'), 'mean_sd')):
            plot_style.save(chi_figure(D, stats, args.core, args.mean_asinh, prior=Pr), f'chi_eff_{name}_m1{tag}')


if __name__ == '__main__':
    main()
