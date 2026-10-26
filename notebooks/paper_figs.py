"""Paper figures at journal column widths (user, 2026-10-26), drawn at their final size with small fonts instead of shrinking the summary
figure:

    python notebooks/paper_figs.py [mass|chi|m2|all] [--core] [--mean-asinh W]
        -> plots/mass_ratio_m1.pdf         Fig. 1: p(m1 | q range) above p(q | m1), the flexible run, one column
           plots/chi_eff_mean_m1.pdf       mu_chi_eff(m1), one row per spin model, one column
           plots/chi_eff_sd_m1.pdf         sigma_chi_eff(m1), one row per spin model, one column
           plots/chi_eff_mean_sd_m1.pdf    both: rows = spin models, columns = mean, sd; two columns
           plots/m2_q_ranges.pdf           p(m2 | q range) for the two q ranges, one row per spin model (user,
                                           2026-10-26); one column

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


def p_m2_given_q_range(r, q_range):
    """p(m2 | q in q_range) per draw, density per Msun on m2_c: each (ln m1, q) cell split into NS x NS sub-cells,
    binned in ln m2 = ln m1 + ln q."""
    du = (np.arange(NS) + 0.5) / NS - 0.5
    lm2 = (A.lnm_c[:, None, None, None] + du[None, :, None, None] * A.dlnm
           + np.log(np.maximum(A.q_c[None, None, :, None] + du[None, None, None, :] * A.dq, 1e-6)))
    idx = np.moveaxis(np.searchsorted(lm2_edges, lm2) - 1, 2, 1).reshape(A.B * A.B, NS * NS)
    keep = np.broadcast_to(((A.q_c >= q_range[0]) & (A.q_c <= q_range[1]))[None], (A.B, A.B)).ravel()
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
    ap.add_argument('what', nargs='?', default='all', choices=('mass', 'chi', 'm2', 'all'))
    ap.add_argument('--core', action='store_true')
    ap.add_argument('--mean-asinh', type=float, default=0.03)
    args = ap.parse_args()
    R = {run: A.load_light(run) for run in (RUNS if args.what != 'mass' else list(RUNS)[:1])}
    plot_style.apply(); plt.rcParams.update(SMALL_RC)       # after pixelpop's import (in load_light)
    core = '_core' if args.core else ''
    if args.what in ('mass', 'all'):
        plot_style.save(mass_figure(R['ppc_gwtc5_k10']), 'mass_ratio_m1')
    if args.what in ('m2', 'all'):
        plot_style.save(m2_figure(R), 'm2_q_ranges')
    if args.what in ('chi', 'all'):
        D = {RUNS[run]: chi_data(r, args.core) for run, r in R.items()}
        for stats, name in ((('mean',), 'mean'), (('sd',), 'sd'), (('mean', 'sd'), 'mean_sd')):
            plot_style.save(chi_figure(D, stats, args.core, args.mean_asinh), f'chi_eff_{name}_m1{core}')


if __name__ == '__main__':
    main()
