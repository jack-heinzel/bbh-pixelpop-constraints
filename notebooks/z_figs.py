"""Figures of the redshift-dependent spin conditionals (notebooks/analysis_z.py caches): the moments of
p(chi | z, m1 band, q band) against z, one row per run.

    python notebooks/z_figs.py [--core] [<run> ...]   (default: ppc_gwtc5_k10_kq5_kz5 ppc_gwtc5_k100_gauss_kz10)
        -> plots/chi_eff_mean_sd_z_m1bands[_core].pdf, chi_eff_mean_sd_z_qbands[_core].pdf, chi_p_mean_sd_z_m1bands.pdf
    python notebooks/z_figs.py --marginals [--core] [<run> ...]   (default: MARG_RUNS)
        -> plots/{chi_eff,chi_p}_mean_sd_marginals_zmodels[_core].pdf

--marginals (user, 2026-11-02): the marginal moments only, no bands, against m1, q and z in separate columns, every run
overplotted: p(chi | m1) and p(chi | q) average the conditional over the other mass variable and the draw's p(z)
(analysis_z MARG), p(chi | z) over the windowed p(m1, q) (CURVES, all m1 and q).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np

import analysis_z as Z

RUNS = {'ppc_gwtc5_k10_kq5_kz5': 'Flexible', 'ppc_gwtc5_k100_gauss_kz10': 'Gaussian',
        'ppc_gwtc5_k100_gauss_kz100_zb100_zadd': r'Gaussian, $F(m_1, q) + G(z)$',
        'ppc_gwtc5_k100_gauss_kz100_zb100_mqadd_zadd': r'Gaussian, $F_1(m_1) + F_2(q) + G(z)$'}
Q_BAND_COLS = ('C0', 'C1')                       # as paper_figs.COLS for SUMMARY_Q_RANGES
# the three Gaussian redshift models: joint (m1, q, z) fields, F(m1, q) + G(z), F1(m1) + F2(q) + G(z)
MARG_RUNS = {'ppc_gwtc5_k100_gauss_kz10': (r'joint $(m_1, q, z)$', 'k'),
             'ppc_gwtc5_k100_gauss_kz100_zb100_zadd': (r'$F(m_1, q) + G(z)$', 'C0'),
             'ppc_gwtc5_k100_gauss_kz100_zb100_mqadd_zadd': (r'$F_1(m_1) + F_2(q) + G(z)$', 'C3')}
M_RANGE, Q_RANGE = (5., 150.), (0.1, 1.)


def marginal_figure(D, spin):
    """Rows: mean and sd of ``spin`` (analysis_z.SPINS); columns: against m1, q and z. D: {label: (colour, MARG,
    CURVES, z edges)}; each run's median (line) and 90% (band)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import paper_figs as F
    s = Z.SPINS.index(spin)
    sym = r'\chi_p' if spin == 'chi_p' else r'\chi_\mathrm{eff}'
    cut = rf', $|\chi_\mathrm{{eff}}| < {F.A.CHI_CORE:g}$' if spin == 'chi_eff_core' else ''
    fig, axes = plt.subplots(2, 3, figsize=(F.TWO_COL, 3.4), sharex='col')
    xs = (F.A.m_c, F.A.q_c)
    for j, st in enumerate(Z.STATS):
        ys = []
        for k, (col, marg, curves, ze) in enumerate(D.values()):
            zx = np.concatenate([[ze[0]], 0.5 * (ze[1:] + ze[:-1]), [ze[-1]]])
            series = [(xs[0], marg[:, s, j, 0]), (xs[1], marg[:, s, j, 1]), (zx, None)]
            for c, (x, y) in enumerate(series):
                l, m, h = F.ext(F.q5(curves[:, 0, 0, s, j])) if c == 2 else F.q5(y)
                vis = (x >= M_RANGE[0]) & (x <= M_RANGE[1]) if c == 0 else (x >= Q_RANGE[0]) if c == 1 else slice(None)
                ys += [l[vis], h[vis]]
                ax = axes[j, c]
                ax.fill_between(x, l, h, color=col, alpha=0.15, lw=0)
                ax.plot(x, l, color=col, lw=0.5, alpha=0.7); ax.plot(x, h, color=col, lw=0.5, alpha=0.7)
                ax.plot(x, m, color=col, lw=1.1)
        lo, hi = min(np.nanmin(y) for y in ys), max(np.nanmax(y) for y in ys)
        for c in range(3):
            ax = axes[j, c]
            if st == 'sd':
                F.log_sd_axis(ax, lo, hi)
            elif spin == 'chi_p':
                ax.set_ylim(0, 1.05 * hi)
            else:
                ax.axhline(0, color='0.5', lw=0.6, ls=':')
                ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
            if c > 0:
                ax.tick_params(labelleft=False)
        axes[j, 0].set_ylabel(rf'${"\\mu" if st == "mean" else "\\sigma"}_{{{sym}}}$' + cut)
    axes[0, 0].set_xscale('log'); axes[0, 0].set_xlim(*M_RANGE)
    axes[0, 1].set_xlim(*Q_RANGE); axes[0, 2].set_xlim(ze[0], ze[-1])
    axes[1, 1].set_xticks((0.2, 0.4, 0.6, 0.8)); axes[1, 2].set_xticks((0.5, 1., 1.5))   # no clash at the joins
    for c, lab in enumerate((r'$m_1\,[\textrm{M}_\odot]$', '$q$', '$z$')):
        axes[1, c].set_xlabel(lab)
    axes[1, 0].set_xticks((5, 10, 20, 50, 100), ['$5$', '$10$', '$20$', '$50$', '$100$'])
    axes[1, 0].minorticks_off()
    handles = [(Patch(color=col, alpha=0.15, lw=0), Line2D([], [], color=col, lw=1.1)) for col, *_ in D.values()]
    fig.legend(handles, list(D), loc='upper center', ncol=len(D), bbox_to_anchor=(0.5, 1.0), **F.LEG)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.17 / fig.get_figheight())); fig.subplots_adjust(hspace=0.06, wspace=0.05)
    return fig


def z_figure(C, ze, spin, by='m1'):
    """Rows: runs; columns: mean and sd of ``spin`` (analysis_z.SPINS) against z, for each m1 band (all q) or each
    q band (all m1): 90% bands and medians, the all-band 90% as black lines."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import paper_figs as F
    s = Z.SPINS.index(spin)
    core = spin == 'chi_eff_core'
    sym = r'\chi_p' if spin == 'chi_p' else r'\chi_\mathrm{eff}'
    cut = rf', $|\chi_\mathrm{{eff}}| < {F.A.CHI_CORE:g}$' if core else ''
    if by == 'm1':
        bands, cols = list(range(1, len(Z.M_BANDS))), F.M1_BAND_COLS
        names = [rf'${a:g} < m_1 < {b:g}\,\textrm{{M}}_\odot$' for a, b in Z.M_BANDS[1:]]
        sel = lambda c, i: c[:, i, 0]
    else:
        bands, cols = list(range(1, len(Z.Q_BANDS))), Q_BAND_COLS
        names = [rf'${a:g} < q < {b:g}$' for a, b in Z.Q_BANDS[1:]]
        sel = lambda c, j: c[:, 0, j]
    zx = np.concatenate([[ze[0]], 0.5 * (ze[1:] + ze[:-1]), [ze[-1]]])
    fig, axes = plt.subplots(len(C), 2, figsize=(F.TWO_COL, 1.65 * len(C) + 0.45), sharex=True, squeeze=False)
    for j, st in enumerate(Z.STATS):
        ys = [sel(c, b)[:, s, j] for c in C.values() for b in [0] + bands]
        lo = min(np.nanmin(np.nanpercentile(y, 5, 0)) for y in ys)
        hi = max(np.nanmax(np.nanpercentile(y, 95, 0)) for y in ys)
        for i, (name, c) in enumerate(C.items()):
            ax = axes[i, j]
            for b, col in zip(bands, cols):
                l, m, h = F.ext(F.q5(sel(c, b)[:, s, j]))
                ax.fill_between(zx, l, h, color=col, alpha=0.25, lw=0); ax.plot(zx, m, color=col, lw=1.0)
            l, _, h = F.ext(F.q5(sel(c, 0)[:, s, j]))
            ax.plot(zx, l, color='k', lw=0.7); ax.plot(zx, h, color='k', lw=0.7)
            if st == 'sd':
                F.log_sd_axis(ax, lo, hi)
            elif spin == 'chi_p':
                ax.set_ylim(0, 1.05 * hi)
            else:
                ax.axhline(0, color='0.5', lw=0.6, ls=':')
                ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
            ax.set_xlim(ze[0], ze[-1])
            if i == len(C) - 1:
                ax.set_xlabel('$z$')
            ax.set_ylabel(rf'${"\\mu" if st == "mean" else "\\sigma"}_{{{sym}}}(z)$' + cut)
            ax.text(0.03, 0.93, name, transform=ax.transAxes, va='top', ha='left',
                    bbox=dict(facecolor='w', alpha=0.8, edgecolor='none', pad=1.5))
    handles = [Line2D([], [], color='k', lw=0.7)] + [(Patch(color=c, alpha=0.25, lw=0), Line2D([], [], color=c, lw=1.0))
                                                    for c in cols[:len(bands)]]
    labels = [r'all $m_1$ (90\%)' if by == 'm1' else r'all $q$ (90\%)'] + names
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 1.0), **F.LEG)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.17 / fig.get_figheight())); fig.subplots_adjust(hspace=0.06)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='*', default=list(RUNS))
    ap.add_argument('--core', action='store_true', help='chi_eff within |chi_eff| < analysis.CHI_CORE')
    ap.add_argument('--marginals', action='store_true', help='marginal moments against m1, q, z, runs overplotted')
    args = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import pixelpop  # noqa: F401  (its import resets matplotlib's rcParams; import it before the style)
    import matplotlib.pyplot as plt
    import plot_style
    import paper_figs as F
    plot_style.apply(); plt.rcParams.update(F.SMALL_RC)
    if args.marginals:
        runs = args.runs if args.runs != list(RUNS) else list(MARG_RUNS)
        D = {}
        for run in runs:
            label, col = MARG_RUNS.get(run, (run, None))
            curves, ze = Z.load_curves(run)
            D[label] = (col, Z.load_marg(run), curves, ze)
        tag = '_core' if args.core else ''
        plot_style.save(marginal_figure(D, 'chi_eff_core' if args.core else 'chi_eff'),
                        f'chi_eff_mean_sd_marginals_zmodels{tag}')
        if not args.core:
            plot_style.save(marginal_figure(D, 'chi_p'), 'chi_p_mean_sd_marginals_zmodels')
        return
    C, ze = {}, None
    for run in args.runs:
        C[RUNS.get(run, run)], ze = Z.load_curves(run)
    spin, tag = ('chi_eff_core', '_core') if args.core else ('chi_eff', '')
    tag += ('_mqzadd' if all('_mqadd' in r for r in args.runs) else '_zadd' if all('_zadd' in r for r in args.runs)
            else '')                                                    # an additive model alone: its own files
    plot_style.save(z_figure(C, ze, spin, 'm1'), f'chi_eff_mean_sd_z_m1bands{tag}')
    plot_style.save(z_figure(C, ze, spin, 'q'), f'chi_eff_mean_sd_z_qbands{tag}')
    if not args.core:
        plot_style.save(z_figure(C, ze, 'chi_p', 'm1'), f'chi_p_mean_sd_z_m1bands{tag}')


if __name__ == '__main__':
    main()
