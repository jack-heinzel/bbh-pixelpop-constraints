"""Figures of the redshift-dependent spin conditionals (notebooks/analysis_z.py caches): the moments of
p(chi | z, m1 band, q band) against z, one row per run.

    python notebooks/z_figs.py [--core] [<run> ...]   (default: ppc_gwtc5_k10_kq5_kz5 ppc_gwtc5_k100_gauss_kz10)
        -> plots/chi_eff_mean_sd_z_m1bands[_core].pdf, chi_eff_mean_sd_z_qbands[_core].pdf, chi_p_mean_sd_z_m1bands.pdf
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
    args = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import pixelpop  # noqa: F401  (its import resets matplotlib's rcParams; import it before the style)
    import matplotlib.pyplot as plt
    import plot_style
    import paper_figs as F
    plot_style.apply(); plt.rcParams.update(F.SMALL_RC)
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
