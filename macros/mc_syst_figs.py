"""Figures and macros for the Monte Carlo systematics (macros/mc_systematics.py's results/post_processing/<run>_mc_syst.npz)
for the Supplemental Material (user, 2026-11-18):

    python macros/mc_syst_figs.py        (login node: pinned, texlive loaded; seconds)
        -> plots/mc_systematics_maps.pdf     sigma_syst / sigma_stat across (m1, q): ln p(m1, q) and the chi_eff mean and
                                             sd; rows = spin models; two columns. Blank where m2 < 3 Msun (undefined).
           plots/mc_systematics_curves.pdf   the same for the one-dimensional quantities: ln p(m1 | q range), the
                                             q-averaged chi_eff mean and sd against m1, the m1-averaged ones against q
           macros/mc_syst.tex, draft/macros/mc_syst.tex
                                             \\mcPrecision{Flex,Gauss} (bits), \\mc{Ten,...}{Mu,Sigma}{Flex,Gauss}
                                             (the Table 1 peak fits), \\mcMax{Map,Curve}{Flex,Gauss}
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'notebooks')); sys.path.insert(0, HERE)
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import plot_style
import analysis as A
from mc_systematics import out_path
from paper_figs import LEG, SMALL_RC, TWO_COL, COLS, Q_NAMES, m_edges
from peak_fits import RUN_NAMES
from reweight import FEATURES, FEATURE_COLOURS

RUNS = {'ppc_gwtc5_k10': 'Flexible', 'ppc_gwtc5_k100_gauss': 'Gaussian'}
VMAX = 0.5
TICKS = (5, 10, 20, 35, 60, 100)
MAPS = (('ln_p_m1_q', r'$\ln p(m_1, q)$'), ('mu_eff_m1_q', r'$\mu_{\chi_{\rm eff}}(m_1, q)$'),
        ('sd_eff_m1_q', r'$\sigma_{\chi_{\rm eff}}(m_1, q)$'))
PEAK_TEX = {name: v[2] for name, v in FEATURES.items()}


def load(run):
    d = np.load(out_path(run))
    return {k[len('ratio/'):]: d[k] for k in d.files if k.startswith('ratio/')}, float(d['precision_bits'])


def m_axis(ax):
    ax.set_xscale('log'); ax.set_xlim(A.MIN_M, 150.)
    ax.set_xticks(TICKS, [f'${t:g}$' for t in TICKS]); ax.minorticks_off()


def maps_figure(R):
    fig, axes = plt.subplots(len(R), len(MAPS), figsize=(TWO_COL, 1.95 * len(R) + 0.5), sharex=True, sharey=True)
    for row, (run, (ratio, _)) in zip(axes, R.items()):
        for ax, (key, lab) in zip(row, MAPS):
            pc = ax.pcolormesh(m_edges, A.q_edges, ratio[key].T, vmin=0., vmax=VMAX, cmap='viridis', rasterized=True)
            for name, ((a, b), (qa, qb), _) in FEATURES.items():
                ax.add_patch(Rectangle((a, qa), b - a, qb - qa, fill=False, ec='w', lw=0.6, ls='--'))
            m_axis(ax); ax.set_ylim(0, 1)
            ax.text(0.04, 0.95, lab, transform=ax.transAxes, ha='left', va='top', color='w')
        row[0].set_ylabel(rf'{RUNS[run]}' '\n' r'$q$')
    for ax in axes[-1]:
        ax.set_xlabel(r'$m_1\,[\textrm{M}_\odot]$')
    fig.subplots_adjust(left=0.08, right=0.9, bottom=0.1, top=0.97, wspace=0.06, hspace=0.08)
    cax = fig.add_axes([0.915, axes[-1, -1].get_position().y0, 0.012,
                        axes[0, -1].get_position().y1 - axes[-1, -1].get_position().y0])
    fig.colorbar(pc, cax=cax, label=r'$\sigma_{\rm syst} / \sigma_{\rm stat}$')
    return fig


def curves_figure(R):
    fig, axes = plt.subplots(len(R), 3, figsize=(TWO_COL, 1.75 * len(R) + 0.55), sharey=True)
    tags = [''] + [f'_q{a:g}-{b:g}' for a, b in A.SUMMARY_Q_RANGES]
    cols, names = ['k'] + COLS, ['all $q$'] + Q_NAMES
    x = A.m_c
    for row, (run, (ratio, _)) in zip(axes, R.items()):
        for tag, c in zip(tags, cols):
            row[0].plot(x, ratio[f'ln_p_m1{tag}'], c=c)
            row[1].plot(x, ratio[f'mu_eff_m1{tag}'], c=c, ls='--')
            row[1].plot(x, ratio[f'sd_eff_m1{tag}'], c=c)
        qc = 0.5 * (A.q_edges[1:] + A.q_edges[:-1])
        row[2].plot(qc, ratio['mu_eff_q'], c='k', ls='--')
        row[2].plot(qc, ratio['sd_eff_q'], c='k')
        row[2].plot(qc, ratio['mu_p_q'], c='C4', ls='--')
        row[2].plot(qc, ratio['sd_p_q'], c='C4')
        for ax in row[:2]:
            m_axis(ax)
            for name, ((a, b), _, _) in FEATURES.items():
                ax.axvspan(a, b, color=FEATURE_COLOURS[name], alpha=0.12, lw=0)
        row[2].set_xlim(0, 1)
        row[0].set_ylabel(rf'{RUNS[run]}' '\n' r'$\sigma_{\rm syst} / \sigma_{\rm stat}$')
        row[0].text(0.04, 0.95, r'$\ln p(m_1 \mid q \in \textrm{range})$', transform=row[0].transAxes, va='top')
        row[1].text(0.04, 0.95, r'$\chi_{\rm eff}$ moments, $q$-averaged', transform=row[1].transAxes, va='top')
        row[2].text(0.04, 0.95, r'$m_1$-averaged: $\chi_{\rm eff}$ (black), $\chi_{\rm p}$ (purple)', transform=row[2].transAxes, va='top')
    for ax in axes.flat:
        ax.set_ylim(0, VMAX)
    for ax in axes[-1, :2]:
        ax.set_xlabel(r'$m_1\,[\textrm{M}_\odot]$')
    axes[-1, 2].set_xlabel(r'$q$')
    for ax in axes[:-1].flat:
        ax.tick_params(labelbottom=False)
    h = [Line2D([], [], c=c, label=n) for c, n in zip(cols, names)]
    h += [Line2D([], [], c='0.4', label='sd'), Line2D([], [], c='0.4', ls='--', label='mean')]
    fig.legend(handles=h, loc='upper center', ncol=len(h), **LEG)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.1, top=0.92, wspace=0.06, hspace=0.1)
    return fig


def write_tex(R):
    lines = ['% Monte Carlo systematics (macros/mc_syst_figs.py from macros/mc_systematics.py): sigma_syst / sigma_stat;',
             '% Precision: population-error precision statistic [bits]; {Mu,Sigma}: Table 1 peak fits;',
             '% MaxMap: largest ratio over the (m1, q) maps of ln p, chi_eff mean, sd; MaxCurve: over the 1D quantities']
    for run, (ratio, prec) in R.items():
        sfx = RUN_NAMES[run]
        lines.append(rf'\newcommand{{\mcPrecision{sfx}}}{{{prec:.1f}}}')
        for name, tex in PEAK_TEX.items():
            for p, P in (('mu', 'Mu'), ('sigma', 'Sigma')):
                lines.append(rf'\newcommand{{\mc{tex}{P}{sfx}}}{{{ratio[f"peak{name}_{p}"].item():.2f}}}')
        mx_map = max(np.nanmax(ratio[k]) for k, _ in MAPS)
        mx_cur = max(np.nanmax(v) for k, v in ratio.items() if v.ndim == 1 and v.size > 1)
        lines += [rf'\newcommand{{\mcMaxMap{sfx}}}{{{mx_map:.2f}}}', rf'\newcommand{{\mcMaxCurve{sfx}}}{{{mx_cur:.2f}}}']
        print(f'{run}: precision {prec:.2f} bits, max map {mx_map:.2f}, max 1D {mx_cur:.2f}, peaks',
              {k: round(float(v.item()), 3) for k, v in ratio.items() if k.startswith('peak')})
    text = '\n'.join(lines) + '\n'
    for path in (os.path.join(HERE, 'mc_syst.tex'), os.path.join(REPO, 'draft', 'macros', 'mc_syst.tex')):
        with open(path, 'w') as f:
            f.write(text)
        print('wrote', path)


if __name__ == '__main__':
    plot_style.apply(); plt.rcParams.update(SMALL_RC)
    R = {run: load(run) for run in RUNS}
    plot_style.save(maps_figure(R), 'mc_systematics_maps')
    plot_style.save(curves_figure(R), 'mc_systematics_curves')
    write_tex(R)
