"""Figures of the window fits: posterior against the empirical-Bayes prior (macros/prior_fits.py), both spin models.

    python macros/window_fits_fig.py
        -> plots/compare_gwtc5_field_vs_gauss_k100_window_fits_prior.pdf   (per-draw distributions, window x statistic)
           plots/compare_gwtc5_field_vs_gauss_k100_window_fits_plane.pdf   (peak location in (m1, mean q))
           plots/compare_gwtc5_field_vs_gauss_k100_window_fits_plane_muq.pdf   (the same in (m1, mu_q))

Needs the joint fits (results/post_processing/<run>_joint_fits.npz) and the prior fits (<run>_prior_fits.npz). The
prior is the field run's (the gauss run's kernel posterior gives the same prior to within the sampling noise).
"""
import os
import sys

import numpy as np

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

import plot_style
from peak_fits import PEAKS
from joint_fits import npz_path, q_moments
from prior_fits import Q_WINDOWS, prior_path

RUNS = {'ppc_gwtc5_k10': ('Flexible', 'C0'), 'ppc_gwtc5_k100_gauss': ('Gaussian', 'C3')}
PRIOR_RUN = 'ppc_gwtc5_k10'
NAME = 'compare_gwtc5_field_vs_gauss_k100_window_fits'
PEAK_LABEL = {'10': r'$\sim 10\,\textrm{M}_\odot$, $q \sim 1$', '35': r'$\sim 35\,\textrm{M}_\odot$, $q \sim 1$',
              '20': r'$\sim 20\,\textrm{M}_\odot$, $q \sim 0.5$', '60': r'$\sim 60\,\textrm{M}_\odot$, $q \sim 0.5$'}


def joint_stats(F):
    """Per-draw statistics of the joint fits (draws with a peak)."""
    F = F[np.all(np.isfinite(F), 1)]
    qm, qs = q_moments(F)
    return {'mu': F[:, 1], 'sigma': F[:, 2], 'mu_q': F[:, 3], 'qmean': qm, 'qspread': qs, 'f_G': F[:, 9]}


def load():
    post = {run: {name: joint_stats(np.load(npz_path(run))[name]) for name in PEAKS} for run in RUNS}
    P = np.load(prior_path(PRIOR_RUN))
    prior = {name: joint_stats(P[f'joint_{name}']) for name in PEAKS}
    for name in PEAKS:
        prior[name]['qfree'] = P[f'qmean_prior_{name}']
        for run in RUNS:
            post[run][name]['qfree'] = np.load(prior_path(run))[f'qmean_post_{name}']
    return post, prior


ROWS = (('mu', r'$\mu\,[\textrm{M}_\odot]$'), ('sigma', r'$\sigma\,[\textrm{M}_\odot]$'),
        ('qfree', r'mean $q$ (fit-free)'), ('qmean', r'mean $q$ of the peak'), ('qspread', r'spread in $q$ of the peak'),
        ('f_G', r'$f_G$'))


def range_of(key, name):
    win = PEAKS[name][1]
    return {'mu': win, 'sigma': (0, 0.5 * (win[1] - win[0])), 'qfree': (0, 1), 'qmean': (0, 1), 'qspread': (0, 0.35),
            'f_G': (0, 1)}[key]


def prior_figure(post, prior):
    fig, axes = plt.subplots(len(ROWS), len(PEAKS), figsize=(4.2 * len(PEAKS), 2.3 * len(ROWS)))
    for j, name in enumerate(PEAKS):
        for i, (key, label) in enumerate(ROWS):
            ax = axes[i, j]
            bins = np.linspace(*range_of(key, name), 41)
            ax.hist(prior[name][key], bins=bins, density=True, color='0.8', lw=0)
            for run, (_, col) in RUNS.items():
                ax.hist(post[run][name][key], bins=bins, density=True, histtype='step', color=col, lw=1.5)
            ax.set_xlim(bins[0], bins[-1]); ax.set_yticks([])
            ax.set_xlabel(label)
            if key == 'qfree':
                w = Q_WINDOWS[name]
                ax.text(0.03, 0.95, rf'${w[0]:g} < m_1 < {w[1]:g}$', transform=ax.transAxes, va='top', fontsize=12)
        axes[0, j].text(0.5, 1.06, PEAK_LABEL[name], transform=axes[0, j].transAxes, ha='center', va='bottom')
    for ax in axes[:, 0]:
        ax.set_ylabel('density')
    handles = [Patch(color='0.8', lw=0)] + [Line2D([], [], color=c, lw=1.5) for _, c in RUNS.values()]
    axes[0, -1].legend(handles, ['prior'] + [l for l, _ in RUNS.values()], loc='upper right', **plot_style.LEG_KW)
    fig.tight_layout(h_pad=0.6)
    return fig


def bounded_kde(xy, lo, hi):
    """Gaussian KDE of the 2D samples ``xy`` truncated to the box [lo, hi] by reflection: the density at a point is
    scipy's KDE (bandwidth from the samples) summed over the point's mirror images in the box's edges and corners, so no
    probability leaks out of the box and the density does not drop at the bounds."""
    kde = gaussian_kde(xy)

    def density(p):
        out = 0.
        for rx in (None, lo[0], hi[0]):
            for ry in (None, lo[1], hi[1]):
                x = p[0] if rx is None else 2 * rx - p[0]
                y = p[1] if ry is None else 2 * ry - p[1]
                out = out + kde(np.vstack([x, y]))
        inside = (p[0] >= lo[0]) & (p[0] <= hi[0]) & (p[1] >= lo[1]) & (p[1] <= hi[1])
        return np.where(inside, out, 0.)
    return density


def hpd_levels(density, xy, fracs=(0.9, 0.5)):
    """Density levels enclosing ``fracs`` of the samples ``xy``."""
    d = np.sort(density(xy))[::-1]
    return [d[int(f * len(d)) - 1] for f in fracs]


PLANE_Y = {'qmean': r'mean $q$ of the peak', 'mu_q': r'$\mu_q$, location of the peak in $q$'}


def plane_figure(post, prior, ykey='qmean'):
    """Peak location in (mu, ykey): bounded-KDE contours, 90% (solid) and 50% (dashed) for each run, 90% for the prior.
    The bounds are the fit's: mu in the window (log), the q statistic in [0, 1]."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (_, win, _) in PEAKS.items():
        lo, hi = (np.log(win[0]), 0.), (np.log(win[1]), 1.)
        X, Y = np.meshgrid(np.linspace(lo[0], hi[0], 150), np.linspace(0, 1, 150))
        grid = np.vstack([X.ravel(), Y.ravel()])
        for s, col, fracs, lws, lss in [(prior[name], '0.55', (0.9,), (1.,), (':',))] + \
                [(post[run][name], col, (0.9, 0.5), (1.5, 1.), ('-', '--')) for run, (_, col) in RUNS.items()]:
            xy = np.vstack([np.log(s['mu']), s[ykey]])
            density = bounded_kde(xy, lo, hi)
            ax.contour(np.exp(X), Y, density(grid).reshape(X.shape), levels=sorted(hpd_levels(density, xy, fracs)),
                       colors=col, linewidths=lws, linestyles=lss)        # levels ascending: 90% first
    ax.set_xscale('log'); ax.set_xlim(4, 110); ax.set_ylim(0, 1)
    ax.set_xticks([5, 10, 20, 50, 100]); ax.set_xticklabels(['5', '10', '20', '50', '100'])
    ax.set_xlabel(r'peak location $\mu\,[\textrm{M}_\odot]$'); ax.set_ylabel(PLANE_Y[ykey])
    handles = ([Line2D([], [], color=c, lw=1.5) for _, c in RUNS.values()] +
               [Line2D([], [], color='0.55', lw=1, ls=':')])
    ax.legend(handles, [l for l, _ in RUNS.values()] + [r'prior (90\%)'], loc='lower left', **plot_style.LEG_KW)
    return fig


def main():
    plot_style.apply()
    post, prior = load()
    plot_style.save(prior_figure(post, prior), f'{NAME}_prior')
    plot_style.save(plane_figure(post, prior), f'{NAME}_plane')
    plot_style.save(plane_figure(post, prior, 'mu_q'), f'{NAME}_plane_muq')


if __name__ == '__main__':
    main()
