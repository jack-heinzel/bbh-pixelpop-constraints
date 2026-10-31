"""Mass bookkeeping between the near-equal-mass peaks and the q ~ 0.5 peaks, per posterior draw, from the joint fits.

    python macros/peak_pairs.py [--modes]
        -> macros/peak_pairs.tex, plots/compare_gwtc5_field_vs_gauss_k100_peak_pairs.pdf
           (--modes: first recompute the fit-free modes, results/post_processing/<run>_peak_modes.npz; a few minutes)

For each pair (lo, hi) = (10, 20), (35, 60) (user, 2026-10-26): the total mass of a binary at the lower peak,
M_lo = mu_lo (1 + q_lo), against the location of the higher peak, mu_hi, as the retained fraction mu_hi / M_lo (0.95
for ~5% radiated; > 1 violates mass conservation for a remnant of the lower peak) and its inverse
R = mu_lo (1 + q_lo) / mu_hi. q_lo is the lower peak's mean q (joint_fits.q_moments; the main macros) or its mu_q
('MuQ' macros). Both peaks come from the same posterior draw (the joint fits' rows are aligned), so correlations carry
through; draws without a peak in either window are dropped. Also: the location ratio mu_hi / mu_lo and the posterior
quantile of a retained fraction of 0.95.

Fit-free version ('Mode' macros; user, 2026-10-26): the lower peak's total mass is the mode of p(M | q in its band),
M = m1 (1 + q) (each (ln m1, q) cell split into NS x NS sub-cells, binned in ln M), and the higher peak's location is
the mode of p(m1 | q in its band) (PEAKS' q bands), each the maximum of ln p refined by a parabola, within MODE_WINDOWS.
Draws whose maximum is at a search-window edge are dropped.
"""
import os
import sys

import numpy as np

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
from peak_fits import MASS_DEC, PEAKS, RUN_NAMES, fmt, pct
from joint_fits import npz_path, q_moments

PAIRS = (('10', '20'), ('35', '60'))
RETAINED = 0.95
RUN_LABELS = {'ppc_gwtc5_k10': ('Flexible', 'C0'), 'ppc_gwtc5_k100_gauss': ('Gaussian', 'C3')}
NAME = 'compare_gwtc5_field_vs_gauss_k100_peak_pairs'
# mode search windows [Msun]: the lower peak's total mass, the higher peak's m1
MODE_WINDOWS = {('10', '20'): ((12., 30.), (15., 30.)), ('35', '60'): ((45., 85.), (40., 100.))}
NS = 4
M_GRID = np.exp(np.linspace(np.log(6.), np.log(400.), 301))


def modes_path(run):
    return os.path.join(REPO, 'results', 'post_processing', f'{run}_peak_modes.npz')


def p_total_mass(p_mass, qband):
    """p(ln M | q in qband) per draw on M_GRID, M = m1 (1 + q); p_mass (draws, Bm, Bq) as analysis.load_light's."""
    import analysis as A
    du = (np.arange(NS) + 0.5) / NS - 0.5
    lM = (A.lnm_c[:, None, None, None] + du[None, :, None, None] * A.dlnm
          + np.log1p(A.q_c[None, None, :, None] + du[None, None, None, :] * A.dq))           # (Bm, NS, Bq, NS)
    lg = np.log(M_GRID)
    idx = np.clip(np.round((lM - lg[0]) / (lg[1] - lg[0])).astype(int), 0, len(lg) - 1)
    idx = np.moveaxis(idx, 2, 1).reshape(A.B * A.B, NS * NS)
    keep = np.broadcast_to(((A.q_c >= qband[0]) & (A.q_c <= qband[1]))[None], (A.B, A.B)).ravel()
    W = np.zeros((A.B * A.B, len(lg)))
    for k in range(NS * NS):
        np.add.at(W, (np.arange(A.B * A.B), idx[:, k]), keep / NS ** 2)
    out = p_mass.reshape(len(p_mass), -1).astype(float) @ W
    return out / (out.sum(1, keepdims=True) * (lg[1] - lg[0]))


def mode(x, y, win):
    """Per-draw mode of densities y (draws, len(x)) within win: the maximum of ln y, refined by a parabola in
    (ln x, ln y); NaN where the maximum is at an edge of win."""
    s = (x >= win[0]) & (x <= win[1])
    lx, ys = np.log(x[s]), np.maximum(y[:, s], 1e-300)
    j = ys.argmax(1); k = np.clip(j, 1, s.sum() - 2); r = np.arange(len(ys))
    y0, y1, y2 = (np.log(ys[r, k + o]) for o in (-1, 0, 1))
    with np.errstate(divide='ignore', invalid='ignore'):
        off = np.clip(0.5 * (y0 - y2) / (y0 - 2 * y1 + y2), -1, 1)
    return np.where((j == 0) | (j == s.sum() - 1), np.nan, np.exp(lx[k] + off * (lx[1] - lx[0])))


def compute_modes(run):
    """The fit-free modes for every pair, per posterior draw (load_light's order), saved to modes_path(run)."""
    sys.path.insert(0, os.path.join(REPO, 'notebooks'))
    import analysis as A
    P = A.load_light(run).p_mass
    out = {}
    for lo, hi in PAIRS:
        (wM, wm), q_lo, q_hi = MODE_WINDOWS[(lo, hi)], PEAKS[lo][0], PEAKS[hi][0]
        out[f'Mtot_{lo}'] = mode(M_GRID, p_total_mass(P, q_lo), wM)
        p_hi = A.p_m1_given_q_range(type('R', (), {'p_mass': P})(), q_hi)
        out[f'm1_{hi}'] = mode(A.m_c, np.nan_to_num(p_hi), wm)
    np.savez(modes_path(run), **out)
    print('saved', modes_path(run))


def pair_stats(run, lo, hi):
    """Per-draw retained fraction (mean q and mu_q), R and location ratio, for draws with both peaks."""
    d = np.load(npz_path(run))
    A, B = d[lo], d[hi]
    ok = np.all(np.isfinite(A), 1) & np.all(np.isfinite(B), 1)
    A, B = A[ok], B[ok]
    qmean = q_moments(A)[0]
    M, M_muq = A[:, 1] * (1 + qmean), A[:, 1] * (1 + A[:, 3])
    out = {'retained': B[:, 1] / M, 'retained_muq': B[:, 1] / M_muq, 'R': M / B[:, 1], 'loc': B[:, 1] / A[:, 1]}
    if os.path.exists(modes_path(run)):
        m = np.load(modes_path(run))
        Mt, mh = m[f'Mtot_{lo}'], m[f'm1_{hi}']
        good = np.isfinite(Mt) & np.isfinite(mh)
        out.update(Mtot_mode=Mt[good], hi_mode=mh[good], retained_mode=mh[good] / Mt[good])
    return out


def write_tex(S):
    lines = ['% Mass bookkeeping between peak pairs, per posterior draw of the joint (m1, q) fits; median and 90%'
             ' (macros/peak_pairs.py). Retained = mu_hi / [mu_lo (1 + q_lo)], q_lo the lower peak\'s mean q (MuQ: its'
             ' mu_q); Ratio = its inverse; LocRatio = mu_hi / mu_lo; QuantileRetained = P(retained < 0.95); Mode: the'
             ' fit-free version from the modes of p(M | q band) and p(m1 | q band)']
    for run, suffix in RUN_NAMES.items():
        lines.append(f'% {run}')
        for lo, hi in PAIRS:
            s, tex = S[run][(lo, hi)], PEAKS[lo][2] + PEAKS[hi][2]
            lines.append(f'% {lo} -> {hi}: {len(s["R"])} draws with both peaks')
            for par, key in (('Retained', 'retained'), ('RetainedMuQ', 'retained_muq'), ('Ratio', 'R'),
                             ('LocRatio', 'loc')):
                lines.append(rf'\newcommand{{\pair{tex}{par}{suffix}}}{{{fmt(s[key])}}}')
            lines.append(rf'\newcommand{{\pair{tex}QuantileRetained{suffix}}}{{{pct(np.mean(s["retained"] < RETAINED))}}}')
            if 'retained_mode' in s:
                lines.append(f'% fit-free: modes of p(M | q band) and p(m1 | q band); {len(s["retained_mode"])} draws')
                for par, key in (('MTotMode', 'Mtot_mode'), ('HiMode', 'hi_mode'), ('RetainedMode', 'retained_mode')):
                    dec = MASS_DEC if par in ('MTotMode', 'HiMode') else None     # masses to 0.1 Msun
                    lines.append(rf'\newcommand{{\pair{tex}{par}{suffix}}}{{{fmt(s[key], dec)}}}')
                lines.append(rf'\newcommand{{\pair{tex}QuantileRetainedMode{suffix}}}'
                             rf'{{{pct(np.mean(s["retained_mode"] < RETAINED))}}}')
    path = os.path.join(HERE, 'peak_pairs.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('wrote', path)


def figure(S):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import plot_style
    plot_style.apply()
    fig, axes = plt.subplots(1, len(PAIRS), figsize=(5.5 * len(PAIRS), 4.2), sharey=False)
    bins = np.linspace(0.6, 1.5, 46)
    for ax, (lo, hi) in zip(axes, PAIRS):
        for run, (label, col) in RUN_LABELS.items():
            s = S[run][(lo, hi)]
            ax.hist(s['retained'], bins=bins, density=True, histtype='step', color=col, lw=1.5)
            ax.hist(s['retained_muq'], bins=bins, density=True, histtype='step', color=col, lw=1, ls='--')
            if 'retained_mode' in s:
                ax.hist(s['retained_mode'], bins=bins, density=True, histtype='step', color=col, lw=1, ls=':')
        ax.axvline(RETAINED, color='k', lw=1, ls='-.')
        ax.axvspan(1., bins[-1], color='0.92', zorder=0, lw=0)          # > 1: more mass than the progenitor binary
        ax.set_xlim(bins[0], bins[-1]); ax.set_yticks([])
        ax.set_xlabel(rf'$\mu_{{{hi}}} / [\mu_{{{lo}}}(1 + q_{{{lo}}})]$')
        ax.text(0.03, 0.95, rf'$\sim {lo} \to \sim {hi}\,\textrm{{M}}_\odot$', transform=ax.transAxes, va='top')
    axes[0].set_ylabel('density')
    handles = ([Line2D([], [], color=c, lw=1.5) for _, c in RUN_LABELS.values()] +
               [Line2D([], [], color='0.3', lw=1.5), Line2D([], [], color='0.3', lw=1, ls='--'),
                Line2D([], [], color='0.3', lw=1, ls=':'), Line2D([], [], color='k', lw=1, ls='-.')])
    labels = ([l for l, _ in RUN_LABELS.values()] + [r'mean $q$', r'$\mu_q$', 'modes (fit-free)',
                                                    rf'{RETAINED:g} retained'])
    axes[-1].legend(handles, labels, loc='upper right', **plot_style.LEG_KW)
    fig.tight_layout()
    plot_style.save(fig, NAME)


def main(modes=False):
    if modes:
        for run in RUN_NAMES:
            compute_modes(run)
    S = {run: {p: pair_stats(run, *p) for p in PAIRS} for run in RUN_NAMES}
    for run in RUN_NAMES:
        for p in PAIRS:
            s = S[run][p]
            print(f'{run} {p[0]} -> {p[1]}: retained {fmt(s["retained"])} (mu_q: {fmt(s["retained_muq"])}), '
                  f'R {fmt(s["R"])}, mu_hi / mu_lo {fmt(s["loc"])}, P(retained < {RETAINED}) '
                  f'{np.mean(s["retained"] < RETAINED):.2f}')
            if 'retained_mode' in s:
                print(f'    fit-free: M_tot mode {fmt(s["Mtot_mode"])}, m1 mode {fmt(s["hi_mode"])}, retained '
                      f'{fmt(s["retained_mode"])}, P(< {RETAINED}) {np.mean(s["retained_mode"] < RETAINED):.2f}')
    write_tex(S)
    figure(S)


if __name__ == '__main__':
    main('--modes' in sys.argv[1:])
