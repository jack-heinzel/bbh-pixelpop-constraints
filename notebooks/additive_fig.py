"""Conditional against marginal spin evolution in the fully additive Gaussian model (user, 2026-11-02): each spin
parameter is offset + F1(ln m1) + F2(q) + G(z), so its dependence on one variable is the same at every value of the
other two. The top row shows those conditional pieces, centred on their population average (the normalization is not
identified: only the shape); the bottom row the population marginals p(chi_eff | m1), p(chi_eff | q), p(chi_eff | z)
(analysis_z MARG, CURVES).

    python notebooks/additive_fig.py      (CPU, pinned on a login node; ~5 min, cached)
        -> plots/chi_eff_sd_conditional_marginal_additive.pdf, plots/chi_eff_mean_conditional_marginal_additive.pdf

Top row, sd: exp(F - <F>) of ln sd_eff, the factor on the Gaussian's width before truncation to the physical range
(the truncated width grows less where the Gaussian is broad); mean: F - <F> of mu_eff, the shift of the Gaussian's
centre. <F> weights by the population: p(m1), p(q) (the draw's windowed mass field) and p(z) (its redshift power law).
The bottom row uses |chi_eff| < analysis.CHI_CORE, as the paper's width figures.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import h5py
import numpy as np

import analysis as A
import analysis_z as Z

RUN = 'ppc_gwtc5_k100_gauss_kz100_zb100_mqadd_zadd'
CACHE = os.path.join(A.REPO, 'results', 'post_processing', f'{RUN}_additive_fields.npz')
KEYS = ('gauss_offsets', '_gauss_sites', 'lnsigma_gauss', 'log_nu_gauss', 'log_ranges_gauss', '_gauss_q_sites',
        'lnsigma_gauss_q', 'log_nu_gauss_q', 'log_ranges_gauss_q', '_gauss_z_sites', 'lnsigma_gauss_z',
        'log_nu_gauss_z', 'log_ranges_gauss_z')
COL = 'C3'                              # the model's colour in z_figs.MARG_RUNS
M_RANGE, Q_RANGE = (5., 150.), (0.1, 1.)


def build():
    """Per draw: the five parameters' F1 (Bm), F2 (Bq), G (Bz) and offsets, and the population weights of m1, q, z."""
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import pixelpop
    from model import axis_eigs, matern_spectrum, to_grid
    _, km, kq, kz, Bz, _ = Z.z_config(RUN)
    B = A.B
    (w_ax, v_ax), = axis_eigs([pixelpop.utils.create_CAR_coupling_matrix(B, 1, isVisible=False)])
    (wz, vz), = axis_eigs([pixelpop.utils.create_CAR_coupling_matrix(Bz, 1, isVisible=False)])
    Vm, Vq, Vz = v_ax[:, :km], v_ax[:, :kq], vz[:, :kz]

    @jax.jit
    def fields(white, lnsig, lognu, logr, white_q, lnsig_q, lognu_q, logr_q, white_z, lnsig_z, lognu_z, logr_z):
        one = lambda w, v, ls, lr, ln, wh, n: jnp.stack([to_grid(matern_spectrum([w], ls[i], lr[i], jnp.exp(ln), n),
                                                                 wh[i], [v]) for i in range(5)])
        return (one(w_ax[:km], Vm, lnsig, logr, lognu, white, B), one(w_ax[:kq], Vq, lnsig_q, logr_q, lognu_q, white_q, B),
                one(wz[:kz], Vz, lnsig_z, logr_z, lognu_z, white_z, Bz))

    pz_of = Z.z_weights(RUN)
    out = {k: [] for k in ('offsets', 'F1', 'F2', 'G', 'W1', 'W2', 'WZ')}
    for f in Z.chain_files(RUN):
        with h5py.File(f, 'r') as h:
            for i in range(len(h['log_likelihood'])):
                F1, F2, G = (np.asarray(x) for x in fields(*(jnp.asarray(h[k][i], dtype=float) for k in KEYS[1:])))
                off = h['gauss_offsets'][i].astype(float)
                lp = h['log_p_mass'][i].astype(float) + A.mass_window()
                p = np.exp(lp - lp.max()); p /= p.sum()                                     # windowed p(m1, q)
                W1, W2, WZ = p.sum(1), p.sum(0), np.asarray(pz_of(float(h['lamb'][i])))
                for k, v in zip(out, (off, F1, F2, G, W1, W2, WZ)):
                    out[k].append(np.asarray(v, np.float32))
    out = {k: np.array(v) for k, v in out.items()}
    np.savez(CACHE, **out)
    return out


def load():
    if os.path.exists(CACHE) and all(os.path.getmtime(CACHE) > os.path.getmtime(f) for f in Z.chain_files(RUN)):
        return dict(np.load(CACHE))
    return build()


def figure(D, marg, curves, ze, stat):
    """stat 'sd' or 'mean': top row the conditional pieces (centred), bottom the marginals; columns m1, q, z."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import paper_figs as F
    j, ip = Z.STATS.index(stat), (2 if stat == 'sd' else 0)            # ln sd_eff or mu_eff
    zc = 0.5 * (ze[1:] + ze[:-1])
    zx = np.concatenate([[ze[0]], zc, [ze[-1]]])
    s_core = Z.SPINS.index('chi_eff_core')
    fig, axes = plt.subplots(2, 3, figsize=(F.TWO_COL, 3.5), sharex='col')
    cols = ((A.m_c, D['F1'], D['W1'], M_RANGE), (A.q_c, D['F2'], D['W2'], Q_RANGE), (zc, D['G'], D['WZ'], (ze[0], ze[-1])))
    top = []
    for c, (x, Fx, W, xr) in enumerate(cols):
        cen = Fx[:, ip] - (Fx[:, ip] * W).sum(1, keepdims=True)
        y = np.exp(cen) if stat == 'sd' else cen
        l, m, h = F.q5(y)
        ax = axes[0, c]
        ax.fill_between(x, l, h, color=COL, alpha=0.2, lw=0); ax.plot(x, m, color=COL, lw=1.1)
        ax.axhline(1. if stat == 'sd' else 0., color='0.4', lw=0.7, ls=':')
        vis = (x >= xr[0]) & (x <= xr[1])
        top += [l[vis], h[vis]]
    lo, hi = min(np.nanmin(t) for t in top), max(np.nanmax(t) for t in top)
    for c in range(3):
        ax = axes[0, c]
        if stat == 'sd':
            ax.set_yscale('log'); ax.set_ylim(max(lo / 1.2, 1 / 8), min(hi * 1.2, 16))
            ticks = [t for t in (1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16) if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
            ax.set_yticks(ticks, [rf'$\times{t:g}$' if t >= 1 else rf'$\times 1/{1 / t:g}$' for t in ticks])
            ax.yaxis.set_minor_formatter(plt.NullFormatter())
        else:
            pad = 0.05 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:+.2f}$' if v else '$0$'))
        if c:
            ax.tick_params(labelleft=False)
    fixed = (r'at fixed $q$, $z$', r'at fixed $m_1$, $z$', r'at fixed $m_1$, $q$')
    for c, t in enumerate(fixed):
        axes[0, c].text(0.03, 0.93, t, transform=axes[0, c].transAxes, va='top', ha='left', fontsize=7,
                        bbox=dict(facecolor='w', alpha=0.8, edgecolor='none', pad=1.2))
    axes[0, 0].set_ylabel('width, relative' if stat == 'sd' else 'mean, relative')
    # bottom: marginals (core)
    bottom = [(A.m_c, marg[:, s_core, j, 0]), (A.q_c, marg[:, s_core, j, 1]), (zx, None)]
    ys = []
    for c, (x, y) in enumerate(bottom):
        l, m, h = F.ext(F.q5(curves[:, 0, 0, s_core, j])) if c == 2 else F.q5(y)
        ax = axes[1, c]
        ax.fill_between(x, l, h, color=COL, alpha=0.2, lw=0); ax.plot(x, m, color=COL, lw=1.1)
        vis = (x >= cols[c][3][0]) & (x <= cols[c][3][1]) if c < 2 else slice(None)
        ys += [l[vis], h[vis]]
    lo, hi = min(np.nanmin(y) for y in ys), max(np.nanmax(y) for y in ys)
    for c in range(3):
        ax = axes[1, c]
        if stat == 'sd':
            F.log_sd_axis(ax, lo, hi)
        else:
            ax.axhline(0, color='0.5', lw=0.6, ls=':')
            ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
        if c:
            ax.tick_params(labelleft=False)
    sym = r'\sigma' if stat == 'sd' else r'\mu'
    axes[1, 0].set_ylabel(rf'${sym}_{{\chi_\mathrm{{eff}}}}$, $|\chi_\mathrm{{eff}}| < {A.CHI_CORE:g}$')
    for c, t in enumerate((r'given $m_1$', r'given $q$', r'given $z$')):
        axes[1, c].text(0.03, 0.93, t, transform=axes[1, c].transAxes, va='top', ha='left', fontsize=7,
                        bbox=dict(facecolor='w', alpha=0.8, edgecolor='none', pad=1.2))
    axes[0, 0].set_xscale('log'); axes[0, 0].set_xlim(*M_RANGE)
    axes[1, 0].set_xticks((5, 10, 20, 50, 100), ['$5$', '$10$', '$20$', '$50$', '$100$']); axes[1, 0].minorticks_off()
    axes[0, 1].set_xlim(*Q_RANGE); axes[1, 1].set_xticks((0.2, 0.4, 0.6, 0.8))
    axes[0, 2].set_xlim(ze[0], ze[-1]); axes[1, 2].set_xticks((0.5, 1., 1.5))
    for c, lab in enumerate((r'$m_1\,[\textrm{M}_\odot]$', '$q$', '$z$')):
        axes[1, c].set_xlabel(lab)
    handles = [(Patch(color=COL, alpha=0.2, lw=0), Line2D([], [], color=COL, lw=1.1))]
    labels = [r'conditional (top) and marginal (bottom), median and 90\%']
    fig.legend(handles, labels, loc='upper center', ncol=1, bbox_to_anchor=(0.5, 1.0), **F.LEG)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.17 / fig.get_figheight())); fig.subplots_adjust(hspace=0.06, wspace=0.05)
    return fig


def main():
    import matplotlib
    matplotlib.use('Agg')
    import pixelpop  # noqa: F401  (its import resets matplotlib's rcParams; import it before the style)
    import matplotlib.pyplot as plt
    import plot_style
    import paper_figs as F
    D = load()
    marg = Z.load_marg(RUN)
    curves, ze = Z.load_curves(RUN)
    plot_style.apply(); plt.rcParams.update(F.SMALL_RC)
    plot_style.save(figure(D, marg, curves, ze, 'sd'), 'chi_eff_sd_conditional_marginal_additive')
    plot_style.save(figure(D, marg, curves, ze, 'mean'), 'chi_eff_mean_conditional_marginal_additive')


if __name__ == '__main__':
    main()
