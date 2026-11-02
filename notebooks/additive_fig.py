"""Conditional against marginal spin evolution in the additive Gaussian models (user, 2026-11-02): each spin parameter is
offset + F1(ln m1) + F2(q) [+ G(z)], so its dependence on one variable is the same at every value of the others. The
top row shows those conditional pieces, centred on their population average (the normalization is not identified: only
the shape); the bottom row the population marginals p(chi_eff | m1), p(chi_eff | q), p(chi_eff | z). Both additive
models are overplotted (RUNS): with redshift dependence (marginals from analysis_z MARG, CURVES: averaged over p(z)) and
without (the per-cell conditional rebuilt here for every draw), so the bottom-row q panel shows how much of the
difference from the joint Gaussian's sigma(q) (paper_figs.chiq) is the average over redshift (user, 2026-11-02).

    python notebooks/additive_fig.py      (CPU, pinned on a login node; cached per run, ~45 min for the run without z)
        -> plots/chi_eff_sd_conditional_marginal_additive.pdf, plots/chi_eff_mean_conditional_marginal_additive.pdf
    python notebooks/additive_fig.py --z0  (the m1 and q marginals of the runs with z at z = 0, not over p(z))
        -> plots/chi_eff_{sd,mean}_conditional_marginal_additive_z0.pdf

Top row, sd: exp(F - <F>) of ln sd_eff, the factor on the Gaussian's width before truncation to the physical range
(the truncated width grows less where the Gaussian is broad); mean: F - <F> of mu_eff, the shift of the Gaussian's
centre. <F> weights by the population: p(m1), p(q) (the draw's windowed mass field) and p(z) (its redshift power law).
The bottom row is over the full chi_eff range (user, 2026-11-02; the core |chi_eff| < 0.8 gives nearly the same).
Runs whose chains are missing are left out.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import glob

import h5py
import numpy as np

import analysis as A
import analysis_z as Z

# run: (label, colour); the first is drawn on top
RUNS = {'ppc_gwtc5_k100_gauss_kz100_zb100_mqadd_zadd': (r'with redshift dependence, $F_{\ln m_1} + F_q + F_z$', 'C3'),
        'ppc_gwtc5_k100_gauss_mqadd': (r'no redshift dependence, $F_{\ln m_1} + F_q$', 'C0')}
MASS_KEYS = ('_gauss_sites', 'lnsigma_gauss', 'log_nu_gauss', 'log_ranges_gauss', '_gauss_q_sites', 'lnsigma_gauss_q',
             'log_nu_gauss_q', 'log_ranges_gauss_q')
Z_KEYS = ('_gauss_z_sites', 'lnsigma_gauss_z', 'log_nu_gauss_z', 'log_ranges_gauss_z')
M_RANGE, Q_RANGE = (5., 150.), (0.1, 1.)


def cache_path(run):
    return os.path.join(A.REPO, 'results', 'post_processing', f'{run}_additive_fields.npz')


def chain_files(run):
    return sorted(glob.glob(os.path.join(A.REPO, 'results', run, 'chain_*_samples.h5')))


def has_z(run):
    return '_zadd' in run


def marginal_fn(run):
    """Jitted (theta (5, Bm, Bq), p (Bm, Bq), the windowed p(m1, q) summing to 1) -> (2 stats, 2 axes (m1, q), B): the
    chi_eff mean and sd given m1 and given q, from the per-cell conditional (the Gaussian with parameters theta,
    truncated to the physical window, as model.py) mixed over p."""
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp as LSE
    from model import NEG, gauss_log_kernel, spin_window_fraction
    B = A.B
    window = spin_window_fraction(A.q_edges, A.e_edges, A.p_edges)
    log_window = jnp.asarray(np.where(window > 0, np.log(np.maximum(window, 1e-300)), NEG))     # (Bq, Be, Bp)
    log_support = jnp.where(log_window > NEG / 2, 0., NEG)
    ec, pc = jnp.asarray(A.e_c), jnp.asarray(A.p_c)
    sd_floor = A.gauss_sd_floor(run)

    @jax.jit
    def marginals(theta, p):
        def slab(m):
            T = gauss_log_kernel(theta[:, m][..., None, None], ec[:, None], pc[None, :], sd_floor) + log_support
            P = jnp.exp(T - LSE(T + log_window, axis=(1, 2))[:, None, None] + log_window)   # (Bq, Be, Bp)
            pe = P.sum(2)
            mu = pe @ ec
            return jnp.stack([mu, jnp.maximum(pe @ ec ** 2 - mu ** 2, 0.)])                 # (2, Bq): mean, var
        mu, var = jnp.moveaxis(jax.lax.map(slab, jnp.arange(B)), 1, 0)                      # (Bm, Bq) each
        out = []
        for ax in (1, 0):                                                                    # given m1, given q
            w = p / jnp.maximum(p.sum(ax, keepdims=True), 1e-300)
            mean = (w * mu).sum(ax)
            out.append(jnp.stack([mean, jnp.sqrt(jnp.maximum((w * (var + mu ** 2)).sum(ax) - mean ** 2, 0.))]))
        return jnp.stack(out, 1)

    return marginals


def build(run):
    """Per draw: the five parameters' offsets, F1 (Bm), F2 (Bq) [, G (Bz)] and the population weights of m1, q [, z];
    for a run without z also MARG (2 stats, 2 axes (m1, q), Bm): the chi_eff mean and sd given m1 and given q, from the
    per-cell conditional (truncated to the physical window, as model.py) mixed over the windowed p(m1, q)."""
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import pixelpop
    from model import axis_eigs, matern_spectrum, to_grid
    B, z = A.B, has_z(run)
    km = int(run.split('_k')[1].split('_')[0])
    (w_ax, v_ax), = axis_eigs([pixelpop.utils.create_CAR_coupling_matrix(B, 1, isVisible=False)])
    Vm = Vq = v_ax[:, :km]
    one = lambda w, v, ls, lr, ln, wh, n: jnp.stack([to_grid(matern_spectrum([w], ls[i], lr[i], jnp.exp(ln), n),
                                                             wh[i], [v]) for i in range(5)])
    keys = MASS_KEYS + (Z_KEYS if z else ())
    if z:
        _, _, _, kz, Bz, _ = Z.z_config(run)
        (wz, vz), = axis_eigs([pixelpop.utils.create_CAR_coupling_matrix(Bz, 1, isVisible=False)])
        Vz = vz[:, :kz]
        pz_of = Z.z_weights(run)

    @jax.jit
    def fields(white, lnsig, lognu, logr, white_q, lnsig_q, lognu_q, logr_q, *zargs):
        out = [one(w_ax[:km], Vm, lnsig, logr, lognu, white, B), one(w_ax[:km], Vq, lnsig_q, logr_q, lognu_q, white_q, B)]
        if zargs:
            white_z, lnsig_z, lognu_z, logr_z = zargs
            out.append(one(wz[:kz], Vz, lnsig_z, logr_z, lognu_z, white_z, Bz))
        return out

    marginals = marginal_fn(run)

    names = ('offsets', 'F1', 'F2', 'W1', 'W2') + (('G', 'WZ') if z else ('MARG',))
    out = {k: [] for k in names}
    files = chain_files(run)
    n = min(len(h5py.File(f, 'r')['log_likelihood']) for f in files)                    # load_light's draw order
    for f in files:
        with h5py.File(f, 'r') as h:
            for i in range(n):
                F = [np.asarray(x) for x in fields(*(jnp.asarray(h[k][i], dtype=float) for k in keys))]
                off = h['gauss_offsets'][i].astype(float)
                lp = h['log_p_mass'][i].astype(float) + A.mass_window()
                p = np.exp(lp - lp.max()); p /= p.sum()                                     # windowed p(m1, q)
                row = {'offsets': off, 'F1': F[0], 'F2': F[1], 'W1': p.sum(1), 'W2': p.sum(0)}
                if z:
                    row.update(G=F[2], WZ=np.asarray(pz_of(float(h['lamb'][i]))))
                else:
                    theta = off[:, None, None] + F[0][:, :, None] + F[1][:, None, :]
                    row['MARG'] = np.asarray(marginals(jnp.asarray(theta), jnp.asarray(p)))
                for k in names:
                    out[k].append(np.asarray(row[k], np.float32))
    out = {k: np.array(v) for k, v in out.items()}
    np.savez(cache_path(run), **out)
    return out


def z0_path(run):
    return os.path.join(A.REPO, 'results', 'post_processing', f'{run}_additive_marg_z0.npz')


def marg_z0(run, D, n=500):
    """For a run with z: MARG (as for a run without z) at z = 0, the conditional at G(z) of the first z bin, for n
    evenly spaced draws (user, 2026-11-02: Fig. 7 conditioned on z = 0); cached. Returns (draws, MARG)."""
    import jax.numpy as jnp
    c = z0_path(run)
    if os.path.exists(c) and all(os.path.getmtime(c) > os.path.getmtime(f) for f in chain_files(run)):
        d = np.load(c)
        return d['draws'], d['MARG']
    fn = marginal_fn(run)
    files = chain_files(run)
    nd = min(len(h5py.File(f, 'r')['log_likelihood']) for f in files)
    pick = np.unique(np.linspace(0, len(D['F1']) - 1, n).round().astype(int))
    out = []
    for k in pick:
        with h5py.File(files[k // nd], 'r') as h:
            lp = h['log_p_mass'][k % nd].astype(float) + A.mass_window()
        p = np.exp(lp - lp.max()); p /= p.sum()
        theta = (D['offsets'][k][:, None, None] + D['F1'][k][:, :, None] + D['F2'][k][:, None, :]
                 + D['G'][k][:, 0][:, None, None]).astype(float)
        out.append(np.asarray(fn(jnp.asarray(theta), jnp.asarray(p)), np.float32))
    np.savez(c, draws=pick, MARG=np.array(out))
    return pick, np.array(out)


def load(run):
    """build(run)'s arrays (cached), plus for a run with z its marginals from analysis_z: MARG (chi_eff, as built for
    a run without z), ZCURVES (draws, 2 stats, Bz) and z_edges."""
    c = cache_path(run)
    fresh = os.path.exists(c) and all(os.path.getmtime(c) > os.path.getmtime(f) for f in chain_files(run))
    D = dict(np.load(c)) if fresh and ('MARG' in np.load(c) or has_z(run)) else build(run)
    if has_z(run):
        D['MARG'] = Z.load_marg(run)[:, Z.SPINS.index('chi_eff')]
        curves, D['z_edges'] = Z.load_curves(run)
        D['ZCURVES'] = curves[:, 0, 0, Z.SPINS.index('chi_eff')]
    return D


def figure(R, stat, z0=False):
    """R: {run: load(run)}. stat 'sd' or 'mean': top row the conditional pieces (centred), bottom the marginals;
    columns m1, q, z (z only for runs with redshift dependence). ``z0``: the m1 and q marginals of runs with z at z = 0
    (D['MARG_Z0'])."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import paper_figs as F
    j, ip = Z.STATS.index(stat), (2 if stat == 'sd' else 0)            # ln sd_eff or mu_eff
    fig, axes = plt.subplots(2, 3, figsize=(F.TWO_COL, 3.5), sharex='col')
    ze = next(D['z_edges'] for D in R.values() if 'z_edges' in D)
    zc = 0.5 * (ze[1:] + ze[:-1])
    zx = np.concatenate([[ze[0]], zc, [ze[-1]]])
    ranges = (M_RANGE, Q_RANGE, (ze[0], ze[-1]))
    top, ys = [], []
    for run, D in reversed(list(R.items())):                          # the first run drawn last, on top
        col = RUNS[run][1]
        cols = [(A.m_c, D['F1'], D['W1']), (A.q_c, D['F2'], D['W2'])] + ([(zc, D['G'], D['WZ'])] if 'G' in D else [])
        for c, (x, Fx, W) in enumerate(cols):
            cen = Fx[:, ip] - (Fx[:, ip] * W).sum(1, keepdims=True)
            l, m, h = F.q5(np.exp(cen) if stat == 'sd' else cen)
            axes[0, c].fill_between(x, l, h, color=col, alpha=0.2, lw=0); axes[0, c].plot(x, m, color=col, lw=1.1)
            vis = (x >= ranges[c][0]) & (x <= ranges[c][1])
            top += [l[vis], h[vis]]
        M = D['MARG_Z0'] if z0 and 'MARG_Z0' in D else D['MARG']
        bottom = [(A.m_c, F.q5(M[:, j, 0])), (A.q_c, F.q5(M[:, j, 1]))]
        if 'ZCURVES' in D:
            bottom.append((zx, F.ext(F.q5(D['ZCURVES'][:, j]))))
        for c, (x, (l, m, h)) in enumerate(bottom):
            axes[1, c].fill_between(x, l, h, color=col, alpha=0.2, lw=0); axes[1, c].plot(x, m, color=col, lw=1.1)
            vis = (x >= ranges[c][0]) & (x <= ranges[c][1])
            ys += [l[vis], h[vis]]
    lo, hi = min(np.nanmin(t) for t in top), max(np.nanmax(t) for t in top)
    for c in range(3):
        ax = axes[0, c]
        ax.axhline(1. if stat == 'sd' else 0., color='0.4', lw=0.7, ls=':')
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
    for c, t in enumerate((r'at fixed $q$, $z$', r'at fixed $m_1$, $z$', r'at fixed $m_1$, $q$')):
        axes[0, c].text(0.03, 0.93, t, transform=axes[0, c].transAxes, va='top', ha='left', fontsize=7,
                        bbox=dict(facecolor='w', alpha=0.8, edgecolor='none', pad=1.2))
    axes[0, 0].set_ylabel('width, relative' if stat == 'sd' else 'mean, relative')
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
    axes[1, 0].set_ylabel(rf'${sym}_{{\chi_\mathrm{{eff}}}}$')
    for c, t in enumerate((r'given $m_1$, at $z = 0$' if z0 else r'given $m_1$', r'given $q$, at $z = 0$' if z0
                           else r'given $q$', r'given $z$')):
        axes[1, c].text(0.03, 0.93, t, transform=axes[1, c].transAxes, va='top', ha='left', fontsize=7,
                        bbox=dict(facecolor='w', alpha=0.8, edgecolor='none', pad=1.2))
    axes[0, 0].set_xscale('log'); axes[0, 0].set_xlim(*M_RANGE)
    axes[1, 0].set_xticks((5, 10, 20, 50, 100), ['$5$', '$10$', '$20$', '$50$', '$100$']); axes[1, 0].minorticks_off()
    axes[0, 1].set_xlim(*Q_RANGE); axes[1, 1].set_xticks((0.2, 0.4, 0.6, 0.8))
    axes[0, 2].set_xlim(ze[0], ze[-1]); axes[1, 2].set_xticks((0.5, 1., 1.5))
    for c, lab in enumerate((r'$m_1\,[\textrm{M}_\odot]$', '$q$', '$z$')):
        axes[1, c].set_xlabel(lab)
    handles = [(Patch(color=RUNS[r][1], alpha=0.2, lw=0), Line2D([], [], color=RUNS[r][1], lw=1.1)) for r in R]
    fig.legend(handles, [RUNS[r][0] for r in R], loc='upper center', ncol=len(R), bbox_to_anchor=(0.5, 1.0), **F.LEG)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.17 / fig.get_figheight())); fig.subplots_adjust(hspace=0.06, wspace=0.05)
    return fig


def main():
    import matplotlib
    matplotlib.use('Agg')
    import pixelpop  # noqa: F401  (its import resets matplotlib's rcParams; import it before the style)
    import matplotlib.pyplot as plt
    import plot_style
    import paper_figs as F
    z0 = '--z0' in sys.argv[1:]
    R = {run: load(run) for run in RUNS if chain_files(run)}
    if z0:
        for run, D in R.items():
            if 'G' in D:
                _, D['MARG_Z0'] = marg_z0(run, D)
    print('runs:', list(R), flush=True)
    plot_style.apply(); plt.rcParams.update(F.SMALL_RC)
    tag = '_z0' if z0 else ''
    plot_style.save(figure(R, 'sd', z0), f'chi_eff_sd_conditional_marginal_additive{tag}')
    plot_style.save(figure(R, 'mean', z0), f'chi_eff_mean_conditional_marginal_additive{tag}')


if __name__ == '__main__':
    main()
