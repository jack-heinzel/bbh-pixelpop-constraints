"""The chi_eff distribution of the detectable population against redshift (user, 2026-11-02): with selection, a
population whose spins do not evolve with z can still give detected binaries whose chi_eff width grows with z, since
heavier binaries are seen farther and their width is larger (they are mostly unequal mass). A spin model that depends
on z alone, fit with the selection correction, converges to exactly this observed distribution when detectability does
not depend on the spins (the selection term is then independent of the spin parameters, so the spin fit follows the
detected events). So sigma_obs(z) is what such an analysis would recover from our population.

    python notebooks/sigma_obs.py [--n N] [<run> ...]      (GPU; sigma_obs.sbatch)  -> results/post_processing/
                                                            <run>_sigma_obs.npz
    python notebooks/sigma_obs.py --figure                 (CPU)  -> plots/chi_eff_sd_z_observed.pdf

Per posterior draw (N evenly spaced over load_light's draw order; every draw by default), the model's own injection
weights (model.setup_model(record_weights=True), the draw's latent sites substituted; w = p_pop / p_draw for every found
injection, as in the selection term) give the detectable population. Per redshift bin of Z_EDGES: the weighted mean and
sd of the injections' chi_eff (all, and |chi_eff| < analysis.CHI_CORE) and chi_p, the mean ln m1 and the fraction with
q < Q_LO, and the weights' effective sample size. The population (not detection-weighted) curves come from analysis_z.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, REPO)
import numpy as np

# run name -> run.py arguments (checked against run.run_name)
RUN_ARGS = {
    'ppc_gwtc5_k100_gauss_kz100_zb100_mqadd_zadd': ['--spin-mass-modes', '100', '--spin-model', 'gauss',
                                                    '--spin-z-modes', '100', '--z-bins', '100', '--z-additive',
                                                    '--mq-additive'],
    'ppc_gwtc5_k100_gauss_kz100_zb100_zadd': ['--spin-mass-modes', '100', '--spin-model', 'gauss', '--spin-z-modes',
                                              '100', '--z-bins', '100', '--z-additive'],
    'ppc_gwtc5_k100_gauss_kz10': ['--spin-mass-modes', '100', '--spin-model', 'gauss', '--spin-z-modes', '10'],
}
FIG_RUN = 'ppc_gwtc5_k100_gauss_kz100_zb100_mqadd_zadd'
Z_EDGES = np.linspace(0., 1.9, 20)
Q_LO = 0.65


def out_path(run):
    return os.path.join(REPO, 'results', 'post_processing', f'{run}_sigma_obs.npz')


def compute(run, n):
    import glob
    import time
    import h5py
    import jax
    import jax.numpy as jnp
    from numpyro import handlers
    import analysis as A
    import run as R
    args = R.make_parser().parse_args(['gwtc5'] + RUN_ARGS[run])
    assert R.run_name(args) == run, (R.run_name(args), run)
    prob_model, init, posteriors, injections, pp = R.build(args, run, record_weights=True)
    keys = sorted(init)
    inj = {k: np.asarray(injections[k]) for k in ('redshift', 'chi_eff', 'chi_p', 'mass_ratio', 'log_mass_1')}
    zb = np.clip(np.digitize(inj['redshift'], Z_EDGES) - 1, 0, len(Z_EDGES) - 2)
    nz = len(Z_EDGES) - 1
    onehot = jnp.asarray(np.eye(nz)[zb].T)                                               # (nz, n_inj)
    e, p = jnp.asarray(inj['chi_eff']), jnp.asarray(inj['chi_p'])
    core = jnp.asarray(np.abs(inj['chi_eff']) < A.CHI_CORE, dtype=float)
    lnm, qlo = jnp.asarray(inj['log_mass_1']), jnp.asarray(inj['mass_ratio'] < Q_LO, dtype=float)
    kw = jax.device_put({'posteriors': posteriors, 'injections': injections})

    @jax.jit
    def stats(lat, kw):
        tr = handlers.trace(handlers.substitute(handlers.seed(prob_model, 0), data=lat)).get_trace(**kw)
        lw = tr['injection_log_weights']['value']
        w = jnp.exp(lw - lw.max())
        W = onehot * w                                                                   # (nz, n_inj)

        def mom(W, x):
            s = jnp.maximum(W.sum(1), 1e-300)
            m = (W @ x) / s
            return jnp.stack([m, jnp.sqrt(jnp.maximum((W @ x ** 2) / s - m ** 2, 0.))])
        spins = jnp.stack([mom(W, e), mom(W * core, e), mom(W, p)])                     # (3, 2, nz)
        s = jnp.maximum(W.sum(1), 1e-300)
        mix = jnp.stack([(W @ lnm) / s, (W @ qlo) / s])                                  # (2, nz)
        neff = W.sum(1) ** 2 / jnp.maximum((W ** 2).sum(1), 1e-300)
        return spins, mix, neff

    files = sorted(glob.glob(os.path.join(REPO, 'results', run, 'chain_*_samples.h5')))
    n_draws = min(len(h5py.File(f, 'r')['log_likelihood']) for f in files)
    order = [(f, i) for f in files for i in range(n_draws)]                              # load_light's draw order
    pick = np.unique(np.linspace(0, len(order) - 1, n or len(order)).round().astype(int))
    out = {'spins': [], 'mix': [], 'neff': []}
    t0 = time.time()
    handles = {f: h5py.File(f, 'r') for f in files}
    for j, k in enumerate(pick):
        f, i = order[k]
        lat = {key: jnp.asarray(handles[f][key][i]) for key in keys}
        for name, v in zip(out, stats(lat, kw)):
            out[name].append(np.asarray(v))
        if (j + 1) % 50 == 0:
            print(f'{run}: {j + 1}/{len(pick)} draws, {(time.time() - t0) / (j + 1):.2f} s each', flush=True)
    np.savez(out_path(run), draws=pick, z_edges=Z_EDGES, q_lo=Q_LO, **{k: np.array(v) for k, v in out.items()})
    print('wrote', out_path(run), flush=True)


def figure(run=FIG_RUN):
    """sigma_chi_eff(z): the population (analysis_z CURVES) against the detectable population (this module), and
    below, the detectable binaries' mean m1 and fraction with q < Q_LO."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import analysis_z as Z
    import paper_figs as F
    d = np.load(out_path(run))
    ze = d['z_edges']; zc = 0.5 * (ze[1:] + ze[:-1])
    curves, zp = Z.load_curves(run)
    zpx = np.concatenate([[zp[0]], 0.5 * (zp[1:] + zp[:-1]), [zp[-1]]])
    s = Z.SPINS.index('chi_eff')
    fig, axes = plt.subplots(2, 1, figsize=(F.ONE_COL, 3.3), sharex=True, gridspec_kw=dict(height_ratios=(2, 1.2)))
    ax = axes[0]
    l, m, h = F.ext(F.q5(curves[:, 0, 0, s, 1]))
    ax.fill_between(zpx, l, h, color='C3', alpha=0.2, lw=0); ax.plot(zpx, m, color='C3', lw=1.1)
    l, m, h = F.q5(d['spins'][:, 0, 1])
    ax.fill_between(zc, l, h, color='k', alpha=0.15, lw=0); ax.plot(zc, m, color='k', lw=1.1)
    ax.plot(zc, l, color='k', lw=0.5); ax.plot(zc, h, color='k', lw=0.5)
    ax.set_ylabel(r'$\sigma_{\chi_\mathrm{eff}}(z)$')
    ax.set_ylim(0, None)
    handles = [(Patch(color='C3', alpha=0.2, lw=0), Line2D([], [], color='C3', lw=1.1)),
               (Patch(color='k', alpha=0.15, lw=0), Line2D([], [], color='k', lw=1.1))]
    ax.legend(handles, ['population', 'detectable binaries'], loc='upper left', **F.LEG)
    ax = axes[1]
    l, m, h = F.q5(np.exp(d['mix'][:, 0]))
    ax.fill_between(zc, l, h, color='C0', alpha=0.2, lw=0); ax.plot(zc, m, color='C0', lw=1.1)
    ax.set_ylabel(r'$\langle m_1 \rangle_\mathrm{det}\,[\textrm{M}_\odot]$', color='C0')
    ax2 = ax.twinx()
    l, m, h = F.q5(d['mix'][:, 1])
    ax2.fill_between(zc, l, h, color='C1', alpha=0.2, lw=0); ax2.plot(zc, m, color='C1', lw=1.1)
    ax2.set_ylabel(rf'det. fraction $q < {Q_LO:g}$', color='C1')
    ax2.set_ylim(0, None)
    ax.set_xlim(ze[0], ze[-1]); ax.set_xlabel('$z$')
    fig.tight_layout(); fig.subplots_adjust(hspace=0.06)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='*', default=list(RUN_ARGS))
    ap.add_argument('--n', type=int, default=0, help='posterior draws, evenly spaced (0: all)')
    ap.add_argument('--figure', action='store_true')
    args = ap.parse_args()
    if args.figure:
        import matplotlib
        matplotlib.use('Agg')
        import pixelpop  # noqa: F401  (its import resets matplotlib's rcParams; import it before the style)
        import matplotlib.pyplot as plt
        import plot_style
        import paper_figs as F
        plot_style.apply(); plt.rcParams.update(F.SMALL_RC)
        plot_style.save(figure(), 'chi_eff_sd_z_observed')
        return
    for run in args.runs:
        compute(run, args.n)


if __name__ == '__main__':
    main()
