"""Empirical-Bayes prior draws of the spin conditional, for comparison with the posterior's spin moments
(user, 2026-10-27: are the chi_p constraints more than the prior?).

    python macros/spin_prior.py [--n N] [--seed S] [<run> ...]   (default: ppc_gwtc5_k10 ppc_gwtc5_k100_gauss, N = 200)
        -> results/post_processing/<run>_spin_prior.npz

For N posterior draws (evenly spaced over the chains, in analysis.load_light's draw order, saved as ``draws``): the
spin model's hyperparameters stay at the draw's values and its white sites are drawn afresh from N(0, 1), so the spin
field is a prior draw given the posterior's kernel (empirical Bayes). The field ('field' model: lnsigma_spin,
log_nu_spin, log_ranges_spin) or the five Gaussian parameter fields ('gauss': gauss_offsets, lnsigma_gauss,
log_nu_gauss, log_ranges_gauss, as model.gauss_fields) are rebuilt with analysis._spin_fns, with the draw's own
(posterior) mass field, so p(q | m1) is the posterior's and only the spins come from the prior. Saved per draw: MOM
(per-cell chi_eff mean, sd, chi_p mean, sd) and MOM_CORE (chi_eff restricted to |chi_eff| < analysis.CHI_CORE).
CPU, about 1-3 s per draw (pin it on a login node).
"""
import glob
import os
import sys
import time

import h5py
import numpy as np

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'notebooks')); sys.path.insert(0, HERE)
import analysis as A

RUNS = ('ppc_gwtc5_k10', 'ppc_gwtc5_k100_gauss')
HYPER = {'field': ('lnsigma_spin', 'log_nu_spin', 'log_ranges_spin'),
         'gauss': ('gauss_offsets', 'lnsigma_gauss', 'log_nu_gauss', 'log_ranges_gauss')}


def prior_path(run):
    return os.path.join(REPO, 'results', 'post_processing', f'{run}_spin_prior.npz')


def load_draws(run, n):
    """The hyperparameters and log_p_mass of n posterior draws, evenly spaced over load_light's order."""
    files = sorted(glob.glob(os.path.join(REPO, 'results', run, 'chain_*_samples.h5')))
    n_draws = min(len(h5py.File(f, 'r')['log_likelihood']) for f in files)
    with h5py.File(files[0], 'r') as h:
        spin_model = A.spin_model_of(h)
    idx = np.linspace(0, len(files) * n_draws - 1, n).astype(int)
    out = {k: [] for k in HYPER[spin_model] + ('log_p_mass',)}
    for i in idx:
        with h5py.File(files[i // n_draws], 'r') as h:
            for k in out:
                out[k].append(h[k][i % n_draws])
    return spin_model, idx, {k: np.asarray(v, float) for k, v in out.items()}


def main(runs, n=200, seed=0):
    import jax.numpy as jnp
    import pixelpop
    from model import axis_eigs, matern_spectrum, to_grid
    rng = np.random.default_rng(seed)
    for run in runs:
        t0 = time.time()
        spin_model, idx, D = load_draws(run, n)
        km = int(run.split('_k')[1].split('_')[0])
        fn, _ = A._spin_fns(km, A.mass_window(), spin_model)
        if spin_model == 'gauss':
            (w_ax, v_ax), = axis_eigs([pixelpop.utils.create_CAR_coupling_matrix(A.B, 1, isVisible=False)])
            Vk = v_ax[:, :km]
        MOM, CORE = [], []
        for j in range(n):
            if spin_model == 'field':
                white = rng.standard_normal((km, km, A.B, A.B))
                out = fn(white, D['lnsigma_spin'][j], D['log_nu_spin'][j], D['log_ranges_spin'][j], D['log_p_mass'][j])
            else:                                                            # model.gauss_fields with fresh sites
                white = rng.standard_normal((5, km, km))
                nu = np.exp(D['log_nu_gauss'][j])
                fields = [to_grid(matern_spectrum([w_ax[:km], w_ax[:km]], D['lnsigma_gauss'][j][i],
                                                  D['log_ranges_gauss'][j][i], nu, A.B * A.B), white[i], [Vk, Vk])
                          for i in range(5)]
                theta = jnp.asarray(D['gauss_offsets'][j])[:, None, None] + jnp.stack(fields)
                out = fn(theta, D['log_p_mass'][j])
            MOM.append(np.asarray(out[2], np.float32)); CORE.append(np.asarray(out[6], np.float32))
            if (j + 1) % 25 == 0:
                print(f'  {run}: {j + 1}/{n} prior draws, {(time.time() - t0) / (j + 1):.1f} s each', flush=True)
        np.savez(prior_path(run), draws=idx, MOM=np.stack(MOM), MOM_CORE=np.stack(CORE))
        print('saved', prior_path(run), flush=True)


if __name__ == '__main__':
    args = sys.argv[1:]
    kw = {}
    for flag in ('--n', '--seed'):
        if flag in args:
            i = args.index(flag); kw[flag[2:]] = int(args[i + 1]); del args[i:i + 2]
    main(args or list(RUNS), **kw)
