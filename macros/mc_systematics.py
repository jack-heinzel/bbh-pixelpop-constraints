"""Monte Carlo systematics of the posterior: sigma_syst / sigma_stat for the population quantities the paper reports
(user, 2026-10-28), per run.

    python macros/mc_systematics.py [--chunk N] [--check N] <run> ...   (GPU; see macros/mc_systematics.sbatch)
        -> results/post_processing/<run>_mc_syst.npz

The Monte Carlo estimate of ln L has an error eps(Lambda) with covariance C(Lambda_n, Lambda_m) between posterior draws
(population_error.log_likelihood_covariance; arXiv:2509.07221). To first order it shifts the posterior mean of a
quantity f by Cov_post(f, eps), whose spread over Monte Carlo realizations is

    sigma_syst^2 = (1 / N^2) sum_{n,m} df_n df_m C(Lambda_n, Lambda_m),   df_n = f(Lambda_n) - mean_n f,

against sigma_stat^2 = Var_post(f). (The runs use the likelihood correction, so the bias part is second order.) With
w~ the importance weights over their mean (per event, and over all generated injections), C is a sum over samples
of w~_n w~_m times c_i = 1 / (M_e (M_e - 1)) for the samples of event e (M_e its real samples) and
N_obs^2 / (T (T - 1)) for the injections (T generated), minus constants that drop out since sum_n df_n = 0. So

    sigma_syst^2 = sum_i c_i g_i^2,   g_i = (1 / N) sum_n df_n w~_n,i,

one pass over the draws per block of samples instead of N^2 covariances. --check N compares this with the explicit
double sum over N draws using population_error's covariance. The per-draw weights come from the model itself
(model.setup_model(record_weights=True)) with each draw's latent sites substituted, in analysis.load_light's draw
order, so they line up with the quantities. Also saved: the precision statistic (bits) of population_error.

Quantities (each per draw; analysis.load_light and its spin cache): ln p(ln m1, q) (windowed); ln p(m1 | q range) for
all q and analysis.SUMMARY_Q_RANGES; per-cell (m1, q) mean and sd of chi_eff (also |chi_eff| < CHI_CORE) and chi_p;
the q-averaged chi_eff / chi_p mean and sd against m1 (all q and each q range; chi_eff also core), as the paper's
figures; and the m1-averaged ones against q (paper_figs.m1_averaged_moments).
"""
import argparse
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'notebooks')); sys.path.insert(0, HERE)


def out_path(run):
    return os.path.join(REPO, 'results', 'post_processing', f'{run}_mc_syst.npz')


def run_args(run):
    """run.py's arguments for a run name (its run_name, inverted)."""
    import run as R
    m = re.fullmatch(r'ppc_(\w+?)_k(\d+)(_split|_gauss)?(?:_sdfloor([\d.e-]+))?(?:_neff([\d.e-]+))?'
                     r'(?:_varcut([\d.e-]+))?(?:_npe(\w+))?(_short)?', run)
    if m is None:
        raise ValueError(f'cannot parse run name {run}')
    cat, k, model, floor, neff, varcut, npe, short = m.groups()
    a = [cat, '--spin-mass-modes', k]
    a += ['--spin-model', model[1:]] if model else []
    a += ['--gauss-sd-floor', floor] if floor else []
    a += ['--event-neff-cut', neff] if neff else []
    a += ['--variance-cut', varcut] if varcut else []
    a += ['--max-samples', '0' if npe == 'all' else npe] if npe else []
    a += ['--short'] if short else []
    args = R.make_parser().parse_args(a)
    assert R.run_name(args) == run, (R.run_name(args), run)
    return args


def quantities(run):
    """{name: (draws, ...) float64} of the reported quantities, in load_light's draw order."""
    import analysis as A
    from paper_figs import add_chi_p_moments, m1_averaged_moments
    r = A.load_light(run)
    add_chi_p_moments(r)
    Q = {}
    with np.errstate(divide='ignore'):
        Q['ln_p_m1_q'] = np.log(r.p_mass.astype(float))
        Q['ln_p_m1'] = np.log(A.p_m1_given_q_range(r, (0., 1.)))
        for lo, hi in A.SUMMARY_Q_RANGES:
            Q[f'ln_p_m1_q{lo:g}-{hi:g}'] = np.log(A.p_m1_given_q_range(r, (lo, hi)))
    for name, x in (('mu_eff', r.MOM[..., 0]), ('sd_eff', r.MOM[..., 1]), ('mu_p', r.MOM[..., 2]),
                    ('sd_p', r.MOM[..., 3]), ('mu_eff_core', r.MOM_CORE[..., 0]), ('sd_eff_core', r.MOM_CORE[..., 1])):
        Q[f'{name}_m1_q'] = x.astype(float)
    for qr in (None,) + tuple(A.SUMMARY_Q_RANGES):
        tag = '' if qr is None else f'_q{qr[0]:g}-{qr[1]:g}'
        M = A.q_averaged_moments(r, qr)
        Mc = A.q_averaged_moments(r, qr, core=True)
        for spin, short in (('chi_eff', 'eff'), ('chi_p', 'p')):
            for st in ('mean', 'sd'):
                Q[f'{"mu" if st == "mean" else "sd"}_{short}_m1{tag}'] = M[spin][st].astype(float)
        for st in ('mean', 'sd'):
            Q[f'{"mu" if st == "mean" else "sd"}_eff_core_m1{tag}'] = Mc['chi_eff'][st].astype(float)
    for spin, short, core in (('chi_eff', 'eff', False), ('chi_eff', 'eff_core', True), ('chi_p', 'p', False)):
        M = m1_averaged_moments(r, core, spin)
        Q[f'mu_{short}_q'], Q[f'sd_{short}_q'] = M['mean'].astype(float), M['sd'].astype(float)
    return r, Q


def latent_draws(run, keys, n_draws):
    """Yield each posterior draw's latent sites (load_light's order: sorted chain files, first n_draws of each)."""
    import glob
    import h5py
    for f in sorted(glob.glob(os.path.join(REPO, 'results', run, 'chain_*_samples.h5'))):
        with h5py.File(f, 'r') as h:
            missing = [k for k in keys if k not in h]
            if missing:
                raise KeyError(f'{f}: latent sites not saved: {missing}')
            for i in range(n_draws):
                yield {k: h[k][i] for k in keys}


def main(runs, chunk, check, block=4096):
    import itertools
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp as LSE
    from numpyro import handlers
    import run as R

    for run in runs:
        t0 = time.time()
        args = run_args(run)
        prob_model, init, posteriors, injections, pp = R.build(args, run, record_weights=True)
        keys = sorted(init)
        r, Q = quantities(run)
        N = min(len(r.p_mass), check) if check else len(r.p_mass)
        names = list(Q)
        shapes = {k: Q[k].shape[1:] for k in names}
        F = np.concatenate([Q[k][:N].reshape(N, -1) for k in names], 1)                     # (N, n_q)
        del Q
        good = np.all(np.isfinite(F), 0)
        F = np.where(good, F, 0.)
        dF = F - F.mean(0)
        sig_stat = np.where(good, dF.std(0), np.nan)
        print(f'{run}: {N} draws, {F.shape[1]} quantities ({good.sum()} finite in every draw), '
              f'{time.time() - t0:.0f} s to load', flush=True)
        lats = list(itertools.islice(latent_draws(run, keys, r.n_draws), N))     # in memory: read once, not per pass
        draws = lambda: iter(lats)

        kw = jax.device_put({'posteriors': posteriors, 'injections': injections})

        @jax.jit
        def log_weights(lat, kw):
            tr = handlers.trace(handlers.substitute(handlers.seed(prob_model, 0), data=lat)).get_trace(**kw)
            return tr['event_log_weights']['value'], tr['injection_log_weights']['value']

        counts = jnp.asarray(pp.event_counts, jnp.float32)
        n_obs, T = len(pp.event_counts), float(injections['total_generated'])

        @jax.jit
        def normalized(ew, iw):
            """w~ over all samples (events flattened, then injections)."""
            ew = jnp.exp(ew - (LSE(ew, axis=1) - jnp.log(counts))[:, None])
            iw = jnp.exp(iw - (LSE(iw) - jnp.log(T)))
            return jnp.concatenate([ew.ravel(), iw])

        ew0, iw0 = log_weights(next(draws()), kw)
        npe, n_inj = ew0.shape[1], iw0.shape[0]
        c = jnp.concatenate([jnp.repeat(1. / (counts * (counts - 1.)), npe),
                             jnp.full(n_inj, n_obs ** 2 / (T * (T - 1.)), jnp.float32)])  # c_i
        n_tot = len(c)
        dFj = jnp.asarray(dF.T / N, jnp.float32)                                           # (n_q, N)
        S = jnp.zeros(F.shape[1], jnp.float32)
        m1, m2 = np.zeros(n_tot), np.zeros(n_tot)                       # sum_n w~ and sum_n w~^2 per sample
        for start in range(0, n_tot, chunk):
            stop = min(start + chunk, n_tot)
            W = jnp.stack([normalized(*log_weights(lat, kw))[start:stop] for lat in draws()])   # (N, chunk)
            m1[start:stop], m2[start:stop] = np.asarray(W.sum(0)), np.asarray((W ** 2).sum(0))
            for b in range(0, stop - start, block):
                G = dFj @ W[:, b:b + block]                                               # (n_q, block)
                S = S + (G ** 2) @ c[start + b:start + b + G.shape[1]]
            del W
            print(f'  samples {stop}/{n_tot}, {time.time() - t0:.0f} s', flush=True)
        sig_syst = np.sqrt(np.maximum(np.asarray(S, float), 0.))
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(good & (sig_stat > 0), sig_syst / sig_stat, np.nan)
        # population_error's precision statistic, (E_n C_nn - E_nm C_nm) / (2 ln 2)
        precision = float(np.sum(np.asarray(c, float) * (m2 / N - (m1 / N) ** 2)) / 2 / np.log(2))
        print(f'  precision statistic {precision:.3f} bits; sigma_syst / sigma_stat median {np.nanmedian(ratio):.3f}, '
              f'99% {np.nanpercentile(ratio, 99):.3f}, max {np.nanmax(ratio):.3f}', flush=True)
        if check:
            import population_error.statistics as PS
            ws = [tuple(jnp.exp(x) for x in log_weights(lat, kw)) for lat in draws()]
            C = np.array([[float(PS.log_likelihood_covariance(ws[n][1], ws[m][1], ws[n][0], ws[m][0], T,
                                                              event_counts=counts)) for m in range(N)]
                          for n in range(N)])
            idx = np.flatnonzero(good)[np.linspace(0, good.sum() - 1, 20).astype(int)]
            explicit = np.sqrt(np.maximum(np.einsum('nq,nm,mq->q', dF[:, idx], C, dF[:, idx]) / N ** 2, 0.))
            print('  explicit (population_error) / factorized sigma_syst:', np.round(explicit / sig_syst[idx], 4))
            prec = (np.mean(np.diag(C)) - np.mean(C)) / 2 / np.log(2)
            print(f'  precision statistic, explicit: {prec:.4f} bits (factorized {precision:.4f})', flush=True)
            continue
        out, off = {}, 0
        for k in names:
            n = int(np.prod(shapes[k]))
            for key, x in (('ratio', ratio), ('syst', sig_syst), ('stat', sig_stat)):
                out[f'{key}/{k}'] = x[off:off + n].reshape(shapes[k])
            off += n
        np.savez(out_path(run), precision_bits=precision, n_draws=N, **out)
        print('saved', out_path(run), f'{time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--chunk', type=int, default=500_000,
                    help='samples per pass over the draws: the pass holds (draws x chunk) weights twice on the GPU, 8 GB '
                         'at 2000 draws (2.5e6 ran out of memory on an A100-80GB)')
    ap.add_argument('--check', type=int, default=0, help='only N draws, compared with the explicit double sum')
    a = ap.parse_args()
    main(a.runs, a.chunk, a.check)
