"""Sample the weakly modelled population (model.py); chains land in results/<run name>/.

    python run.py gwtc5 --spin-mass-modes 10 --device 0 --mem-fraction .4 [--short]

The chain starts from a random draw from the prior of every latent site (keyed by --seed and --chain; --check-only
evaluates the same draw), unless --init-from gives one (e.g. warmstart.py's output). Chains of one run (--chain 0, 1,
...; separate jobs) share its directory as chain_<N>_samples.h5, each from its own prior draw and sampler key.

Run name: ppc_<catalog>_k<spin mass modes>[_<spin model>][_neff<floor>][_npe<max samples>][_no231123][_short][_<tag>];
the spin model ('field', the default, 'split' or 'gauss'; model.py) is left out of the name for 'field'.

This node caps processes + threads at 500 per user, and XLA sizes its thread pools from the CPU affinity, so the job
pins itself (--cpus cores, the least busy ones by default) and serializes XLA compilation before jax is imported.
"""
import argparse
import os
import time

REPO = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(REPO, 'results')
SHORT = dict(warmup=2000, tot_samples=500, thinning=5)
MAX_SAMPLES = 20000


def make_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('catalog', choices=['gwtc5', 'nrsur', 'physical', 'mixed'])
    ap.add_argument('--spin-mass-modes', type=int, default=10,
                    help='mass modes per axis (ln m1, q) of the spin field; 0: spins independent of mass')
    ap.add_argument('--spin-model', choices=['field', 'split', 'gauss'], default='field',
                    help="spin conditional: one 4D Matern field ('field'), mass-independent + mass-varying fields with "
                         "separate amplitudes ('split'), or a truncated Gaussian with field-valued parameters ('gauss')")
    ap.add_argument('--gauss-sd-floor', type=float, default=0.02,
                    help="spin_model gauss: soft floor on its sds (model.GAUSS_SD_FLOOR); not 0.02 adds '_sdfloor<x>'")
    ap.add_argument('--spin-bins', type=int, nargs=2, default=[100, 100], metavar=('B_CHIEFF', 'B_CHIP'))
    ap.add_argument('--bins', type=int, default=100, help='mass grid bins per axis')
    ap.add_argument('--event-neff-cut', type=float, default=100.)
    ap.add_argument('--max-samples', type=int, default=MAX_SAMPLES,
                    help='usable PE samples kept per event (those in grid bins injections reach); 0: keep all')
    ap.add_argument('--no-likelihood-correction', action='store_true')
    ap.add_argument('--exclude-events', nargs='*', default=[])
    ap.add_argument('--short', action='store_true', help=f'a first look, not converged: {SHORT}')
    ap.add_argument('--warmup', type=int, default=4000)
    ap.add_argument('--samples', type=int, default=1000)
    ap.add_argument('--thinning', type=int, default=5)
    ap.add_argument('--maxtreedepth', type=int, default=7)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--chain', type=int, default=0, help='chain index within the run (its own prior draw and sampler key)')
    ap.add_argument('--init-from', default=None,
                    help='npz of latent-site values (default: a prior draw, keyed by --seed and --chain)')
    ap.add_argument('--tag', default='')
    ap.add_argument('--check-only', action='store_true',
                    help='print ln L, Var(ln L) and the smallest event Neff at the initial point, and the time per '
                         'value+gradient, then exit')
    ap.add_argument('--device', default='0')
    ap.add_argument('--mem-fraction', default='.4')
    ap.add_argument('--cpus', type=int, default=8)
    ap.add_argument('--cpu-pick', choices=['idle', 'block'], default='idle')
    return ap


def run_name(args):
    return (f'ppc_{args.catalog}_k{args.spin_mass_modes}'
            + (f'_{args.spin_model}' if args.spin_model != 'field' else '')
            + (f'_sdfloor{args.gauss_sd_floor:g}' if args.spin_model == 'gauss' and args.gauss_sd_floor != 0.02 else '')
            + (f'_neff{args.event_neff_cut:g}' if args.event_neff_cut != 100. else '')
            + (f'_npe{args.max_samples or "all"}' if args.max_samples != MAX_SAMPLES else '')
            + ('_nocorr' if args.no_likelihood_correction else '')
            + ('_no' + '_'.join(e.replace('GW', '') for e in args.exclude_events) if args.exclude_events else '')
            + ('_short' if args.short else '') + (f'_{args.tag}' if args.tag else '')
            + (f'_seed{args.seed}' if args.seed != 1 else ''))


def idle_cores(n, seconds=1.):
    """The n least busy cores over ``seconds`` (from /proc/stat), among those this process may use."""
    def read():
        out = {}
        with open('/proc/stat') as f:
            for line in f:
                if line.startswith('cpu') and line[3].isdigit():
                    p = line.split(); v = list(map(int, p[1:]))
                    out[int(p[0][3:])] = (v[3] + v[4], sum(v))
        return out
    a = read(); time.sleep(seconds); b = read()
    allowed = os.sched_getaffinity(0) or set(a)
    busy = {c: 1. - (b[c][0] - a[c][0]) / max(1, b[c][1] - a[c][1]) for c in a if c in allowed}
    return set(sorted(busy, key=busy.get)[:n])


def setup_environment(args):
    """Pin CPUs, select the GPU and cap XLA's threads. Must run before jax is imported."""
    if args.cpu_pick == 'idle':
        os.sched_setaffinity(0, idle_cores(args.cpus))
    else:
        start = (int(args.device.split(',')[0]) * args.cpus) % os.cpu_count()
        os.sched_setaffinity(0, set(range(start, start + args.cpus)))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[var] = '1'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = args.mem_fraction
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'true')
    os.environ.setdefault('XLA_FLAGS', '--xla_gpu_force_compilation_parallelism=1 '
                          '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')


def build(args, name, record_weights=False):
    """(model, initial values, posteriors, injections, PixelPopData) for these arguments."""
    import data
    import model
    pp, posteriors, injections = data.build_pixelpop_data(name, args.catalog, bins=args.bins,
                                                          event_neff_cut=args.event_neff_cut,
                                                          exclude_events=tuple(args.exclude_events),
                                                          max_samples=args.max_samples, n_spin_bins=tuple(args.spin_bins))
    prob_model, init = model.setup_model(pp, spin_mass_modes=args.spin_mass_modes, n_spin_bins=tuple(args.spin_bins),
                                         likelihood_correction=not args.no_likelihood_correction,
                                         record_weights=record_weights, spin_model=args.spin_model,
                                         gauss_sd_floor=args.gauss_sd_floor)
    return prob_model, init, posteriors, injections, pp


def chain_key(seed, chain, use):
    """PRNG key for one ``use`` ('init' or 'sampler') of one chain: distinct for every (seed, chain, use)."""
    import jax
    return jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(seed), {'init': 1, 'sampler': 2}[use]), chain)


def prior_draw(prob_model, init, model_kwargs, seed, chain=0):
    """Every latent site of ``init`` drawn from its prior (one forward pass of the model), keyed by ``seed``, ``chain``."""
    from numpyro import handlers
    key = chain_key(seed, chain, 'init')
    tr = handlers.trace(handlers.seed(prob_model, key)).get_trace(**model_kwargs)
    return {k: tr[k]['value'] for k in init}


def main():
    args = make_parser().parse_args()
    setup_environment(args)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from numpyro import distributions as dist
    from pixelpop.models.probabilistic import inference_loop
    import model

    name = run_name(args)
    if not args.check_only and os.path.exists(os.path.join(RESULTS, name, f'chain_{args.chain}_samples.h5')):
        raise SystemExit(f'{name} already has chain {args.chain}; use --chain, --tag or --seed for a new one')
    os.makedirs(RESULTS, exist_ok=True)
    prob_model, init, posteriors, injections, pp = build(args, name)
    if args.init_from:
        loaded = np.load(args.init_from)
        unknown = sorted(set(loaded.files) - set(init))
        if unknown:
            raise ValueError(f'{args.init_from}: not latent sites of this model: {unknown}')
        for k in loaded.files:
            init[k] = jnp.asarray(loaded[k])
    else:
        init = prior_draw(prob_model, init, {'posteriors': posteriors, 'injections': injections}, args.seed, args.chain)
    if args.check_only:
        from numpyro.infer.util import log_density
        kw = {'posteriors': posteriors, 'injections': injections}
        with jax.default_device(jax.devices()[0]):
            tr = __import__('numpyro').handlers.trace(__import__('numpyro').handlers.substitute(
                __import__('numpyro').handlers.seed(prob_model, 0), data=init)).get_trace(**kw)
            g = lambda k: np.asarray(tr[k]['value'])
            print(f'initial point: ln L {float(g("log_likelihood")):.2f}, correction {float(g("log_likelihood_correction")):.2f}, '
                  f'Var {float(g("log_likelihood_variance")):.2f}, min event Neff {g("event_neff").min():.0f}', flush=True)
            vg = jax.jit(jax.value_and_grad(lambda p: log_density(prob_model, (), kw, p)[0]))
            jax.block_until_ready(vg(init)); t0 = time.time()
            for _ in range(5):
                jax.block_until_ready(vg(init))
            print(f'value+grad {1e3 * (time.time() - t0) / 5:.1f} ms', flush=True)
        return
    s = dict(SHORT) if args.short else dict(warmup=args.warmup, tot_samples=args.samples, thinning=args.thinning)
    print(f'run {name} chain {args.chain}: {s}, spin model {args.spin_model}, spin mass modes {args.spin_mass_modes}, '
          f'spin bins {args.spin_bins}', flush=True)
    scalar_spin = [k for k in model.SPIN_SITES[args.spin_model] if k.startswith('lnsigma') and k != 'lnsigma_gauss']
    print_keys = (['log_likelihood', 'log_likelihood_variance', 'log_likelihood_correction', 'lnsigma', *scalar_spin,
                   'detection_efficiency'] + [h for h, (_, d) in pp.priors.items() if d is not dist.Delta])
    inference_loop(prob_model, model_kwargs={'posteriors': posteriors, 'injections': injections}, initial_value=init,
                   warmup=s['warmup'], tot_samples=s['tot_samples'], thinning=s['thinning'],
                   num_samples=max(1, min(100, s['tot_samples'] // 10)), pacc=0.7, maxtreedepth=args.maxtreedepth,
                   parallel=[args.chain], run_dir=RESULTS, name=name, print_keys=print_keys,
                   dense_mass=model.mass_matrix_blocks(args.spin_model), pixelpop_data=pp,
                   rng_key=chain_key(args.seed, args.chain, 'sampler'))


if __name__ == '__main__':
    main()
