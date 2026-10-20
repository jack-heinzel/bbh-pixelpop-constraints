"""Sample the weakly modelled population (model.py); chains land in results/<run name>/.

    python run.py nrsur --spin-mass-modes 10 --device 0 --mem-fraction .4 [--short]

Run name: ppc_<catalog>_k<spin mass modes>[_neff<floor>][_no231123][_short][_<tag>].

This node caps processes + threads at 500 per user, and XLA sizes its thread pools from the CPU affinity, so the job
pins itself (--cpus cores, the least busy ones by default) and serializes XLA compilation before jax is imported.
"""
import argparse
import os
import time

REPO = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(REPO, 'results')
SHORT = dict(warmup=2000, tot_samples=500, thinning=5)


def make_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('catalog', choices=['nrsur', 'physical', 'mixed'])
    ap.add_argument('--spin-mass-modes', type=int, default=10,
                    help='mass modes per axis (ln m1, q) of the spin field; 0: spins independent of mass')
    ap.add_argument('--spin-bins', type=int, nargs=2, default=[100, 100], metavar=('B_CHIEFF', 'B_CHIP'))
    ap.add_argument('--bins', type=int, default=100, help='mass grid bins per axis')
    ap.add_argument('--event-neff-cut', type=float, default=100.)
    ap.add_argument('--no-likelihood-correction', action='store_true')
    ap.add_argument('--exclude-events', nargs='*', default=[])
    ap.add_argument('--short', action='store_true', help=f'a first look, not converged: {SHORT}')
    ap.add_argument('--warmup', type=int, default=4000)
    ap.add_argument('--samples', type=int, default=1000)
    ap.add_argument('--thinning', type=int, default=5)
    ap.add_argument('--maxtreedepth', type=int, default=7)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--init-from', default=None, help='npz of latent-site values')
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
            + (f'_neff{args.event_neff_cut:g}' if args.event_neff_cut != 100. else '')
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
                                                          exclude_events=tuple(args.exclude_events))
    prob_model, init = model.setup_model(pp, spin_mass_modes=args.spin_mass_modes, n_spin_bins=tuple(args.spin_bins),
                                         likelihood_correction=not args.no_likelihood_correction,
                                         record_weights=record_weights)
    return prob_model, init, posteriors, injections, pp


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
    if not args.check_only and os.path.isdir(os.path.join(RESULTS, name)) and any(f.endswith('_samples.h5') for f in os.listdir(os.path.join(RESULTS, name))):
        raise SystemExit(f'{name} already has chains; use --tag or --seed for a new run')
    os.makedirs(RESULTS, exist_ok=True)
    prob_model, init, posteriors, injections, pp = build(args, name)
    if args.init_from:
        loaded = np.load(args.init_from)
        unknown = sorted(set(loaded.files) - set(init))
        if unknown:
            raise ValueError(f'{args.init_from}: not latent sites of this model: {unknown}')
        for k in loaded.files:
            init[k] = jnp.asarray(loaded[k])
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
    print(f'run {name}: {s}, spin mass modes {args.spin_mass_modes}, spin bins {args.spin_bins}', flush=True)
    print_keys = ['log_likelihood', 'log_likelihood_variance', 'log_likelihood_correction', 'lnsigma', 'lnsigma_spin',
                  'detection_efficiency'] + [h for h, (_, d) in pp.priors.items() if d is not dist.Delta]
    inference_loop(prob_model, model_kwargs={'posteriors': posteriors, 'injections': injections}, initial_value=init,
                   warmup=s['warmup'], tot_samples=s['tot_samples'], thinning=s['thinning'],
                   num_samples=max(1, min(100, s['tot_samples'] // 10)), pacc=0.7, maxtreedepth=args.maxtreedepth,
                   parallel=1, run_dir=RESULTS, name=name, print_keys=print_keys, dense_mass=model.mass_matrix_blocks(),
                   pixelpop_data=pp, rng_key=jax.random.PRNGKey(args.seed))


if __name__ == '__main__':
    main()
