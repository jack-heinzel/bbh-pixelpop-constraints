"""Redraw the side-by-side mass-ratio / spin summary figure (analysis.summary_figure) from the rebuilt spin cache.

    python notebooks/summary_fig.py <name> <run> [<run> ...] [--m-min 5] [--mean] [--mean-asinh W] [--core]
        -> plots/compare_<name>_mass_ratio_spin_summary[_mean][_asinh<W>][_core][_m<m-min>].pdf
--mean adds the mu_chi_eff(m1) row; --core restricts the chi_eff rows to |chi_eff| < analysis.CHI_CORE (needs MOM_CORE
in the cache: rebuilt by post.sbatch since 2026-10-22).

Reads only the mass field and the chi_eff moments (analysis.load_light): about 0.7 GB per long run, so it runs on a
login node (pin it: taskset -c <core>, with the texlive module loaded for usetex). The cache must exist
(results/post_processing/for_plotting/<run>/; post.sbatch or python notebooks/analysis.py <run>).
"""
import argparse
import os
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import matplotlib
matplotlib.use('Agg')
import plot_style
import analysis as A

ap = argparse.ArgumentParser()
ap.add_argument('name')
ap.add_argument('runs', nargs='+')
ap.add_argument('--m-min', type=float, default=A.MIN_M, help='left edge of the m1 axis (Msun)')
ap.add_argument('--mean', action='store_true', help='add the mean chi_eff row')
ap.add_argument('--core', action='store_true', help='chi_eff rows restricted to |chi_eff| < CHI_CORE')
ap.add_argument('--mean-asinh', type=float, default=None, metavar='W',
                help='mean row on a shared asinh(mu / W) scale instead of a y range per column')
args = ap.parse_args()
R = [A.load_light(run) for run in args.runs]
plot_style.apply()                  # after pixelpop's import (in load_light), which resets matplotlib's rcParams
models = [r.spin_model for r in R]
heads = [A.SPIN_MODEL_TITLES[r.spin_model] + (f' ($k = {r.k}$)' if models.count(r.spin_model) > 1 else '') for r in R]
fig, peaks = A.summary_figure(R, heads, m_min=args.m_min, show_mean=args.mean or bool(args.mean_asinh), core=args.core,
                              mean_asinh=args.mean_asinh)
suffix = (('_mean' if args.mean or args.mean_asinh else '') + (f'_asinh{args.mean_asinh:g}' if args.mean_asinh else '')
          + ('_core' if args.core else '')
          + ('' if args.m_min == A.MIN_M else f'_m{args.m_min:g}'))
plot_style.save(fig, f'compare_{args.name}_mass_ratio_spin_summary{suffix}')
