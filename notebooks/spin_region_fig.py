"""Draw analysis.spin_region_figure for one run: p(chi_eff, chi_p | m1 region, q region), median 2D density with
1D slices, columns over m1 regions and rows over q regions.

    python notebooks/spin_region_fig.py <run> [--high 50-75] [--transpose] [--marginals]
        -> plots/<run>_spin_regions[_high50-75][_marg][_T].pdf

--marginals: side panels with the 1D marginals p(chi_eff | region), p(chi_p | region) instead of slices.

Needs REGION in the spin cache (post.sbatch since 2026-10-23). Reads only REGION (~0.8 GB per long run), so it runs on
a login node (taskset, texlive module).
"""
import argparse
import os
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import matplotlib
matplotlib.use('Agg')
import plot_style
import analysis as A

ap = argparse.ArgumentParser()
ap.add_argument('run')
ap.add_argument('--high', default='55-65', choices=['55-65', '50-75'], help='the highest m1 region (Msun)')
ap.add_argument('--marginals', action='store_true', help='1D marginals in the side panels instead of slices')
ap.add_argument('--transpose', action='store_true', help='rows over m1, columns over q (portrait)')
args = ap.parse_args()
R = A.load_regions(args.run)
plot_style.apply()
high = tuple(float(x) for x in args.high.split('-'))
m_regions = A.REGION_M[:3] + (high,)
fig = A.spin_region_figure(R, m_regions, transpose=args.transpose, marginals=args.marginals, title=A.SPIN_MODEL_TITLES[R.spin_model] + (f' ($k = {R.k}$)' if R.k != 10 else ''))
plot_style.save(fig, f'{args.run}_spin_regions' + ('' if args.high == '55-65' else f'_high{args.high}')
                + ('_marg' if args.marginals else '') + ('_T' if args.transpose else ''))
