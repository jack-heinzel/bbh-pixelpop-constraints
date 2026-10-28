"""Initial values for run.py --init-from: every latent site of one posterior draw of an earlier run with the same model
(user, 2026-10-28: the variance-cut runs, whose random prior starts lie far outside Var(ln L) < cut and freeze).

    python init_from_draw.py <run> [--out PATH]      -> results/inits/<run>_minvar.npz

The draw is the one with the smallest Var(ln L) over all chains. Latent sites: the white sites ('_...') and the
hyperparameters of the mass field (model.L_SITES), the spin model (model.SPIN_SITES) and the parametric models
('lamb').
"""
import argparse
import glob
import os

import h5py
import numpy as np

from model import L_SITES, SPIN_SITES

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMETRIC = ('lamb',)


def main(run, out=None):
    files = sorted(glob.glob(os.path.join(HERE, 'results', run, 'chain_*_samples.h5')))
    best = min(((float(v), f, i) for f in files
                for i, v in enumerate(h5py.File(f, 'r')['log_likelihood_variance'][:])), key=lambda t: t[0])
    var, f, i = best
    with h5py.File(f, 'r') as h:
        model = 'split' if 'lnsigma_spin1' in h else 'gauss' if 'gauss_offsets' in h else 'field'
        keys = [k for k in h if k.startswith('_')] + [k for k in L_SITES + SPIN_SITES[model] + PARAMETRIC if k in h]
        sites = {k: h[k][i] for k in keys}
        lnl = float(h['log_likelihood'][i])
    out = out or os.path.join(HERE, 'results', 'inits', f'{run}_minvar.npz')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, **sites)
    print(f'{run}: draw {i} of {os.path.basename(f)}, Var(ln L) {var:.2f}, ln L {lnl:.1f}; {sorted(keys)} -> {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--out')
    a = ap.parse_args()
    main(a.run, a.out)
