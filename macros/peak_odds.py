"""Is the ~33 Msun feature a peak rather than a steepening of a falling power law? (draft/main.tex, Mass peaks; user,
2026-10-30): the integrated rate in m1 in [30, 35] Msun against its neighbors [25, 30] and [35, 40],

    R_W = int_W dm1 int_Q dq p(m1, q),

per posterior draw and per draw of the mass field's empirical-Bayes prior (prior_fits.prior_p_mass: each posterior draw's
kernel parameters, fresh white noise), for q marginalized ('All': every q the m2 > 3 Msun window allows) and for
q in [0.8, 1] ('Hi'). The m1 bins (uniform in ln m1) enter with the fraction of their width inside W, so the windows
need not fall on bin edges. Under the stationary prior the rate in a window scales with its width in ln m1, which
favors the widest, [25, 30]; the odds ratio removes that.

    python macros/peak_odds.py   -> macros/peak_odds.tex

Macros \\peakOdds<Q><Stat><Run>, Q in (All, Hi), Run in (Flex, Gauss), Stat:
    P, PPrior: P(R_mid > R_lo and R_mid > R_hi) under the posterior and the prior;
    BF: posterior odds / prior odds of that event ('> x' when every posterior draw is a peak, from 1 - P < 1/n);
    PLo, PHi (and PLoPrior, PHiPrior): P(R_mid > R_lo), P(R_mid > R_hi) separately;
    RatioLo, RatioHi: R_mid / R_lo, R_mid / R_hi (median and 90%).
"""
import os
import sys

import numpy as np

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'notebooks')); sys.path.insert(0, HERE)
import analysis as A
from peak_fits import RUN_NAMES, fmt
from prior_fits import load_hyper, prior_p_mass

WINDOWS = {'lo': (25., 30.), 'mid': (30., 35.), 'hi': (35., 40.)}
Q_RANGES = {'All': (0., 1.), 'Hi': (0.8, 1.)}


def m1_weights(window):
    """Fraction of each ln m1 bin inside ``window`` (Msun), times its width: integrates a density per d ln m1."""
    lo, hi = np.log(window)
    return np.clip(np.minimum(A.lnm_edges[1:], hi) - np.maximum(A.lnm_edges[:-1], lo), 0., None)


def rates(p_mass, q_range):
    """(draws, 3): the integrated rate (as a fraction of the population) in each window of WINDOWS."""
    t = (A.q_c >= q_range[0]) & (A.q_c <= q_range[1])
    p_m = p_mass[:, :, t].astype(float).sum(2) * A.dq                                   # (draws, Bm) per d ln m1
    return np.stack([p_m @ m1_weights(w) for w in WINDOWS.values()], 1)


def bf(post, prior):
    """Posterior odds over prior odds of the boolean arrays ``post``, ``prior``; a lower bound if all of post is True."""
    n_fail = np.sum(~post)
    b = (len(post) - max(n_fail, 1)) / max(n_fail, 1) / (prior.mean() / (1 - prior.mean()))
    s = f'{float(f"{b:.2g}"):g}'
    return f'> {s}' if n_fail == 0 else s


def pstr(p):
    return f'{p:.3f}' if 0.99 < p < 1 else f'{p:.2f}'


def main():
    lines = ['% Integrated rate in m1 [30, 35] against [25, 30] and [35, 40] (macros/peak_odds.py), q marginalized (All) '
             'and q in [0.8, 1] (Hi); posterior and empirical-Bayes prior']
    for run, suffix in RUN_NAMES.items():
        post = A.load_light(run).p_mass
        prior = prior_p_mass(load_hyper(run))
        for qname, qr in Q_RANGES.items():
            Rp, Rr = rates(post, qr), rates(prior, qr)
            gt = lambda R, j: R[:, 1] > R[:, j]
            peak_p, peak_r = gt(Rp, 0) & gt(Rp, 2), gt(Rr, 0) & gt(Rr, 2)
            vals = {'P': pstr(peak_p.mean()), 'PPrior': pstr(peak_r.mean()), 'BF': bf(peak_p, peak_r),
                    'PLo': pstr(gt(Rp, 0).mean()), 'PHi': pstr(gt(Rp, 2).mean()),
                    'PLoPrior': pstr(gt(Rr, 0).mean()), 'PHiPrior': pstr(gt(Rr, 2).mean()),
                    'RatioLo': fmt(Rp[:, 1] / Rp[:, 0]), 'RatioHi': fmt(Rp[:, 1] / Rp[:, 2])}
            print(f'{run} q {qname} {qr}: ' + ', '.join(f'{k} {v}' for k, v in vals.items())
                  + f'; prior ratios {fmt(Rr[:, 1] / Rr[:, 0])}, {fmt(Rr[:, 1] / Rr[:, 2])}', flush=True)
            for k, v in vals.items():
                lines.append(rf'\newcommand{{\peakOdds{qname}{k}{suffix}}}{{{v}}}')
        del post, prior
    path = os.path.join(HERE, 'peak_odds.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('wrote', path)


if __name__ == '__main__':
    main()
