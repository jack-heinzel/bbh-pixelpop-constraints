"""Broadening of chi_eff with decreasing mass ratio, from sigma_chieff(q) averaged over p(m1 | q) (paper_figs.chiq):
per posterior draw, the probability that the width at Q_HI is below the width at Q_LO (user, 2026-10-28), and the same
for the empirical-Bayes spin prior draws (macros/spin_prior.py).

    python macros/chi_q_broadening.py   -> macros/chi_q_broadening.tex

Macros, per run (suffix Flex / Gauss) and for chi_eff restricted to |chi_eff| < analysis.CHI_CORE ('Core') or not:
SdQLo, SdQHi (sigma_chieff at Q_LO and Q_HI, median and 90%), SdRatio (sigma(Q_LO) / sigma(Q_HI)), Prob
(P[sigma(Q_HI) < sigma(Q_LO)]) and ProbPrior (the same under the prior), e.g. \\chiQBroadProbCoreFlex.
"""
import os
import sys

import numpy as np

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'notebooks')); sys.path.insert(0, HERE)
import analysis as A
from paper_figs import m1_averaged_moments, m1_averaged_prior
from peak_fits import RUN_NAMES, fmt

Q_LO, Q_HI = 0.35, 0.85
pfmt = lambda p: f'{p:.3f}' if 0.99 < p < 1 else f'{p:.2f}'      # 0.996 is not 1.00 (2000 draws)


def q_bin(q):
    return int(np.clip(np.searchsorted(A.q_edges, q) - 1, 0, A.B - 1))


def broadening(sd):
    """(sigma(Q_LO), sigma(Q_HI)) per draw from sd (draws, Bq)."""
    return sd[:, q_bin(Q_LO)], sd[:, q_bin(Q_HI)]


def main():
    lines = [f'% Broadening of chi_eff with decreasing q (macros/chi_q_broadening.py): sigma_chieff(q) averaged over'
             f' p(m1 | q); SdQLo / SdQHi at q = {Q_LO:g} / {Q_HI:g} (median and 90%), SdRatio = sigma({Q_LO:g}) /'
             f' sigma({Q_HI:g}), Prob = P[sigma({Q_HI:g}) < sigma({Q_LO:g})], ProbPrior: the same for the'
             f' empirical-Bayes spin prior; Core: |chi_eff| < {A.CHI_CORE:g}']
    for run, suffix in RUN_NAMES.items():
        r = A.load_light(run)
        lines.append(f'% {run}')
        for core, tag in ((True, 'Core'), (False, '')):
            lo, hi = broadening(m1_averaged_moments(r, core)['sd'])
            plo, phi = broadening(m1_averaged_prior(r, core)['sd'])
            ok, pok = np.isfinite(lo) & np.isfinite(hi), np.isfinite(plo) & np.isfinite(phi)
            prob, pprob = np.mean(hi[ok] < lo[ok]), np.mean(phi[pok] < plo[pok])
            print(f'{run} {tag or "all"}: sigma({Q_LO:g}) {fmt(lo[ok])}, sigma({Q_HI:g}) {fmt(hi[ok])}, ratio '
                  f'{fmt(lo[ok] / hi[ok])}, P {prob:.3f} (prior {pprob:.3f}), {ok.sum()} draws')
            for par, x in (('SdQLo', lo[ok]), ('SdQHi', hi[ok]), ('SdRatio', lo[ok] / hi[ok])):
                lines.append(rf'\newcommand{{\chiQBroad{par}{tag}{suffix}}}{{{fmt(x)}}}')
            lines.append(rf'\newcommand{{\chiQBroadProb{tag}{suffix}}}{{{pfmt(prob)}}}')
            lines.append(rf'\newcommand{{\chiQBroadProbPrior{tag}{suffix}}}{{{pfmt(pprob)}}}')
        del r
    path = os.path.join(HERE, 'chi_q_broadening.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('wrote', path)


if __name__ == '__main__':
    main()
