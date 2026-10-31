"""Spin moments in the four peak regions of the spin-peaks figure (paper_figs.SPIN_PEAK_REGIONS; draft/main.tex, the
10, 20, 35 and 60 Msun features; user, 2026-10-30): for each region R = {m1 in [a, b], q in [c, d]} (bins by centre),
the mean and standard deviation of chi_eff and of chi_p under p(chi | R), the mixture of the per-cell conditionals with
the windowed mass weights,

    E[x | R] = sum_cells w mu,   Var[x | R] = sum_cells w (sd^2 + mu^2) - E[x | R]^2,   w = p(ln m1, q) / sum_R p,

per posterior draw (the spin cache's per-cell MOM) and per empirical-Bayes spin prior draw (macros/spin_prior.py: the
draw's posterior mass field and kernels, fresh white noise). 'Core' restricts chi_eff to |chi_eff| < analysis.CHI_CORE
(MOM_CORE; each cell weighted by its probability inside the core), as the width figures.

    python macros/peak_spins.py   -> macros/peak_spins.tex

Macros \\peakSpin<Peak><Stat><Run>, Peak in (Ten, ThirtyFive, Twenty, Sixty), Run in (Flex, Gauss), Stat: EffMean,
EffSd, EffMeanCore, EffSdCore, PMean, PSd (median and 90%), PEffPos (P(core mean chi_eff > 0)), and each with suffix Prior.
Runs on a login node, pinned (< 3 GB per run).
"""
import glob
import os
import sys

import h5py
import numpy as np

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'notebooks')); sys.path.insert(0, HERE)
import analysis as A
from paper_figs import SPIN_PEAK_REGIONS
from peak_fits import RUN_NAMES, fmt, pct

PEAK_NAMES = ('Ten', 'ThirtyFive', 'Twenty', 'Sixty')         # SPIN_PEAK_REGIONS order
pfmt = pct


def region_moments(p_mass, MOM, MOM_CORE, region):
    """{stat: (draws,)} for one region: chi_eff and chi_p mean and sd, and the core chi_eff mean and sd."""
    (m0, m1), (q0, q1) = region
    s, t = (A.m_c >= m0) & (A.m_c <= m1), (A.q_c >= q0) & (A.q_c <= q1)
    w = p_mass[:, s][:, :, t].astype(float)
    sub = lambda X: X[:, s][:, :, t].astype(float)
    M, C = sub(MOM), sub(MOM_CORE)

    def mix(w, mu, sd):
        w = w / w.sum((1, 2), keepdims=True)
        mean = (w * mu).sum((1, 2))
        return mean, np.sqrt(np.maximum((w * (sd ** 2 + mu ** 2)).sum((1, 2)) - mean ** 2, 0.))

    out = {}
    out['EffMean'], out['EffSd'] = mix(w, M[..., 0], M[..., 1])
    out['PMean'], out['PSd'] = mix(w, M[..., 2], M[..., 3])
    out['EffMeanCore'], out['EffSdCore'] = mix(w * (1 - C[..., 2]), C[..., 0], C[..., 1])
    return out


def main():
    lines = ['% Spin moments in the peak regions (macros/peak_spins.py): mean and sd of chi_eff, chi_p given m1, q in '
             'the region (median and 90%), posterior and empirical-Bayes spin prior; Core: |chi_eff| < '
             f'{A.CHI_CORE:g}; PEffPos: P(core mean chi_eff > 0)']
    for run, suffix in RUN_NAMES.items():
        r = A.load_light(run)
        files = sorted(glob.glob(os.path.join(REPO, 'results', run, 'chain_*_samples.h5')))
        MOM = np.concatenate([h5py.File(A.post_path(run, f), 'r')['MOM'][:r.n_draws] for f in files])
        MOM_CORE = r.MOM_CORE
        d = np.load(os.path.join(REPO, 'results', 'post_processing', f'{run}_spin_prior.npz'))
        lines.append(f'% {run}: {r.n_draws} posterior draws, {len(d["draws"])} prior draws')
        for name, region in zip(PEAK_NAMES, SPIN_PEAK_REGIONS):
            post = region_moments(r.p_mass, MOM, MOM_CORE, region)
            prior = region_moments(r.p_mass[d['draws']], d['MOM'], d['MOM_CORE'], region)
            lines.append(f'% {name}: m1 in {region[0]} Msun, q in {region[1]}')
            for tag, res in (('', post), ('Prior', prior)):
                for k, x in res.items():
                    lines.append(rf'\newcommand{{\peakSpin{name}{k}{tag}{suffix}}}{{{fmt(x)}}}')
                lines.append(rf'\newcommand{{\peakSpin{name}PEffPos{tag}{suffix}}}{{{pfmt(np.mean(res["EffMeanCore"] > 0))}}}')
            print(f'{run} {name} {region}: ' + ', '.join(f'{k} {fmt(x)} (prior {fmt(prior[k])})' for k, x in post.items())
                  + f"; P(core mean chi_eff > 0) {pfmt(np.mean(post['EffMeanCore'] > 0))}", flush=True)
        del r, MOM, MOM_CORE, d
    path = os.path.join(HERE, 'peak_spins.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('wrote', path)


if __name__ == '__main__':
    main()
