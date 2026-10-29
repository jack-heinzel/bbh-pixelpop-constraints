"""Mass-ratio contrasts between the peaks (draft/temp.tex, SM "Mass ratio contrast"; user, 2026-10-29): the mean mass
ratio over q in [Q0, 1] in a window W around each peak,

    <q>_W = int_W dln m1 int_Q0^1 dq q p(ln m1, q) / int_W dln m1 int_Q0^1 dq p(ln m1, q),

and Delta_{10,20} = <q>_10 - <q>_20, Delta_{35,20} = <q>_35 - <q>_20, Delta_{35,60} = <q>_35 - <q>_60, per posterior
draw and per draw of the mass field's empirical-Bayes prior (macros/prior_fits.prior_p_mass: each posterior draw's kernel
parameters, fresh white noise).

    python macros/q_contrast.py   -> macros/q_contrast.tex

Macros per contrast (TenTwenty, ThirtyFiveTwenty, ThirtyFiveSixty) and run (Flex, Gauss): \\qContrast<Name>{Med, P, Z,
ESS, PPrior, ZPrior}<Run>. Med: median and 90% of Delta; P: P(Delta > 0); Z: median / sigma (sigma: half the central 68%
interval); ESS: the effective sample size of Delta over the chains (numpyro's estimator); PPrior, ZPrior: the same under
the prior. Also the window means <q>_W: \\qWindow<Peak><Run>.
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

Q0 = 0.4
WINDOWS = {'Ten': (8., 12.), 'Twenty': (15., 25.), 'ThirtyFive': (28., 40.), 'Sixty': (50., 70.)}
CONTRASTS = (('Ten', 'Twenty'), ('ThirtyFive', 'Twenty'), ('ThirtyFive', 'Sixty'))


def pfmt(p, n):
    """P to 3 decimals near 1; '> 1 - 1/n' when no draw of n is below (P = 1 is not resolvable)."""
    return f'> {1 - 1 / n:.4f}'.rstrip('0') if p == 1 else f'{p:.3f}' if 0.99 < p < 1 else f'{p:.2f}'


def window_q(p_mass, window, q0=Q0):
    """<q> over m1 bins with centres in ``window`` and q bins with centres >= q0, per draw."""
    s = (A.m_c >= window[0]) & (A.m_c <= window[1])
    t = A.q_c >= q0
    assert np.all(A.q_c[t].min() * A.m_c[s] >= A.MIN_M), 'q0 must lie above 3 Msun / m1 in the window'
    pq = p_mass[:, s][:, :, t].astype(float).sum(1)
    return (pq @ A.q_c[t]) / pq.sum(1)


def z_score(d):
    lo, md, hi = np.percentile(d, [16, 50, 84])
    return md / (0.5 * (hi - lo))


def ess(d, n_chains):
    from numpyro.diagnostics import effective_sample_size
    return float(effective_sample_size(np.asarray(d).reshape(n_chains, -1)))


def main():
    import glob
    lines = [f'% Mass-ratio contrasts between the peaks (macros/q_contrast.py): <q> over q >= {Q0:g} in the windows '
             + ', '.join(f'{k} [{a:g}, {b:g}]' for k, (a, b) in WINDOWS.items())
             + '; Med (90%), P(Delta > 0), Z = median / half the central 68%, ESS, and the prior\'s P, Z']
    for run, suffix in RUN_NAMES.items():
        n_chains = len(glob.glob(os.path.join(REPO, 'results', run, 'chain_*_samples.h5')))
        post = A.load_light(run).p_mass
        prior = prior_p_mass(load_hyper(run))
        Qp = {k: window_q(post, w) for k, w in WINDOWS.items()}
        Qr = {k: window_q(prior, w) for k, w in WINDOWS.items()}
        del post, prior
        lines.append(f'% {run}: {len(Qp["Ten"])} posterior and prior draws')
        for k, x in Qp.items():
            lines.append(rf'\newcommand{{\qWindow{k}{suffix}}}{{{fmt(x)}}}')
        for a, b in CONTRASTS:
            d, dr = Qp[a] - Qp[b], Qr[a] - Qr[b]
            name = a + b
            vals = {'Med': fmt(d), 'P': pfmt(np.mean(d > 0), len(d)), 'Z': f'{z_score(d):.1f}',
                    'ESS': f'{ess(d, n_chains):.0f}', 'PPrior': pfmt(np.mean(dr > 0), len(dr)),
                    'ZPrior': f'{z_score(dr) + 0.:.1f}'.replace('-0.0', '0.0')}
            print(f'{run} Delta_{a},{b}: ' + ', '.join(f'{k} {v}' for k, v in vals.items()), flush=True)
            for k, v in vals.items():
                lines.append(rf'\newcommand{{\qContrast{name}{k}{suffix}}}{{{v}}}')
    path = os.path.join(HERE, 'q_contrast.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('wrote', path)


if __name__ == '__main__':
    main()
