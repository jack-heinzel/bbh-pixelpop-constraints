"""Posterior smoothing length scales of the spin fields, and the field correlation between feature locations, as LaTeX
macros (user, 2026-11-24), for the flagship runs (peak_fits.RUN_NAMES).

    python macros/spin_ranges.py        (login node, pinned; seconds)
        -> macros/spin_ranges.tex, draft/macros/spin_ranges.tex

Ranges are the Matern practical range rho of each field's spectrum (model.matern_spectrum: kappa = sqrt(8 nu) / rho;
the correlation falls to ~0.1 at distance rho), sampled in bins with a prior uniform in ln rho over [0, ln B]
(B = 100 bins per axis: prior median 10 bins), converted to axis units with the grid spacings (A.dlnm in ln m1,
A.dq in q, 0.02 in chi_eff, 0.01 in chi_p).
  Flexible: one joint field over (ln m1, q, chi_eff, chi_p) ('log_ranges_spin'); its mass ranges smooth every spin
  property together. Macros \\spinRange{Lnm,Q,ChiEff,ChiP}Flex, \\spinNuFlex.
  Gaussian: one field per Gaussian parameter over (ln m1, q) ('log_ranges_gauss', A.GP order), shared nu. Macros
  \\spinRange{MuEff,MuP,SdEff,SdP}{Lnm,Q}Gauss, \\spinNuGauss.
Amplitudes: sigma = exp(lnsigma), the field's standard deviation averaged over its grid sites (matern_spectrum's
normalization). Flexible: of the joint log density F(ln m1, q, chi_eff, chi_p) (natural-log units), prior lnsigma ~
N(0, 2) (pixelpop's coupling_prior); \\spinAmpFlex. Gaussian: of each parameter field over (ln m1, q), in its own units
(mu_eff, mu_p: chi; SdEff, SdP: ln sd, so a fractional width change), prior lnsigma ~ N(ln 0.1, 1) for the means and
N(ln 0.5, 1) for the ln sd (model.GAUSS_LNSIGMA_CENTRES); \\spinAmp{MuEff,MuP,SdEff,SdP}Gauss.
Correlations: the Matern correlation of the field (Flexible) or of mu_chi_p (Gaussian) between two feature locations
along one axis, per draw: \\spinCorr{TenTwenty,TwentyThirtyFive,ThirtyFiveSixty}{suffix} along ln m1 (feature masses
10, 21, 33, 59 Msun) and \\spinCorrQ{suffix} between q = 0.5 and q = 0.9. Continuum Matern, ignoring the lattice and
the K mass-mode truncation of the Flexible model (which only removes shorter scales).
"""
import glob
import os
import sys

import numpy as np

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'notebooks')); sys.path.insert(0, HERE)
import h5py
from scipy.special import gamma, kv

import analysis as A
from peak_fits import RUN_NAMES, fmt

UNITS = {'Lnm': A.dlnm, 'Q': A.dq, 'ChiEff': 0.02, 'ChiP': 0.01}
GAUSS_TEX = {'mu_eff': 'MuEff', 'mu_p': 'MuP', 'ln_sd_eff': 'SdEff', 'ln_sd_p': 'SdP'}
FEATURE_M = {'Ten': 10., 'Twenty': 21., 'ThirtyFive': 33., 'Sixty': 59.}
PAIRS = (('Ten', 'Twenty'), ('Twenty', 'ThirtyFive'), ('ThirtyFive', 'Sixty'))
Q_PAIR = (0.5, 0.9)


def load(run, key):
    return np.concatenate([h5py.File(f, 'r')[key][:]
                           for f in sorted(glob.glob(os.path.join(REPO, 'results', run, 'chain_*_samples.h5')))])


def matern_corr(d, rho, nu):
    """Matern correlation at distance d (same units as rho), practical range rho, smoothness nu."""
    x = np.maximum(np.sqrt(8 * nu) * d / rho, 1e-12)
    return 2 ** (1 - nu) / gamma(nu) * x ** nu * kv(nu, x)


def corr_lines(sfx, rho_m, rho_q, nu):
    """Correlation macros from ranges in bins."""
    out = []
    for a, b in PAIRS:
        d = np.log(FEATURE_M[b] / FEATURE_M[a]) / A.dlnm
        out.append(rf'\newcommand{{\spinCorr{a}{b}{sfx}}}{{{fmt(matern_corr(d, rho_m, nu))}}}')
    d = (Q_PAIR[1] - Q_PAIR[0]) / A.dq
    out.append(rf'\newcommand{{\spinCorrQ{sfx}}}{{{fmt(matern_corr(d, rho_q, nu))}}}')
    return out


def main():
    lines = ['% Spin-field smoothing: Matern practical ranges in axis units (ln m1, q, chi_eff, chi_p) and smoothness nu,'
             ' median and 90% (macros/spin_ranges.py); prior: ln rho uniform over [0, ln 100] bins (median 10 bins,'
             ' 0.42 in ln m1, 0.1 in q). Corr: field correlation between feature masses 10, 21, 33, 59 Msun along ln m1'
             ' (Flexible: the joint spin field; Gaussian: mu_chi_p), CorrQ: between q = 0.5 and 0.9; Amp: field sd'
             ' (Flexible: of the log density, prior lnsigma ~ N(0, 2); Gaussian: in chi for the means, prior median 0.1,'
             ' in ln sd for the widths, prior median 0.5)']
    for run, sfx in RUN_NAMES.items():
        lines.append(f'% {run}')
        if sfx == 'Flex':
            R, nu = np.exp(load(run, 'log_ranges_spin')), np.exp(load(run, 'log_nu_spin'))
            for j, ax in enumerate(UNITS):
                lines.append(rf'\newcommand{{\spinRange{ax}{sfx}}}{{{fmt(R[:, j] * UNITS[ax])}}}')
            lines.append(rf'\newcommand{{\spinNu{sfx}}}{{{fmt(nu)}}}')
            lines.append(rf'\newcommand{{\spinAmp{sfx}}}{{{fmt(np.exp(load(run, "lnsigma_spin")))}}}')
            lines += corr_lines(sfx, R[:, 0], R[:, 1], nu)
        else:
            R, nu = np.exp(load(run, 'log_ranges_gauss')), np.exp(load(run, 'log_nu_gauss'))
            for name, tex in GAUSS_TEX.items():
                i = A.GP.index(name)
                for j, ax in enumerate(('Lnm', 'Q')):
                    lines.append(rf'\newcommand{{\spinRange{tex}{ax}{sfx}}}{{{fmt(R[:, i, j] * UNITS[ax])}}}')
            lines.append(rf'\newcommand{{\spinNu{sfx}}}{{{fmt(nu)}}}')
            amp = np.exp(load(run, 'lnsigma_gauss'))
            for name, tex in GAUSS_TEX.items():
                lines.append(rf'\newcommand{{\spinAmp{tex}{sfx}}}{{{fmt(amp[:, A.GP.index(name)])}}}')
            i = A.GP.index('mu_p')
            lines += corr_lines(sfx, R[:, i, 0], R[:, i, 1], nu)
    text = '\n'.join(lines) + '\n'
    for path in (os.path.join(HERE, 'spin_ranges.tex'), os.path.join(REPO, 'draft', 'macros', 'spin_ranges.tex')):
        with open(path, 'w') as f:
            f.write(text)
        print('wrote', path)
    print(text)


if __name__ == '__main__':
    main()
