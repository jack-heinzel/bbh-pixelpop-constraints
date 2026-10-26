"""One joint fit of the whole (m1, q) plane per hyperparameter draw: four peaks and a power-law background.

    python macros/global_fit.py [--cpus N] [--draws K] [<run> ...]   (default: ppc_gwtc5_k10 ppc_gwtc5_k100_gauss)
        -> macros/global_fit.tex, plots/<run>_global_fit.pdf, results/post_processing/<run>_global_fit.npz

The windowed p(m1, q) (density per Msun per unit q) on every supported cell with m1 in FIT_RANGE is fitted with
(user, 2026-10-26: the principled version of the window fits in joint_fits.py)

    f(m1, q) = sum_k  A_k N(m1; mu_k, sigma_k) TN(q; mu_qk, sigma_qk)  +  B (m1 / M_REF)^(-alpha) U(q)

over the peaks k in PEAKS (TN: a Gaussian truncated to [3 Msun / m1, 1], normalized at each m1, as in joint_fits.py;
U: uniform on the same range, user 2026-10-26: a background flat in q cannot take over the near-equal-mass continuum,
which a truncated-Gaussian background did, leaving the highest peak to model a broad low-q continuum),
by minimizing the generalized KL divergence D(p || f) (cross entropy with free normalization; peak_fits.deviance).
The peaks are only ordered, not confined to mass ranges (user, 2026-10-26): their locations are a stick-breaking
sequence in ln m1 over FIT_RANGE,

    ln mu_1 = ln m_lo + t_1 (ln m_hi - ln m_lo),    ln mu_k = ln mu_{k-1} + t_k (ln m_hi - ln mu_{k-1}),    t_k in [0, 1],

so mu_1 <= mu_2 <= ... inside FIT_RANGE, and named by their order (PEAKS); widths are relative, sigma_k = r_k mu_k with
r_k in R_BOUNDS; mu_q in (0, 1). Each draw starts from typical values (START, from the window fits) with the
amplitudes read off the draw, and from variants of them (STARTS); the lowest divergence wins. In a draw whose best fit
has a peak at an edge of FIT_RANGE or merged with the previous one (t_k at 0 or 1) or with r_k on a bound, that peak is
absent (its PeakFrac counts it; its summaries leave the draw out); the other peaks stand.
Reported per peak: mu, sigma, mu_q, sigma_q, Frac (its share of the fitted probability in FIT_RANGE), PeakFrac; for
the background: alpha, Frac. (The parameter vector keeps background mu_qb, sigma_qb slots, pinned and unused, so a
truncated-Gaussian background can be switched back on with BKG_Q_FLAT = False.) Run with macros/global_fit.sbatch (draws spread over --cpus processes).
"""
import argparse
import os
import sys
import time
import warnings
import multiprocessing as mp

import numpy as np
from scipy.optimize import least_squares

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'notebooks')); sys.path.insert(0, HERE)
import analysis as A
from joint_fits import log_tn
from peak_fits import RUN_NAMES, deviance, fmt

FIT_RANGE = (5., 150.)        # below: the steep low-mass rise; above: almost no events (both need extra components)
M_REF = 20.
# the peaks in order of mass, name: LaTeX name (named after where they sit, not confined there)
PEAKS = {'10': 'Ten', '20': 'Twenty', '35': 'ThirtyFive', '60': 'Sixty'}
R_BOUNDS = (0.02, 0.6)        # sigma_k / mu_k
BKG_Q_FLAT = True             # background uniform in q (True) or a truncated Gaussian in q (False)
# typical (mu, sigma, mu_q, sigma_q) per peak, from the window fits (2026-10-26)
START = {'10': (9.5, 1.0, 0.95, 0.11), '20': (20.5, 2.5, 0.48, 0.13), '35': (32.5, 3.5, 0.95, 0.11),
         '60': (58., 7.5, 0.55, 0.10)}
# variants of the start: (scale on every sigma, background (alpha, mu_qb, sigma_qb))
STARTS = ((1.0, (2.5, 0.6, 0.3)), (0.6, (2.5, 0.6, 0.3)), (1.0, (3.5, 0.3, 0.5)), (1.6, (1.5, 0.8, 0.2)))
NP = len(PEAKS)
COLS = [f'{p}_{k}' for k in PEAKS for p in ('lnA', 'mu', 'sigma', 'mu_q', 'sigma_q')] + \
       ['lnB', 'alpha', 'mu_qb', 'sigma_qb'] + [f'frac_{k}' for k in PEAKS] + ['frac_bkg']


def grid():
    msel = (A.m_c >= FIT_RANGE[0]) & (A.m_c <= FIT_RANGE[1])
    M, Q = np.meshgrid(A.m_c[msel], A.q_c, indexing='ij')
    return msel, M, Q, Q * M >= A.MIN_M, M * A.dlnm * A.dq


def bounds():
    """Bounds of the internal parameters: per peak (lnA, t, r, mu_q, sigma_q), then (lnB, alpha, mu_qb, sigma_qb)."""
    lo = [-60., 0., R_BOUNDS[0], 0., 0.02] * NP + [-60., -10., 0., 0.02]
    hi = [20., 1., R_BOUNDS[1], 1., 3.] * NP + [20., 10., 1., 3.]
    if BKG_Q_FLAT:                                   # pin the unused background q parameters
        lo[-2:], hi[-2:] = [0.5, 1.], [0.501, 1.001]          # (wider than the 1e-6 clip margin of the starts)
    return lo, hi


def to_phys(x):
    """Internal parameters (t_k, r_k) -> physical (mu_k, sigma_k): the ordered stick-breaking locations."""
    x = np.array(x, dtype=float)
    l0, l1 = np.log(FIT_RANGE)
    lmu = l0
    for k in range(NP):
        i = 5 * k
        lmu = lmu + x[i + 1] * (l1 - lmu)
        mu = np.exp(lmu)
        x[i + 1], x[i + 2] = mu, x[i + 2] * mu
    return x


def to_internal(mus, sigmas):
    """(t_k, r_k) for given ordered peak locations and widths."""
    l0, l1 = np.log(FIT_RANGE)
    t, prev = [], l0
    for mu in mus:
        t.append((np.log(mu) - prev) / (l1 - prev)); prev = np.log(mu)
    return t, [s / mu for s, mu in zip(sigmas, mus)]


def log_parts(x, m, q):
    """log of each peak's density and of the background's, at cells (m, q): (NP + 1, cells); x physical."""
    qmin = A.MIN_M / m
    out = []
    for k in range(NP):
        lnA, mu, sigma, mu_q, sd_q = x[5 * k:5 * k + 5]
        out.append(lnA - 0.5 * ((m - mu) / sigma) ** 2 + log_tn(q, qmin, mu_q, sd_q))
    lnB, alpha, mu_b, sd_b = x[5 * NP:]
    log_q = -np.log(1 - qmin) if BKG_Q_FLAT else log_tn(q, qmin, mu_b, sd_b)
    out.append(lnB - alpha * np.log(m / M_REF) + log_q)
    return np.array(out)


def log_model(x, m, q):
    lp = log_parts(x, m, q)
    mx = lp.max(0)
    return mx + np.log(np.exp(lp - mx).sum(0))


def fit_draw(d):
    """Fit one draw's density d (Bm_fit, Bq). Returns COLS (NaN for a peak on a bound; all NaN if every fit failed)."""
    msel, M, Q, support, area = GRID
    lo, hi = BOUNDS
    ok = support & (d > 0) & np.isfinite(d)
    m, q, y, a = M[ok], Q[ok], d[ok].astype(float), area[ok]
    ly = np.log(y)
    marg = (d * support).sum(1) * A.dq                                    # p(m1) on the fit range
    mm = A.m_c[msel]
    at = lambda v: max(np.interp(v, mm, marg), 1e-30)
    best = None
    for s_scale, (al0, mqb0, sqb0) in STARTS:
        mus = [v[0] for v in START.values()]
        t0, r0 = to_internal(mus, [v[1] * s_scale for v in START.values()])
        x0 = []
        for k, (mu0, s0, mq0, sq0) in enumerate(START.values()):
            x0 += [np.log(0.7 * at(mu0)), t0[k], r0[k], mq0, sq0]          # A_k: about the marginal at the peak
        x0 += [np.log(0.3 * at(M_REF)), al0, mqb0, sqb0]
        x0 = np.clip(x0, np.add(lo, 1e-6), np.subtract(hi, 1e-6))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = least_squares(lambda x: deviance(log_model(to_phys(x), m, q), y, ly, a), x0, bounds=(lo, hi),
                                    method='trf', x_scale='jac', max_nfev=3000)
        except (ValueError, FloatingPointError):
            continue
        if best is None or res.cost < best.cost:
            best = res
    out = np.full(len(COLS), np.nan)
    if best is None:
        return out
    xi = best.x
    x = to_phys(xi)
    parts = np.exp(log_parts(x, M[support], Q[support])) * area[support]
    fr = parts.sum(1) / parts.sum()
    for k in range(NP):            # absent: at a range edge or merged (t on a bound), or width on a bound
        i = 5 * k
        if (xi[i + 1] < 1e-4 or xi[i + 1] > 1 - 1e-4
                or np.isclose(xi[i + 2], lo[i + 2], rtol=1e-4) or np.isclose(xi[i + 2], hi[i + 2], rtol=1e-4)):
            x[i:i + 5] = np.nan; fr[k] = np.nan
    out[:len(x)] = x; out[len(x):] = fr
    return out


def _init(dens_shared):
    global GRID, BOUNDS, DENS
    GRID, BOUNDS, DENS = grid(), bounds(), dens_shared


def _work(i):
    return fit_draw(DENS[i])


def model_params(row):
    """The fitted parameters of one draw (a row of the results), with absent peaks switched off (zero amplitude,
    harmless placeholders for their NaN shape parameters)."""
    x = np.array(row[:5 * NP + 4], dtype=float)
    for k in range(NP):
        if np.isnan(x[5 * k + 1]):
            x[5 * k:5 * k + 5] = [-np.inf, 1., 1., 0.5, 1.]
    return x


def diagnostic(run, dens, P, path, n_model=200):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import plot_style
    plot_style.apply()
    msel, M, Q, support, area = grid()
    mm = A.m_c[msel]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={'width_ratios': [1.2, 1]})
    ax = axes[0]
    marg = (dens * support).sum(2) * A.dq
    lo, md, hi = np.percentile(marg, [5, 50, 95], 0)
    ax.fill_between(mm, lo, hi, color='C0', alpha=0.3, lw=0); ax.plot(mm, md, color='C0', label=r'$p(m_1)$ (all $q$)')
    good = np.where(np.isfinite(P[:, 5 * NP]))[0][:n_model]
    comps = []
    for j in good:
        lp = log_parts(model_params(P[j]), M, Q)
        comps.append((np.exp(lp) * support).sum(2) * A.dq)                         # (NP + 1, Bm_fit)
    comps = np.array(comps)
    ax.plot(mm, np.median(comps.sum(1), 0), color='k', label='global fit')
    for k, name in enumerate(list(PEAKS) + ['bkg']):
        ax.plot(mm, np.median(comps[:, k], 0), color='0.4' if name == 'bkg' else f'C{k + 1}', ls=':' if name == 'bkg' else '--',
                lw=1, label='power law' if name == 'bkg' else rf'peak {name}')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(FIT_RANGE)
    ax.set_ylim(np.percentile(lo, 2) / 3, hi.max() * 3)
    ax.set_xlabel(r'$m_1\,[M_\odot]$'); ax.set_ylabel(r'$p(m_1)\,[M_\odot^{-1}]$')
    ax.legend(ncol=2, **dict(plot_style.LEG_KW, fontsize=11))
    ax = axes[1]
    med = np.where(support, np.median(dens, 0), np.nan); vmax = np.nanmax(med)
    medges = np.exp(A.lnm_edges[np.r_[np.where(msel)[0], np.where(msel)[0][-1] + 1]])
    pc = ax.pcolormesh(medges, A.q_edges, med.T, norm=LogNorm(vmin=vmax * 1e-3, vmax=vmax), cmap='viridis',
                       shading='flat', rasterized=True)
    model = np.where(support, np.median([np.exp(log_model(model_params(P[j]), M, Q)) for j in good[:50]], 0), np.nan)
    ax.contour(M, Q, np.log10(model), levels=np.log10(vmax) - np.array([2., 1., 0.5]), colors='w', linewidths=1)
    ax.set_xscale('log'); ax.set_xlim(FIT_RANGE); ax.set_ylim(0, 1); ax.tick_params(which='both', direction='out')
    ax.set_xlabel(r'$m_1\,[M_\odot]$'); ax.set_ylabel(r'$q$')
    cb = fig.colorbar(pc, ax=ax); cb.set_label(r'median $p(m_1, q)$; white: global fit')
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    print('saved', path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='*', default=list(RUN_NAMES))
    ap.add_argument('--cpus', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 1)))
    ap.add_argument('--draws', type=int, default=0, help='fit only every n-th draw (a quick test); 0: all')
    args = ap.parse_args()
    lines = ['% Global (m1, q) fit per draw: four Gaussian x truncated-Gaussian peaks + power law x truncated Gaussian,'
             f' m1 in {FIT_RANGE} Msun, minimizing the KL divergence; median and 90% over draws with the peak'
             ' (macros/global_fit.py)']
    msel = grid()[0]
    for run in args.runs:
        r = A.load_light(run)
        dens = (r.p_mass[:, msel] / A.m_c[None, msel, None]).astype(np.float32)
        del r
        if args.draws:
            dens = dens[::args.draws]
        t0 = time.time()
        # spawn, not fork: jax (imported by load_light) is multithreaded; the workers never import it
        with mp.get_context('spawn').Pool(args.cpus, initializer=_init, initargs=(dens,)) as pool:
            P = np.array(pool.map(_work, range(len(dens)), chunksize=4))
        print(f'{run}: {len(dens)} draws in {time.time() - t0:.0f} s on {args.cpus} processes', flush=True)
        np.savez(os.path.join(REPO, 'results', 'post_processing', f'{run}_global_fit.npz'), P=P, columns=np.array(COLS))
        suffix = RUN_NAMES.get(run, run.replace('_', ''))
        lines.append(f'% {run}: {len(P)} draws, {int(np.isnan(P[:, 5 * NP]).sum())} with every fit failed')
        for k, (name, tex) in enumerate(PEAKS.items()):
            c = 5 * k
            absent = int(np.isnan(P[:, c + 1]).sum())
            print(f'  peak {name} (number {k + 1} in mass order): mu {fmt(P[:, c + 1])}, sigma {fmt(P[:, c + 2])}, mu_q {fmt(P[:, c + 3])}, '
                  f'sigma_q {fmt(P[:, c + 4])}, frac {fmt(P[:, 5 * NP + 4 + k])}  [absent in {absent} of {len(P)}]',
                  flush=True)
            lines.append(f'% peak {name} (number {k + 1} in mass order): absent in {absent} of {len(P)} draws')
            for par, col in (('Mu', c + 1), ('Sigma', c + 2), ('QMu', c + 3), ('QSigma', c + 4),
                             ('Frac', 5 * NP + 4 + k)):
                lines.append(rf'\newcommand{{\global{tex}{par}{suffix}}}{{{fmt(P[:, col])}}}')
            lines.append(rf'\newcommand{{\global{tex}PeakFrac{suffix}}}{{{1 - absent / len(P):.3f}}}')
        b = 5 * NP
        print(f'  background: alpha {fmt(P[:, b + 1])}, frac {fmt(P[:, -1])}', flush=True)
        bkg = (('Alpha', b + 1), ('Frac', len(COLS) - 1)) + (() if BKG_Q_FLAT else (('QMu', b + 2), ('QSigma', b + 3)))
        for par, col in bkg:
            lines.append(rf'\newcommand{{\globalBkg{par}{suffix}}}{{{fmt(P[:, col])}}}')
        diagnostic(run, dens, P, os.path.join(REPO, 'plots', f'{run}_global_fit.pdf'))
    if not args.draws:
        with open(os.path.join(HERE, 'global_fit.tex'), 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print('wrote', os.path.join(HERE, 'global_fit.tex'))


if __name__ == '__main__':
    main()
