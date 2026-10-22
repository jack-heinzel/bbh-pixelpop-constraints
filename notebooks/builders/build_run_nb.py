"""Write notebooks/<run>.ipynb: diagnostics, mass and spin results of one run in results/<run>/.

Usage:  python notebooks/builders/build_run_nb.py ppc_nrsur_k5_short

Execute it on CPU, pinned (it imports jax through pixelpop; the notebook sets the single-thread XLA flags itself):
    taskset -c <4 cores> jupyter nbconvert --to notebook --execute --inplace notebooks/<run>.ipynb
Reads only the small datasets of the chain (not the white-noise sites), so it is safe next to GPU jobs.
"""
import os
import sys

import nbformat as nbf

run = sys.argv[1]
catalog = run.split('_')[1]
here = os.path.dirname(os.path.abspath(__file__))
repo = os.path.abspath(os.path.join(here, '..', '..'))      # this machine's checkout
out = os.path.join(here, '..', f'{run}.ipynb')

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
md(rf"""# `{run}`

The weakly modelled population of `model.py`:

$$p(\ln m_1, q, \chi_\mathrm{{eff}}, \chi_p) = p_\mathrm{{mass}}(\ln m_1, q)\; p(\chi_\mathrm{{eff}}, \chi_p \mid \ln m_1, q)$$

- **Mass:** PixelPop's Matérn field on 100 × 100 bins in $(\ln m_1, q)$, $m_1 \in [3, 200]\,M_\odot$, times the
  $m_2 > 3\,M_\odot$ window.
- **Spins:** a Matérn log density on 100 × 100 bins in $(\chi_\mathrm{{eff}}, \chi_p)$, on the physical region
  $\chi_p < \chi_{{p,\max}}(\chi_\mathrm{{eff}}, q)$. Its shape changes with mass through the lowest $k \times k$ mass
  modes. There are no gates, no channels and no 2G template.

Catalog `{catalog}`. The cells below read the chain as it is on disk, so re-running the notebook picks up a chain that
has grown or finished. **Caveats** are printed with the chain summary: chain length, number of chains, and whether the
run finished."""),
code(r"""import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '1')
import sys
REPO = '""" + repo + r"""'
sys.path.insert(0, REPO)
import glob, pickle
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
from pixelpop.models.gwpop_models import PlanckWindow_SecondaryMass
%matplotlib inline
import plot_style
plot_style.apply()
from model import SPIN_REF_MASSES

RUN = '""" + run + r"""'
CATALOG = '""" + catalog + r"""'
RUN_DIR = os.path.join(REPO, 'results', RUN)
MIN_M, MAX_M = 3., 200.
lnm_edges = np.linspace(np.log(MIN_M), np.log(MAX_M), 101)
q_edges = np.linspace(0., 1., 101)
e_edges = np.linspace(-1., 1., 101)
p_edges = np.linspace(0., 1., 101)
centres = lambda e: 0.5 * (e[1:] + e[:-1])
lnm_c, q_c, e_c, p_c = map(centres, (lnm_edges, q_edges, e_edges, p_edges))
dlnm, dq, de, dp = (np.diff(e)[0] for e in (lnm_edges, q_edges, e_edges, p_edges))"""),
md("## Chain summary and convergence"),
code(r"""SKIP = ('_eigenbasis_sites', '_spin_sites', '_spin0_sites', '_spin1_sites', '_gauss_sites')   # white noise: not needed
files = sorted(glob.glob(os.path.join(RUN_DIR, 'chain_*_samples.h5')))
chains = []
for f in files:
    with h5py.File(f, 'r') as h:
        chains.append({k: h[k][()] for k in h if k not in SKIP})
n_draws = min(len(c['log_likelihood']) for c in chains)
S = {k: np.stack([c[k][:n_draws] for c in chains]) for k in chains[0]}     # (chains, draws, ...)
flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in S.items()}
print(f'{RUN}: {len(files)} chain(s), {n_draws} draws each')
for f in files:
    meta = f.replace('_samples.h5', '_metadata.txt')
    if os.path.exists(meta):
        print(open(meta).read())
for log in sorted(glob.glob(os.path.join(REPO, 'results', 'logs', f'{RUN}.log'))
                  + glob.glob(os.path.join(REPO, 'results', 'logs', f'{RUN}_long_chain*.log'))):
    tail = open(log, errors='replace').read().replace('\r', '\n').strip().splitlines()[-1]
    print(f'last line of {os.path.basename(log)}:', tail[:160])"""),
code(r"""SPIN_MODEL = 'split' if 'lnsigma_spin1' in S else 'gauss' if 'gauss_offsets' in S else 'field'
print('spin model:', SPIN_MODEL)
HYPER = [k for k in ('lnsigma', 'log_nu_spde', 'log_ranges', 'lnsigma_spin', 'log_nu_spin', 'log_ranges_spin',
                     'lnsigma_spin0', 'log_nu_spin0', 'log_ranges_spin0', 'lnsigma_spin1', 'log_nu_spin1',
                     'log_ranges_spin1', 'gauss_offsets', 'lnsigma_gauss', 'log_nu_gauss', 'log_ranges_gauss', 'lamb',
                     'log_likelihood', 'log_likelihood_variance', 'log_likelihood_correction', 'selection_neff')
         if k in S]
GP = ('mu_eff', 'mu_p', 'ln_sd_eff', 'ln_sd_p', 'atanh_rho')
LABELS = {'log_ranges': ('ln m1', 'q'), 'log_ranges_spin': ('ln m1 mode', 'q mode', 'chi_eff', 'chi_p'),
          'log_ranges_spin0': ('chi_eff', 'chi_p'), 'log_ranges_spin1': ('ln m1 mode', 'q mode', 'chi_eff', 'chi_p'),
          'gauss_offsets': GP, 'lnsigma_gauss': GP,
          'log_ranges_gauss': tuple(f'{g}, {a}' for g in GP for a in ('ln m1', 'q'))}
rows = []
for k in HYPER:
    x = S[k].reshape(S[k].shape[0], S[k].shape[1], -1)
    for j in range(x.shape[2]):
        name = k if x.shape[2] == 1 else f'{k}[{LABELS.get(k, range(x.shape[2]))[j]}]'
        v = x[:, :, j]
        rows.append(dict(parameter=name, median=np.median(v), lo5=np.percentile(v, 5), hi95=np.percentile(v, 95),
                         ess=float(effective_sample_size(v)),
                         rhat=float(split_gelman_rubin(v)) if v.shape[1] >= 4 else np.nan))
emin = S['event_neff'].min(-1)
rows.append(dict(parameter='min event Neff', median=np.median(emin), lo5=np.percentile(emin, 5),
                 hi95=np.percentile(emin, 95), ess=float(effective_sample_size(emin)), rhat=float(split_gelman_rubin(emin))))
summary = pd.DataFrame(rows).set_index('parameter')
print('ESS and split R-hat are over all chains;'
      + (' with one chain, R-hat compares its two halves only.' if len(files) == 1 else ''))
summary.round(3)"""),
code(r"""fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
for i, c in enumerate(range(S['log_likelihood'].shape[0])):
    for ax, y in zip(axes, (S['log_likelihood'][c], S['log_likelihood_variance'][c], emin[c], S['selection_neff'][c])):
        ax.plot(y, lw=0.8, color=f'C{c}')
for ax, lab in zip(axes, (r'$\ln \mathcal{L}$', r'$\mathrm{Var}(\ln \mathcal{L})$', r'min event $N_\mathrm{eff}$',
                          r'selection $N_\mathrm{eff}$')):
    ax.set_ylabel(lab)
axes[-1].set_xlabel('draw')
fig.tight_layout()
plot_style.save(fig, f'{RUN}_traces')"""),
md(r"""## Events with the smallest effective sample size

The likelihood tapers events below $N_\mathrm{eff} = 100$; these are the events closest to that floor, by their median
over the draws."""),
code(r"""DATA_DIRS = {'gwtc5': os.path.join(REPO, 'data', 'gwtc5_ifar1_chieffchip', 'data')}
directory = DATA_DIRS.get(CATALOG, f'/home/jack.heinzel/public_html/mass-gap-investigation/data/{CATALOG}_o1o4b_chieffchip/data')
import re
with open(os.path.join(directory, 'posteriors.pkl'), 'rb') as f:
    names = [re.findall(r'GW\d{6}(?:_\d{6})?', os.path.basename(str(n)))[-1] for n in pickle.load(f)['event_names']]
if len(names) != flat['event_neff'].shape[1]:
    print(f'event names ({len(names)}) do not match the chain ({flat["event_neff"].shape[1]} events): not labelled')
    names = [str(i) for i in range(flat['event_neff'].shape[1])]
neff = pd.DataFrame({'median': np.median(flat['event_neff'], 0), 'lo5': np.percentile(flat['event_neff'], 5, 0),
                     'frac < 100': (flat['event_neff'] < 100).mean(0)}, index=names)
neff.sort_values('median').head(12).round(2)"""),
md(r"""## Masses

The saved `log_p_mass` is the mass field normalized on the grid **without** the $m_2 > 3\,M_\odot$ window, which acts
only on the event and injection weights. Here the window is added on bin centres, and the density renormalized,
before any marginal is taken."""),
code(r"""logW = np.asarray(PlanckWindow_SecondaryMass({'log_mass_1': lnm_c[:, None], 'mass_ratio': q_c[None]}, MIN_M, 0.))
lp = flat['log_p_mass'] + logW[None]
lp -= np.log(np.exp(lp - lp.max((1, 2), keepdims=True)).sum((1, 2), keepdims=True) * dlnm * dq) + lp.max((1, 2), keepdims=True)
p_mass = np.exp(lp)                                   # (draws, ln m1, q), density per (d ln m1 dq), windowed
p_lnm = p_mass.sum(2) * dq                            # p(ln m1)
p_q = p_mass.sum(1) * dlnm                            # p(q)
print('window: fraction of bins kept', (logW > -1).mean(), '; normalization check', np.allclose(p_lnm.sum(1) * dlnm, 1))"""),
code(r"""def band(x, axis=0):
    return np.percentile(x, [5, 50, 95], axis=axis)

N_FINE = 50
m_edges = np.exp(lnm_edges)
m_fine = np.concatenate([np.linspace(a, b, N_FINE, endpoint=False) for a, b in zip(m_edges[:-1], m_edges[1:])])
idx_fine = np.repeat(np.arange(100), N_FINE)
p_m1 = p_lnm[:, idx_fine] / m_fine[None]               # p(m1) = p(ln m1) / m1, smooth within each bin
lo, med, hi = band(p_m1)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
ax = axes[0]
ax.fill_between(m_fine, lo, hi, color='C0', alpha=0.3, lw=0)
ax.plot(m_fine, med, color='C0')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(MIN_M, MAX_M); ax.set_ylim(np.percentile(lo, 1) / 3, hi.max() * 3)
ax.set_xlabel(r'$m_1\,[M_\odot]$'); ax.set_ylabel(r'$p(m_1)\,[M_\odot^{-1}]$')
ax = axes[1]
lo, med, hi = band(np.concatenate([p_q, p_q[:, -1:]], 1), 0)
ax.fill_between(q_edges, lo, hi, step='post', color='C0', alpha=0.3, lw=0)
ax.step(q_edges, med, where='post', color='C0')
ax.set_yscale('log'); ax.set_xlim(0, 1)
ax.set_xlabel(r'$q$'); ax.set_ylabel(r'$p(q)$')
fig.tight_layout()
plot_style.save(fig, f'{RUN}_mass_marginals')"""),
code(r"""med2 = np.median(p_mass, 0)
fig, ax = plt.subplots(figsize=(7, 5.5))
vmax = np.log10(med2.max()); vmin = vmax - 4
pc = ax.pcolormesh(np.exp(lnm_edges), q_edges, np.log10(np.maximum(med2, 1e-300)).T, vmin=vmin, vmax=vmax,
                   cmap='viridis', shading='flat', rasterized=True)
ax.plot(np.exp(lnm_edges), np.clip(MIN_M / np.exp(lnm_edges), 0, 1), color='w', lw=1, ls='--')
ax.set_xscale('log'); ax.set_xlim(MIN_M, MAX_M); ax.set_ylim(0, 1)
ax.tick_params(which='both', direction='out')
ax.set_xlabel(r'$m_1\,[M_\odot]$'); ax.set_ylabel(r'$q$')
cb = fig.colorbar(pc, ax=ax); cb.set_label(r'$\log_{10}$ median $p(\ln m_1, q)$')
fig.tight_layout()
plot_style.save(fig, f'{RUN}_mass_joint')"""),
md(r"""### The primary-mass distribution at different mass ratios

$p(m_1 \mid q \in [q_\mathrm{lo}, q_\mathrm{hi}])$ from the windowed field: the field is summed over the $q$ bins in
each range, normalized within the range, and drawn as a density in linear $m_1$ ($p(m_1) = p(\ln m_1)/m_1$ within each
bin). The legend gives each range's share of the population, $P(q \in \mathrm{range})$. The dotted line marks the
smallest $m_1$ allowed by $m_2 > 3\,M_\odot$ at the range's lower edge; below it only the upper part of the range
contributes."""),
code(r"""Q_RANGES = [(0.1, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
fig, axes = plt.subplots(len(Q_RANGES), 1, figsize=(8, 2.6 * len(Q_RANGES)), sharex=True)
for j, ((qlo, qhi), ax) in enumerate(zip(Q_RANGES, axes)):
    sel = (q_edges[:-1] >= qlo - 1e-9) & (q_edges[1:] <= qhi + 1e-9)
    p_lnm_q = p_mass[:, :, sel].sum(2) * dq                       # (draws, ln m1): p(ln m1, q in range)
    share = p_lnm_q.sum(1) * dlnm
    y = (p_lnm_q / share[:, None])[:, idx_fine] / m_fine[None]
    lo, med, hi = band(y)
    ax.fill_between(m_fine, lo, hi, color=f'C{j}', alpha=0.3, lw=0)
    ax.plot(m_fine, med, color=f'C{j}', label=rf'${qlo:g} < q < {qhi:g}$; $P = {np.median(share):.2f}$')
    ax.axvline(MIN_M / qlo, color='0.5', ls=':', lw=1)
    ax.set_yscale('log'); ax.set_ylim(max(np.percentile(lo, 1), med.max() * 1e-6), hi.max() * 3)
    ax.set_ylabel(r'$p(m_1 \mid q)\,[M_\odot^{-1}]$')
    ax.legend(loc='upper right', **plot_style.LEG_KW)
axes[-1].set_xscale('log'); axes[-1].set_xlim(MIN_M, MAX_M); axes[-1].set_xlabel(r'$m_1\,[M_\odot]$')
fig.tight_layout()
plot_style.save(fig, f'{RUN}_m1_given_q')"""),
md(r"""## Spins: the population marginal

The population's $(\chi_\mathrm{eff}, \chi_p)$ distribution is the spin conditional averaged over the mass distribution.
The saved `log_p_spin_marginal` averages with the mass weights **before** the $m_2 > 3\,M_\odot$ window. The data
never see the windowed-out corner ($q < 3/m_1$), so the field is unconstrained there. It carries a sizeable,
prior-driven share of the unwindowed mass probability (printed below), and with it that corner's spins. So the
marginal is rebuilt here from the saved spin field (`_spin_sites` and the spin hyperparameters, exactly as `model.py`
builds it, on every draw; `notebooks/analysis.py`, cached in `results/post_processing/for_plotting/`) and averaged
with the **windowed** mass weights. The rebuild is checked against the saved marginal, using the unwindowed weights;
differences of order $10^{-3}$ come from the chain being stored in float32."""),
code(r"""sys.path.insert(0, os.path.join(REPO, 'notebooks'))
import analysis as A
r = RES = A.load_run(RUN)        # spins: every draw, cached in results/post_processing/for_plotting/<RUN>/ (RES: r is reused below)
print('spins', r.spin_source)
P_raw, P_win, MOM = r.P_raw, r.P_win, r.MOM          # MOM: (draws, ln m1, q, 4)
tv_win = 0.5 * np.abs(P_win - P_raw).sum((1, 2))
removed = r.window_removed
print(f'rebuild vs saved marginal, total variation: max {r.rebuild_tv:.1e}')
print('mass probability the window removes: median %.3f, 90%% [%.3f, %.3f]' % tuple(np.percentile(removed, [50, 5, 95])))
print('windowed vs unwindowed spin marginal, total variation: median %.3f, 90%% [%.3f, %.3f]'
      % tuple(np.percentile(tv_win, [50, 5, 95])))"""),
code(r"""def spin_1d(P):
    return P.sum(2) / de, P.sum(1) / dp

fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
med2 = np.median(P_win, 0) / (de * dp)
ax = axes[0]
vmax = np.log10(med2.max()); vmin = vmax - 3
pc = ax.pcolormesh(e_edges, p_edges, np.log10(np.maximum(med2, 1e-300)).T, vmin=vmin, vmax=vmax, cmap='viridis',
                   shading='flat', rasterized=True)
ax.tick_params(which='both', direction='out')
ax.set_xlabel(r'$\chi_\mathrm{eff}$'); ax.set_ylabel(r'$\chi_p$')
cb = fig.colorbar(pc, ax=ax); cb.set_label(r'$\log_{10}$ median $p(\chi_\mathrm{eff}, \chi_p)$')
for P, col, lab in ((P_win, 'C0', r'$m_2 > 3\,M_\odot$ window'), (P_raw, 'C1', 'unwindowed (saved)')):
    for ax, y, edges in zip(axes[1:], spin_1d(P), (e_edges, p_edges)):
        lo, med, hi = band(np.concatenate([y, y[:, -1:]], 1))
        ax.fill_between(edges, lo, hi, step='post', color=col, alpha=0.3 if col == 'C0' else 0.15, lw=0)
        ax.step(edges, med, where='post', color=col, ls='-' if col == 'C0' else '--', label=lab)
for ax, edges, lab in ((axes[1], e_edges, r'\chi_\mathrm{eff}'), (axes[2], p_edges, r'\chi_p')):
    ax.set_xlim(edges[0], edges[-1]); ax.set_ylim(0)
    ax.set_xlabel(rf'${lab}$'); ax.set_ylabel(rf'$p({lab})$')
axes[1].set_xlim(-0.6, 0.8)
axes[2].legend(**plot_style.LEG_KW)
fig.tight_layout()
plot_style.save(fig, f'{RUN}_spin_marginal')

def moments(P):
    p_e, p_p = spin_1d(P)
    mean_e = (p_e * e_c).sum(1) * de
    return pd.DataFrame({'mean chi_eff': mean_e,
                         'sd chi_eff': np.sqrt((p_e * e_c ** 2).sum(1) * de - mean_e ** 2),
                         'P(chi_eff < 0)': (p_e * (e_c < 0)).sum(1) * de,
                         'mean chi_p': (p_p * p_c).sum(1) * dp})
fmt = lambda df: df.quantile([0.05, 0.5, 0.95]).T.apply(lambda r: f'{r[0.5]:.3f} [{r[0.05]:.3f}, {r[0.95]:.3f}]', axis=1)
print('population spin moments, median [5%, 95%] over draws')
pd.DataFrame({'windowed': fmt(moments(P_win)), 'unwindowed (saved)': fmt(moments(P_raw))})"""),
md(r"""## Spins as a function of mass

`log_p_spin_q1_m` holds the spin conditional $p(\chi_\mathrm{eff}, \chi_p \mid m_1, q)$ in the top $q$ bin
($q \approx 1$) at """ + ', '.join(f'{m:g}' for m in (10., 18., 35., 60.)) + r"""$\,M_\odot$, the masses saved by the
model (`model.SPIN_REF_MASSES`). These slices show how much the spin distribution is allowed, and needs, to change with
mass. At $q \approx 1$ the physical region is $\chi_p < \sqrt{1 - \max(0, 2|\chi_\mathrm{eff}| - 1)^2}$."""),
code(r"""Pm = np.exp(flat['log_p_spin_q1_m'])                   # (draws, n_masses, chi_eff, chi_p), probability per bin
Pm /= Pm.sum((2, 3), keepdims=True)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for j, m in enumerate(SPIN_REF_MASSES):
    for ax, y, edges in ((axes[0], Pm[:, j].sum(2) / de, e_edges), (axes[1], Pm[:, j].sum(1) / dp, p_edges)):
        lo, med, hi = band(np.concatenate([y, y[:, -1:]], 1))
        ax.fill_between(edges, lo, hi, step='post', color=f'C{j}', alpha=0.3, lw=0)
        ax.step(edges, med, where='post', color=f'C{j}', label=rf'$m_1 = {m:g}\,M_\odot$')
axes[0].set_xlim(-0.6, 0.8); axes[1].set_xlim(0, 1)
for ax, lab in zip(axes, (r'\chi_\mathrm{eff}', r'\chi_p')):
    ax.set_ylim(0); ax.set_xlabel(rf'${lab}$'); ax.set_ylabel(rf'$p({lab} \mid m_1, q \approx 1)$')
axes[0].legend(**plot_style.LEG_KW)
fig.tight_layout()
plot_style.save(fig, f'{RUN}_spin_given_m1')

rows = {}
for j, m in enumerate(SPIN_REF_MASSES):
    pe_j = Pm[:, j].sum(2) / de; pp_j = Pm[:, j].sum(1) / dp
    mean_e = (pe_j * e_c).sum(1) * de
    rows[f'{m:g} Msun'] = {
        'mean chi_eff': np.percentile(mean_e, [5, 50, 95]),
        'sd chi_eff': np.percentile(np.sqrt((pe_j * e_c ** 2).sum(1) * de - mean_e ** 2), [5, 50, 95]),
        'P(chi_eff < 0)': np.percentile((pe_j * (e_c < 0)).sum(1) * de, [5, 50, 95]),
        'mean chi_p': np.percentile((pp_j * p_c).sum(1) * dp, [5, 50, 95])}
table = pd.DataFrame({m: {k: f'{v[1]:.3f} [{v[0]:.3f}, {v[2]:.3f}]' for k, v in r.items()} for m, r in rows.items()})
print('median [5%, 95%] over draws, at q ~ 1')
table"""),
code(r"""# difference between the heaviest and lightest slice, per draw: is the mass dependence resolved?
pe = Pm.sum(3) / de
mean_e = (pe * e_c).sum(2) * de                          # (draws, n_masses)
d = mean_e[:, -1] - mean_e[:, 0]
print(f'mean chi_eff at {SPIN_REF_MASSES[-1]:g} minus at {SPIN_REF_MASSES[0]:g} Msun (q ~ 1): '
      f'median {np.median(d):.3f}, 90% [{np.percentile(d, 5):.3f}, {np.percentile(d, 95):.3f}], P(> 0) {(d > 0).mean():.2f}')
sd = np.sqrt((pe * e_c ** 2).sum(2) * de - mean_e ** 2)
d = sd[:, -1] - sd[:, 0]
print(f'sd chi_eff, same difference: median {np.median(d):.3f}, 90% [{np.percentile(d, 5):.3f}, {np.percentile(d, 95):.3f}], '
      f'P(> 0) {(d > 0).mean():.2f}')"""),
md(r"""## Spin moments as a function of primary mass

The mean and standard deviation of $\chi_\mathrm{eff}$ and $\chi_p$ under the spin conditional
$p(\chi_\mathrm{eff}, \chi_p \mid m_1, q)$, in every $m_1$ bin, from the rebuilt field. Bands are 90% over draws. These
are properties of the conditional at fixed $(m_1, q)$, so they need no mass window; bins with $q\,m_1 < 3\,M_\odot$
are dropped because the population has no support there. Where the data are sparse the bands widen towards the
zero-field limit, a flat density on the physical region: mean $\chi_\mathrm{eff} = 0$, sd $\chi_\mathrm{eff} = 0.51$–$0.53$,
mean $\chi_p = 0.45$–$0.47$, sd $\chi_p = 0.27$–$0.28$ (for $q = 0.3$–$1$). The field's prior spreads draws around that
limit, not onto it.

First at $q \approx 1$ (the top $q$ bin, $0.99 < q < 1$):"""),
code(r"""MOM_LABELS = [r'mean $\chi_\mathrm{eff}$', r'sd $\chi_\mathrm{eff}$', r'mean $\chi_p$', r'sd $\chi_p$']
m_c = np.exp(lnm_c)

def moment_panels(axes, iq, col, label=None):
    ok = q_c[iq] * m_c >= MIN_M
    for k, ax in enumerate(axes):
        lo, med, hi = band(MOM[:, ok, iq, k])
        ax.fill_between(m_c[ok], lo, hi, color=col, alpha=0.3, lw=0)
        ax.plot(m_c[ok], med, color=col, label=label)

fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
moment_panels(axes.ravel(), -1, 'C0')
for ax, lab in zip(axes.ravel(), MOM_LABELS):
    ax.set_ylabel(lab); ax.set_xscale('log'); ax.set_xlim(MIN_M, MAX_M)
axes[0, 0].axhline(0, color='0.5', lw=0.8, ls=':')
for ax in axes[1]:
    ax.set_xlabel(r'$m_1\,[M_\odot]$')
fig.tight_layout()
plot_style.save(fig, f'{RUN}_spin_moments_q1')

rows = []
for m in (8., 10., 15., 20., 30., 40., 60., 100.):
    im = int(np.argmin(np.abs(m_c - m)))
    rows.append({'m1 [Msun]': f'{m_c[im]:.1f}', **{lab: '%.3f [%.3f, %.3f]' % (np.median(MOM[:, im, -1, k]),
                                                                           *np.percentile(MOM[:, im, -1, k], [5, 95]))
                 for k, lab in enumerate(('mean chi_eff', 'sd chi_eff', 'mean chi_p', 'sd chi_p'))}})
print('q ~ 1: median [5%, 95%] over draws')
pd.DataFrame(rows).set_index('m1 [Msun]')"""),
md(r"""Then in $q$ slices, one row per slice (the $q$ bin nearest each value), so the bands do not overlap. A slice's
curve starts at $m_1 = 3\,M_\odot / q$."""),
code(r"""Q_SLICES = (0.3, 0.5, 0.7, 1.0)
fig, axes = plt.subplots(len(Q_SLICES), 4, figsize=(18, 3.2 * len(Q_SLICES)), sharex=True, sharey='col')
for r, qv in enumerate(Q_SLICES):
    iq = int(np.argmin(np.abs(q_c - qv)))
    moment_panels(axes[r], iq, f'C{r}', label=rf'$q = {q_c[iq]:.3f}$')
    axes[r, 0].legend(loc='lower left', **plot_style.LEG_KW)
    axes[r, 0].axhline(0, color='0.5', lw=0.8, ls=':')
for c, lab in enumerate(MOM_LABELS):
    axes[-1, c].set_xlabel(r'$m_1\,[M_\odot]$')
    for r in range(len(Q_SLICES)):
        axes[r, c].set_ylabel(lab)
axes[0, 0].set_xscale('log'); axes[0, 0].set_xlim(MIN_M, MAX_M)
fig.tight_layout()
plot_style.save(fig, f'{RUN}_spin_moments_q_slices')"""),
md(r"""## Spin moments given primary mass alone

The moments above are of the conditional at fixed $(m_1, q)$. The population at a given $m_1$ mixes all mass ratios,
with the windowed $p(q \mid m_1)$, so its spread has two parts (law of total variance):
$$\mathrm{sd}^2(\chi \mid m_1) = \underbrace{\mathrm{E}_q\!\left[\mathrm{sd}^2(\chi \mid m_1, q)\right]}_{\text{within}^2}
+ \underbrace{\mathrm{Var}_q\!\left[\mathrm{E}(\chi \mid m_1, q)\right]}_{\text{between}^2}.$$
*Within* is the width at fixed $q$; *between* is the spread of the per-$q$ means. A broad $p(\chi_\mathrm{eff} \mid m_1)$
at some mass can come from either, which the $q \approx 1$ slice alone does not show."""),
code(r"""Q = A.q_averaged_moments(RES)
fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
for row, (name, lab) in enumerate((('chi_eff', r'\chi_\mathrm{eff}'), ('chi_p', r'\chi_p'))):
    for ax, key, ylab in ((axes[row, 0], 'mean', rf'mean ${lab}$'), (axes[row, 1], 'sd', rf'sd ${lab}$')):
        lo, med, hi = band(Q[name][key])
        ax.fill_between(m_c, lo, hi, color='C0', alpha=0.3, lw=0)
        ax.plot(m_c, med, color='C0', label=r'given $m_1$')
        mom = MOM[:, :, -1, {'mean': 0, 'sd': 1}[key] + 2 * row]
        ax.plot(m_c, np.median(mom, 0), color='C1', ls='--', label=r'at $q \approx 1$ (median)')
        ax.set_ylabel(ylab)
axes[0, 0].axhline(0, color='0.5', lw=0.8, ls=':')
axes[0, 1].legend(**plot_style.LEG_KW)
for ax in axes.ravel():
    ax.set_xscale('log'); ax.set_xlim(MIN_M, MAX_M)
for ax in axes[1]:
    ax.set_xlabel(r'$m_1\,[M_\odot]$')
fig.tight_layout()
plot_style.save(fig, f'{RUN}_spin_moments_given_m1')

rows = []
for m in (8., 12., 15., 18., 20., 25., 30., 35., 45., 60., 80., 100.):
    im = int(np.argmin(np.abs(m_c - m)))
    rows.append({'m1 [Msun]': f'{m_c[im]:.1f}', **{f'{k} chi_eff': A.fmt(Q['chi_eff'][k][:, im])
                                                  for k in ('mean', 'sd', 'within', 'between')}})
print('chi_eff given m1 (averaged over the windowed p(q | m1)): median [5%, 95%] over draws')
pd.DataFrame(rows).set_index('m1 [Msun]')"""),
md(r"""Where the $\chi_\mathrm{eff}$ distribution is broader or narrower than at the $35\,M_\odot$ peak: per-draw differences
of sd($\chi_\mathrm{eff} \mid m_1$) from its value at the reference mass, then the same for its *within* and *between*
parts. The panel splits sd into the two parts across $m_1$."""),
code(r"""fig, ax = plt.subplots(figsize=(8, 4.8))
for key, col, ls, lab in (('sd', 'C0', '-', 'total'), ('within', 'C2', '--', r'within $q$'),
                          ('between', 'C3', ':', r'between $q$ (spread of means)')):
    lo, med, hi = band(Q['chi_eff'][key])
    ax.fill_between(m_c, lo, hi, color=col, alpha=0.2, lw=0)
    ax.plot(m_c, med, color=col, ls=ls, label=lab)
ax.axvline(A.WIDTH_REF, color='0.5', lw=0.8, ls=':')
ax.set_xscale('log'); ax.set_xlim(MIN_M, MAX_M); ax.set_ylim(0)
ax.set_xlabel(r'$m_1\,[M_\odot]$'); ax.set_ylabel(r'sd $\chi_\mathrm{eff}$ given $m_1$')
ax.legend(**plot_style.LEG_KW)
fig.tight_layout()
plot_style.save(fig, f'{RUN}_chieff_width_split')
pd.concat({k: A.width_contrasts(Q, key=k) for k in ('sd', 'within', 'between')}, axis=1)"""),
md(r"""## Mass ratio and spin width together

One figure on a shared $m_1$ axis (`analysis.summary_figure`): (1) $p(m_1)$: the 90% interval over all $q$ (black lines) and $p(m_1 \mid 0.3 < q < 0.65)$,
$p(m_1 \mid 0.8 < q < 1)$ (90% bands, medians); (2) the median $p(q \mid m_1)$, with the 90% (solid) and 99% (dashed)
intervals of the mean $\langle q \rangle$ given $m_1$; (3) the median $\chi_\mathrm{eff}$ width at fixed $(m_1, q)$,
with the 25–75% (solid) and 5–95% (dashed) quantiles of $p(q \mid m_1)$; (4) $\sigma_{\chi_\mathrm{eff}}(m_1)$, the sd
of $p(\chi_\mathrm{eff} \mid m_1)$, over all $q$ (black lines, 90%) and over the same two $q$ ranges (bands). Dotted
lines: the $p(m_1)$ peaks; dashed: twice those masses. Read panel 3 between the quantile lines: outside them there are
few events and the width follows the prior."""),
code(r"""fig, peaks = A.summary_figure([RES])                    # analysis.summary_figure (also used by the comparisons)
print('p(m1) peaks: ' + ', '.join(f'{m:.1f}' for m in peaks[0]) + ' Msun; twice: '
      + ', '.join(f'{2 * m:.1f}' for m in peaks[0]))
plot_style.save(fig, f'{RUN}_mass_ratio_spin_summary')"""),
md(r"""## Model-specific parameters

- **`split`:** $\sigma_1$ is the marginal sd of the mass-varying part $G_1$ of the spin log density, and $\sigma_0$
  that of the mass-independent part $G_0$. Both have PixelPop's prior, $\ln\sigma \sim N(0, 2)$. A $\sigma_1$
  posterior pushed well above its prior's low tail means the data ask for spins that change with mass.
- **`gauss`:** the correlation $\rho(m_1)$ of the Gaussian at $q \approx 1$ (its other parameters show up through the
  moments above). The parameters $\mu$ and $\sigma$ are those of the Gaussian *before* truncation to the physical
  region, so they differ from the moments, which are of the truncated distribution.
- **`field`:** nothing extra."""),
code(r"""if SPIN_MODEL == 'split':
    import scipy.stats as st
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.linspace(-6, 4, 400)
    for k, col, lab in (('lnsigma_spin0', 'C0', r'$\ln\sigma_0$ (mass-independent)'),
                        ('lnsigma_spin1', 'C1', r'$\ln\sigma_1$ (mass-varying)')):
        ax.hist(flat[k], bins=30, density=True, color=col, alpha=0.5, label=lab)
    ax.plot(x, st.norm(0, 2).pdf(x), color='0.4', ls='--', label='prior')
    ax.set_xlabel(r'$\ln\sigma$'); ax.set_ylabel('density'); ax.legend(**plot_style.LEG_KW)
    fig.tight_layout(); plot_style.save(fig, f'{RUN}_split_sigmas')
    l1 = flat['lnsigma_spin1']
    print('ln sigma_1: median %.2f, 90%% [%.2f, %.2f]; P(sigma_1 > sigma_0) %.2f'
          % (np.median(l1), *np.percentile(l1, [5, 95]), (l1 > flat['lnsigma_spin0']).mean()))
    print('prior P(ln sigma_1 < 5th percentile of the posterior) = %.3f' % st.norm(0, 2).cdf(np.percentile(l1, 5)))
elif SPIN_MODEL == 'gauss':
    th = flat['gauss_params']                                    # (draws, 5, ln m1, q)
    m_c = np.exp(lnm_c)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    lo, med, hi = band(np.tanh(th[:, 4, :, -1]))
    ax.fill_between(m_c, lo, hi, color='C0', alpha=0.3, lw=0); ax.plot(m_c, med, color='C0')
    ax.axhline(0, color='0.5', lw=0.8, ls=':')
    ax.set_xscale('log'); ax.set_xlim(MIN_M, MAX_M); ax.set_ylim(-1, 1)
    ax.set_xlabel(r'$m_1\,[M_\odot]$'); ax.set_ylabel(r'$\rho(\chi_\mathrm{eff}, \chi_p \mid m_1, q \approx 1)$')
    fig.tight_layout(); plot_style.save(fig, f'{RUN}_gauss_rho')
else:
    print('field model: no extra parameters')"""),
]
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbf.write(nb, out)
print(f'wrote {os.path.normpath(out)}')
