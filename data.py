"""Data loading and PixelPopData construction.

Catalogs (all (ln m1, q, chi_eff, chi_p) collections, read from ``<directory>/{posteriors,injections}.pkl``):

* ``gwtc5`` (default): the O4b paper's O1-O4b set at FAR < 1/yr (IFAR > 1 yr), 259 BBHs, NRSur7dq4 > Mixed >
  SpinTaylor PE, variable NPE capped at 20000; VT the corrected April 2026 mixture at IFAR 1, SNR 10. Collected by
  ``collect/`` into
  ``data/gwtc5_ifar1_chieffchip/data/`` (gitignored); set PPC_DATA_DIR to read it from elsewhere.
* ``nrsur``, ``physical``, ``mixed``: mass-gap-investigation's IFAR > 4 yr collections (236 BBHs, PE capped at 15000
  samples), from ``PPC_DATA_ROOT/<catalog>_o1o4b_chieffchip/data/``; kept to compare with the earlier runs.

Conventions (as mass-gap-investigation's run_utils): variable-NPE rectangular posteriors with ``event_counts``, short
events padded with prior = +inf; run.py keeps at most 20000 usable samples per event (``subsample``); log_mass_1 with its Jacobian; PE samples outside the grid flagged with prior = +inf,
injections outside it dropped; redshift and the m2 > 3 Msun window are the GWTC-6 parametric models.

Import only after the thread / CUDA environment is set (run.py), since this imports jax.
"""
import os
import pickle

import numpy as np
from jax import numpy as jnp
from numpyro import distributions as dist

import pixelpop
from pixelpop.models.gwtc_defaults import gwtc6_default

REPO = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get('PPC_DATA_ROOT', '/home/jack.heinzel/public_html/mass-gap-investigation/data')
CATALOGS = {
    'gwtc5': os.environ.get('PPC_DATA_DIR', os.path.join(REPO, 'data', 'gwtc5_ifar1_chieffchip', 'data')),
    **{c: os.path.join(DATA_ROOT, f'{c}_o1o4b_chieffchip', 'data') for c in ('nrsur', 'physical', 'mixed')},
}
MIN_M, MAX_M, MAX_Z = 3., 200., 1.9
OTHER_PARAMS = ['redshift', 'mass_ratio_window']
WINDOW_PRIORS = {'mlow_2': ([MIN_M], dist.Delta), 'delta_m_2': ([0.], dist.Delta)}


def load_data(catalog, exclude_events=()):
    """Posteriors, injections and real per-event sample counts. ``exclude_events``: drop events whose file name contains
    any of these strings (their PE samples only; the selection function is unchanged)."""
    if catalog not in CATALOGS:
        raise ValueError(f'catalog must be one of {tuple(CATALOGS)}, got {catalog!r}')
    directory = CATALOGS[catalog]
    with open(os.path.join(directory, 'posteriors.pkl'), 'rb') as f:
        loaded = pickle.load(f)
    with open(os.path.join(directory, 'injections.pkl'), 'rb') as f:
        injections = pickle.load(f)
    if exclude_events:
        names = [str(n) for n in loaded['event_names']]
        keep = np.array([not any(e in n for e in exclude_events) for n in names])
        dropped = [os.path.basename(n) for n, k in zip(names, keep) if not k]
        if len(dropped) != len(exclude_events):
            raise ValueError(f'exclude_events {exclude_events} matched {dropped}')
        loaded['posteriors'] = {k: np.asarray(v)[keep] for k, v in loaded['posteriors'].items()}
        loaded['event_counts'] = np.asarray(loaded['event_counts'])[keep]
        print(f'excluded {dropped}')
    posteriors = {k: jnp.asarray(v) for k, v in loaded['posteriors'].items()}
    event_counts = np.asarray(loaded['event_counts'])
    injections = {k: (jnp.asarray(v) if np.ndim(v) else v) for k, v in injections.items()}
    nobs, npe = posteriors['mass_1'].shape
    print(f'{catalog}: {nobs} events ({npe} columns; real samples per event min/median/max {event_counts.min()}/'
          f'{int(np.median(event_counts))}/{event_counts.max()}), {len(injections["prior"])} found injections')
    for data, remove in ((posteriors, False), (injections, True)):
        pixelpop.utils.data.convert_m1_to_lm1(data)     # prior -> log_prior, + log m1
        pixelpop.utils.data.clean_par(data, 'log_mass_1', np.log(MIN_M), np.log(MAX_M), remove=remove)
        pixelpop.utils.data.clean_par(data, 'redshift', 0., MAX_Z, remove=remove)
    return posteriors, injections, event_counts


def subsample(posteriors, injections, event_counts, max_samples, bins, n_spin_bins, seed=0):
    """At most ``max_samples`` columns, filled greedily with usable samples: each event keeps a random subset of its
    real samples that are inside the grid (finite prior) and in a (ln m1, q) bin and a (chi_eff, chi_p) bin some
    injection reaches (the bins of PixelPopData and model.setup_model), padded with prior = +inf copies. The other
    samples carry zero weight anyway; leaving them out of the count shifts each event's ln L by the constant
    -ln(usable / real samples). Returns the posteriors and the new event counts."""
    from pixelpop.models.gwpop_models import bbh_maxima, bbh_minima
    from model import spin_bins
    axes = [np.asarray(jnp.linspace(np.log(MIN_M), np.log(MAX_M), bins + 1)),          # as pixelpop.place_in_bins
            np.asarray(jnp.linspace(bbh_minima['mass_ratio'], bbh_maxima['mass_ratio'], bins + 1))]
    spin_edges = np.linspace(-1., 1., n_spin_bins[0] + 1), np.linspace(0., 1., n_spin_bins[1] + 1)   # as model.py

    def mass_index(x):
        im, iq = (np.digitize(np.asarray(x[k]), a) - 1 for k, a in zip(('log_mass_1', 'mass_ratio'), axes))
        return np.where((im >= 0) & (im < bins) & (iq >= 0) & (iq < bins), im * bins + iq, -1)

    def spin_index(x):
        ie, ip = spin_bins(x, *spin_edges)
        return ie * n_spin_bins[1] + ip

    im = mass_index(injections)
    mass_reached = np.zeros(bins * bins, bool); mass_reached[im[im >= 0]] = True
    spin_reached = np.zeros(int(np.prod(n_spin_bins)), bool); spin_reached[spin_index(injections)] = True
    pm = mass_index(posteriors)
    nobs, npe = pm.shape
    usable = ((np.arange(npe)[None] < np.asarray(event_counts)[:, None]) & np.isfinite(np.asarray(posteriors['log_prior']))
              & (pm >= 0) & mass_reached[np.maximum(pm, 0)] & spin_reached[spin_index(posteriors)])
    n_usable = usable.sum(1)
    if n_usable.min() == 0:
        raise ValueError(f'events with no usable samples: {np.flatnonzero(n_usable == 0)}')
    width = int(min(max_samples, n_usable.max()))
    rng = np.random.default_rng(seed)
    cols = []
    for u in usable:
        idx = np.flatnonzero(u)
        if len(idx) > width:
            idx = np.sort(rng.choice(idx, width, replace=False))
        cols.append(np.concatenate([idx, rng.choice(idx, width - len(idx))]))
    counts = np.minimum(n_usable, width)
    rows = np.arange(nobs)[:, None]
    out = {k: jnp.asarray(np.asarray(v)[rows, np.stack(cols)]) for k, v in posteriors.items()}
    out['log_prior'] = jnp.where(jnp.asarray(np.arange(width)[None] >= counts[:, None]), jnp.inf, out['log_prior'])
    real = np.asarray(event_counts)
    print(f'subsample: {width} columns; usable samples per event min/median/max {n_usable.min()}/{int(np.median(n_usable))}/'
          f'{n_usable.max()} ({1 - n_usable.sum() / real.sum():.2e} of real samples unusable, at most '
          f'{(1 - n_usable / real).max():.2%} for one event); kept per event min/median/max {counts.min()}/'
          f'{int(np.median(counts))}/{counts.max()}')
    return out, counts


def build_pixelpop_data(name, catalog, bins=100, event_neff_cut=100., exclude_events=(), max_samples=None,
                        n_spin_bins=(100, 100), variance_cut=np.inf):
    """PixelPopData on the (ln m1, q) grid (ln m1 in [ln 3, ln 200], q in [0, 1]). ``variance_cut``: the soft cut on
    Var(ln L) (pixelpop's UncertaintyCut is its square root; inf, the default: none). ``max_samples``: keep at most that
    many usable samples per event (``subsample``; None keeps them all)."""
    posteriors, injections, event_counts = load_data(catalog, exclude_events)
    if max_samples:
        posteriors, event_counts = subsample(posteriors, injections, event_counts, max_samples, bins, n_spin_bins)
    models = {p: gwtc6_default.models[p] for p in OTHER_PARAMS}
    hypers = {p: list(gwtc6_default.hyperparameters[p]) for p in OTHER_PARAMS}
    priors = {h: gwtc6_default.priors[h] for p in OTHER_PARAMS for h in hypers[p]}
    priors.update(WINDOW_PRIORS)
    pp = pixelpop.utils.PixelPopData(
        name=name, posteriors=posteriors, injections=injections,
        pixelpop_parameters=['log_mass_1', 'mass_ratio'], other_parameters=OTHER_PARAMS, bins=bins,
        minima={'log_mass_1': np.log(MIN_M)}, maxima={'log_mass_1': np.log(MAX_M)},
        parametric_models=models, parameter_to_hyperparameters=hypers, priors=priors,
        UncertaintyCut=np.sqrt(variance_cut), event_counts=event_counts, spde_matern=True, EventNeffCut=event_neff_cut,
        SelectionNeffCut=0.)
    return pp, posteriors, injections
