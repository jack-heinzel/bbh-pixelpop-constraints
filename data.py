"""Data loading and PixelPopData construction.

Catalogs (all (ln m1, q, chi_eff, chi_p) collections, read from ``<directory>/{posteriors,injections}.pkl``):

* ``gwtc5`` (default): the O4b paper's O1-O4b set at FAR < 1/yr (IFAR > 1 yr), 259 BBHs, NRSur7dq4 > Mixed >
  SpinTaylor PE, variable NPE capped at 20000; VT the corrected April 2026 mixture at IFAR 1, SNR 10. Collected by
  ``collect/`` into
  ``data/gwtc5_ifar1_chieffchip/data/`` (gitignored); set PPC_DATA_DIR to read it from elsewhere.
* ``nrsur``, ``physical``, ``mixed``: mass-gap-investigation's IFAR > 4 yr collections (236 BBHs, PE capped at 15000
  samples), from ``PPC_DATA_ROOT/<catalog>_o1o4b_chieffchip/data/``; kept to compare with the earlier runs.

Conventions (as mass-gap-investigation's run_utils): variable-NPE rectangular posteriors with ``event_counts``, short
events padded with prior = +inf; log_mass_1 with its Jacobian; PE samples outside the grid flagged with prior = +inf,
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


def build_pixelpop_data(name, catalog, bins=100, event_neff_cut=100., exclude_events=()):
    """PixelPopData on the (ln m1, q) grid (ln m1 in [ln 3, ln 200], q in [0, 1]); no variance cut."""
    posteriors, injections, event_counts = load_data(catalog, exclude_events)
    models = {p: gwtc6_default.models[p] for p in OTHER_PARAMS}
    hypers = {p: list(gwtc6_default.hyperparameters[p]) for p in OTHER_PARAMS}
    priors = {h: gwtc6_default.priors[h] for p in OTHER_PARAMS for h in hypers[p]}
    priors.update(WINDOW_PRIORS)
    pp = pixelpop.utils.PixelPopData(
        name=name, posteriors=posteriors, injections=injections,
        pixelpop_parameters=['log_mass_1', 'mass_ratio'], other_parameters=OTHER_PARAMS, bins=bins,
        minima={'log_mass_1': np.log(MIN_M)}, maxima={'log_mass_1': np.log(MAX_M)},
        parametric_models=models, parameter_to_hyperparameters=hypers, priors=priors,
        UncertaintyCut=np.inf, event_counts=event_counts, spde_matern=True, EventNeffCut=event_neff_cut,
        SelectionNeffCut=0.)
    return pp, posteriors, injections
