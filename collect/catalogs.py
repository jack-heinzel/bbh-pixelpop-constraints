"""The GWTC-5 O1-O4b BBH catalog at FAR < 1/yr (IFAR > 1 yr), as in the O4b population paper's August reruns.

Events, files, labels and VT are those of ~/public_html/o4b_population/pixelpop-o4/code/collection/
collect_o4b_chieff_chip.ini (REFERENCE_INI), 259 BBHs:

* files: the rp.o4 GWTC-2.1 and GWTC-3 (v1) data releases, and the rp.o4 GWTC-4.1 (Stable_Release-3) and GWTC-5
  (Stable_Release-8) ``bbh_only`` trees, which rp.o4 scraped at FAR < 1/yr;
* the reference ini's ignore list: O1-O3 events below the FAR threshold or containing a neutron star (or m2 < 3 Msun:
  GW190814, GW190917);
* labels: the first of PREFERRED_LABELS present in each file (gwpopulation_pipe's rule), i.e. NRSur7dq4, else Mixed,
  else IMRPhenomXPHM-SpinTaylor: C01:Mixed for O1-O3, C00:NRSur7dq4 else C00:Mixed for O4a, NRSur7dq4 else SpinTaylor
  for O4b;
* VT: the corrected, clipped April 2026 multirun mixture, IFAR > 1 yr, O1/O2 semianalytic SNR > 10.

The one difference is the collector: the reference used upstream ``gwpopulation_pipe_collection``, which downsamples
every event to the smallest event's count (1993 samples); here pixelpop's collector keeps each event's own samples
up to ``samples-per-posterior`` (variable NPE). check_labels.py checks the events, files and labels against the
reference's event_data.json.
"""
import json
import os
import re
from glob import glob

import numpy as np

REPO = '/home/jack.heinzel/public_html/bbh-pixelpop-constraints'
REFERENCE = '/home/jack.heinzel/public_html/o4b_population/pixelpop-o4'
REFERENCE_INI = f'{REFERENCE}/code/collection/collect_o4b_chieff_chip.ini'
REFERENCE_EVENTS = f'{REFERENCE}/data/o4b_chieff_chip/data/event_data.json'
INI = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'collect_gwtc5_ifar1_chieffchip.ini')
PREFERRED_LABELS = ['C00:NRSur7dq4', 'C00:Mixed', 'C00:IMRPhenomXPHM-SpinTaylor',
                    'C01:NRSur7dq4', 'C01:Mixed', 'C01:IMRPhenomXPHM-SpinTaylor']

# The cosmology each sample-regex entry's samples were drawn with, verified from the samples (d_L(z) reproduces the
# stored luminosity_distance; dry_run.py re-checks every file), NOT from the file metadata: GWTC-5 files store
# 'Planck15_lal' but their samples are astropy Planck15 (see mass-gap-investigation's CLAUDE.md).
EXACT_PLANCK15 = 'Planck15_astropy'
RELEASE_COSMOLOGY = {
    'GWTC2.1': 'Planck15_LAL',
    'GWTC3': 'Planck15_LAL',
    'GWTC4': EXACT_PLANCK15,
    'GWTC5': EXACT_PLANCK15,
}
# events whose chosen label was drawn in another cosmology than the rest of their release (checked from d_L(z), to
# 2e-8): GW250114's C00:NRSur7dq4 group in Stable_Release-8 is Planck15_LAL (its metadata says Planck15; the file's
# other labels are astropy Planck15)
COSMOLOGY_OVERRIDES = {'GW250114_082203': 'Planck15_LAL'}
EVENT_RE = re.compile(r'GW\d{6}(?:_\d{6})?')


def event_name(filename):
    return EVENT_RE.findall(os.path.basename(filename))[-1]


def ini_entries(ini=INI):
    """(sample-regex {release: pattern}, ignore substrings) of a collection ini."""
    text = open(ini).read()
    block = re.search(r'^sample-regex=\{(.*?)\}', text, re.M | re.S).group(1)
    regex = {k.strip(): v.strip() for k, v in (e.split(':', 1) for e in block.split(',') if ':' in e)}
    ignore = re.search(r'^ignore=\[(.*)\]$', text, re.M).group(1)
    return regex, [i.strip() for i in ignore.split(',') if i.strip()]


def catalog_files(ini=INI):
    """event -> dict(release, file, label, labels_present): every file the ini's sample-regex and ignore list keep,
    with the first of PREFERRED_LABELS it holds (None if it holds none)."""
    import h5py
    regex, ignore = ini_entries(ini)
    events = {}
    for release, pattern in regex.items():
        for filename in sorted(glob(pattern)):
            if any(i in filename for i in ignore):
                continue
            with h5py.File(filename, 'r') as f:
                present = [k for k in f.keys() if k not in ('history', 'version')]
            label = next((l for l in PREFERRED_LABELS if l in present), None)
            event = event_name(filename)
            if event in events:
                raise ValueError(f'{event} in {events[event]["file"]} and {filename}')
            events[event] = dict(release=release, file=filename, label=label, labels_present=present)
    return events


def reference_events():
    """event -> (label, file) of the reference collection."""
    return {event_name(f): (info['label'], f) for f, info in json.load(open(REFERENCE_EVENTS)).items()}


def event_labels(events):
    """event name -> [label], for the collector."""
    return {e: [i['label']] for e, i in events.items()}


def event_cosmologies(events):
    """event name -> the cosmology its samples were drawn with."""
    return {e: COSMOLOGY_OVERRIDES.get(e, RELEASE_COSMOLOGY[i['release']]) for e, i in events.items()}


# ---- exact Planck15 for the collector's PE prior (as mass-gap-investigation/code/collect/catalogs.py) ----
# EXACT_PLANCK15 is astropy's full Planck15 (radiation and massive neutrinos), which bilby samples O4 events with.
# gwpopulation's 'Planck15' is wcosmo's radiation-free approximation: 6e-4 off in d_L.
class _AstropyCosmology:
    """An astropy cosmology exposing the unitless interface gwpopulation_pipe calls."""

    def __init__(self, cosmology):
        self.cosmology = cosmology

    def differential_comoving_volume(self, redshift):
        return self.cosmology.differential_comoving_volume(np.asarray(redshift)).value

    def luminosity_distance(self, redshift):
        return self.cosmology.luminosity_distance(np.asarray(redshift)).value


def register_exact_planck15():
    """Make EXACT_PLANCK15 resolvable where the collector evaluates the PE prior (idempotent)."""
    from gwpopulation_pipe import data_collection
    if getattr(data_collection.get_cosmology, '_exact_planck15', False):
        return
    from astropy.cosmology import Planck15
    original, exact = data_collection.get_cosmology, _AstropyCosmology(Planck15)

    def get_cosmology(name):
        return exact if name == EXACT_PLANCK15 else original(name)

    get_cosmology._exact_planck15 = True
    data_collection.get_cosmology = get_cosmology


def resolve_cosmology(name):
    """The object a cosmology name resolves to in the collector (for checks)."""
    register_exact_planck15()
    from gwpopulation_pipe import data_collection
    return data_collection.get_cosmology(name)
