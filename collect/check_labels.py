"""Check a collection loaded exactly the reference's events, files and labels, with the declared cosmologies.

Usage:  python check_labels.py collect_gwtc5_ifar1_chieffchip.ini

Compares <run-dir>/data/event_data.json with the reference collection's (catalogs.REFERENCE_EVENTS) and with
catalogs.event_cosmologies, and checks posteriors.pkl's event_counts are min(samples in the file,
samples-per-posterior). Exits non-zero on any difference, so run_collect.sh stops before a bad collection is used.
"""
import json
import os
import pickle
import re
import sys
from collections import Counter

import h5py
import numpy as np

from catalogs import catalog_files, event_cosmologies, event_name, reference_events


def ini_value(ini, key):
    return re.search(rf'^{key}=(.*)$', open(ini).read(), re.M).group(1).strip()


def check(ini):
    directory = os.path.join(ini_value(ini, 'run-dir'), 'data')
    cap = int(ini_value(ini, 'samples-per-posterior'))
    reference = reference_events()
    cosmologies = event_cosmologies(catalog_files(ini))
    loaded = {event_name(f): (info['label'], f, info.get('cosmology'))
              for f, info in json.load(open(os.path.join(directory, 'event_data.json'))).items()}
    problems = []
    for e, (label, filename, cosmology) in sorted(loaded.items()):
        if e not in reference:
            problems.append(f'UNEXPECTED {e}: {label} from {filename}')
            continue
        if label != reference[e][0]:
            problems.append(f'MISMATCH {e}: loaded {label!r}, reference {reference[e][0]!r}')
        if os.path.realpath(filename) != os.path.realpath(reference[e][1]):
            problems.append(f'WRONG FILE {e}: {filename}, reference {reference[e][1]}')
        if cosmology != cosmologies.get(e):
            problems.append(f'WRONG COSMOLOGY {e}: used {cosmology!r}, wanted {cosmologies.get(e)!r}')
    problems += [f'MISSING {e}' for e in sorted(set(reference) - set(loaded))]

    with open(os.path.join(directory, 'posteriors.pkl'), 'rb') as f:
        posteriors = pickle.load(f)
    for name, count in zip(posteriors['event_names'], posteriors['event_counts']):
        e = event_name(str(name))
        if e in loaded:
            with h5py.File(loaded[e][1], 'r') as f:
                n = len(f[loaded[e][0]]['posterior_samples'])
            if count != min(n, cap):
                problems.append(f'COUNT {e}: {count} samples, wanted min({n}, {cap})')

    counts = np.asarray(posteriors['event_counts'])
    print(f'{ini}: {len(loaded)} events collected, {len(reference)} in the reference; samples per event '
          f'min/median/max {counts.min()}/{int(np.median(counts))}/{counts.max()}, '
          f'width {posteriors["posteriors"]["mass_1"].shape[1]}')
    for label, n in sorted(Counter(l for l, _, _ in loaded.values()).items()):
        print(f'  {n:4d}  {label}')
    for p in problems:
        print('  ' + p)
    print('  OK' if not problems else '  FAILED')
    return not problems


if __name__ == '__main__':
    sys.exit(0 if all([check(ini) for ini in sys.argv[1:]]) else 1)
