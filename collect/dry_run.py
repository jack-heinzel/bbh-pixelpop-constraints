"""Resolve every file the ini would load to a label and check it against the reference, without loading the samples.

Usage:  python dry_run.py collect_gwtc5_ifar1_chieffchip.ini

Fails if a file holds none of the preferred labels (upstream would then silently read its first label), if the
events, files or labels differ from the reference collection (catalogs.REFERENCE_EVENTS), or if an event's declared
cosmology does not reproduce its stored luminosity_distance from its stored redshift. Takes a minute, so run it
before a collection.
"""
import os
import sys
from collections import Counter

import h5py
import numpy as np

from catalogs import catalog_files, event_cosmologies, reference_events, resolve_cosmology

COSMOLOGY_TOLERANCE = 1e-4
_COSMOLOGIES = {}


def cosmology_problem(event, filename, label, name):
    """The declared cosmology fails to reproduce the samples' d_L(z), or None."""
    if name not in _COSMOLOGIES:
        _COSMOLOGIES[name] = resolve_cosmology(name)
    with h5py.File(filename, 'r') as f:
        p = f[label]['posterior_samples']
        n = min(len(p), 2000)
        z, dl = p['redshift'][:n], p['luminosity_distance'][:n]
    model = _COSMOLOGIES[name].luminosity_distance(z)
    err = float(np.max(np.abs(np.asarray(getattr(model, 'value', model)) / dl - 1)))
    if err > COSMOLOGY_TOLERANCE:
        return f'COSMOLOGY {event}: {name} gives d_L(z) off by {err:.1e} in {filename}'


def dry_run(ini):
    events = catalog_files(ini)
    cosmologies = event_cosmologies(events)
    reference = reference_events()
    problems = []
    for event, info in events.items():
        if info['label'] is None:
            problems.append(f'NO PREFERRED LABEL {event}: {info["labels_present"]} in {info["file"]}')
            continue
        if event not in reference:
            problems.append(f'NOT IN REFERENCE {event}')
        elif (info['label'], os.path.realpath(info['file'])) != (reference[event][0],
                                                                 os.path.realpath(reference[event][1])):
            problems.append(f'DIFFERS FROM REFERENCE {event}: {info["label"]} {info["file"]} vs {reference[event]}')
        problem = cosmology_problem(event, info['file'], info['label'], cosmologies[event])
        if problem:
            problems.append(problem)
    problems += [f'MISSING {e}' for e in sorted(set(reference) - set(events))]

    with_label = [i for i in events.values() if i['label']]
    counts = []
    for i in with_label:
        with h5py.File(i['file'], 'r') as f:
            counts.append(len(f[i['label']]['posterior_samples']))
    counts = np.array(counts)
    print(f'{ini}: {len(events)} events resolve, {len(reference)} in the reference')
    for (release, label), n in sorted(Counter((i['release'], i['label']) for i in events.values()).items()):
        print(f'  {n:4d}  {release} {label}')
    print(f'  samples per event: min {counts.min()}, median {int(np.median(counts))}, max {counts.max()}')
    for p in problems[:40]:
        print('  ' + p)
    if len(problems) > 40:
        print(f'  ... and {len(problems) - 40} more')
    print('  OK' if not problems else '  FAILED')
    return not problems


if __name__ == '__main__':
    sys.exit(0 if all([dry_run(ini) for ini in sys.argv[1:]]) else 1)
