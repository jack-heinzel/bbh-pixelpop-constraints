"""Collect the catalog of catalogs.py (posteriors.pkl, injections.pkl, event_data.json in the ini's run-dir/data).

Usage:
    python collect.py collect_gwtc5_ifar1_chieffchip.ini

Wraps ``pixelpop.utils.collection.main`` (as mass-gap-investigation/code/collect/collect.py): unlike upstream
``gwpopulation_pipe_collection`` it does NOT downsample every event to the smallest event's sample count. Events
longer than ``samples-per-posterior`` are downsampled to that cap; shorter ones are padded up to the common width
with ``prior=+inf`` (zero-weight) rows. The posteriors are pickled as

    {'posteriors': rect_dict, 'event_counts': (Nobs,), 'event_names': [...]}

Each event is given its label (catalogs.catalog_files) with ``only_listed_events=True``, and ``event_cosmologies``
sets the cosmology its PE prior is evaluated in (catalogs.RELEASE_COSMOLOGY). Needs the pixelpop per-file cosmology
edit (see mass-gap-investigation's CLAUDE.md).
"""
import sys
from collections import Counter

from pixelpop.utils.collection import main

from catalogs import catalog_files, event_cosmologies, event_labels, register_exact_planck15

events = catalog_files(sys.argv[1])
labels = event_labels(events)
cosmologies = event_cosmologies(events)
print(f'Collecting {len(labels)} events, labels {dict(Counter(l[0] for l in labels.values()))}, '
      f'cosmologies {dict(Counter(cosmologies.values()))}')
register_exact_planck15()
main(event_labels=labels, only_listed_events=True, event_cosmologies=cosmologies)
