#!/bin/bash
# gauss with all 100 x 100 mass modes, short chain, on GPU 1 after the split short (launcher PID $1) (2026-10-20).
# Warm start from the baseline k = 10 short (else the k = 5 short). Launch a frozen copy.
cd "$(dirname "$(realpath "$0")")"
LAUNCH=launch_frozen_short_variants_20261020.sh
LOG=results/logs/launch.log
now() { TZ=America/New_York date '+%F %H:%M %Z'; }
while kill -0 "$1" 2>/dev/null; do sleep 60; done
src=
for run in ppc_gwtc5_k10_short ppc_gwtc5_k5_short; do
    [ -f results/$run/chain_0_samples.h5 ] && { src=$(realpath results/$run/chain_0_samples.h5); break; }
done
echo "$(now) queue: split done; starting gauss k=100 on GPU 1 from ${src:-default}" >> $LOG
SRC=$src bash $LAUNCH gwtc5 100 1 .4 - --spin-model gauss
