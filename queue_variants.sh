#!/bin/bash
# Run the spin-model variants after the baseline gwtc5 shorts, one per GPU (2026-10-20). Launch a frozen copy.
# GPU 1: after the baseline k = 5 short (launcher PID $2, done first): split k = 10, warm-started from that chain.
# GPU 0: after the baseline k = 10 short (launcher PID $1): gauss k = 10, warm-started from that chain.
# A missing baseline chain falls back to launch_short.sh's default source.
cd "$(dirname "$(realpath "$0")")"
LAUNCH=launch_frozen_short_variants_20261020.sh
LOG=results/logs/launch.log
now() { TZ=America/New_York date '+%F %H:%M %Z'; }

after() {   # after PID BASELINE_RUN GPU MODEL
    while kill -0 "$1" 2>/dev/null; do sleep 60; done
    local chain=results/$2/chain_0_samples.h5 src=
    [ -f "$chain" ] && src=$(realpath "$chain")
    echo "$(now) queue: $2 done; starting $4 k=10 on GPU $3 from ${src:-default}" >> $LOG
    SRC=$src bash $LAUNCH gwtc5 10 "$3" .4 - --spin-model "$4"
}

after "$2" ppc_gwtc5_k5_short 1 split &
after "$1" ppc_gwtc5_k10_short 0 gauss &
wait
