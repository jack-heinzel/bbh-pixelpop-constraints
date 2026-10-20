#!/bin/bash
# Warm start, short chain, then long chain for one spin-mass-mode count k on one GPU (2026-10-20).
# Usage: launch.sh CATALOG K GPU MEMFRAC [extra run.py args...]   (launch a frozen copy: never edit a script while it runs)
cd "$(dirname "$(realpath "$0")")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=~/.conda/envs/gwjax313/bin/python
cat=$1; k=$2; d=$3; mf=$4; shift 4
SRC=/home/jack.heinzel/public_html/mass-gap-investigation/results/pixelpop/lm1q_hierarchical_nrsur_varcutinf_short_bm3_s100_flat_ellipse_neff100_lcorr_sep2g/chain_0_samples.h5
L=results/logs; LOG=$L/launch.log; W=results/warmstart_ppc_${cat}_k${k}.npz
A="$cat --spin-mass-modes $k --device $d --mem-fraction $mf $*"
echo "$(date '+%F %H:%M') GPU $d: warmstart k=$k" >> $LOG
$PY warmstart.py $A --source $SRC --out $W > $L/warmstart_${cat}_k${k}.log 2>&1 || { echo "$(date '+%F %H:%M') GPU $d: warmstart k=$k FAILED" >> $LOG; exit 1; }
for mode in short long; do
    extra=$([ $mode = short ] && echo --short)
    echo "$(date '+%F %H:%M') GPU $d: start $mode k=$k" >> $LOG
    $PY run.py $A $extra --init-from $W > $L/ppc_${cat}_k${k}_${mode}.log 2>&1
    rc=$?; echo "$(date '+%F %H:%M') GPU $d: end $mode k=$k (exit $rc)" >> $LOG
    [ $rc = 0 ] || exit $rc
done
