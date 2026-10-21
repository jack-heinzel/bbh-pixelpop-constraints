#!/bin/bash
# Short chain, then long chain for one spin-mass-mode count k on one GPU, each from a prior draw (--seed) (2026-10-20).
# Usage: launch.sh CATALOG K GPU MEMFRAC [extra run.py args...]   (launch a frozen copy: never edit a script while it runs)
cd "$(dirname "$(realpath "$0")")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=~/venvs/gwjax313/bin/python
cat=$1; k=$2; d=$3; mf=$4; shift 4
L=results/logs; LOG=$L/launch.log
mkdir -p $L
A="$cat --spin-mass-modes $k --device $d --mem-fraction $mf $*"
for mode in short long; do
    extra=$([ $mode = short ] && echo --short)
    echo "$(date '+%F %H:%M') GPU $d: start $mode k=$k" >> $LOG
    $PY run.py $A $extra > $L/ppc_${cat}_k${k}_${mode}.log 2>&1
    rc=$?; echo "$(date '+%F %H:%M') GPU $d: end $mode k=$k (exit $rc)" >> $LOG
    [ $rc = 0 ] || exit $rc
done
