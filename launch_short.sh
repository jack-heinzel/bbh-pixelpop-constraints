#!/bin/bash
# A short chain only, for one spin-mass-mode count k on one GPU: from a prior draw (run.py's default), or warm-started
# from the best sample of the chain in $SRC (warmstart.py).
# Usage: [SRC=<chain h5>] launch_short.sh CATALOG K GPU MEMFRAC TAG [extra run.py args...]
#        (TAG '-' for none; launch a frozen copy). Extra args go to run.py (and warmstart.py), e.g. --spin-model split.
cd "$(dirname "$(realpath "$0")")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# the venv on Rusty (python/3.13.2 module), the conda env on the LIGO cluster
PY=$( [ -x ~/venvs/gwjax313/bin/python ] && echo ~/venvs/gwjax313/bin/python || echo ~/.conda/envs/gwjax313/bin/python )
cat=$1; k=$2; d=$3; mf=$4; tag=$5; shift 5
T=$([ "$tag" = - ] && echo "" || echo "--tag $tag")
A="$cat --spin-mass-modes $k --device $d --mem-fraction $mf $*"
# file names follow run.py's run name (it includes the spin model), so variants never overwrite each other
NAME=$($PY -c "import sys, run; print(run.run_name(run.make_parser().parse_args(sys.argv[1:])))" $A $T --short) || exit 1
L=results/logs; LOG=$L/launch.log; W=results/warmstart_${NAME}.npz
mkdir -p $L
now() { TZ=America/New_York date '+%F %H:%M %Z'; }
INIT=
if [ -n "$SRC" ]; then
    echo "$(now) GPU $d: warmstart $NAME from $SRC" >> $LOG
    $PY warmstart.py $A --source $SRC --out $W > $L/warmstart_${NAME}.log 2>&1 || { echo "$(now) GPU $d: warmstart $NAME FAILED" >> $LOG; exit 1; }
    INIT="--init-from $W"
fi
echo "$(now) GPU $d: start $NAME ${INIT:-(prior draw)}" >> $LOG
$PY run.py $A $T --short $INIT > $L/${NAME}.log 2>&1
rc=$?; echo "$(now) GPU $d: end $NAME (exit $rc)" >> $LOG
exit $rc
