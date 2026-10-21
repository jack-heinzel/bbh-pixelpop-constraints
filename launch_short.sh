#!/bin/bash
# Warm start, then a short chain only, for one spin-mass-mode count k on one GPU.
# Usage: [SRC=<chain h5>] launch_short.sh CATALOG K GPU MEMFRAC TAG [extra run.py args...]
#        (TAG '-' for none; launch a frozen copy). Extra args go to warmstart.py and run.py, e.g. --spin-model split.
cd "$(dirname "$(realpath "$0")")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=~/.conda/envs/gwjax313/bin/python
cat=$1; k=$2; d=$3; mf=$4; tag=$5; shift 5
# warm-start source: $SRC if set, else mass-gap-investigation's bm3 short
SRC=${SRC:-/home/jack.heinzel/public_html/mass-gap-investigation/results/pixelpop/lm1q_hierarchical_nrsur_varcutinf_short_bm3_s100_flat_ellipse_neff100_lcorr_sep2g/chain_0_samples.h5}
T=$([ "$tag" = - ] && echo "" || echo "--tag $tag")
A="$cat --spin-mass-modes $k --device $d --mem-fraction $mf $*"
# file names follow run.py's run name (it includes the spin model), so variants never overwrite each other
NAME=$($PY -c "import sys, run; print(run.run_name(run.make_parser().parse_args(sys.argv[1:])))" $A $T --short) || exit 1
L=results/logs; LOG=$L/launch.log; W=results/warmstart_${NAME}.npz
now() { TZ=America/New_York date '+%F %H:%M %Z'; }
echo "$(now) GPU $d: warmstart $NAME from $SRC" >> $LOG
$PY warmstart.py $A --source $SRC --out $W > $L/warmstart_${NAME}.log 2>&1 || { echo "$(now) GPU $d: warmstart $NAME FAILED" >> $LOG; exit 1; }
echo "$(now) GPU $d: start $NAME" >> $LOG
$PY run.py $A $T --short --init-from $W > $L/${NAME}.log 2>&1
rc=$?; echo "$(now) GPU $d: end $NAME (exit $rc)" >> $LOG
exit $rc
