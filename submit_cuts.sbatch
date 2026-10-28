#!/bin/bash
# Rusty: robustness of the two paper runs to the likelihood-uncertainty cuts (user, 2026-10-28): the Flexible spin model
# (field, k = 10) and the Gaussian (gauss, k = 100), each with event Neff > 50, event Neff > 30, Var(ln L) < 1 and
# Var(ln L) < 4 (the variance cuts keep the baseline event Neff > 100); chains 0 and 1 of each, long chains only, each
# on its own GPU from its own prior draw, after the check at the starting point. 16 tasks (the gpu QOS limit).
# Submit from the repository:
#     sbatch submit_cuts.sbatch                         (all 16)
#     sbatch --array=8-15 submit_cuts.sbatch            (only the Gaussian ones)
#     sbatch --array=3 --export=ALL,CHAIN=2 submit_cuts.sbatch   (one task as a new chain index, e.g. to replace a frozen chain)
#     sbatch --array=4-7,12-15 --export=ALL,WARM=1 submit_cuts.sbatch   (the variance cuts, warm-started)
# WARM=1 starts from the baseline run's lowest-Var(ln L) posterior draw (init_from_draw.py: results/inits/), since random
# prior starts (Var(ln L) ~ 1e2-1e4) lie far outside a variance cut and the chains freeze (2026-10-28, job 7359370).
# Task t: model = t / 8, cut = (t / 2) % 4, chain = t % 2. Slurm copies this script at submission.
#SBATCH --job-name=ppc-cuts
#SBATCH --partition=gpu
#SBATCH --array=0-15
#SBATCH --gpus=1
#SBATCH --constraint=a100-80gb|h100|h200
#SBATCH --exclude=workergpu168
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/slurm-%A_%a.out

cd "$SLURM_SUBMIT_DIR"
module purge                      # no cuda/cudnn modules: jax brings its own CUDA libraries
unset JAX_PLATFORMS
PY=~/venvs/gwjax313/bin/python
t=$SLURM_ARRAY_TASK_ID
MODELS=("--spin-mass-modes 10 --spin-model field" "--spin-mass-modes 100 --spin-model gauss")
NAMES=(k10 k100_gauss)
CUTS=("--event-neff-cut 50" "--event-neff-cut 30" "--variance-cut 1" "--variance-cut 4")
CUT_NAMES=(neff50 neff30 varcut1 varcut4)
m=$((t / 8)); i=$(((t / 2) % 4)); c=${CHAIN:-$((t % 2))}
cat=gwtc5
L=results/logs; LOG=$L/launch.log
mkdir -p $L
# the GPU is ours alone (Slurm shows it as device 0), so XLA may take most of its memory
A="$cat ${MODELS[$m]} ${CUTS[$i]} --chain $c --device 0 --mem-fraction .9 --cpus $SLURM_CPUS_PER_TASK"
[ -n "$WARM" ] && A="$A --init-from results/inits/ppc_${cat}_${NAMES[$m]}_minvar.npz"
R="${NAMES[$m]}_${CUT_NAMES[$i]}"
J="job $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID"
echo "$(date '+%F %H:%M') $J on $(hostname) ($(nvidia-smi --query-gpu=name --format=csv,noheader)): $R chain $c${WARM:+ (warm)}" >> $LOG
$PY run.py $A --check-only || exit 1
echo "$(date '+%F %H:%M') $J: start long $R chain $c" >> $LOG
$PY run.py $A > $L/ppc_${cat}_${R}_long_chain${c}.log 2>&1
rc=$?; echo "$(date '+%F %H:%M') $J: end long $R chain $c (exit $rc)" >> $LOG
exit $rc
