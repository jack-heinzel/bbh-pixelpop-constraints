#!/bin/bash
# Rusty: long chains of the spin-prior variants (user, 2026-10-21: long chains only, no shorts): tasks 0-3 split and
# gauss at k = 10, tasks 4-5 gauss with all 100 x 100 mass modes; chains 0 and 1 of each, each on its own GPU from its
# own prior draw; check at the starting point first.
# All tasks run at once (Rusty allows 16 GPUs per user). Submit from the repository:
#     sbatch submit_variants.sbatch [extra run.py args]              (tasks 0-3: split, gauss at k = 10)
#     sbatch --array=4-5 submit_variants.sbatch                       (gauss k = 100, chains 0 and 1)
#     sbatch --array=2 --export=ALL,CHAIN=2 submit_variants.sbatch     (one task, as a new chain index: gauss chain 2)
# Slurm copies this script at submission, so editing it does not affect queued or running jobs.
#SBATCH --job-name=ppc-var
#SBATCH --partition=gpu
#SBATCH --array=0-3
#SBATCH --gpus=1
#SBATCH --constraint=a100-80gb|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/slurm-%A_%a.out

cd "$SLURM_SUBMIT_DIR"
module purge                      # no cuda/cudnn modules: jax brings its own CUDA libraries
unset JAX_PLATFORMS
PY=~/venvs/gwjax313/bin/python
M=(split split gauss gauss gauss gauss); C=(0 1 0 1 0 1); K=(10 10 10 10 100 100)
model=${M[$SLURM_ARRAY_TASK_ID]}; c=${CHAIN:-${C[$SLURM_ARRAY_TASK_ID]}}
k=${K[$SLURM_ARRAY_TASK_ID]}; cat=gwtc5
L=results/logs; LOG=$L/launch.log
mkdir -p $L
# the GPU is ours alone (Slurm shows it as device 0), so XLA may take most of its memory
A="$cat --spin-mass-modes $k --spin-model $model --chain $c --device 0 --mem-fraction .9 --cpus $SLURM_CPUS_PER_TASK $*"
J="job $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID"
echo "$(date '+%F %H:%M') $J on $(hostname) ($(nvidia-smi --query-gpu=name --format=csv,noheader)): $model k=$k chain $c $*" >> $LOG
$PY run.py $A --check-only || exit 1
echo "$(date '+%F %H:%M') $J: start long $model k=$k chain $c" >> $LOG
$PY run.py $A > $L/ppc_${cat}_k${k}_${model}_long_chain${c}.log 2>&1
rc=$?; echo "$(date '+%F %H:%M') $J: end long $model k=$k chain $c (exit $rc)" >> $LOG
exit $rc
