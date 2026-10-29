#!/bin/bash
# Rusty: redshift-dependent spin conditionals p(chi_eff, chi_p | m1, q, z) (user, 2026-10-29; model.py spin_z_modes),
# z on 10 bins over [0, max_z]:
#   tasks 0-1: Flexible field, 10 ln m1 x 5 q x 5 z modes x all 100 x 100 spin modes (~2.5e6 sites), longer chains
#              (4000 warmup, 2000 draws at thinning 5);   -> ppc_gwtc5_k10_kq5_kz5
#   tasks 2-4: Gaussian, its five parameter fields on 100 x 100 x 10 (ln m1, q, z) modes, chains 0-2 (a spare: Gaussian
#              chains sometimes freeze in warmup);           -> ppc_gwtc5_k100_gauss_kz10
# Each on its own GPU from its own prior draw, after the check at the starting point (which prints value+gradient ms).
# Submit from the repository:
#     sbatch submit_redshift.sbatch
#     sbatch --array=2 --export=ALL,CHAIN=3 submit_redshift.sbatch      (one task as a new chain index)
# Slurm copies this script at submission, so editing it does not affect queued or running jobs.
#SBATCH --job-name=ppc-z
#SBATCH --partition=gpu
#SBATCH --array=0-4
#SBATCH --gpus=1
#SBATCH --constraint=a100-80gb|h100|h200
#SBATCH --exclude=workergpu168
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --output=results/logs/slurm-%A_%a.out

cd "$SLURM_SUBMIT_DIR"
module purge                      # no cuda/cudnn modules: jax brings its own CUDA libraries
unset JAX_PLATFORMS
PY=~/venvs/gwjax313/bin/python
t=$SLURM_ARRAY_TASK_ID
if [ $t -lt 2 ]; then
    M="--spin-mass-modes 10 --spin-q-modes 5 --spin-z-modes 5 --samples 2000"; R=k10_kq5_kz5; c=${CHAIN:-$t}
else
    M="--spin-mass-modes 100 --spin-model gauss --spin-z-modes 10"; R=k100_gauss_kz10; c=${CHAIN:-$((t - 2))}
fi
cat=gwtc5
L=results/logs; LOG=$L/launch.log
mkdir -p $L
# the GPU is ours alone (Slurm shows it as device 0), so XLA may take most of its memory
A="$cat $M --chain $c --device 0 --mem-fraction .9 --cpus $SLURM_CPUS_PER_TASK"
J="job $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID"
echo "$(date '+%F %H:%M') $J on $(hostname) ($(nvidia-smi --query-gpu=name --format=csv,noheader)): $R chain $c" >> $LOG
$PY run.py $A --check-only || exit 1
echo "$(date '+%F %H:%M') $J: start long $R chain $c" >> $LOG
$PY run.py $A > $L/ppc_${cat}_${R}_long_chain${c}.log 2>&1
rc=$?; echo "$(date '+%F %H:%M') $J: end long $R chain $c (exit $rc)" >> $LOG
exit $rc
