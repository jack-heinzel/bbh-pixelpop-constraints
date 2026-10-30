#!/bin/bash
# Rusty: fully additive Gaussian spin model (user, 2026-10-30; model.py spin_mass_additive + spin_z_additive): each of
# the five Gaussian parameters is offset + F1(ln m1) + F2(q) + G(z), three 1D Matern fields on 100 modes each (100 bins
# per axis; z over [0, max_z]), each with its own amplitude, nu and range, mean modes pinned. No m1-q-z interaction, so
# neither a q-dependent mass trend nor a redshift trend can hide in an unobserved corner.
#     -> ppc_gwtc5_k100_gauss_kz100_zb100_mqadd_zadd, chains 0-2, each on its own GPU, after the check at its start.
# Chains >= WARM_FROM (default 1) start from the baseline Gaussian's lowest-Var(ln L) draw for the mass field, the
# redshift slope and the five offsets (results/inits/ppc_gwtc5_k100_gauss_minvar_massoffsets.npz), the 1D spin fields
# near zero (setup_model's initial values); lower chains from a prior draw (two of three such starts failed for _zadd).
# Submit from the repository:
#     sbatch submit_mqzadd.sbatch
#     sbatch --array=0 --export=ALL,CHAIN=3 submit_mqzadd.sbatch      (one task as a new chain index)
# Slurm copies this script at submission, so editing it does not affect queued or running jobs.
#SBATCH --job-name=ppc-mqzadd
#SBATCH --partition=gpu
#SBATCH --array=0-2
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
c=${CHAIN:-$SLURM_ARRAY_TASK_ID}
M="--spin-mass-modes 100 --spin-model gauss --spin-z-modes 100 --z-bins 100 --z-additive --mq-additive"
R=k100_gauss_kz100_zb100_mqadd_zadd
cat=gwtc5
L=results/logs; LOG=$L/launch.log
mkdir -p $L
# the GPU is ours alone (Slurm shows it as device 0), so XLA may take most of its memory
A="$cat $M --chain $c --device 0 --mem-fraction .9 --cpus $SLURM_CPUS_PER_TASK"
[ "$c" -ge "${WARM_FROM:-1}" ] && A="$A --init-from results/inits/ppc_gwtc5_k100_gauss_minvar_massoffsets.npz"
J="job $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID"
echo "$(date '+%F %H:%M') $J on $(hostname) ($(nvidia-smi --query-gpu=name --format=csv,noheader)): $R chain $c" >> $LOG
$PY run.py $A --check-only || exit 1
echo "$(date '+%F %H:%M') $J: start long $R chain $c" >> $LOG
$PY run.py $A > $L/ppc_${cat}_${R}_long_chain${c}.log 2>&1
rc=$?; echo "$(date '+%F %H:%M') $J: end long $R chain $c (exit $rc)" >> $LOG
exit $rc
