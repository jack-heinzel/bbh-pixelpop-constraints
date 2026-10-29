#!/bin/bash
# Rusty: Gaussian spin model with an additive redshift shift (user, 2026-10-29; model.py spin_z_additive): each of the
# five Gaussian parameters is offset + F(ln m1, q) (100 x 100 modes, as ppc_gwtc5_k100_gauss) + G(z) (a 1D Matern field
# on 100 modes, 100 z bins over [0, max_z], its own amplitude, nu and range). G is shared by every (m1, q), so it cannot
# put a trend only in the unobserved low-mass, high-z corner (as the joint (m1, q, z) fields of
# ppc_gwtc5_k100_gauss_kz10 can).   -> ppc_gwtc5_k100_gauss_kz100_zb100_zadd, chains 0-2 (a spare: Gaussian chains
# sometimes freeze in warmup). Each on its own GPU from its own prior draw, after the check at the starting point.
# Submit from the repository:
#     sbatch submit_zadd.sbatch
#     sbatch --array=0 --export=ALL,CHAIN=3 submit_zadd.sbatch      (one task as a new chain index)
#     WARM=1 (--export=ALL,CHAIN=3,WARM=1): start from the baseline Gaussian's lowest-Var(ln L) draw
#     (results/inits/ppc_gwtc5_k100_gauss_minvar.npz, init_from_draw.py) for F, the offsets and the mass field, with G
#     near zero (setup_model's initial values)
# Slurm copies this script at submission, so editing it does not affect queued or running jobs.
#SBATCH --job-name=ppc-zadd
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
M="--spin-mass-modes 100 --spin-model gauss --spin-z-modes 100 --z-bins 100 --z-additive"; R=k100_gauss_kz100_zb100_zadd
cat=gwtc5
L=results/logs; LOG=$L/launch.log
mkdir -p $L
# the GPU is ours alone (Slurm shows it as device 0), so XLA may take most of its memory
A="$cat $M --chain $c --device 0 --mem-fraction .9 --cpus $SLURM_CPUS_PER_TASK"
[ "${WARM:-0}" = 1 ] && A="$A --init-from results/inits/ppc_gwtc5_k100_gauss_minvar.npz"
J="job $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID"
echo "$(date '+%F %H:%M') $J on $(hostname) ($(nvidia-smi --query-gpu=name --format=csv,noheader)): $R chain $c" >> $LOG
$PY run.py $A --check-only || exit 1
echo "$(date '+%F %H:%M') $J: start long $R chain $c" >> $LOG
$PY run.py $A > $L/ppc_${cat}_${R}_long_chain${c}.log 2>&1
rc=$?; echo "$(date '+%F %H:%M') $J: end long $R chain $c (exit $rc)" >> $LOG
exit $rc
