#!/bin/bash
# Rusty: mass-additive Gaussian spin model without redshift dependence (user, 2026-11-02; model.py spin_mass_additive):
# each of the five Gaussian parameters is offset + F1(ln m1) + F2(q), two 1D Matern fields on 100 modes each (100 bins
# per axis), each with its own amplitude, nu and range, mean modes pinned; spins independent of z. Separates the two
# differences between the Gaussian (Fig. 6) and F1 + F2 + G(z) (Fig. 7) marginals: additivity and the z average.
#     -> ppc_gwtc5_k100_gauss_mqadd, chains 1-2, each on its own GPU, after the check at its start.
# Chains >= WARM_FROM (default 1) start from the baseline Gaussian's lowest-Var(ln L) draw for the mass field, the
# redshift slope and the five offsets (results/inits/ppc_gwtc5_k100_gauss_minvar_massoffsets.npz), the 1D spin fields
# near zero (setup_model's initial values); chain 0 would start from a prior draw (those froze for _zadd and _mqadd_zadd).
# QONLY=1 (user, 2026-11-02): no F1(ln m1) either, the spins depend on q alone (--q-only) -> ..._mqadd_qonly
# M1ONLY=1 (user, 2026-11-02): no F2(q), the spins depend on ln m1 alone (--m1-only) -> ..._mqadd_m1only
# Submit from the repository:
#     sbatch submit_mqadd.sbatch
#     sbatch --array=0 --export=ALL,CHAIN=3 submit_mqadd.sbatch       (one task as a new chain index)
# Slurm copies this script at submission, so editing it does not affect queued or running jobs.
#SBATCH --job-name=ppc-mqadd
#SBATCH --partition=gpu
#SBATCH --array=1-2
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
M="--spin-mass-modes 100 --spin-model gauss --mq-additive"
R=k100_gauss_mqadd
[ "${QONLY:-0}" = 1 ] && M="$M --q-only" && R=${R}_qonly
[ "${M1ONLY:-0}" = 1 ] && M="$M --m1-only" && R=${R}_m1only
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
