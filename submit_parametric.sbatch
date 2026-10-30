#!/bin/bash
# Rusty: parametric check of the q-split mass spectrum (user, 2026-10-29; parametric.py): power law + two peaks in m1
# separately for q > 0.7 and q < 0.7 (all hyperparameters separate, mixing fraction zeta), power law in q in each range,
# mass-independent Gaussian (chi_eff, chi_p), power-law redshift; chains 0, 1 from setup_parametric's starting values
# (ln L -7417, Var 1.2, min event Neff 43; prior draws start at min Neff ~1).   -> ppc_gwtc5_plpp_qsplit0.7
# SPIN_SPLIT=1 (user, 2026-10-30): a separate spin Gaussian in each q range (--spin-split)
#     -> ppc_gwtc5_plpp_qsplit0.7_spinsplit
# Submit from the repository:
#     sbatch submit_parametric.sbatch
#     sbatch --export=ALL,SPIN_SPLIT=1 submit_parametric.sbatch
# SHARED_CONT=1 (user, 2026-10-30): one continuum (alpha, mmin, delta_m) for both ranges (--shared-continuum)
#     -> ..._sharedcont
# QSPLIT=<q> (default 0.7): the mass-ratio split (user, 2026-10-30: 0.65, as the 10 Msun peak leaks over 0.7)
# Slurm copies this script at submission, so editing it does not affect queued or running jobs.
#SBATCH --job-name=ppc-plpp
#SBATCH --partition=gpu
#SBATCH --array=0-1
#SBATCH --gpus=1
#SBATCH --constraint=a100-80gb|h100|h200
#SBATCH --exclude=workergpu168
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=results/logs/slurm-%A_%a.out

cd "$SLURM_SUBMIT_DIR"
module purge                      # no cuda/cudnn modules: jax brings its own CUDA libraries
unset JAX_PLATFORMS
PY=~/venvs/gwjax313/bin/python
c=${CHAIN:-$SLURM_ARRAY_TASK_ID}
QS=${QSPLIT:-0.7}; R=plpp_qsplit$QS; cat=gwtc5
L=results/logs; LOG=$L/launch.log
mkdir -p $L
A="$cat --parametric --q-split $QS --chain $c --device 0 --mem-fraction .9 --cpus $SLURM_CPUS_PER_TASK"
[ "${SPIN_SPLIT:-0}" = 1 ] && A="$A --spin-split" && R=${R}_spinsplit
[ "${SHARED_CONT:-0}" = 1 ] && A="$A --shared-continuum" && R=${R}_sharedcont
J="job $SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID"
echo "$(date '+%F %H:%M') $J on $(hostname) ($(nvidia-smi --query-gpu=name --format=csv,noheader)): $R chain $c" >> $LOG
$PY run.py $A --check-only || exit 1
echo "$(date '+%F %H:%M') $J: start long $R chain $c" >> $LOG
$PY run.py $A > $L/ppc_${cat}_${R}_long_chain${c}.log 2>&1
rc=$?; echo "$(date '+%F %H:%M') $J: end long $R chain $c (exit $rc)" >> $LOG
exit $rc
