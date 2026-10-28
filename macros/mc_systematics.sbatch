#!/bin/bash
# Rusty, one GPU: Monte Carlo systematics, sigma_syst / sigma_stat of the reported quantities (macros/mc_systematics.py),
# for every run with chains and a spin cache. First a 20-draw check of the factorized sum against population_error's
# explicit covariances (first run only), then the full computation. Submit from the repository:
#     sbatch macros/mc_systematics.sbatch [<run> ...]   (default: the five runs below)
#SBATCH --job-name=ppc-mcsyst
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=a100-80gb|h100|h200
#SBATCH --exclude=workergpu168
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=results/logs/slurm-mc-syst-%j.out

cd "$SLURM_SUBMIT_DIR"
module purge                      # no cuda/cudnn modules: jax brings its own CUDA libraries
unset JAX_PLATFORMS
export XLA_PYTHON_CLIENT_MEM_FRACTION=.9 OMP_NUM_THREADS=1
RUNS=${*:-ppc_gwtc5_k10 ppc_gwtc5_k100_gauss ppc_gwtc5_k10_gauss ppc_gwtc5_k10_split ppc_gwtc5_k5}
PY=~/venvs/gwjax313/bin/python
echo "$(date '+%F %H:%M') on $(hostname) ($(nvidia-smi --query-gpu=name --format=csv,noheader)): $RUNS"
JAX_PLATFORMS=cuda $PY macros/mc_systematics.py --check 20 ${RUNS%% *} || exit 1
JAX_PLATFORMS=cuda $PY macros/mc_systematics.py $RUNS || exit 1
echo "$(date '+%F %H:%M') done"
