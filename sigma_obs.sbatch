#!/bin/bash
# Rusty, one GPU: chi_eff moments of the detectable population against z (notebooks/sigma_obs.py), from each draw's own
# injection weights, for the three Gaussian redshift models, then the figure. Submit from the repository:
#     sbatch sigma_obs.sbatch [<run> ...]      (default: sigma_obs.RUN_ARGS; N=<draws> to change the 500 draws)
#SBATCH --job-name=ppc-sigobs
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=a100-80gb|h100|h200
#SBATCH --exclude=workergpu168
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=results/logs/slurm-sigma-obs-%j.out

cd "$SLURM_SUBMIT_DIR"
module purge                      # no cuda/cudnn modules: jax brings its own CUDA libraries
module load texlive/20240312      # plot_style uses usetex
unset JAX_PLATFORMS
export XLA_PYTHON_CLIENT_MEM_FRACTION=.9 OMP_NUM_THREADS=1
PY=~/venvs/gwjax313/bin/python
echo "$(date '+%F %H:%M') on $(hostname) ($(nvidia-smi --query-gpu=name --format=csv,noheader)): ${*:-default runs}"
JAX_PLATFORMS=cuda $PY notebooks/sigma_obs.py --n ${N:-0} "$@" || exit 1    # N = 0: every draw (user, 2026-11-03: paper plots use all draws)
JAX_PLATFORMS=cpu $PY notebooks/sigma_obs.py --figure || exit 1
JAX_PLATFORMS=cpu $PY notebooks/sigma_obs.py --figure --traces 500 || exit 1
echo "$(date '+%F %H:%M') done"
