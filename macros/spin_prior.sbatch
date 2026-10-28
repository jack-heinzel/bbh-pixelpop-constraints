#!/bin/bash
# Rusty, one GPU: empirical-Bayes spin prior draws (macros/spin_prior.py) for both runs, 2000 draws each
# (~0.05-0.1 s per draw on a GPU; ~2 s on a CPU). Writes results/post_processing/<run>_spin_prior.npz. Submit from
# the repository:
#     sbatch macros/spin_prior.sbatch [--n N] [<run> ...]
#SBATCH --job-name=ppc-spin-prior
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=a100-80gb|h100|h200
#SBATCH --exclude=workergpu168
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=results/logs/slurm-spin-prior-%j.out

cd "$SLURM_SUBMIT_DIR"
module purge                      # no cuda/cudnn modules: jax brings its own CUDA libraries
unset JAX_PLATFORMS
echo "$(date '+%F %H:%M') on $(hostname) ($(nvidia-smi --query-gpu=name --format=csv,noheader))"
JAX_PLATFORMS=cuda ~/venvs/gwjax313/bin/python macros/spin_prior.py --n 2000 "$@" || exit 1
echo "$(date '+%F %H:%M') done"
