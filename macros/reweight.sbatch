#!/bin/bash
# Rusty, one GPU: population-reweighted event posteriors (macros/reweight.py compute) for the two baseline runs, every
# posterior draw (2 x 1000 each). ~0.07 s per draw on an A100, a few minutes per run. Submit from the repository:
#     sbatch macros/reweight.sbatch [<run> ...]
# Then, on the login node (pinned, texlive loaded): python macros/reweight.py report
#SBATCH --job-name=ppc-reweight
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=a100-80gb|h100|h200
#SBATCH --exclude=workergpu168
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=results/logs/slurm-reweight-%j.out

cd "$SLURM_SUBMIT_DIR"
module purge                      # no cuda/cudnn modules: jax brings its own CUDA libraries
unset JAX_PLATFORMS
export XLA_PYTHON_CLIENT_MEM_FRACTION=.9 OMP_NUM_THREADS=1
RUNS=${*:-ppc_gwtc5_k10 ppc_gwtc5_k100_gauss}
echo "$(date '+%F %H:%M') on $(hostname) ($(nvidia-smi --query-gpu=name --format=csv,noheader)): $RUNS"
JAX_PLATFORMS=cuda ~/venvs/gwjax313/bin/python macros/reweight.py compute $RUNS || exit 1
echo "$(date '+%F %H:%M') done"
