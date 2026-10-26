#!/bin/bash
# Rusty, CPU: the joint (m1, q) window fits for the given runs (default: both), then macros/joint_fits.tex from every
# run with saved fits. ~0.5 s per draw and window: ~1.3 h per run. Submit from the repository:
#     sbatch macros/joint_fits.sbatch [<run> ...]
#SBATCH --job-name=ppc-joint
#SBATCH --partition=genx
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=results/logs/slurm-joint-%j.out

cd "$SLURM_SUBMIT_DIR"
module purge
module load texlive/20240312      # plot_style uses usetex
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
echo "$(date '+%F %H:%M') on $(hostname): $*"
~/venvs/gwjax313/bin/python macros/joint_fits.py "$@" || exit 1
echo "$(date '+%F %H:%M') done"
