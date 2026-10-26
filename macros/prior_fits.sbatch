#!/bin/bash
# Rusty, CPU: the window fits on empirical-Bayes prior draws of the mass field (macros/prior_fits.py), for the given
# runs (default: both). Four windows in parallel, ~0.5 s per draw and window: ~20-30 min per run. Submit from the
# repository:
#     sbatch macros/prior_fits.sbatch [<run> ...]
#SBATCH --job-name=ppc-prior-fits
#SBATCH --partition=genx
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=results/logs/slurm-prior-fits-%j.out

cd "$SLURM_SUBMIT_DIR"
module purge
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
echo "$(date '+%F %H:%M') on $(hostname): $*"
~/venvs/gwjax313/bin/python macros/prior_fits.py "$@" || exit 1
echo "$(date '+%F %H:%M') done"
