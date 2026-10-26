#!/bin/bash
# Rusty, CPU: the global (m1, q) fit per draw (macros/global_fit.py), both runs, draws spread over the job's cores:
#     -> macros/global_fit.tex, plots/<run>_global_fit.pdf, results/post_processing/<run>_global_fit.npz
# About 4 s per draw on one core (2 x 2000 draws: ~4.5 CPU hours). Submit from the repository:
#     sbatch macros/global_fit.sbatch
# Slurm copies this script at submission.
#SBATCH --job-name=ppc-global
#SBATCH --partition=genx
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=results/logs/slurm-global-%j.out

cd "$SLURM_SUBMIT_DIR"
module purge
module load texlive/20240312      # plot_style uses usetex
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
echo "$(date '+%F %H:%M') on $(hostname), $SLURM_CPUS_PER_TASK cpus"
~/venvs/gwjax313/bin/python macros/global_fit.py --cpus $SLURM_CPUS_PER_TASK || exit 1
echo "$(date '+%F %H:%M') done"
