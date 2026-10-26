#!/bin/bash
# Rusty, CPU: the peak fits behind the paper's macros, both runs, then the joint (m1, q) fits:
#     macros/peak_fits.py  -> macros/peak_fits.tex,  plots/<run>_peak_fits.pdf
#     macros/joint_fits.py -> macros/joint_fits.tex, plots/<run>_joint_fits.pdf, results/post_processing/<run>_joint_fits.npz
# The spin cache must be complete (post.sbatch). Submit from the repository:
#     sbatch macros/fits.sbatch
# genx shares nodes, which suits a small single-threaded job. Slurm copies this script at submission.
#SBATCH --job-name=ppc-fits
#SBATCH --partition=genx
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=results/logs/slurm-fits-%j.out

cd "$SLURM_SUBMIT_DIR"
module purge
module load texlive/20240312      # plot_style uses usetex
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
PY=~/venvs/gwjax313/bin/python
echo "$(date '+%F %H:%M') on $(hostname)"
$PY macros/peak_fits.py || exit 1
echo "$(date '+%F %H:%M') peak fits done"
$PY macros/joint_fits.py || exit 1
echo "$(date '+%F %H:%M') done"
