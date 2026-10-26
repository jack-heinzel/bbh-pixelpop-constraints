#!/bin/bash
# Rusty, CPU: conditional-prior ("inpainting") significance of the window peaks (macros/inpaint.py), both runs, at
# buffers of 0 and 10 m1 bins. ~10 min per run and buffer; writes ~3-6 GB of realizations per run and buffer to
# results/post_processing/. Submit from the repository:
#     sbatch macros/inpaint.sbatch
#SBATCH --job-name=ppc-inpaint
#SBATCH --partition=genx
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=results/logs/slurm-inpaint-%j.out

cd "$SLURM_SUBMIT_DIR"
module purge
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
echo "$(date '+%F %H:%M') on $(hostname)"
for b in 0 10; do
    ~/venvs/gwjax313/bin/python macros/inpaint.py --buffer $b || exit 1
done
echo "$(date '+%F %H:%M') done"
