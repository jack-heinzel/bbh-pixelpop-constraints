#!/bin/bash
# Rusty, one GPU: rebuild the redshift-dependent spin conditionals on every draw (notebooks/analysis_z.py; caches in
# results/post_processing/for_plotting/<run>/chain_<c>_z.h5), then draw their figures (notebooks/z_figs.py).
# Submit from the repository once the chains are done:
#     sbatch post_z.sbatch [<run> ...]      (default: ppc_gwtc5_k10_kq5_kz5 ppc_gwtc5_k100_gauss_kz10)
#     sbatch --export=ALL,PER_RUN=1 post_z.sbatch <run> ...   (figures for each run on its own)
#     sbatch --export=ALL,MARGINALS=1 post_z.sbatch <run> ...  (only z_figs.py --marginals: m1, q, z marginals, runs
#                                                               overplotted; rebuilds caches without MARG)
#SBATCH --job-name=ppc-post-z
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=a100-80gb|h100|h200
#SBATCH --exclude=workergpu168
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=results/logs/slurm-post-z-%j.out

cd "$SLURM_SUBMIT_DIR"
module purge                      # no cuda/cudnn modules: jax brings its own CUDA libraries
module load texlive/20240312      # plot_style uses usetex
unset JAX_PLATFORMS
export XLA_PYTHON_CLIENT_MEM_FRACTION=.9 OMP_NUM_THREADS=1
RUNS=${*:-ppc_gwtc5_k10_kq5_kz5 ppc_gwtc5_k100_gauss_kz10}
PY=~/venvs/gwjax313/bin/python
echo "$(date '+%F %H:%M') on $(hostname) ($(nvidia-smi --query-gpu=name --format=csv,noheader)): $RUNS"
JAX_PLATFORMS=cuda $PY notebooks/analysis_z.py $RUNS || exit 1
if [ -n "$MARGINALS" ]; then              # MARGINALS=1: only the marginal comparison (runs overplotted)
    JAX_PLATFORMS=cpu $PY notebooks/z_figs.py --marginals $RUNS || exit 1
    JAX_PLATFORMS=cpu $PY notebooks/z_figs.py --marginals --core $RUNS || exit 1
    echo "$(date '+%F %H:%M') done"; exit 0
fi
if [ -n "$PER_RUN" ]; then GROUPS_=($RUNS); else GROUPS_=("$RUNS"); fi    # PER_RUN=1: each run's figures alone
for runs in "${GROUPS_[@]}"; do
    JAX_PLATFORMS=cpu $PY notebooks/z_figs.py $runs || exit 1
    JAX_PLATFORMS=cpu $PY notebooks/z_figs.py --core $runs || exit 1
done
echo "$(date '+%F %H:%M') done"
