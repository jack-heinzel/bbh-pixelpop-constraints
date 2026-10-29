#!/bin/bash
# Rusty, one GPU: the per-cell chi_eff-chi_p cross moments on every draw (notebooks/analysis.py rebuild_cross ->
# results/post_processing/for_plotting/<run>/chain_<c>_cross.h5), then the correlation figures
# (notebooks/paper_figs.py corr, with and without --core). Submit from the repository:
#     sbatch post_cross.sbatch [<run> ...]      (default: ppc_gwtc5_k10 ppc_gwtc5_k100_gauss)
#SBATCH --job-name=ppc-cross
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=a100-80gb|h100|h200
#SBATCH --exclude=workergpu168
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=results/logs/slurm-post-cross-%j.out

cd "$SLURM_SUBMIT_DIR"
module purge                      # no cuda/cudnn modules: jax brings its own CUDA libraries
module load texlive/20240312      # plot_style uses usetex
unset JAX_PLATFORMS
export XLA_PYTHON_CLIENT_MEM_FRACTION=.9 OMP_NUM_THREADS=1
RUNS=${*:-ppc_gwtc5_k10 ppc_gwtc5_k100_gauss}
PY=~/venvs/gwjax313/bin/python
echo "$(date '+%F %H:%M') on $(hostname) ($(nvidia-smi --query-gpu=name --format=csv,noheader)): $RUNS"
JAX_PLATFORMS=cuda $PY - $RUNS <<'PYEOF' || exit 1
import glob, os, sys
sys.path.insert(0, 'notebooks'); sys.path.insert(0, '.')
import analysis as A
for run in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(A.REPO, 'results', run, 'chain_*_samples.h5'))):
        print('wrote', A.rebuild_cross(run, f), flush=True)
PYEOF
JAX_PLATFORMS=cpu $PY notebooks/paper_figs.py corr || exit 1
JAX_PLATFORMS=cpu $PY notebooks/paper_figs.py corr --core || exit 1
echo "$(date '+%F %H:%M') done"
