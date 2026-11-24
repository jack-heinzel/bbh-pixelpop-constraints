#!/bin/bash
# Rusty, one GPU: the extra-regions spin cache (analysis.rebuild_regions for paper_figs.EXTRA_REGIONS, Fig. 2's
# 15-30 and 45-75 Msun regions; user, 2026-11-18) for the two baseline runs, every draw, then the population-reweighted
# posteriors with the same feature regions (macros/reweight.py compute). Submit from the repository:
#     sbatch notebooks/regions.sbatch
# Then, on the login node (pinned, texlive loaded): paper_figs.py spin / talk, macros/peak_spins.py,
# macros/reweight.py report, macros/mc_syst_figs.py.
#SBATCH --job-name=ppc-regions
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=a100-80gb|h100|h200
#SBATCH --exclude=workergpu168
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=results/logs/slurm-regions-%j.out

cd "$SLURM_SUBMIT_DIR"
module purge                      # no cuda/cudnn modules: jax brings its own CUDA libraries
module load texlive/20240312      # paper_figs imports plot_style
unset JAX_PLATFORMS
export XLA_PYTHON_CLIENT_MEM_FRACTION=.9 OMP_NUM_THREADS=1 JAX_PLATFORMS=cuda
PY=~/venvs/gwjax313/bin/python
echo "$(date '+%F %H:%M') on $(hostname) ($(nvidia-smi --query-gpu=name --format=csv,noheader))"
$PY - ppc_gwtc5_k10 ppc_gwtc5_k100_gauss <<'PYEOF' || exit 1
import glob, os, sys
sys.path[:0] = ['notebooks', 'macros', '.']
import analysis as A
import paper_figs as P
print('extra regions:', P.EXTRA_REGIONS, flush=True)
for run in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(A.REPO, 'results', run, 'chain_*_samples.h5'))):
        print('->', A.rebuild_regions(run, f, P.EXTRA_REGIONS), flush=True)
PYEOF
$PY macros/reweight.py compute || exit 1
echo "$(date '+%F %H:%M') done"
