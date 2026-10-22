#!/bin/bash
# Rusty: rebuild the notebooks' spin quantities on every draw (results/post_processing/for_plotting/<run>/chain_<c>.h5,
# notebooks/analysis.py) on one GPU, then execute the main notebooks. Submit from the repository:
#     sbatch post.sbatch                                    (the runs below)
#     RUNS=ppc_gwtc5_k10_gauss NOTEBOOKS=ppc_gwtc5_k10_gauss,compare_gwtc5_k10_field_vs_gauss sbatch post.sbatch
# RUNS and NOTEBOOKS are lists separated by commas or spaces.
# Chains whose cache is complete are skipped. Slurm copies this script at submission, so editing it does not affect
# queued or running jobs.
#SBATCH --job-name=ppc-post
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=a100-80gb|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=results/logs/slurm-post-%j.out

cd "$SLURM_SUBMIT_DIR"
module purge                      # no cuda/cudnn modules: jax brings its own CUDA libraries
module load texlive/20240312      # plot_style uses usetex
unset JAX_PLATFORMS
PY=~/venvs/gwjax313/bin/python
RUNS=${RUNS:-ppc_gwtc5_k10 ppc_gwtc5_k5 ppc_gwtc5_k10_npe10000_short}
NOTEBOOKS=${NOTEBOOKS:-ppc_gwtc5_k10 ppc_gwtc5_k5 compare_gwtc5_k5_vs_k10_long compare_gwtc5_k10_short_vs_long}
RUNS=${RUNS//,/ }; NOTEBOOKS=${NOTEBOOKS//,/ }

echo "$(date '+%F %H:%M') on $(hostname) ($(nvidia-smi --query-gpu=name --format=csv,noheader))"
$PY - $RUNS <<'EOF' || exit 1
import glob, os, sys
sys.path.insert(0, 'notebooks'); sys.path.insert(0, '.')
import h5py
import analysis as A
for run in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(A.REPO, 'results', run, 'chain_*_samples.h5'))):
        with h5py.File(f, 'r') as h:
            n = len(h['log_likelihood'])
        p = A.post_path(run, f)
        if os.path.exists(p):
            with h5py.File(p, 'r') as g:
                if g.attrs['n_draws'] >= n:
                    print('complete:', p, flush=True)
                    continue
        print('wrote', A.rebuild_chain(run, f), flush=True)
EOF
for nb in $NOTEBOOKS; do
    echo "$(date '+%F %H:%M') execute notebooks/$nb.ipynb"
    ~/venvs/gwjax313/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
        notebooks/$nb.ipynb || echo "FAILED: $nb"
done
echo "$(date '+%F %H:%M') done"
