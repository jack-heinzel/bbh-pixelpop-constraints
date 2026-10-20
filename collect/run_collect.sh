#!/bin/bash
# Collect the FAR < 1/yr O1-O4b BBH catalog (the O4b paper set; catalogs.py) (PE + VT at IFAR > 1) and check the result.
# Run from this directory:  bash run_collect.sh > logs/collect.log 2>&1
#
# Thread caps: the node limits processes + threads per user, so unconstrained BLAS/XLA thread pools hit
# "pthread_create failed". The ini uses backend=numpy.
set -e

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

PY=~/.conda/envs/gwjax313/bin/python
INI=collect_gwtc5_ifar1_chieffchip.ini

$PY dry_run.py $INI               # every file resolves to its label and cosmology
$PY collect.py $INI
$PY check_labels.py $INI
