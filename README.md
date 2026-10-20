# bbh-pixelpop-constraints

Weakly modelled constraints on the binary black hole population (O1–O4b, 236 events) with PixelPop: what the data say
about masses and spins, and how spins change with mass, without gates, channels or a second-generation template.

## Model (`model.py`)

    p(ln m1, q, chi_eff, chi_p) = p_mass(ln m1, q) · p(chi_eff, chi_p | ln m1, q)

- **Mass:** PixelPop's Matérn field on a 100 × 100 (ln m1, q) grid, m1 ∈ [3, 200] M☉, all modes.
- **Spins:** a Matérn log-density on 100 × 100 (chi_eff, chi_p) bins, restricted to the physical window
  chi_p < chi_p,max(chi_eff, q), whose shape changes smoothly with mass: the field is factorized over the lowest k × k
  Laplacian modes of the (ln m1, q) plane (k = 10 baseline, k = 5 sensitivity), all spin modes. A zero field is flat on
  the allowed region; there is no isotropic or template mean.
- **Other parameters:** redshift power law and the m2 > 3 M☉ window (GWTC-6 parametric models).
- **Likelihood:** rate-marginalized, selection-bias correction N(N+1)σ²_sel/2, event effective-sample-size floor 100,
  no variance cut.

The model is the `mass_gate='none'`, `frac_2g_gate='none'` case of mass-gap-investigation's `hierarchical.py`;
`tests/test_model.py` checks they give the same likelihood.

## Data

The PE and injection collections live in mass-gap-investigation (`data/<catalog>_o1o4b_chieffchip/data/`): the same
236 BBHs (IFAR > 4 yr, GW190814 excluded) and VT mixture as there; catalog `nrsur` uses NRSur7dq4 PE where available.
Set `PPC_DATA_ROOT` to read them from elsewhere.

## Running

```bash
# environment: ~/.conda/envs/gwjax313 (pixelpop 0.4.0.dev1 with the per-file cosmology collection edit)
python run.py nrsur --spin-mass-modes 10 --device 0 --mem-fraction .4 --check-only     # ln L at the start, speed
python warmstart.py nrsur --spin-mass-modes 10 --device 0 --source <chain h5> --out results/warm.npz
python run.py nrsur --spin-mass-modes 10 --device 0 --mem-fraction .4 [--short] --init-from results/warm.npz
bash launch.sh nrsur 10 0 .4          # warm start, short chain, long chain (launch a frozen copy)
JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
    taskset -c <4 cores> python -m pytest tests -q
```

Chains go to `results/<run name>/chain_0_samples.h5` (gitignored). Run names: `ppc_<catalog>_k<k>[_short][...]`.
