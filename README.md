# bbh-pixelpop-constraints

Weakly modelled constraints on the binary black hole population (O1–O4b, 259 events at IFAR > 1 yr) with PixelPop: what the data say
about masses and spins, and how spins change with mass, without gates, channels or a second-generation template.

## Model (`model.py`)

    p(ln m1, q, chi_eff, chi_p) = p_mass(ln m1, q) · p(chi_eff, chi_p | ln m1, q)

- **Mass:** PixelPop's Matérn field on a 100 × 100 (ln m1, q) grid, m1 ∈ [3, 200] M☉, all modes.
- **Spins:** a Matérn log-density on 100 × 100 (chi_eff, chi_p) bins, restricted to the physical window
  chi_p < chi_p,max(chi_eff, q), whose shape changes smoothly with mass: the field is factorized over the lowest k × k
  Laplacian modes of the (ln m1, q) plane (k = 10 baseline, k = 5 sensitivity), all spin modes. A zero field is flat on
  the allowed region; there is no isotropic or template mean.
- **Spin-prior variants** (`--spin-model`, `model.py`): `split`, F = G0(chi) + G1(ln m1, q, chi) with separate
  amplitudes, G1 holding only mass-varying modes and a separable (mass x spin) Matérn spectrum, so sigma_1 measures the
  mass dependence; `gauss`, a correlated 2D Gaussian in (chi_eff, chi_p) truncated to the physical window, whose mean,
  ln sd and atanh rho are each an offset plus a k × k-mode Matérn field on (ln m1, q). All models save the windowed
  spin marginal and the conditional's per-(m1, q) mean and sd of chi_eff and chi_p. `warmstart.py` maps a field start
  onto either (exactly for split, via moments for gauss).
- **Other parameters:** redshift power law and the m2 > 3 M☉ window (GWTC-6 parametric models).
- **Likelihood:** rate-marginalized, selection-bias correction N(N+1)σ²_sel/2, event effective-sample-size floor 100,
  no variance cut.

The model is the `mass_gate='none'`, `frac_2g_gate='none'` case of mass-gap-investigation's `hierarchical.py`;
`tests/test_model.py` checks they give the same likelihood.

## Data (`collect/`)

Catalog `gwtc5`: the O4b population paper's August-rerun set
(`~/public_html/o4b_population/pixelpop-o4/code/collection/collect_o4b_chieff_chip.ini`), 259 O1–O4b BBHs at
**FAR < 1/yr (IFAR > 1 yr)**. Same files, ignore list, labels, VT and thresholds as that ini:

- files: rp.o4's GWTC-2.1 and GWTC-3 (v1) data releases, GWTC-4.1 Stable_Release-3 and GWTC-5 Stable_Release-8
  `bbh_only` trees (scraped at FAR < 1/yr); the ini's ignore list removes sub-threshold and NS-containing O1–O3 events
  and GW190814, GW190917;
- labels: first present of NRSur7dq4 > Mixed > IMRPhenomXPHM-SpinTaylor (C01:Mixed for O1–O3; C00:NRSur7dq4 else
  C00:Mixed for O4a; NRSur7dq4 else SpinTaylor for O4b);
- VT: `multirun-mixtures_20260410130052UTC_corrected/mixture-semi_o1_o2-real_o3_o4a_o4b-cartesian_spins_...-clipped.hdf`,
  IFAR > 1, O1/O2 semianalytic SNR > 10.

The difference is the collector: the reference used upstream `gwpopulation_pipe_collection`, which cuts every event to
the smallest one's 1993 samples. Here pixelpop's collector keeps each event's own samples (**variable NPE**, cap 20000;
per-event counts 1993–134k, median 17k), padding short events with prior = +inf rows. Each event's PE prior is
evaluated in the cosmology its samples were drawn with (Planck15_LAL for O1–O3 and for GWTC-5's GW250114 NRSur7dq4
group, astropy Planck15 for the rest of O4; `dry_run.py` verifies d_L(z) per file). `dry_run.py` and `check_labels.py`
check events, files and labels against the reference's `event_data.json`.

```bash
cd collect && bash run_collect.sh > logs/collect.log 2>&1   # dry run, collect, check
```

Output: `data/gwtc5_ifar1_chieffchip/data/{posteriors,injections}.pkl` (gitignored; `PPC_DATA_DIR` reads it from
elsewhere). The earlier catalogs `nrsur`, `physical`, `mixed` (mass-gap-investigation's IFAR > 4 yr collections, 236
events, `PPC_DATA_ROOT`) stay loadable for comparison.

## Running

```bash
# environment: ~/.conda/envs/gwjax313 (pixelpop 0.4.0.dev1 with the per-file cosmology collection edit)
python run.py gwtc5 --spin-mass-modes 10 --device 0 --mem-fraction .4 --check-only     # ln L at the start, speed
python warmstart.py gwtc5 --spin-mass-modes 10 --device 0 --source <chain h5> --out results/warm.npz
python run.py gwtc5 --spin-mass-modes 10 --device 0 --mem-fraction .4 [--short] --init-from results/warm.npz
bash launch.sh gwtc5 10 0 .4          # warm start, short chain, long chain (launch a frozen copy)
[SRC=<chain h5>] bash launch_short.sh gwtc5 10 0 .4 - --spin-model split   # warm start + short only
python notebooks/builders/build_run_nb.py <run>   # notebooks/<run>.ipynb; execute on pinned CPUs
JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
    taskset -c <4 cores> python -m pytest tests -q
```

Chains go to `results/<run name>/chain_0_samples.h5` (gitignored). Run names: `ppc_<catalog>_k<k>[_short][...]`.
