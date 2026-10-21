# bbh-pixelpop-constraints

Companion to `~/public_html/mass-gap-investigation` (the hierarchical-merger interpretation paper). This repository
holds the **weakly modelled** analysis only: PixelPop mass field × a slowly mass-dependent (chi_eff, chi_p) field, with
**no gates, no channels and no 2G template** (user, 2026-10-20). Keep it that way: model choices that impose structure
belong in mass-gap-investigation. See README.md for the model.

## Decisions (user, 2026-10-20)

- Spins in (chi_eff, chi_p), not component spins.
- No Planck gates; no 2G gates or 2G spin component of any kind.
- Spin field mass dependence: k = 10 mass modes per axis (baseline), k = 5 (sensitivity).
- Catalog `gwtc5` (user, 2026-10-20, replacing `nrsur`): the O4b paper's August-rerun set (o4b_population/pixelpop-o4,
  FAR < 1/yr, 259 events: its rp.o4 directories, ignore list, NRSur > Mixed > SpinTaylor labels and VT), collected with
  pixelpop's variable-NPE loader (cap 20000) instead of upstream's downsampling; built by `collect/` (README). With GW231123; corrected likelihood, event
  Neff floor 100, no variance cut (as mass-gap-investigation).
- PE samples (user, 2026-10-20): at most 20000 per event (`run.py --max-samples`, `data.subsample`; 10000 for the
  short chains, now `ppc_gwtc5_k*_npe10000_short`, chosen for speed before the jax 0.9 environment), drawn
  greedily from the usable ones (inside the grid, in (ln m1, q) and (chi_eff, chi_p) bins injections reach; fixed seed).
  Unusable samples are 9.7e-4 of the total, at most 2.3% for one event (GW230704, mostly z > 1.9).
- Initialization (user, 2026-10-20, replacing the warm start): every chain starts from a random prior draw of all latent
  sites, keyed by `--seed`; `launch.sh` runs the short then the long chain, no `warmstart.py`.

## Conventions (shared with mass-gap-investigation)

- Environment `~/venvs/gwjax313` (venv on the `python/3.13.2` module; activate with `gwjax`; jax 0.9.0 + CUDA 13, numpyro 0.21.0, as on LIGO: jax 0.12.0 makes ln L + gradient 2.5-3.8x slower on an A100, `bench.py`, 2026-10-20). The node caps processes + threads at 500 per user: GPU jobs pin themselves
  (`run.py --cpus`, least busy cores) before jax is imported; CPU tests need `taskset` and the XLA single-thread flags.
- At most two GPUs at once, one job per GPU. Never edit a shell script while it runs: launch frozen copies.
- Never `rm -f`/`rm -rf` variable or glob paths; never `pkill -f`/`pgrep -f` with self-matching patterns.
- Compute the likelihood at the starting point before inference (`run.py --check-only`, same `--seed` as the run).
- Report times in US Eastern (the node clock is Pacific).
- Plots: follow `~/public_html/mass-spin-correlations/plots/CLAUDE.md`; `plot_style.apply()`, no titles, PDF to `plots/`.
  PixelPop rates must be windowed (m2 > 3 M☉) before plotting; the saved `log_p_mass` is unwindowed.
- Commit or push only when asked.

## Runs

- 2026-10-20 09:36 EDT: `launch_frozen_20261020.sh nrsur 10 0 .4` and `... nrsur 5 1 .4` (GPUs 0, 1): warm start from
  mass-gap-investigation's `lm1q_hierarchical_nrsur_varcutinf_short_bm3_s100_flat_ellipse_neff100_lcorr_sep2g` best
  sample (mass field copied; its 3 × 3-mode spin field embedded, amplitudes rescaled to the same field), then a short
  chain (2000 warmup, 500 samples at thinning 5), then a long chain (4000 + 1000 at thinning 5).
  At the starting points: cold start ln L −7220, Var 91–104, min event Neff 23–29; warm start (k = 10) ln L −6633,
  Var 122, min Neff 13 (the high-spin events, which the source fit with its 2G ellipse). Value+gradient 24 ms (k = 10),
  10 ms (k = 5) on an A100.
- 2026-10-20 11:35 EDT: both stopped for a cluster reboot (user), launchers and runs killed by PID. k = 5 short was sampling
  (partial chain in `results/ppc_nrsur_k5_short/`); k = 10 short was at warmup step 1827/2000 (no samples). Relaunch with
  `launch.sh` (frozen copy); `run.py` refuses a run directory that already has chains, so the k = 5 short needs `--tag` or
  moving its partial chain aside.
