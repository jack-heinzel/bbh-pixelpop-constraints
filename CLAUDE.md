# bbh-pixelpop-constraints

Companion to `~/public_html/mass-gap-investigation` (the hierarchical-merger interpretation paper). This repository
holds the **weakly modelled** analysis only: PixelPop mass field × a slowly mass-dependent (chi_eff, chi_p) field, with
**no gates, no channels and no 2G template** (user, 2026-10-20). Keep it that way: model choices that impose structure
belong in mass-gap-investigation. See README.md for the model.

## Decisions (user, 2026-10-20)

- Spins in (chi_eff, chi_p), not component spins.
- No Planck gates; no 2G gates or 2G spin component of any kind.
- Spin field mass dependence: k = 10 mass modes per axis (baseline), k = 5 (sensitivity).
- Spin-prior variants (user, 2026-10-20: "try these in order, with short chains"), `run.py --spin-model`: `field`
  (baseline), then `split` (G0(chi) + G1(m1, q, chi), separate amplitudes, separable G1 spectrum), then `gauss`
  (truncated 2D Gaussian whose mean, ln sd and atanh rho are Matern fields on (ln m1, q)). A two-component mixture
  with a Gaussian was discussed but is a channel/template, so it belongs in mass-gap-investigation unless the user
  relaxes the rule above.
- Catalog `gwtc5` (user, 2026-10-20, replacing `nrsur`): the O4b paper's August-rerun set (o4b_population/pixelpop-o4,
  FAR < 1/yr, 259 events: its rp.o4 directories, ignore list, NRSur > Mixed > SpinTaylor labels and VT), collected with
  pixelpop's variable-NPE loader (cap 20000) instead of upstream's downsampling; built by `collect/` (README). With GW231123; corrected likelihood, event
  Neff floor 100, no variance cut (as mass-gap-investigation).

## Conventions (shared with mass-gap-investigation)

- Environment `~/.conda/envs/gwjax313`. The node caps processes + threads at 500 per user: GPU jobs pin themselves
  (`run.py --cpus`, least busy cores) before jax is imported; CPU tests need `taskset` and the XLA single-thread flags.
- At most two GPUs at once, one job per GPU. Never edit a shell script while it runs: launch frozen copies.
- Never `rm -f`/`rm -rf` variable or glob paths; never `pkill -f`/`pgrep -f` with self-matching patterns.
- Compute the likelihood at the starting point before inference (`run.py --check-only`, `warmstart.py`).
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
- 2026-10-20 18:16 EDT: baseline `gwtc5` shorts, `launch_frozen_short_20261020.sh gwtc5 10 0 .4 -` and `... 5 1 .4 -`
  (warm start from the mass-gap bm3 short as before; on gwtc5 it gives ln L -7336, Var 123, min event Neff 3). k = 10 runs
  at 5.4 s/it (GPU 0 shared with another user's job, 92% busy), k = 5 at 1.3 s/it.
- 2026-10-20 18:40 EDT: `launch_frozen_queue_variants_v2_20261020.sh 630157 630156` waits on the baseline launchers, then
  runs `split` k = 10 on GPU 1 (warm start from `ppc_gwtc5_k5_short`) and `gauss` k = 10 on GPU 0 (from
  `ppc_gwtc5_k10_short`), via `launch_frozen_short_variants_20261020.sh`. Kill the queue by its PIDs to cancel. The
  baseline chains were started before model.py saved `spin_moments` / `log_p_spin_marginal_windowed`, so their
  notebooks rebuild these from the spin field; the variants save them.
- 2026-10-20 23:35 EDT: `launch_frozen_queue_gauss_k100_20261020.sh 796189`: after the split short (GPU 1), `gauss` with
  all 100 x 100 mass modes (`--spin-mass-modes 100 --spin-model gauss`, user: "we could try to use all modes"; only
  gauss can, its parameter fields have 5 x 1e4 sites against 1e8 for field/split at k = 100), warm-started from the
  baseline k = 10 short. Its field start uses the source's k (warmstart.py); tested on CPU from the partial k = 10
  chain: gauss k = 100 start ln L -7341 (field start -7276), Var 1.9, min event Neff 135.

