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
  relaxes the rule above. Variants on Rusty (user, 2026-10-21): long chains only (no shorts), `submit_variants.sbatch`
  (split, gauss at k = 10, chains 0 and 1). A Gaussian + linear-space perturbation mixture was raised again
  (2026-10-21) and is on hold.
- Catalog `gwtc5` (user, 2026-10-20, replacing `nrsur`): the O4b paper's August-rerun set (o4b_population/pixelpop-o4,
  FAR < 1/yr, 259 events: its rp.o4 directories, ignore list, NRSur > Mixed > SpinTaylor labels and VT), collected with
  pixelpop's variable-NPE loader (cap 20000) instead of upstream's downsampling; built by `collect/` (README). With GW231123; corrected likelihood, event
  Neff floor 100, no variance cut (as mass-gap-investigation).
- PE samples (user, 2026-10-20): at most 20000 per event (`run.py --max-samples`, `data.subsample`; 10000 for the
  short chains, now `ppc_gwtc5_k*_npe10000_short`, chosen for speed before the jax 0.9 environment), drawn
  greedily from the usable ones (inside the grid, in (ln m1, q) and (chi_eff, chi_p) bins injections reach; fixed seed).
  Unusable samples are 9.7e-4 of the total, at most 2.3% for one event (GW230704, mostly z > 1.9).
- Initialization (user, 2026-10-20, replacing the warm start): every chain starts from a random prior draw of all latent
  sites, keyed by `--seed`; `launch.sh` runs the short then the long chain, no `warmstart.py`. Warm starts are low
  priority (user, 2026-10-21); `launch_short.sh` warm-starts only when `SRC` is set.

## Conventions (shared with mass-gap-investigation)

- Environment: on Rusty `~/venvs/gwjax313` (venv on the `python/3.13.2` module; activate with `gwjax`), on the LIGO
  cluster (ldas-pcdev12) `~/.conda/envs/gwjax313`; both jax 0.9.0 + CUDA 13, numpyro 0.21.0 (jax 0.12.0 makes ln L +
  gradient 2.5-3.8x slower on an A100, `bench.py`, 2026-10-20). The launch and collection scripts pick whichever exists. The node caps processes + threads at 500 per user: GPU jobs pin themselves
  (`run.py --cpus`, least busy cores) before jax is imported; CPU tests need `taskset` and the XLA single-thread flags.
- One job per GPU. On the LIGO cluster at most two GPUs at once; on Rusty up to 16 (the `gpu` QOS limit per user:
  16 GPUs, 16 jobs, 288 CPUs, 4000 GB; user, 2026-10-21). Never edit a shell script while it runs: launch frozen copies.
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
- Merge of the Rusty branch (2026-10-21, commit 0424e73: prior-draw init, `--chain`, `--max-samples` / `data.subsample`).
  The LIGO-cluster runs of 2026-10-20/21 above (`ppc_gwtc5_k{5,10}_short`, `..._k10_split_short`, `..._k{10,100}_gauss_short`)
  predate it: warm-started, and with every collected sample (up to the collection's 20000 cap, unusable ones included
  at zero weight), so their ln L differ from post-merge runs by a per-event constant.
- 2026-10-21 (Rusty): long chains `ppc_gwtc5_k10`, `ppc_gwtc5_k5` (2 chains x 1000 draws each, `submit.sbatch`) done.
  Notebook spin quantities are rebuilt on every draw and cached in `results/post_processing/for_plotting/<run>/`
  (`notebooks/analysis.py`); build them and execute the main notebooks with `sbatch post.sbatch` (job 7300264: 9 min on
  one GPU, 0.04-0.07 s per draw). Rusty login nodes cap each user at 1 CPU and 8 GB in total, so do this on a GPU node,
  not the login node. Older notebooks are in `notebooks/exploratory/`.
- 2026-10-21 17:02-20:38 EDT (Rusty): variant long chains, `submit_variants.sbatch` (jobs 7300920, 7303103, 7303640):
  split k = 10 chains 0, 1; gauss k = 10 chains 0, 1 (+ chain 2); gauss k = 100 chains 0, 1. Three of these froze
  early in warmup (step size ~1e-2 -> ~1e-6, parameters stuck far from the posterior: gauss k = 10 chain 0, moved to
  `results/ppc_gwtc5_k10_gauss/stuck_chain0/`; split chain 1; gauss k = 100 chain 0). The starting point's Var(ln L) /
  min Neff does not predict which chains freeze. Gauss k = 10 notebooks use chain 1 (chain 2 pending). Notebook
  finding: sd chi_eff given m1 (averaged over p(q | m1)) is broad at ~18-20 and >~45 Msun, narrow at the ~10 and ~33
  peaks, mostly from the width at fixed q (low q), not from per-q mean shifts; the q ~ 1 slice hides it.
  Notebooks run on the login node pinned (`taskset`), or jax exceeds the 256 processes + threads per user.
- 2026-10-21/22 (Rusty): replacements as chain 2 (jobs 7303103, 7304010, 7304012; split chain 1 moved to
  `results/ppc_gwtc5_k10_split/stuck_chain1/`, gauss k = 100 chain 0 wrote nothing). Healthy chains: split k = 10 chains
  0, 2; gauss k = 10 chains 1, 2; gauss k = 100 chains 1, 2 (ln L -7301, -7298, -7300 vs field -7295). Notebooks and
  `compare_gwtc5_spin_models` via `post.sbatch` (job 7308668). Gauss k = 100 matches gauss k = 10 (chi_eff widths within
  ~0.01), so k = 10 misses no structure; all four models show the chi_eff width pattern above, split least confidently
  (P ~ 0.85 vs 0.98-0.99 for gauss).

