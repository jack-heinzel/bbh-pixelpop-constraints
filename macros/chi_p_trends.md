# Mean χ_p against m1 and q, across models and cuts (2026-11-24)

From `macros/chi_p_trends.py`, using all 2000 draws per run. Mean χ_p is the population mean of the per-cell conditional, averaged over the windowed p(m1, q). Values are median [90%] with P = posterior probability. Columns:

- **χ_p(21 M☉)**: mean χ_p at m1 ≈ 21 M☉, averaged over q in [0.3, 0.65].
- **P(≥ 0.54)**: the probability that this mean is at least 0.54, the expectation for a second-generation primary (spin χ_f = 0.69) merging with a nonspinning first-generation secondary at q ≈ 0.5.
- **60 − 10**: the change in mean χ_p from m1 = 10 to 60 M☉, averaged over q.
- **q trend**: mean χ_p at q ≈ 0.95 minus q ≈ 0.4, for m1 in 15–30 M☉.

| Run | χ_p(21 M☉) | P(≥ 0.54) | 60 − 10 | P(rise) | q trend (P > 0) |
|---|---|---|---|---|---|
| Flexible (k10) | 0.31 [0.23, 0.42] | 0.002 | +0.20 | 0.98 | −0.07 (0.19) |
| Flexible, Neff > 30 | 0.29 [0.19, 0.41] | 0.002 | +0.21 | 0.96 | −0.10 (0.16) |
| Flexible, Neff > 50 | 0.30 [0.20, 0.41] | 0.001 | +0.21 | 0.96 | −0.09 (0.16) |
| Flexible, Var < 1 | 0.32 [0.25, 0.41] | 0.000 | +0.15 | 0.98 | −0.02 (0.39) |
| Flexible, Var < 4 | 0.31 [0.23, 0.41] | 0.001 | +0.19 | 0.97 | −0.07 (0.20) |
| Flexible, k5 | 0.31 [0.22, 0.43] | 0.002 | +0.20 | 0.97 | −0.07 (0.20) |
| Split (k10) | 0.30 [0.22, 0.39] | 0.001 | +0.09 | 0.91 | −0.02 (0.27) |
| Gaussian, k10 | 0.30 [0.21, 0.39] | 0.002 | +0.10 | 0.93 | −0.01 (0.47) |
| Gaussian, k100 (baseline) | 0.30 [0.21, 0.38] | 0.001 | +0.07 | 0.88 | 0.00 (0.48) |
| Gaussian, Neff > 30 † | 0.36 [0.33, 0.38] | 0.000 | −0.01 | 0.43 | +0.01 (0.64) |
| Gaussian, Neff > 50 | 0.28 [0.20, 0.38] | 0.000 | +0.11 | 0.92 | −0.01 (0.43) |
| Gaussian, Var < 1 | 0.32 [0.26, 0.38] | 0.000 | +0.01 | 0.73 | 0.00 (0.54) |
| Gaussian, Var < 4 | 0.30 [0.23, 0.38] | 0.000 | +0.05 | 0.84 | 0.00 (0.47) |
| Additive, f1(m1) + f2(q) | 0.29 [0.20, 0.41] | 0.000 | +0.20 | 0.97 | 0.00 (0.47) |
| Additive, f1(m1) only | 0.28 [0.20, 0.37] | 0.001 | +0.23 | 0.98 | n/a |
| Additive, f2(q) only | 0.33 [0.26, 0.42] | 0.000 | +0.03 | 0.83 | −0.03 (0.30) |
| *Prior, at (21 M☉, q = 0.5)* | *Flexible 0.38 [0.07, 0.85]; Gaussian 0.50 [0.12, 0.88]* | *0.30 / 0.40* | *0* | *0.5* | |

† The intervals are suspiciously narrow, which points to a poorly mixed chain. Set aside.

## Summary

- **χ_p ≈ 0.54 at 21 M☉ is excluded in every run.** P ≤ 0.004, against 0.30–0.40 under the prior, so roughly 100–400 to 1. A pure 2G+1G population is ruled out. If first-generation mergers have χ_p ≈ 0.21 (the 10 M☉ value), a fraction of ≲ 60% second-generation mergers is still allowed.
- **Mean χ_p rises with m1 where the model lets it vary freely with mass.** The rise is +0.20 (P ≈ 0.97–0.98) in the Flexible model, the additive model and the m1-only model, and it is stable under the Neff and Var cuts. The Gaussian and split models give a weaker rise (+0.07 to +0.11, P = 0.84–0.93), and it vanishes in the Gaussian model with Var < 1. Most of the rise lies above the 33 M☉ peak (10 → 33 M☉: only +0.04 to +0.10), where χ_p is poorly measured: a ≲ 2σ, model-dependent hint.
- **No q dependence at fixed m1 in any run.** Even the q-only additive model, which can vary χ_p only with q, finds nothing. The apparent drop of χ_p with q when averaging over all masses comes from p(m1 | q): low-q systems sit at higher m1.
