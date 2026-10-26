# Peak significance in the PixelPop mass field: notes (2026-10-26)

Question: is there statistical evidence for a *peak* at ~10, ~20, ~35 and ~60 $M_\odot$ (the first two near equal mass,
the others at $q \sim 0.5$)? This is different from asking whether the data *inform* the field there.

## What has been tried here, and why it falls short

| approach | what it measures | problem |
|---|---|---|
| Gaussian + power-law window fits (`macros/peak_fits.py`, `macros/joint_fits.py`) | interpretable peak parameters: $\mu$, $\sigma$, $\mu_q$, $f_G$ | the fit always finds "a peak"; it says nothing about significance |
| Same fits on empirical-Bayes prior draws (`macros/prior_fits.py`) | which statistics the data constrain beyond the prior | random prior fields give a fitted peak in 70–85% of draws, with median $f_G \approx 0.6$; it measures how tightly things are located, not whether there is a bump |
| Savage–Dickey at $f_G = 0$ | a Bayes factor against "no Gaussian in the window" | $f_G = 0$ means an exact power law, which almost no field satisfies; capped at about $N_{\rm post} P_{\rm prior}$; needs the full hyperprior, not the empirical-Bayes one |
| Conditional-prior inpainting (`macros/inpaint.py`, `macros/conditional_prior.md`) | whether the window's field is unusual given the field outside | the kernel is fitted to the field, so the posterior realization is a typical prior draw by construction; low power |
| Likelihood ratio, posterior field against the inpainted field | how much the data lose when the window is smoothed | measures how strongly the data inform the field there, not whether a peak is present; positive even for noise-fitting |

## 1. Excess over a straight line in $(\ln m_1, \ln p)$

The approach used before: in each window, draw a straight line in $\ln p$ against $\ln m_1$ (a power law) from the
window's lower end to its upper end. Then measure the excess of the density above that chord.

Per posterior draw $i$, on $p_i(m_1 \mid q \in Q)$ or $p_i(m_1, q \in Q)$ (windowed, $m_2 > 3\,M_\odot$):

- chord: $\ell_i(\ln m) = \ln p_i(m_a) + \frac{\ln m - \ln m_a}{\ln m_b - \ln m_a}\,[\ln p_i(m_b) - \ln p_i(m_a)]$;
- candidate statistics:
  - the maximum log residual, $\max_{m \in [m_a, m_b]} [\ln p_i(m) - \ell_i(\ln m)]$. This is the `bump` statistic in
    `macros/inpaint.py` (on $p(m_1, q \in Q)$, with the end bins taken as the window's first and last bins);
  - the integrated positive excess, $\int_{m_a}^{m_b} [p_i(m) - e^{\ell_i}]_+\,dm$, as a fraction of the window's
    probability;
  - the signed integrated excess, without $[\cdot]_+$, which is symmetric under the null and so better behaved.
- report the posterior on the statistic and $P(\text{statistic} > 0 \mid d)$.

Notes:

- The chord is a convex-hull test: any concave bump in $\ln p$ sits above it. A curving but featureless density, such
  as the turnover at the low-mass end, also gives an excess. So the window ends matter: they should sit where the
  density is not itself curving.
- **The look-elsewhere effect is the main problem.** The windows and their ends were chosen by eye after seeing the
  field. Two ways to handle it:
  - fix the windows *a priori* (e.g. in an earlier catalogue) and quote local significances;
  - scan: compute the statistic over all windows of a family (fixed width in $\ln m_1$, sliding over the grid, and
    the q bands) and use the **maximum** over the scan as the test statistic. The null distribution of that maximum
    (from §3) gives a global p-value.
- Without §3, $P(\text{statistic} > 0 \mid d)$ is a posterior probability, not a p-value. Under a featureless
  population, noise in the field already gives positive residuals about half the time, or more for $[\cdot]_+$.

## 2. A parametric model with explicit peaks

Fit a parametric mass model with components at ~10, 20, 35 and 60 $M_\odot$ (e.g. a power law plus Gaussians, each
with its own $q$ pairing), and check whether each peak's mixing fraction is constrained away from zero. Alternatively,
compute the Bayes factor against the model without that peak (Savage–Dickey on the mixing fraction, or nested
sampling).

Problems:

- **Misspecification.** The parametric model is wrong somewhere, and a misspecified model fits the data by distorting
  whatever component is most flexible. A Gaussian can soak up a shoulder, a turnover or a mis-modelled power law. A
  nonzero mixing fraction then means "a Gaussian improves this wrong model", not "there is a peak".
- The Bayes factor depends on the priors on the peak's location, width and fraction.
- It is still useful as a cross-check, especially if the peaks it finds match PixelPop's locations and the result is
  robust to the background model (e.g. a broken power law against a smoothing spline).

## 3. Gold standard: mock catalogues from a featureless population

This gives an actual p-value for a given feature, under this model and this analysis.

1. **Null population.** Take the PixelPop population (e.g. the posterior median field, or a few posterior draws) and
   remove the feature, e.g. the ~35 $M_\odot$ peak. Ways to do it:
   - replace $\ln p$ in the window by the chord of §1;
   - replace it by the inpainted conditional mean from outside (`inpaint.load_realizations` returns this field as
     `smooth`).

   Keep everything else, including the spin field and the other peaks, so the null differs only in the feature being
   tested.
2. **Mock catalogues.** Draw detected events from the null population through the selection function, e.g. by
   reweighting the injection set to the null population and resampling it, with $N_{\rm det}$ matching the real
   catalogue (259) or Poisson-fluctuating around it. Then simulate PE posteriors for each event. **This is the
   hardest step.** Options, roughly in order of cost:
   - Gaussian likelihoods in (chirp mass, $q$, $\chi_{\rm eff}$) with SNR-dependent widths calibrated on the real
     events;
   - reuse the shapes of real events' PE samples, shifted to the mock event's parameters;
   - full PE on simulated signals (unaffordable at this scale).
3. **Inference.** Run *exactly* the same PixelPop inference (same grid, priors, spin model and chain settings) and the
   same post-processing on each mock.
4. **Null distribution.** Compute the significance statistic on each mock: the §1 chord excess, or the scan maximum
   to include the look-elsewhere effect. Its distribution over mocks is the null. The real catalogue's value gives
   $p = \#\{\text{mocks} \geq \text{observed}\} / N_{\rm mock}$.
5. **Power (injection–recovery).** Repeat with the feature left in the population, to show the statistic actually
   detects peaks of the inferred size.

Cost: each mock is a full inference (one long chain is GPU-hours here). $N_{\rm mock} \sim 50$–100 resolves
$p \sim 0.01$–0.02. Anything smaller needs either a tail fit to the null distribution, or a cheaper statistic that
behaves like the full inference.

Choices to state:

- the statistic, the windows and the scan family are fixed *before* looking at the mocks;
- the null keeps the other features. Testing all four peaks needs one null per peak, or a single fully smooth null if
  the question is "is there any peak";
- the posterior uncertainty in the null population is handled either by drawing each mock's null from a different
  posterior draw (a parametric-bootstrap / posterior-predictive null) or by fixing the median.

## Pieces already in this repository

- `macros/inpaint.py`:
  - `bump`: the max chord residual of §1, per posterior draw;
  - `load_realizations(run, peak, buffer)`: the posterior field, the inpainted smooth-continuation field (a ready-made
    null population for §3) and conditional realizations;
  - output in `results/post_processing/<run>_inpaint_b<buffer>.{npz,h5}`.
- Field ↔ latent map for running a modified field through `model.py`: every non-DC mode has a nonzero amplitude, so
  white sites $= (v_m^\top x\, v_q) / a$ (`model.matern_spectrum` gives $a$).
- Window definitions: `macros/peak_fits.py` (`PEAKS`).
