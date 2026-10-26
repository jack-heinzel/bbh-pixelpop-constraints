# Conditional prior for a window: brainstorm (2026-10-26)

Goal: a reference distribution for the fits in a window (e.g. $m_1 \in [25, 40]\,M_\odot$). For each posterior draw,
keep the kernel parameters $(\ln\sigma, \ln\nu, \ln\rho)$ **and** the field realization *outside* the window. Then draw
the field *inside* the window from the Gaussian-field prior, conditioned on both. Compare the posterior's window to
these draws.

## 1. What it is called

- **Constrained realizations** (Hoffman & Ribak 1991) in cosmology. CMB **inpainting** of masked regions, such as the
  Galactic plane, is the same operation: draw the field inside a mask, consistent with the observed field outside and
  the power spectrum.
- **Conditional simulation** in geostatistics (the sampling counterpart of kriging).
- **Matheron's rule** / **pathwise conditioning** in the GP/ML literature (Wilson et al. 2020, *Efficiently sampling
  functions from Gaussian process posteriors*). It turns a prior draw into a conditional draw with one linear update.
- **Conditional of a GMRF on a subset of sites** (Rue & Held 2005, ch. 2). It is easy because it uses the precision
  matrix, not the covariance.

The test itself, the field in the window against what the field outside predicts, is a **cross-validatory / conflict
check** (cf. "node-splitting", Presanis et al.). It asks whether the window holds structure the smooth continuation from
outside does not predict.

## 2. Implementation

### The singular direction is only the DC mode

In site space the prior covariance is

$$\Sigma = V\,\mathrm{diag}(a^2)\,V^\top, \qquad V = v_m \otimes v_q,$$

with $a$ the Matern SPDE spectrum (`model.matern_spectrum`). It has rank $B^2 - 1$ because the DC amplitude is zeroed.
As a result, the prior forces $\sum_{\rm sites} L = 0$. Conditioning on the outside values would then fix the inside sum
exactly. That constraint is an artefact, since the softmax ignores the constant.

**Fix (intrinsic GMRF):** give the DC mode a *flat* prior (infinite variance, precision $0$) instead of zero variance.
Then:

- the conditional is well defined as long as any outside site exists;
- one can condition directly on the saved `log_p_mass`, because the normalization constant no longer matters.

### Work with the precision, not the covariance

$$Q = V\,\mathrm{diag}(1/a^2)\,V^\top \quad (\text{with } 1/a^2_{\rm DC} = 0).$$

With window sites $W$ and outside sites $O$:

$$x_W \mid x_O \sim \mathcal N\!\left(-Q_{WW}^{-1} Q_{WO}\, x_O,\; Q_{WW}^{-1}\right).$$

For non-integer $\nu$, $Q$ is not sparse, but $\Sigma_{OO}^{-1}$ (a $9000^2$ inverse per draw) is never needed. Only two
pieces are:

- **$Q_{WO}\,x_O$:** zero-pad $x_O$ over $W$, apply $Q$ spectrally (forward transform, multiply by $1/a^2$, inverse
  transform), then read off the $W$ entries.
- **$Q_{WW} = V_W\,\mathrm{diag}(1/a^2)\,V_W^\top$:** $V_W$ keeps the rows of $V$ for the window sites. The result is
  $n_W \times n_W$, which is small.

Then sample $x_W = \text{mean} + L^{-\top} z$, where $Q_{WW} = L L^\top$ and $z \sim \mathcal N(0, I)$.

### A full-q strip decouples

If the window is $m_1 \in [a, b]$ at *every* $q$, as for the joint fits, the $q$ eigenvectors are still a complete
orthonormal basis inside the window. So $Q$ block-diagonalizes by $q$-mode:

- transform along $q$;
- for each of the 100 $q$-modes, solve a 1D conditioning of the ~11–26 $m_1$ bins in the window given the rest;
- that is 100 tiny Cholesky factorizations per draw.

### A q-band rectangle is still cheap

For the 1D fits' $q$ bands, the decoupling breaks, but $n_W \approx 800$ sites, so a dense conditional costs
milliseconds.

### Equivalent Matheron form

Draw a full prior field $f$ spectrally (as `prior_fits.py` does now), then correct it:

$$x_W = f_W + Q_{WW}^{-1} Q_{WO}\,(f_O - x_O).$$

This is the same distribution with no explicit square root; pick whichever is simpler to write.

### Unphysical cells

Treat the strip as including the $q < 3/m_1$ cells. They carry no data, so they only take part through the field's
smoothness. Resample them along with everything else and apply the $m_2 > 3\,M_\odot$ window afterwards.

## 3. Caveats to decide on

1. **Leakage makes the test conservative.** The outside field is a posterior realization conditioned on *all* events,
   including those in the window. Through smoothness, the field near the window's edges already "knows" about the
   peak. The $m_1$ correlation range is about 11 bins, roughly $0.46$ in $\ln m_1$ or a factor of 1.6 in mass, so a
   window like 25–40 $M_\odot$ is only about one correlation length wide.
   - A fix: condition only on sites more than a **buffer** (~1 correlation length) away from the window, and resample
     the buffer too.
   - Trade-off: a wider buffer means the prior has more freedom, so the test loses power.
   - The gold standard is a refit with the window's data removed. That is not clean here, because events' PE samples
     straddle the window.
2. **The kernel parameters were also fitted with the peaks present.** A larger $\ln\sigma$, needed to fit the peaks,
   widens the conditional prior. This also errs on the conservative side.
3. **The answer depends on window width relative to the correlation length.** A narrow window is nearly determined by
   interpolation from its edges, so any localized excess looks significant. A wide window is closer to the
   unconditioned prior. Report the window widths alongside the result.
4. **Choosing the test statistic.** For each posterior draw $i$, draw several conditional samples given draw $i$'s
   outside field. Compare a statistic of the posterior's own window field against those conditional samples (a per-draw
   rank or p-value), then aggregate over draws. Two kinds of statistic:
   - **Fit-based:** $f_G$, the contrast between the peak and the smooth continuation, the $q$-location.
   - **Fit-free:** the Mahalanobis distance of the posterior's $x_W$ under the conditional,
     $(x_W - \text{mean})^\top Q_{WW} (x_W - \text{mean}) \sim \chi^2_{n_W}$. It needs no fit, but it is diluted by
     cells where the posterior is still just the prior (unphysical or data-poor cells). Projecting the residual onto a
     few smooth "bump" templates would sharpen it.

## 4. Cost

Conditioning costs milliseconds per draw, so the window fits remain the main cost, as in `macros/prior_fits.py`. It can
reuse that script's field code (`prior_p_mass`), swapping the unconditioned prior draw for the conditional one.
