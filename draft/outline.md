# PRL outline (draft, 2026-10-26)

PRL limits (check the current *Information for Authors*): about **3750 words** for text plus captions, with figures and
tables converted to words; **abstract ≤ 600 characters**; a **justification statement** at submission; methods detail
and robustness checks go in the **Supplemental Material (SM)**, which does not count toward the length.

Figure word-equivalents (APS rule of thumb, aspect = width / height): one-column figure ≈ 150 / aspect + 20 words,
two-column figure ≈ 300 / (0.5 aspect) + 40 words.

## Word budget

| item | words |
|---|---|
| Fig. 1 (mass ratio + spin summary) | ~270 (one column, as proposed below) to ~470 (two columns, `summary.pdf` as is) |
| Fig. 2 (spins of the four peaks) | ~150–200 (one row, two columns, aspect ~4) |
| Fig. 3 (peak locations + mass bookkeeping) | ~130 (one column) |
| Table I (peak numbers) | ~100 |
| **figures + table** | **~650–900** |
| **text + captions** | **~2850–3100** |

## Sections: what to fill in

1. **Title.** The current one ("…reveals peaks and correlations") is fine. A sharper option: "Template-free evidence
   for hierarchical structure in the binary black hole mass, mass-ratio and spin distribution" (only if the
   significance wording supports "evidence"; otherwise "hints of").
2. **Abstract (≤ 600 characters).** One sentence each: the data (259 events, GWTC-5 / O4b), the method (nonparametric,
   no templates), the result (near-equal-mass peaks at `\jointTenMuFlex`, `\jointThirtyFiveMuFlex`; q ≈ 0.5 peaks at
   `\jointTwentyMuFlex`, `\jointSixtyMuFlex`, about twice the mass; broad χ_eff there), and the meaning (consistent
   with hierarchical mergers; tentative; the full O4 sample will decide).
3. **Introduction (~500 words).** The current text is good but long for PRL; trim the second and third paragraphs to
   about half. Fill the empty `\cite{}`s (LIGO, Virgo, KAGRA, GWTC catalogues, PixelPop). End with the result
   paragraph (line 83, now with macros).
4. **Method (~400 words; details in the SM).**
   - Data: the catalogue (`gwtc5`, 259 events, FAR < 1/yr), at most 20000 PE samples per event, injections and VT.
   - Mass: PixelPop Matérn field on (ln m₁, q), with m₂ > 3 M☉.
   - Spins: the conditional (χ_eff, χ_p) field (k = 10 mass modes), and the Gaussian model whose parameters are Matérn
     fields (the current equations; move the covariance matrix to the SM to save space).
   - Redshift: a power law. The "correlated redshift" note (`\jh`) becomes a sentence in the outlook.
   - Sampling: NUTS, chains, the likelihood correction, the Neff floor.
5. **Results: mass structure (~700 words).**
   - Fig. 1: p(m₁ | q range) and p(q | m₁). Two families: near-equal mass at ~10 and ~33 M☉, q ≈ 0.5 at ~21 and
     ~59 M☉.
   - Table I: μ, σ, mean q, peak fraction per peak (the joint-fit macros, Flex; Gauss in the SM).
   - The peaks are measured: the μ intervals are 3–9 times narrower than the prior's (SM figure).
6. **Results: spins (~500 words).**
   - Fig. 1, bottom row: σ_χeff(m₁) is broad at ~18–20 and > 45 M☉, narrow at the near-equal-mass peaks; it comes from
     the width at fixed q (low q).
   - Fig. 2: the (χ_eff, χ_p) distribution in each of the four peak regions.
   - Robust across the spin models (field, split, gauss k = 10 and 100); details in the SM.
7. **Mass bookkeeping (~400 words).**
   - Fig. 3: retained fraction μ_hi / M_lo.
   - 35 → 60: `\pairThirtyFiveSixtyRetainedModeFlex`, consistent with ~5% radiated.
   - 10 → 20: `\pairTenTwentyRetainedModeFlex`, with 0.95 at the `\pairTenTwentyQuantileRetainedModeFlex` quantile:
     a mild tension. Possible readings: the dynamical first-generation spectrum differs from the (field-dominated,
     χ_eff > 0) 10 M☉ peak, or pairing is selective. This model cannot separate the channels (SM: χ_eff slices).
8. **Significance and outlook (~350 words).**
   - Tentative, stated plainly. Posterior-only significance measures do not work in a model this flexible (random
     prior fields give fitted peaks 70–85% of the time; SM).
   - A calibrated p-value needs mock catalogues (planned).
   - Forecast: ~2–4× more events (the full O4 sample) for a decisive test.
   - Next: correlate redshift; channel-mixture interpretation (companion paper).

## Figures: making them fit

All current figures are drawn at 15–24 inches wide with 16 pt fonts. Shrunk to PRL widths (3.4 in one column, 7.0 in
two), the text ends up at ~4–7 pt, which is too small. Redraw them at their final size instead (`figsize` in inches,
fonts ~8–9 pt), rather than scaling with `width=\linewidth`.

### Fig. 1: mass ratio and spin summary (`analysis.summary_figure`, `notebooks/summary_fig.py`)

- **Now:** two columns (Flexible | Gaussian) × 3 rows, 15.8 × 11.2 in (`summary.pdf`); the version with the μ row
  (`summary_mean.pdf`) is 15.8 × 15.0 in.
- **Proposal:** Flexible only, one column, 3 rows (p(m₁ | q ranges); p(q | m₁); σ_χeff), 3.4 × ~5.5 in, ≈ 270 words.
  The Gaussian column moves to the SM with the same layout. Alternative: keep both models but overlay the Gaussian as
  lines on the Flexible bands, saving a column.
- **To figure out:** the legend placement at one-column width; whether the μ_χeff row (`summary_mean`) earns its
  space (probably SM); the colour-bar width; the shared x labels.

### Fig. 2: spins of the four peaks (`analysis.spin_region_figure`, `notebooks/spin_region_fig.py`)

- **Now:** a 4 × 4 grid of (m₁, q) regions plus totals, 23.9 × 14.3 in (or transposed); ~400 words at full width
  (~1000 for the transposed version), and unreadable when shrunk.
- **Proposal:** only the four peak regions, (8–12, q ~ 1), (30–40, q ~ 1), (16–22, q ~ 0.5), (55–65, q ~ 0.5): one
  row of four (χ_eff, χ_p) panels, two columns wide (7.0 × ~1.9 in), ≈ 150–200 words. The Gaussian model as contours
  on the Flexible heatmap, or a second row. Drop the slice side panels in the main text (keep them in the SM's full
  grid).
- **To figure out:** a shared colour scale across the four panels against per-panel scales; contour levels for the
  model comparison; the panel labels (mass range and q range in one corner).

### Fig. 3: peak locations and mass bookkeeping (`macros/window_fits_fig.py`, `macros/peak_pairs.py`)

- **Now:** the plane plot at 6.2 × 4.6 in (one column after redrawing); the peak-pairs histograms at 10.7 × 3.9 in.
- **Proposal:** one column, two stacked panels: (a) the (μ, mean q) plane without the prior contours (they are
  fragmented; the prior comparison goes in the SM); (b) the retained-fraction histograms for both pairs in one panel,
  fit-free only (two colours for the pairs, line style for the spin model), with the 0.95 line. 3.4 × ~4.5 in,
  ≈ 130 words.
- **To figure out:** whether (b) fits as an inset in (a); how to show the 0.95 / > 1 region compactly.

### Table I: peak numbers

- Rows: the four peaks. Columns: μ, σ, mean q (joint fits), and the 1D-fit μ for comparison, all macros. Flexible
  only; the Gaussian values in an SM table.

## Supplemental Material

1. Data: event list, PE sample selection, injections; the likelihood correction and Neff floor.
2. Models and priors (Matérn fields, spin models, `split`), and the sampler settings; the chains that froze and
   their replacements.
3. Robustness: k = 5 against k = 10; gauss k = 100 against k = 10; split; the short (NPE 10000) chains.
4. The window fits: 1D and joint fits, their diagnostics (`plots/<run>_{peak,joint}_fits.pdf`), and the comparison
   with the prior (`..._window_fits_prior.pdf`).
5. Full spin-region grids for both models (the current `spin_regions_*.pdf`).
6. The fit-free total-mass measurement; the χ_eff-slice check and why the model cannot separate the channels.
7. Significance: why posterior-only measures fail here; the plan for mock catalogues (`macros/peak_significance.md`).
8. Leave-one-out on influential events (GW231123, the high-spin low-Neff events), if run.

## Housekeeping

- Figure paths: `draft/plots/...` do not resolve when compiling in `draft/` (where `\input{acronyms}` resolves);
  use `plots/...` (`\graphicspath` already includes `plots`) and `\bibliography{main}`.
- Duplicate `\label{fig:placeholder}`s.
- `notebooks/draft_figures.sh`: add the redrawn PRL figures and Fig. 3.
- `bash notebooks/draft_figures.sh macros` after any refit, so the numbers in `draft/macros/` stay current.
