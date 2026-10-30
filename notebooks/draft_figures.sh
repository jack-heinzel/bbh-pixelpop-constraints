#!/bin/bash
# Redraw the figures used in draft/ from the spin cache and copy them into draft/plots/ under their short names.
# Runs on a login node (one pinned core, < 3 GB); the cache must be complete (post.sbatch).
#     bash notebooks/draft_figures.sh [summary|regions|macros|paper|excess|all]      (default: all)
# paper: notebooks/paper_figs.py (the column-width paper figures), copied to draft/plots/ under the same names.
# macros: copies the LaTeX number macros (macros/*_fits.tex, macros/peak_pairs.tex) to draft/macros/.
cd "$(dirname "$0")/.."
module load texlive/20240312 2>/dev/null       # plot_style uses usetex
PY="taskset -c ${CORE:-5} $HOME/venvs/gwjax313/bin/python"
RUNS="ppc_gwtc5_k10 ppc_gwtc5_k100_gauss"
NAME=gwtc5_field_vs_gauss_k100
S=plots/compare_${NAME}_mass_ratio_spin_summary
what=${1:-all}

if [ "$what" = summary ] || [ "$what" = all ]; then
    $PY notebooks/summary_fig.py $NAME $RUNS --m-min 5 || exit 1
    $PY notebooks/summary_fig.py $NAME $RUNS --m-min 5 --mean-asinh 0.03 --core || exit 1
    cp ${S}_m5.pdf draft/plots/summary.pdf
    cp ${S}_mean_asinh0.03_core_m5.pdf draft/plots/summary_mean.pdf
fi
if [ "$what" = regions ] || [ "$what" = all ]; then
    for run in $RUNS; do
        $PY notebooks/spin_region_fig.py $run || exit 1
        $PY notebooks/spin_region_fig.py $run --transpose || exit 1
    done
    for t in "" _T; do
        cp plots/ppc_gwtc5_k10_spin_regions$t.pdf draft/plots/spin_regions_flexible$t.pdf
        cp plots/ppc_gwtc5_k100_gauss_spin_regions$t.pdf draft/plots/spin_regions_gaussian$t.pdf
    done
fi
if [ "$what" = macros ] || [ "$what" = all ]; then
    mkdir -p draft/macros
    cp macros/peak_fits.tex macros/joint_fits.tex macros/peak_pairs.tex macros/peak_excess.tex macros/chi_q_broadening.tex macros/q_contrast.tex macros/peak_odds.tex draft/macros/
fi
if [ "$what" = paper ] || [ "$what" = all ]; then
    $PY notebooks/paper_figs.py all || exit 1
    $PY notebooks/paper_figs.py chi --core || exit 1                  # |chi_eff| < 0.8
    $PY notebooks/paper_figs.py chiq --core --prior || exit 1
    $PY notebooks/paper_figs.py chiq --core --lvk || exit 1           # with the GWTC-5 chi_eff | q models
    for f in mass_ratio_m1 chi_eff_sd_m1 chi_eff_mean_m1 chi_eff_mean_sd_m1 spin_peak_regions spin_peak_regions_marginals peak_bookkeeping \
             m2_q_ranges m2_m1_windows chi_eff_sd_m1_core chi_eff_sd_q_core_prior chi_eff_sd_q_core_lvk \
             chi_eff_mean_sd_q_core_lvk; do
        cp plots/$f.pdf draft/plots/$f.pdf
    done
fi
if [ "$what" = excess ] || [ "$what" = all ]; then      # macros/peak_excess.py: significance figures and macros
    $PY macros/peak_excess.py || exit 1
    cp plots/peak_excess.pdf plots/peak_excess_own_q.pdf draft/plots/
    cp macros/peak_excess.tex draft/macros/
fi
echo "draft figures done"
