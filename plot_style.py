"""House plotting style, from ~/public_html/mass-spin-correlations/plots/CLAUDE.md.

    import plot_style
    plot_style.apply()
    ...
    ax.legend(**plot_style.LEG_KW)
    plot_style.save(fig, 'pixelpop_lm1q_joint')      # -> plots/pixelpop_lm1q_joint.pdf

No titles on figures (paper captions explain them); PDF output; LaTeX serif text;
ticks inward on all four sides (outward on pcolormesh panels, where inward ticks
vanish against a dark background).
"""
import os

import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))
PLOTS = f'{REPO}/plots'

RC = {
    'text.usetex': True,
    'font.family': 'serif',
    'font.size': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'pdf.fonttype': 42,          # TrueType, for the Chrome PDF viewer
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'xtick.bottom': True,
    'ytick.left': True,
    'ytick.right': True,
}

# legends drawn at 14 pt so they fit inside the axes
LEG_KW = dict(handlelength=1.2, handletextpad=0.5, borderaxespad=0.5, fontsize=14)
LEG_KW_MULTICOL = dict(LEG_KW, columnspacing=0.8)


def apply():
    plt.rcParams.update(RC)


def save(fig, name):
    """Save ``fig`` as plots/<name>.pdf; ``name`` should say which model/data made it."""
    os.makedirs(PLOTS, exist_ok=True)
    path = os.path.join(PLOTS, f'{name}.pdf')
    fig.savefig(path, bbox_inches='tight')
    print(f'Saved {path}')
    return path
