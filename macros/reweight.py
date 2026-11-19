"""Population-reweighted event posteriors (user, 2026-11-14): each event's PE samples reweighted by the inferred
population, the number of events in each mass feature, and PE against reweighted posteriors in (m1, q) and
(chi_eff, chi_p).

    python macros/reweight.py compute [<run> ...]    (GPU; macros/reweight.sbatch)
        -> results/post_processing/<run>_reweighted.npz
    python macros/reweight.py report [<run> ...]     (CPU, login node pinned, texlive loaded)
        -> macros/reweight.tex (+ draft/macros/), plots/reweighted_vs_pe{,_gaussian}.pdf,
           plots/reweighted_vs_pe_stacked{,_gaussian}.pdf, plots/reweighted_vs_pe_features{,_gaussian}.pdf,
           plots/reweighted_vs_pe_contours{,_gaussian}.pdf (per-event 90% contours, PE | reweighted)

Weights. For posterior draw Lambda_n, the model's own per-sample event weights (model.setup_model(record_weights=True),
the draw's latent sites substituted, in analysis.load_light's draw order as macros/mc_systematics.py),
    ln w_nij = ln p_pop(theta_ij | Lambda_n) - ln pi_PE(theta_ij),
including the redshift model and the m2 > 3 Msun window, are normalized over each event's samples, w~_nij. The
population-reweighted posterior of event i is p(theta | d_i, data) = (1 / N) sum_n p(theta | d_i, Lambda_n): the
samples theta_ij with the draw-averaged weights W_ij = mean_n w~_nij, or equivalently one sample per draw drawn with
probabilities w~_nij (both saved). Normalizing within each draw divides out the event's own evidence Z_i(Lambda),
which is the factor Z_i / xi through which event i enters p(Lambda | data); so the average over the full hyperposterior
equals L(d_i | theta) int p(theta | Lambda) / xi(Lambda) p(Lambda | data without i) dLambda, the leave-one-out form
(Galaudage+ 2020, Moore & Gerosa 2021) and the exact marginal of theta_i in the joint hierarchical posterior: no double
counting (user, 2026-11-14). Averaging p(theta | Lambda) over draws first and then multiplying by L would double count.
The PE posterior is the unweighted usable samples (PE prior), as in the model.

Feature counts. Feature k is the region peak_excess.EVENT_WINDOWS[k] in m1 x the feature's own q range
(peak_fits.PEAKS). Per draw, P_nik = sum_j w~_nij [theta_ij in k] is event i's probability of lying in feature k,
and N_nk = sum_i P_nik the expected number of events there; the macros give its median and 90% interval over draws
(hyperposterior uncertainty; the Poisson-binomial scatter of which events are in, sum_i P (1 - P), is printed), and
the same sum for the PE posteriors (as peak_excess's NEventsPE; the PE prior there, so not a population statement).
Also the effective number of contributing events, Neff_k = (sum_i P_ik)^2 / sum_i P_ik^2 (user, 2026-11-14): N_k
when N_k events each have P = 1, larger when the count is spread thinly over many events (N_k / P for events of equal
P), so Neff_k / N_k shows how localized the support is.
Counts are always sums of probabilities, never the number of events with P > 0.5 (user, 2026-11-14): that can be small
or zero even when the data say a lot about a region, if single events are broad compared with it. The figures weight
events by P the same way.
"""
import argparse
import glob
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'notebooks')); sys.path.insert(0, HERE)

RUNS = ('ppc_gwtc5_k10', 'ppc_gwtc5_k100_gauss')
MODEL_LABELS = {'ppc_gwtc5_k10': 'Flexible', 'ppc_gwtc5_k100_gauss': 'Gaussian'}
PARAMS = ('mass_1', 'mass_ratio', 'chi_eff', 'chi_p', 'redshift')
# feature regions: Fig. 2's (paper_figs.SPIN_PEAK_REGIONS; user, 2026-11-18: 15-30 and 45-75 Msun for the ~20 and
# ~60 Msun features, were peak_excess.EVENT_WINDOWS' 15-25 and 50-70)
FEATURES = {'10': ((8., 12.), (0.8, 1.0), 'Ten'), '35': ((30., 40.), (0.8, 1.0), 'ThirtyFive'),
            '20': ((15., 30.), (0.3, 0.65), 'Twenty'), '60': ((45., 75.), (0.3, 0.65), 'Sixty')}
FEATURE_COLOURS = {'10': 'C0', '35': 'C3', '20': 'C2', '60': 'C1'}
GREY_BELOW = 0.1               # figures: an event is coloured by its most probable feature if that P exceeds this
# (user, 2026-11-14) counts are sums of membership probabilities, never N(P > 0.5): with broad single-event
# posteriors N(P > 0.5) can be small or zero even when the data say a lot about the region


def out_path(run):
    return os.path.join(REPO, 'results', 'post_processing', f'{run}_reweighted.npz')


def feature_masks(m1, q):
    """(..., n_features) bool: each sample in each feature region."""
    return np.stack([(m1 >= a) & (m1 <= b) & (q >= c) & (q <= d) for (a, b), (c, d), _ in FEATURES.values()], -1)


# ------------------------------------------------------------------------------------------------ GPU: the weights
def compute(run, seed=0, n_max=None, path=None):
    import h5py
    import jax
    import jax.numpy as jnp
    from numpyro import handlers
    import data as D
    import run as R
    from mc_systematics import latent_draws, run_args

    t0 = time.time()
    args = run_args(run)
    prob_model, init, posteriors, injections, pp = R.build(args, run, record_weights=True)
    keys = sorted(init)
    files = sorted(glob.glob(os.path.join(REPO, 'results', run, 'chain_*_samples.h5')))
    n_draws = min(len(h5py.File(f, 'r')['log_likelihood']) for f in files)
    counts = np.asarray(pp.event_counts)
    nobs, npe = np.asarray(posteriors['mass_ratio']).shape
    real = np.arange(npe)[None] < counts[:, None]
    with open(os.path.join(D.CATALOGS[args.catalog], 'posteriors.pkl'), 'rb') as f:
        names = [os.path.basename(str(n)) for n in pickle.load(f)['event_names']]
    assert len(names) == nobs, (len(names), nobs)
    pe = {'mass_1': np.exp(np.asarray(posteriors['log_mass_1']))}
    pe.update({k: np.asarray(posteriors[k]) for k in PARAMS[1:]})
    masks = jnp.asarray(feature_masks(pe['mass_1'], pe['mass_ratio']) & real[..., None], jnp.float32)
    kw = jax.device_put({'posteriors': posteriors, 'injections': injections})
    print(f'{run}: {nobs} events x {npe} columns, {len(files)} chains x {n_draws} draws, built in '
          f'{time.time() - t0:.0f} s', flush=True)

    @jax.jit
    def step(lat, key, kw, masks):
        tr = handlers.trace(handlers.substitute(handlers.seed(prob_model, 0), data=lat)).get_trace(**kw)
        lw = tr['event_log_weights']['value']
        lw = lw - jax.scipy.special.logsumexp(lw, axis=1, keepdims=True)
        w = jnp.exp(lw)
        return (w, jnp.einsum('ij,ijk->ik', w, masks), 1. / jnp.sum(w ** 2, 1),
                jax.random.categorical(key, lw, axis=1))

    W = jnp.zeros((nobs, npe), jnp.float32)
    P, neff, idx = [], [], []
    key = jax.random.PRNGKey(seed)
    for n, lat in enumerate(latent_draws(run, keys, n_draws)):
        if n_max is not None and n >= n_max:
            break
        w, p, ne, j = step(lat, jax.random.fold_in(key, n), kw, masks)
        W = W + w
        P.append(np.asarray(p)); neff.append(np.asarray(ne)); idx.append(np.asarray(j))
        if (n + 1) % 200 == 0:
            print(f'  {n + 1} draws, {time.time() - t0:.0f} s', flush=True)
    N = len(P)
    W = np.asarray(W / N, np.float32)
    path = path or out_path(run)
    np.savez_compressed(path, names=np.array(names), counts=counts, W=W, P_feature=np.stack(P).astype(np.float32),
                        neff=np.stack(neff).astype(np.float32), resample_index=np.stack(idx).astype(np.int32),
                        features=np.array(list(FEATURES)), **{k: v.astype(np.float32) for k, v in pe.items()})
    print(f'{run}: {N} draws -> {path} ({time.time() - t0:.0f} s)', flush=True)


# ------------------------------------------------------------------------------------------------ CPU: the report
def load(run):
    d = dict(np.load(out_path(run)))
    d['real'] = np.arange(d['W'].shape[1])[None] < d['counts'][:, None]
    d['W_pe'] = d['real'] / d['counts'][:, None].astype(np.float32)
    return d


def weighted_quantiles(x, w, qs):
    """Per event (rows) weighted quantiles of x."""
    o = np.argsort(x, 1)
    xs, c = np.take_along_axis(x, o, 1), np.cumsum(np.take_along_axis(w, o, 1), 1)
    c /= c[:, -1:]
    return np.stack([xs[np.arange(len(x)), np.argmax(c >= q, 1)] for q in qs], -1)


def counts_table(d):
    """{feature: dict} of the reweighted and PE counts."""
    P = d['P_feature'].astype(float)                                            # (N, nobs, K)
    Pbar = P.mean(0)
    Ppe = (feature_masks(d['mass_1'], d['mass_ratio']) * d['W_pe'][..., None]).sum(1)     # (nobs, K)
    out = {}
    for k, name in enumerate(FEATURES):
        Nk = P[:, :, k].sum(1)
        neff = Nk ** 2 / (P[:, :, k] ** 2).sum(1)                              # per draw
        out[name] = dict(N=np.percentile(Nk, [5, 50, 95]), N_pe=Ppe[:, k].sum(),
                         neff=np.percentile(neff, [5, 50, 95]), neff_pe=Ppe[:, k].sum() ** 2 / (Ppe[:, k] ** 2).sum(),
                         scatter=np.sqrt((Pbar[:, k] * (1 - Pbar[:, k])).sum()),
                         top=sorted(zip(Pbar[:, k], Ppe[:, k], d['names']), reverse=True))
    return out, Pbar, Ppe


def write_tex(tables):
    from peak_fits import RUN_NAMES
    lines = ['% Population-reweighted event counts per feature (macros/reweight.py): m1 window x the feature\'s q range;',
             '% NRw: expected number of events (sum of membership probabilities), median and 90% over draws;',
             '% NPe: the sum of membership probabilities of the PE posteriors (PE prior);',
             '% NeffRw, NeffPe: effective number of contributing events, (sum_i P_i)^2 / sum_i P_i^2 (per draw: median, 90%)']
    for run, tab in tables.items():
        sfx = RUN_NAMES[run]
        for name, t in tab.items():
            tex = FEATURES[name][2]
            lo, med, hi = t['N']
            nlo, nmed, nhi = t['neff']
            lines += [rf'\newcommand{{\rw{tex}NRw{sfx}}}{{{med:.0f}^{{+{hi - med:.0f}}}_{{-{med - lo:.0f}}}}}',
                      rf'\newcommand{{\rw{tex}NeffRw{sfx}}}{{{nmed:.0f}^{{+{nhi - nmed:.0f}}}_{{-{nmed - nlo:.0f}}}}}']
            if sfx == 'Flex':
                lines += [rf'\newcommand{{\rw{tex}NPe}}{{{t["N_pe"]:.0f}}}',
                          rf'\newcommand{{\rw{tex}NeffPe}}{{{t["neff_pe"]:.0f}}}']
    text = '\n'.join(lines) + '\n'
    for path in (os.path.join(HERE, 'reweight.tex'), os.path.join(REPO, 'draft', 'macros', 'reweight.tex')):
        with open(path, 'w') as f:
            f.write(text)
        print(f'wrote {path}')


def event_styles(Pbar):
    """Per event: the colour of its most probable feature (grey if that probability is below GREY_BELOW), an opacity
    growing with that probability, and the probability (to draw the most probable last)."""
    k = np.argmax(Pbar, 1)
    p = Pbar.max(1)
    cols = np.where(p > GREY_BELOW, np.array([FEATURE_COLOURS[f] for f in FEATURES])[k], '0.6')
    alpha = np.where(p > GREY_BELOW, 0.2 + 0.75 * p, 0.3)
    return cols, alpha, p


def shift_figure(d, Pbar, label):
    """Each event's PE median (open) and reweighted median (filled), joined, in (m1, q) and (chi_eff, chi_p); colour:
    its most probable feature (event_styles), opacity growing with that probability. Feature regions as rectangles."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from paper_figs import LEG, TWO_COL, feature_masses
    cols, alpha, pmax = event_styles(Pbar)
    med = {k: (weighted_quantiles(d[k], d['W_pe'], [0.5])[:, 0], weighted_quantiles(d[k], d['W'], [0.5])[:, 0])
           for k in PARAMS[:4]}
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COL, 3.0))
    order = np.argsort(pmax)                                   # most probable members on top
    for ax, (x, y) in zip(axes, (('mass_1', 'mass_ratio'), ('chi_eff', 'chi_p'))):
        for z, i in enumerate(order):
            z = 2 + z / len(order)
            ax.plot([med[x][0][i], med[x][1][i]], [med[y][0][i], med[y][1][i]], color=cols[i], lw=0.5,
                    alpha=alpha[i], zorder=z)
            ax.scatter(med[x][0][i], med[y][0][i], s=7, facecolor='none', edgecolor=cols[i], lw=0.5, alpha=alpha[i],
                       zorder=z)
            ax.scatter(med[x][1][i], med[y][1][i], s=7, color=cols[i], lw=0, alpha=alpha[i], zorder=z)
    for name, ((a, b), (c, e), _) in FEATURES.items():
        axes[0].add_patch(Rectangle((a, c), b - a, e - c, fill=False, ec=FEATURE_COLOURS[name], lw=0.7, ls='--',
                                    zorder=1))
    fx = feature_masses()
    ticks = [5., 10., 20., 30., 50., 100.]
    axes[0].set_xscale('log'); axes[0].set_xlim(5., 150.); axes[0].set_ylim(0., 1.)
    axes[0].set_xticks(ticks, [f'${t:g}$' for t in ticks]); axes[0].minorticks_off()
    axes[0].set_xlabel(r'median $m_1\,[\textrm{M}_\odot]$'); axes[0].set_ylabel(r'median $q$')
    axes[1].set_xlim(-0.4, 0.6); axes[1].set_ylim(0., 1.)
    axes[1].set_xlabel(r'median $\chi_\mathrm{eff}$'); axes[1].set_ylabel(r'median $\chi_\mathrm{p}$')
    axes[1].text(0.97, 0.95, label, transform=axes[1].transAxes, va='top', ha='right',
                 bbox=dict(facecolor='w', alpha=0.8, edgecolor='none', pad=1.5))
    h = [Line2D([], [], marker='o', ls='', mfc='none', mec='k', ms=3, mew=0.5),
         Line2D([], [], marker='o', ls='', color='k', ms=3)]
    h += [Line2D([], [], marker='o', ls='', color=FEATURE_COLOURS[f], ms=3) for f in FEATURES]
    lab = ['PE', 'population-reweighted'] + [rf'${fx[f]}\,\textrm{{M}}_\odot$ feature' for f in FEATURES]
    fig.legend(h, lab, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.0), **LEG)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.2 / fig.get_figheight()))
    return fig


HIST = {'mass_1': (np.geomspace(3., 200., 81), r'$m_1\,[\textrm{M}_\odot]$', True),
        'mass_ratio': (np.linspace(0., 1., 41), r'$q$', False),
        'chi_eff': (np.linspace(-1., 1., 81), r'$\chi_\mathrm{eff}$', False),
        'chi_p': (np.linspace(0., 1., 41), r'$\chi_\mathrm{p}$', False)}


def stacked(ax, d, k, w, ev, **kw):
    """Sum over events of their posteriors in parameter k (sample weights w), each event weighted by ``ev`` (bool or
    membership probability), as a step histogram (density per unit of the axis: per ln m1 for m1)."""
    edges, _, logx = HIST[k]
    x = d[k].ravel(); ww = (w * np.asarray(ev, float)[:, None]).ravel()
    h, _ = np.histogram(x, edges, weights=ww)
    width = np.diff(np.log(edges)) if logx else np.diff(edges)
    ax.stairs(h / width, edges, **kw)


def stacked_figure(d, label):
    """The summed posteriors of all events, PE (grey) against reweighted (black), in m1, q, chi_eff, chi_p."""
    import matplotlib.pyplot as plt
    from paper_figs import LEG, TWO_COL
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 4, figsize=(TWO_COL, 1.9))
    allev = np.ones(len(d['names']), bool)
    for ax, k in zip(axes, HIST):
        stacked(ax, d, k, d['W_pe'], allev, color='0.55', lw=0.9)
        stacked(ax, d, k, d['W'], allev, color='k', lw=0.9)
        edges, xl, logx = HIST[k]
        ax.set_xlabel(xl); ax.set_xlim(edges[0], edges[-1])
        if logx:
            ax.set_xscale('log'); ax.set_xlim(5., 150.)
            t = [5., 10., 20., 50., 100.]
            ax.set_xticks(t, [f'${v:g}$' for v in t]); ax.minorticks_off()
        if k == 'chi_eff':
            ax.set_xlim(-0.6, 0.8)
        ax.set_yticks([])
    axes[0].set_ylabel('summed posteriors')
    fig.legend([Line2D([], [], color='0.55'), Line2D([], [], color='k')], ['PE', f'population-reweighted ({label})'],
               loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0), **LEG)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.2 / fig.get_figheight()))
    return fig


def features_figure(d, Pbar, label):
    """Rows: features; columns: m1, q, chi_eff, chi_p. The posteriors of all events summed with weights P_ik, their
    (reweighted) probability of lying in feature k: PE (grey) against reweighted (feature colour), the same events and
    weights in both, so both integrate to sum_i P_ik; the feature region dashed."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from paper_figs import LEG, TWO_COL, feature_masses
    fx = feature_masses()
    fig, axes = plt.subplots(len(FEATURES), 4, figsize=(TWO_COL, 1.15 * len(FEATURES) + 0.5))
    for k, (row, (name, ((a, b), (c, e), _))) in enumerate(zip(axes, FEATURES.items())):
        sel = Pbar[:, k]
        col = FEATURE_COLOURS[name]
        for ax, p in zip(row, HIST):
            stacked(ax, d, p, d['W_pe'], sel, color='0.55', lw=0.9)
            stacked(ax, d, p, d['W'], sel, color=col, lw=1.0)
            edges, xl, logx = HIST[p]
            ax.set_xlim(edges[0], edges[-1]); ax.set_yticks([])
            if logx:
                ax.set_xscale('log'); ax.set_xlim(5., 150.)
                t = [5., 10., 20., 50., 100.]
                ax.set_xticks(t, [f'${v:g}$' for v in t]); ax.minorticks_off()
            if p == 'chi_eff':
                ax.set_xlim(-0.6, 0.8)
            if k == len(FEATURES) - 1:
                ax.set_xlabel(xl)
            else:
                ax.set_xticklabels([])
        for v in (a, b):
            row[0].axvline(v, color=col, lw=0.6, ls='--')
        for v in (c, e):
            row[1].axvline(v, color=col, lw=0.6, ls='--')
        row[0].set_ylabel(rf'{fx[name]}$\,\textrm{{M}}_\odot$' + '\n' + rf'$\sum P = {sel.sum():.1f}$', fontsize=7)
    fig.legend([Line2D([], [], color='0.55'), Line2D([], [], color='k')],
               ['PE', f'population-reweighted ({label}); each event weighted by its probability of being in the '
                'feature'],
               loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0), **LEG)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.22 / fig.get_figheight())); fig.subplots_adjust(hspace=0.1, wspace=0.05)
    return fig


CONTOUR_GRIDS = {'mass': (('mass_1', np.geomspace(3., 200., 121), True), ('mass_ratio', np.linspace(0., 1., 61), False)),
                 'spin': (('chi_eff', np.linspace(-1., 1., 101), False), ('chi_p', np.linspace(0., 1., 51), False))}
CRED = 0.9                                         # credible level of the per-event contours


def event_contour(ax, d, i, w, space, **kw):
    """Event i's CRED credible contour (highest-density region of a weighted 2D histogram, smoothed by 1.2 bins with
    reflection at the edges) in ``space`` ('mass': (ln m1, q), 'spin': (chi_eff, chi_p))."""
    from scipy.ndimage import gaussian_filter
    (kx, ex, logx), (ky, ey, _) = CONTOUR_GRIDS[space]
    n = d['counts'][i]
    x, y, ww = d[kx][i, :n], d[ky][i, :n], w[i, :n]
    H, _, _ = np.histogram2d(np.log(x) if logx else x, y, [np.log(ex) if logx else ex, ey], weights=ww)
    H = gaussian_filter(H, 1.2, mode='reflect')
    h = np.sort(H.ravel())[::-1]
    c = np.cumsum(h) / h.sum()
    level = h[min(np.searchsorted(c, CRED), len(h) - 1)]
    xc = 0.5 * (ex[1:] + ex[:-1]) if not logx else np.sqrt(ex[1:] * ex[:-1])
    ax.contour(xc, 0.5 * (ey[1:] + ey[:-1]), H.T, levels=[level], **kw)


def contour_figure(d, Pbar, label):
    """2 x 2: per-event CRED contours, PE (left) against population-reweighted (right), in (m1, q) (top) and
    (chi_eff, chi_p) (bottom); colour: the event's most probable feature, opacity growing with that probability
    (event_styles)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from paper_figs import LEG, TWO_COL, feature_masses
    fx = feature_masses()
    cols, alpha, pmax = event_styles(Pbar)
    fig, axes = plt.subplots(2, 2, figsize=(TWO_COL, 5.6), sharex='row', sharey='row')
    order = np.argsort(pmax)                                       # most probable members drawn last (on top)
    for row, space in zip(axes, ('mass', 'spin')):
        for ax, w, title in zip(row, (d['W_pe'], d['W']), ('PE', f'population-reweighted ({label})')):
            for i in order:
                event_contour(ax, d, i, w, space, colors=[cols[i]], linewidths=0.35 + 0.35 * pmax[i],
                              alpha=alpha[i])
    t = [5., 10., 20., 30., 50., 100.]
    axes[0, 0].set_xscale('log'); axes[0, 0].set_xlim(5., 150.); axes[0, 0].set_ylim(0., 1.)
    axes[0, 0].set_xticks(t, [f'${v:g}$' for v in t]); axes[0, 0].minorticks_off()
    axes[1, 0].set_xlim(-0.6, 0.8); axes[1, 0].set_ylim(0., 1.)
    axes[1, 0].set_xticks([-0.4, -0.2, 0., 0.2, 0.4, 0.6])
    for ax in axes[0]:
        ax.set_xlabel(r'$m_1\,[\textrm{M}_\odot]$')
    for ax in axes[1]:
        ax.set_xlabel(r'$\chi_\mathrm{eff}$')
    axes[0, 0].set_ylabel(r'$q$'); axes[1, 0].set_ylabel(r'$\chi_\mathrm{p}$')
    h = [Line2D([], [], color=FEATURE_COLOURS[f], lw=0.8) for f in FEATURES] + [Line2D([], [], color='0.6', lw=0.6)]
    lab = [rf'${fx[f]}\,\textrm{{M}}_\odot$ feature' for f in FEATURES] + ['other events']
    fig.legend(h, lab, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.0), **LEG)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.38 / fig.get_figheight())); fig.subplots_adjust(hspace=0.22, wspace=0.05)
    for ax, head in zip(axes[0], ('PE', f'population-reweighted ({label})')):
        pos = ax.get_position()
        fig.text(0.5 * (pos.x0 + pos.x1), pos.y1 + 0.008, head, ha='center', va='bottom', fontsize=8)
    return fig


def report(runs):
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import plot_style
    from paper_figs import SMALL_RC
    plot_style.apply(); plt.rcParams.update(SMALL_RC)
    tables = {}
    for run in runs:
        d = load(run)
        tab, Pbar, Ppe = counts_table(d)
        tables[run] = tab
        ne = d['neff']
        print(f'{run}: {d["P_feature"].shape[0]} draws; reweighted Neff per event per draw: median '
              f'{np.median(ne):.0f}, min over events of the median {np.median(ne, 0).min():.0f} '
              f'({d["names"][np.argmin(np.median(ne, 0))]})')
        for name, t in tab.items():
            lo, med, hi = t['N']
            print(f'  {name:>2} Msun feature {FEATURES[name][:2]}: reweighted N = {med:.1f} ({lo:.1f}-{hi:.1f}), '
                  f'Neff {t["neff"][1]:.1f} ({t["neff"][0]:.1f}-{t["neff"][2]:.1f}), membership scatter '
                  f'+-{t["scatter"]:.1f}; PE N = {t["N_pe"]:.1f}, Neff {t["neff_pe"]:.1f}')
            for p, ppe, nm in t['top'][:12]:
                if p < 0.1:
                    break
                print(f'      {nm:<40} P_rw {p:.2f}  P_pe {ppe:.2f}')
        sfx = '' if run == 'ppc_gwtc5_k10' else '_gaussian'
        lab = MODEL_LABELS[run]
        plot_style.save(shift_figure(d, Pbar, lab), f'reweighted_vs_pe{sfx}')
        plot_style.save(stacked_figure(d, lab), f'reweighted_vs_pe_stacked{sfx}')
        plot_style.save(features_figure(d, Pbar, lab), f'reweighted_vs_pe_features{sfx}')
        plot_style.save(contour_figure(d, Pbar, lab), f'reweighted_vs_pe_contours{sfx}')
    write_tex(tables)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=('compute', 'report'))
    ap.add_argument('runs', nargs='*', default=list(RUNS))
    ap.add_argument('--n', type=int, default=None, help='compute: first N draws only (a test; written to --out)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    if a.what == 'compute':
        for run in a.runs:
            compute(run, n_max=a.n, path=a.out)
    else:
        report(a.runs)
