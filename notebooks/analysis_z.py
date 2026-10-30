"""Post-processing of the redshift-dependent spin conditionals (model.py spin_z_modes > 0; user, 2026-10-29):
rebuild p(chi_eff, chi_p | m1, q, z) on every draw from the saved latents, exactly as model.py builds it, on a GPU.

    python notebooks/analysis_z.py <run> ...      (GPU; post_z.sbatch)  -> results/post_processing/for_plotting/<run>/
                                                                           chain_<c>_z.h5

Per draw (float32):
  MOM       (Bm, Bq, Bz, 4)   the conditional's chi_eff mean, sd, chi_p mean, sd per (m1, q, z) cell
  MOM_CORE  (Bm, Bq, Bz, 3)   chi_eff mean, sd within |chi_eff| < analysis.CHI_CORE, and the probability outside it
  P_WIN_Z   (Bz, Be, Bp)      p(chi_eff, chi_p | z) per z bin, over the windowed p(m1, q) (per-bin probabilities)
  CURVES    (n_m, n_q, 3, 2, Bz)  the moments of p(chi | z, m1 in M_BANDS[i], q in Q_BANDS[j]) for chi_eff, chi_eff
                              (core) and chi_p ('mean', 'sd'), over the windowed p(m1, q) restricted to the bands
(MOM, MOM_CORE and P_WIN_Z only up to MAX_CELL_BZ z bins: at 100 they would take ~30 GB per run), and ``dmom``, the
largest difference between the rebuilt MOM and the chain's saved spin_moments (a check of the
rebuild; NaN when the chain did not save them). The mass distribution does not depend on z in these models, so every
z bin uses the draw's own windowed p(m1, q); p(z) (the redshift power law) only matters for z-marginals, which can be
formed from P_WIN_Z and the draw's 'lamb'.
"""
import glob
import os
import re
import sys
import time

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import analysis as A

M_BANDS = ((A.MIN_M, A.MAX_M), (8., 12.), (18., 24.), (30., 40.), (55., 65.))     # all m1, then the four peaks
Q_BANDS = ((0., 1.), (0.3, 0.65), (0.8, 1.0))                                    # all q, then summary ranges
SPINS = ('chi_eff', 'chi_eff_core', 'chi_p')
STATS = ('mean', 'sd')
MAX_CELL_BZ = 20


def z_config(run):
    """(spin model, km, kq, kz, Bz, z_max) of a redshift-dependent run, from its name (run.py's run_name); spin model
    'gauss_zadd' for model.py's spin_z_additive, 'gauss_mqzadd' with spin_mass_additive too."""
    km = int(re.search(r'_k(\d+)', run).group(1))
    kq = int(m.group(1)) if (m := re.search(r'_kq(\d+)', run)) else km
    kz = int(re.search(r'_kz(\d+)', run).group(1))
    Bz = int(m.group(1)) if (m := re.search(r'_zb(\d+)', run)) else 10
    from data import MAX_Z
    spin_model = ('gauss_mqzadd' if '_mqadd' in run and '_zadd' in run else 'gauss_zadd' if '_zadd' in run
                  else 'gauss' if '_gauss' in run else 'field')
    return spin_model, km, kq, kz, Bz, MAX_Z


def z_edges(run):
    *_, Bz, z_max = z_config(run)
    return np.linspace(0., z_max, Bz + 1)


def cache_path(run, chain_file):
    return A.post_path(run, chain_file).replace('.h5', '_z.h5')


def band_masks():
    """(n_m, n_q, Bm, Bq) 0/1: the (m1, q) bins (by centre) of each (M_BANDS, Q_BANDS) pair."""
    W = np.zeros((len(M_BANDS), len(Q_BANDS), A.B, A.B))
    for i, (m0, m1) in enumerate(M_BANDS):
        for j, (q0, q1) in enumerate(Q_BANDS):
            W[i, j] = ((A.m_c >= m0) & (A.m_c <= m1))[:, None] * ((A.q_c >= q0) & (A.q_c <= q1))[None]
    return W


def _z_fns(run):
    """Jitted rebuild of one draw: (MOM, MOM_CORE, P_WIN_Z, CURVES), and the chain keys it takes."""
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp as LSE
    import pixelpop
    from model import NEG, axis_eigs, gauss_log_kernel, matern_spectrum, spin_window_fraction, to_grid
    spin_model, km, kq, kz, Bz, _ = z_config(run)
    sd_floor = A.gauss_sd_floor(run)
    B = A.B
    (w_ax, v_ax), = axis_eigs([pixelpop.utils.create_CAR_coupling_matrix(B, 1, isVisible=False)])
    (wz, vz), = axis_eigs([pixelpop.utils.create_CAR_coupling_matrix(Bz, 1, isVisible=False)])
    Vm, Vq, Vz = v_ax[:, :km], v_ax[:, :kq], vz[:, :kz]
    window = spin_window_fraction(A.q_edges, A.e_edges, A.p_edges)
    log_window = jnp.asarray(np.where(window > 0, np.log(np.maximum(window, 1e-300)), NEG))[:, None]    # (Bq, 1, ..)
    log_support = jnp.where(log_window > NEG / 2, 0., NEG)
    ec, pc = jnp.asarray(A.e_c), jnp.asarray(A.p_c)
    core = jnp.asarray((np.abs(A.e_c) < A.CHI_CORE).astype(float))
    lW, logdV = jnp.asarray(A.mass_window()), np.log(A.dlnm * A.dq)
    masks = jnp.asarray(band_masks())

    def stats(slab_T, log_p_mass):
        lw = log_p_mass + logdV + lW
        w_win = jnp.exp(lw - LSE(lw))                                                   # windowed p(m1, q)

        def slab(m):
            T = slab_T(m) + log_support
            P = jnp.exp(T - LSE(T + log_window, axis=(2, 3))[..., None, None] + log_window)    # (Bq, Bz, Be, Bp)
            pe, pq = P.sum(3), P.sum(2)
            me, mp = pe @ ec, pq @ pc
            mom = jnp.stack([me, jnp.sqrt(jnp.maximum(pe @ ec ** 2 - me ** 2, 0.)),
                             mp, jnp.sqrt(jnp.maximum(pq @ pc ** 2 - mp ** 2, 0.))], -1)
            pe_in = pe * core; inside = jnp.maximum(pe_in.sum(2), 1e-300)
            mu_in = (pe_in @ ec) / inside
            mom_core = jnp.stack([mu_in, jnp.sqrt(jnp.maximum((pe_in @ ec ** 2) / inside - mu_in ** 2, 0.)),
                                  1. - pe_in.sum(2)], -1)
            return mom, mom_core, jnp.einsum('q,qzep->zep', w_win[m], P)
        mom, mom_core, pz = jax.lax.map(slab, jnp.arange(B), batch_size=max(1, 40 // Bz))

        def moments(w, mu, sd):
            """mean and sd of the mixture over (m1, q) with weights w (n_m, n_q, Bm, Bq[, Bz]) per z."""
            w = w / jnp.maximum(w.sum((2, 3), keepdims=True), 1e-300)
            mean = jnp.einsum('ijmq...,mq...->ij...', w, mu)
            ex2 = jnp.einsum('ijmq...,mq...->ij...', w, sd ** 2 + mu ** 2)
            return jnp.stack([mean, jnp.sqrt(jnp.maximum(ex2 - mean ** 2, 0.))], -2)    # (n_m, n_q, 2, Bz)
        W = (masks * w_win)[..., None] * jnp.ones(Bz)                                   # (n_m, n_q, Bm, Bq, Bz)
        curves = jnp.stack([moments(W, mom[..., 0], mom[..., 1]),
                            moments(W * (1. - mom_core[..., 2]), mom_core[..., 0], mom_core[..., 1]),
                            moments(W, mom[..., 2], mom[..., 3])], 2)                   # (n_m, n_q, 3, 2, Bz)
        return mom, mom_core, pz.sum(0), curves

    if spin_model == 'field':
        pin = np.zeros((km, kq, kz, B, B), bool); pin[..., 0, 0] = True; pin = jnp.asarray(pin)

        @jax.jit
        def fn(white, lnsig, lognu, logr, log_p_mass):
            G = to_grid(matern_spectrum([w_ax[:km], w_ax[:kq], wz[:kz], w_ax, w_ax], lnsig, logr, jnp.exp(lognu),
                                        B * B * Bz * B * B, pin=pin),
                        white, [jnp.eye(km), jnp.eye(kq), jnp.eye(kz), v_ax, v_ax])
            return stats(lambda m: jnp.einsum('k,ql,zn,klnep->qzep', Vm[m], Vq, Vz, G), log_p_mass)
        keys = ('_spin_sites', 'lnsigma_spin', 'log_nu_spin', 'log_ranges_spin', 'log_p_mass')
    elif spin_model == 'gauss_zadd':
        @jax.jit
        def fn(white, offsets, lnsig, lognu, logr, white_z, lnsig_z, lognu_z, logr_z, log_p_mass):
            F = [to_grid(matern_spectrum([w_ax[:km], w_ax[:kq]], lnsig[i], logr[i], jnp.exp(lognu), B * B), white[i],
                         [Vm, Vq]) for i in range(5)]
            G = [to_grid(matern_spectrum([wz[:kz]], lnsig_z[i], logr_z[i], jnp.exp(lognu_z), Bz), white_z[i], [Vz])
                 for i in range(5)]
            theta = (offsets[:, None, None] + jnp.stack(F))[..., None] + jnp.stack(G)[:, None, None]   # (5, Bm, Bq, Bz)
            return stats(lambda m: gauss_log_kernel(theta[:, m][..., None, None], ec[:, None], pc[None, :], sd_floor),
                         log_p_mass)
        keys = ('_gauss_sites', 'gauss_offsets', 'lnsigma_gauss', 'log_nu_gauss', 'log_ranges_gauss', '_gauss_z_sites',
                'lnsigma_gauss_z', 'log_nu_gauss_z', 'log_ranges_gauss_z', 'log_p_mass')
    elif spin_model == 'gauss_mqzadd':
        @jax.jit
        def fn(white, offsets, lnsig, lognu, logr, white_q, lnsig_q, lognu_q, logr_q, white_z, lnsig_z, lognu_z, logr_z,
               log_p_mass):
            one = lambda w, v, ls, lr, ln, wh, n: jnp.stack([to_grid(matern_spectrum([w], ls[i], lr[i], jnp.exp(ln), n),
                                                                     wh[i], [v]) for i in range(5)])
            F1 = one(w_ax[:km], Vm, lnsig, logr, lognu, white, B)                          # (5, Bm)
            F2 = one(w_ax[:kq], Vq, lnsig_q, logr_q, lognu_q, white_q, B)                  # (5, Bq)
            G = one(wz[:kz], Vz, lnsig_z, logr_z, lognu_z, white_z, Bz)                     # (5, Bz)
            theta = (offsets[:, None, None, None] + F1[:, :, None, None] + F2[:, None, :, None]
                     + G[:, None, None, :])                                                 # (5, Bm, Bq, Bz)
            return stats(lambda m: gauss_log_kernel(theta[:, m][..., None, None], ec[:, None], pc[None, :], sd_floor),
                         log_p_mass)
        keys = ('_gauss_sites', 'gauss_offsets', 'lnsigma_gauss', 'log_nu_gauss', 'log_ranges_gauss', '_gauss_q_sites',
                'lnsigma_gauss_q', 'log_nu_gauss_q', 'log_ranges_gauss_q', '_gauss_z_sites', 'lnsigma_gauss_z',
                'log_nu_gauss_z', 'log_ranges_gauss_z', 'log_p_mass')
    else:
        @jax.jit
        def fn(white, offsets, lnsig, lognu, logr, log_p_mass):
            axes = [(w_ax[:km], Vm), (w_ax[:kq], Vq), (wz[:kz], Vz)]
            fields = [to_grid(matern_spectrum([a[0] for a in axes], lnsig[i], logr[i], jnp.exp(lognu), B * B * Bz),
                              white[i], [a[1] for a in axes]) for i in range(5)]
            theta = offsets[:, None, None, None] + jnp.stack(fields)                    # (5, Bm, Bq, Bz)
            return stats(lambda m: gauss_log_kernel(theta[:, m][..., None, None], ec[:, None], pc[None, :], sd_floor),
                         log_p_mass)
        keys = ('_gauss_sites', 'gauss_offsets', 'lnsigma_gauss', 'log_nu_gauss', 'log_ranges_gauss', 'log_p_mass')
    return fn, keys


OUT = ('MOM', 'MOM_CORE', 'P_WIN_Z', 'CURVES')


def rebuild_chain(run, chain_file, log_every=100):
    """Rebuild every draw of one chain of a redshift-dependent run (OUT) into cache_path(run, chain_file)."""
    import jax.numpy as jnp
    fn, keys = _z_fns(run)
    keep = set(OUT) if z_config(run)[4] <= MAX_CELL_BZ else {'CURVES'}
    path = cache_path(run, chain_file)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.partial'
    t0 = time.time()
    with h5py.File(chain_file, 'r') as h, h5py.File(tmp, 'w') as g:
        n = len(h['log_likelihood'])
        g.attrs.update(n_draws=n, source=chain_file, m_bands=np.array(M_BANDS), q_bands=np.array(Q_BANDS),
                       spins=np.array(SPINS, dtype='S'), z_edges=z_edges(run))
        out, dmom = None, g.create_dataset('dmom', (n,))
        for i in range(n):
            res = [np.asarray(x, np.float32) for x in fn(*(jnp.asarray(h[k][i], dtype=float) for k in keys))]
            if out is None:
                out = {k: g.create_dataset(k, (n,) + v.shape, 'f4', chunks=(1,) + v.shape, compression='gzip',
                                           compression_opts=1) for k, v in zip(OUT, res) if k in keep}
            for k, v in zip(OUT, res):
                if k in keep:
                    out[k][i] = v
            dmom[i] = np.abs(res[0] - h['spin_moments'][i]).max() if 'spin_moments' in h else np.nan
            if log_every and (i + 1) % log_every == 0:
                print(f'{run} {os.path.basename(chain_file)}: {i + 1}/{n} draws, {(time.time() - t0) / (i + 1):.2f} s '
                      f'each, max |MOM - saved| {np.nanmax(dmom[:i + 1]):.1e}', flush=True)
    os.replace(tmp, path)
    return path


def chain_files(run):
    return sorted(glob.glob(os.path.join(A.REPO, 'results', run, 'chain_*_samples.h5')))


def load_curves(run):
    """CURVES of every chain (first n_draws of each, as load_light): (draws, n_m, n_q, 3, 2, Bz), and the z edges."""
    files = chain_files(run)
    n = min(h5py.File(cache_path(run, f), 'r').attrs['n_draws'] for f in files)
    return (np.concatenate([h5py.File(cache_path(run, f), 'r')['CURVES'][:n] for f in files]).astype(float),
            z_edges(run))


if __name__ == '__main__':
    for run in sys.argv[1:]:
        for f in chain_files(run):
            if os.path.exists(cache_path(run, f)) and os.path.getmtime(cache_path(run, f)) > os.path.getmtime(f):
                print('up to date:', cache_path(run, f), flush=True)       # newer than its chain file
                continue
            print('wrote', rebuild_chain(run, f), flush=True)
