"""Mean chi_p against m1 and q across the spin models and cut runs (user, 2026-11-24); the numbers in
macros/chi_p_trends.md. Per draw, the per-cell mean chi_p (the chain's spin_moments, or the for_plotting cache)
averaged over the windowed p(m1, q) in the stated q (m1) ranges. Login node, pinned, ~5 min:

    taskset -c 0 python macros/chi_p_trends.py      -> results/post_processing/chi_p_trends/<run>.npz + printed summary
"""
import os, sys, glob
REPO='/mnt/home/jheinzel/bbh-pixelpop-constraints'
sys.path[:0]=[REPO, REPO+'/notebooks', REPO+'/macros']
os.environ.setdefault('JAX_PLATFORMS','cpu')
import numpy as np, h5py
import analysis as A
mc, qc = A.m_c, 0.5*(A.q_edges[1:]+A.q_edges[:-1])
im = lambda m: int(np.argmin(abs(mc-m))); iq = lambda q: int(np.argmin(abs(qc-q)))
RUNS = ['ppc_gwtc5_k10','ppc_gwtc5_k10_neff30','ppc_gwtc5_k10_neff50','ppc_gwtc5_k10_varcut1','ppc_gwtc5_k10_varcut4',
        'ppc_gwtc5_k5','ppc_gwtc5_k10_split','ppc_gwtc5_k10_gauss','ppc_gwtc5_k100_gauss','ppc_gwtc5_k100_gauss_neff30',
        'ppc_gwtc5_k100_gauss_neff50','ppc_gwtc5_k100_gauss_varcut1','ppc_gwtc5_k100_gauss_varcut4',
        'ppc_gwtc5_k100_gauss_mqadd','ppc_gwtc5_k100_gauss_mqadd_m1only','ppc_gwtc5_k100_gauss_mqadd_qonly']
win = A.mass_window()
mq = lambda lo, hi: ((qc >= lo) & (qc <= hi)).astype(float)
mm = lambda lo, hi: ((mc >= lo) & (mc <= hi)).astype(float)
out = {}
for run in RUNS:
    files = sorted(glob.glob(f'{REPO}/results/{run}/chain_*_samples.h5'))
    n = min(len(h5py.File(f,'r')['log_likelihood']) for f in files)
    R = {k: [] for k in ('vm_all','vm_hi','vm_lo','vq_20','vq_60','cell')}
    for f in files:
        with h5py.File(f,'r') as h:
            if 'spin_moments' in h:
                MOM = lambda i: h['spin_moments'][i, ..., 2]
            else:
                g = h5py.File(A.post_path(run, f),'r'); MOM = lambda i: g['MOM'][i, ..., 2]
            for i in range(n):
                lp = h['log_p_mass'][i].astype(float) + win; p = np.exp(lp - lp.max())
                mu = np.asarray(MOM(i), float)
                def vs_m(w): pw = p*w[None]; return (pw*mu).sum(1)/np.maximum(pw.sum(1),1e-300)
                def vs_q(w): pw = p*w[:,None]; return (pw*mu).sum(0)/np.maximum(pw.sum(0),1e-300)
                R['vm_all'].append(vs_m(np.ones(len(qc)))); R['vm_hi'].append(vs_m(mq(.8,1))); R['vm_lo'].append(vs_m(mq(.3,.65)))
                R['vq_20'].append(vs_q(mm(15,30))); R['vq_60'].append(vs_q(mm(45,75))); R['cell'].append(mu[im(21), iq(0.5)])
    R = {k: np.array(v) for k, v in R.items()}
    out[run] = R
    os.makedirs(f'{REPO}/results/post_processing/chi_p_trends', exist_ok=True); np.savez(f'{REPO}/results/post_processing/chi_p_trends/{run}.npz', **R)
    s = lambda x: f'{np.median(x):.2f} [{np.percentile(x,5):.2f},{np.percentile(x,95):.2f}]'
    c = lambda d: f'{np.median(d):+.2f} P>0 {np.mean(d>0):.2f}'
    va, vl, vh = R['vm_all'], R['vm_lo'], R['vm_hi']
    print(f"\n== {run} ({len(va)} draws)")
    print(f"  all q: 10 {s(va[:,im(10)])}  21 {s(va[:,im(21)])}  33 {s(va[:,im(33)])}  60 {s(va[:,im(60)])}")
    print(f"  60-10 {c(va[:,im(60)]-va[:,im(10)])}; 33-10 {c(va[:,im(33)]-va[:,im(10)])}; 60-33 {c(va[:,im(60)]-va[:,im(33)])}")
    print(f"  q.8-1 60-10 {c(vh[:,im(60)]-vh[:,im(10)])}; q.3-.65 60-21 {c(vl[:,im(60)]-vl[:,im(21)])}")
    print(f"  21 Msun, q 0.3-0.65: {s(vl[:,im(21)])} P(>=0.54) {np.mean(vl[:,im(21)]>=0.54):.3f}; q .8-1: {s(vh[:,im(21)])}; cell (21,0.5): {s(R['cell'])} P(>=0.54) {np.mean(R['cell']>=0.54):.3f}")
    print(f"  q dep (0.95-0.4): m1 15-30 {c(R['vq_20'][:,iq(.95)]-R['vq_20'][:,iq(.4)])}; m1 45-75 {c(R['vq_60'][:,iq(.95)]-R['vq_60'][:,iq(.4)])}")
