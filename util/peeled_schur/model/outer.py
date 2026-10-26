import t2, tg, numpy as np, scipy.sparse.linalg as sla, sys
def run(m, cfgs, nrhs=2):
    rng = np.random.default_rng(3); n = m.W.shape[0]
    bs = [rng.standard_normal(n)+1j*rng.standard_normal(n) for _ in range(nrhs)]
    tgW = t2.TwoGrid(m.Wu, m.P1, 4, m.rings4, 'jacobi', 0.7, scale=False)   # m.W is already scaled
    tgW.A = m.W; tgW.lu = __import__('scipy.sparse.linalg', fromlist=['splu']).splu((tgW.P.T@m.W@tgW.P).tocsc()); tgW.B = tg.BlockInv(m.W, m.rings4)
    tgT = t2.TwoGrid(m.T2, m.P1, 2, m.rings2, 'zebra'); tgK = t2.TwoGrid(m.K.astype(complex), m.P1, 1, m.rings1, 'zebra')
    luM = sla.splu(m.M.tocsc().astype(complex)); Dn = tg.nodal_inv(m.g); Xc = {k: tg.cheb_inv(m.M, Dn, k) for k in (2, 3, 4, 6)}
    Xs = {'exact': lambda v: luM.solve(v.astype(complex))}
    for k in Xc: Xs[f'cheb{k}'] = (lambda v, X=Xc[k]: X@v)
    out = []
    for lab, c in cfgs:
        if c == 'prod':
            P = lambda r: tgW.vcycle(r, 0, 6)
        else:
            kT, kK, xk, nT = c
            P = t2.make_PT2(m, lambda v: tgT.solve(v, 0, kT, nT), lambda v: tgK.solve(v, 0, kK), Xs[xk], 'K')
        its = [t2.fgmres(m.W, P, b, 0.1) for b in bs]; its6 = [t2.fgmres(m.W, P, b, 1e-6) for b in bs]
        out.append(f"{lab}: {max(its)} ({max(its6)})")
    return " | ".join(out)
cfgs = [('prod V(0,6)', 'prod'), ('T2 2 K1 exX', (2, 1, 'exact', 1)), ('T2 2 K1 ch4', (2, 1, 'cheb4', 1)),
        ('T2 2 K1 ch2', (2, 1, 'cheb2', 1)), ('T2 1 K1 ch4', (1, 1, 'cheb4', 1)), ('T2 2 K2 ch6', (2, 2, 'cheb6', 1))]
base = dict(A=1e4, nu=0.5, e=0.1, hh=1.0, kap=0.0)
nx = int(sys.argv[1]) if len(sys.argv) > 1 else 8
print("FGMRES its on W, rtol 0.1 (rtol 1e-6); rho pedestal 10:1 width 0.05")
for lab, p in [("base", base), ("A=1e6", dict(base, A=1e6)), ("A=1e2", dict(base, A=1e2)), ("nu=50", dict(base, nu=50.0)),
               ("hh=30", dict(base, hh=30.0)), ("hh=0", dict(base, hh=0.0, e=0.01)), ("kap=0.3", dict(base, kap=0.3))]:
    m = t2.Model(nx, nx, 4, p, True, ratio=10, width=0.05)
    print(f"{lab:7s} {run(m, cfgs)}", flush=True)
