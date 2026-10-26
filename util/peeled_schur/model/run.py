import tg, numpy as np, sys
ks = (1, 2, 3, 4, 6)
def case(nx, ny, asp, p, semi=True, extra=(), its=True):
    pb = tg.Problem(nx, ny, asp, p, semi)
    sm = {'coll': pb.coll(), 'dist': pb.dist()}
    for name, kw in extra: sm[name] = pb.dist(**kw)
    print(f"A={p['A']:.0e} nu={p['nu']:g} e={p['e']:g} hh={p['hh']:g} kap={p['kap']:g} asp={asp} {'semi' if semi else 'full'}  n={pb.W.shape[0]}")
    rng = np.random.default_rng(1); b = rng.standard_normal(pb.W.shape[0])+0j
    for name, P in sm.items():
        f = [tg.factor(pb, P, k) for k in ks]
        s = f"   {name:10s} V(0,k) factor " + " ".join(f"{k}:{x:.3f}" for k, x in zip(ks, f))
        if its: s += "  | FGMRES its 1e-6 k=2/4/6: " + "/".join(str(tg.fgmres_its(pb.W, pb.tg_apply(P, k), b, 1e-6)) for k in (2, 4, 6))
        print(s, flush=True)
base = dict(A=1e4, nu=0.5, e=0.0, hh=0.0, kap=0.0)
nx = int(sys.argv[1]) if len(sys.argv) > 1 else 16
case(nx, nx, 4, base)
for A in [1e2, 1e6]: case(nx, nx, 4, dict(base, A=A))
for nu in [0.0, 5.0, 50.0]: case(nx, nx, 4, dict(base, nu=nu))
for hh in [1.0, 30.0]: case(nx, nx, 4, dict(base, hh=hh), extra=[('dist-nohh', dict(hyper=False))])
case(nx, nx, 4, dict(base, kap=0.3))
case(nx, nx, 1, base, semi=False)
