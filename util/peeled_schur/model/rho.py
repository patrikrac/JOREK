import tg, numpy as np, sys
def ops_dense(pb, P, om):
    """dense two-grid pieces: S = I - om P W (smoother), C = I - P_c Wc^-1 P_c^T W"""
    n = pb.W.shape[0]; Wd = pb.W.toarray()
    PW = P(Wd.astype(complex)); S = np.eye(n) - om*PW
    Pm = pb.P.toarray(); C = np.eye(n) - Pm@pb.lu.solve((Pm.T@Wd).astype(complex))
    return S, C
def report(pb, sms, ks=(1, 2, 4, 6), oms=(0.5, 0.7, 0.9, 1.0)):
    for name, P in sms:
        best = {}
        for om in oms:
            S, C = ops_dense(pb, P, om); E = C.copy()
            for k in range(1, max(ks)+1):
                E = S@E
                if k in ks:
                    r = max(abs(np.linalg.eigvals(E)))
                    if r < best.get(k, (9e9,))[0]: best[k] = (r, om)
        print(f"   {name:14s} rho(S^k C) best over omega: " + "  ".join(f"k{k}: {r:.3f}(om{om})" for k, (r, om) in best.items()), flush=True)
if __name__ == "__main__":
    nx = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    base = dict(A=1e4, nu=0.5, e=0.1, hh=1.0, kap=0.0)
    cases = [("base", base, 4, True), ("A=1e2", dict(base, A=1e2), 4, True), ("A=1e6", dict(base, A=1e6), 4, True),
             ("nu=0", dict(base, nu=0.0), 4, True), ("nu=50", dict(base, nu=50.0), 4, True),
             ("hh=0,e=0", dict(base, hh=0.0, e=0.0), 4, True), ("hh=30", dict(base, hh=30.0), 4, True),
             ("kap=0.3", dict(base, kap=0.3), 4, True), ("asp1 full", base, 1, False)]
    for lab, p, asp, semi in cases:
        pb = tg.Problem(nx, nx, asp, p, semi); print(lab, p, f"asp={asp}", "semi" if semi else "full", flush=True)
        report(pb, [('coll', pb.coll()), ('dist', pb.dist()), ('dist exT2', pb.dist(tkind='exact')),
                    ('dist exX,Q', pb.dist(xkind='exact', qkind='exact'))])
