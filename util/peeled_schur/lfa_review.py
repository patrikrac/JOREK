"""LFA of what lfa_peeling.py leaves out (review of peeled_schur_preconditioner.md).
Constant coefficients, bicubic Hermite, grid Alfven number A = 1e4 unless noted. Needs lfa_mass_lumping.py.
  (1) compression W_comp = Pc int dZ v dZ u (Z radial x or along the rings y), in Q (design) or left out of P;
  (2) resistive flux response S_B = A G* M^-1 K (M + eK)^-1 G, e = theta dt eta / (opz h^2), against
      X = diag(M)^-1 (design X_0), diag(M + eK)^-1, the node-block inverse and Chebyshev on it;
  (3) X_1 = degree-1 Chebyshev on diag(M) with an underestimated top b: SPD or not.
Prints kappa = max|lambda| / min|lambda| of P^-1 S, [min|lambda|, max|lambda|], and the S-norm field of
values' lower bound (FoV)."""
import numpy as np
from lfa_mass_lumping import grid, ops, H, sym1, Me, Ke, kron, LUMP


def stats(S, P):
    ev = np.linalg.eigvals(np.linalg.solve(P, S))
    w, V = np.linalg.eigh(S); Sh = V@(np.sqrt(w)[..., None]*H(V))
    T = Sh@np.linalg.solve(P, Sh)
    return abs(ev).min(), abs(ev).max(), abs(ev).max()/abs(ev).min(), np.linalg.eigvalsh((T+H(T))/2).min()


def setup(aspect=1, kappa=0.0):
    hx, hy = 1/aspect, 1.0; tx, ty = grid(64); M, K, G, _ = ops(tx, ty, hx, hy, kappa)
    Mx, My = hx*sym1(Me, tx), hy*sym1(Me, ty); Kx, Ky = sym1(Ke, tx)/hx, sym1(Ke, ty)/hy
    return M, K, G, kron(Kx, My), kron(Mx, Ky), np.linalg.inv(M)


def cheb(Mop, Bi, k):
    """degree k-1 Chebyshev approximation of Mop^-1 preconditioned by Bi (as lfa_mass_lumping.cheb_X)"""
    T = Bi@Mop; ev = np.linalg.eigvals(T).real; a, b = ev.min()*0.999, ev.max()*1.001
    th, de = (a+b)/2, (b-a)/2; sig = th/de; rho = 1/sig
    r = np.broadcast_to(np.eye(4), Mop.shape).copy(); d = r/th; x = 0*r
    for _ in range(k):
        x = x+d; r = r-T@d; rn = 1/(2*sig-rho); d = rn*rho*d+2*rn/de*r; rho = rn
    return x@Bi


if __name__ == "__main__":
    print("(1) compression, kappa [min, max] FoV:   W_comp in Q (design)   |   W_comp left out of P")
    for A in [1e2, 1e4]:
        M, K, G, KZx, KZy, Mi = setup(); T = M + A*H(G)@Mi@G
        for Pc in [0.1, 1, 10, 100]:
            for nm, KZ in [('x', KZx), ('y', KZy)]:
                S = K + Pc*KZ + A*H(G)@Mi@K@Mi@G
                q = stats(S, (K+Pc*KZ)@Mi@T); z = stats(S, K@Mi@T)
                print(f"  A {A:5.0e} Pc {Pc:5g} Z={nm}: {q[2]:7.2f} [{q[0]:.3f}, {q[1]:.3f}] {q[3]:6.3f}"
                      f"   |   {z[2]:7.2f} [{z[0]:.3f}, {z[1]:7.3f}] {z[3]:6.3f}")
    print("(2) resistive response, kappa [min, max]:")
    A = 1e4; M, K, G, _, _, Mi = setup()
    m1, k1 = Me[:2, :2] + Me[2:, 2:], Ke[:2, :2] + Ke[2:, 2:]
    bM = kron(m1[None], m1[None])[0]; bK = kron(k1[None], m1[None])[0] + kron(m1[None], k1[None])[0]
    for e in [0, 1e-2, 1e-1, 1, 10]:
        Mr = M + e*K; S = K + A*H(G)@Mi@K@np.linalg.inv(Mr)@G
        cand = [('exact X_ex', np.linalg.inv(Mr)), ('X_0 (design)', np.broadcast_to(np.diag(1/np.diag(bM)), M.shape)),
                ('diag(M+eK)^-1', np.broadcast_to(np.diag(1/np.diag(bM + e*bK)), M.shape))]
        nb = np.broadcast_to(np.linalg.inv(bM + e*bK), M.shape)
        cand += [('node block', nb), ('node block + Cheb 1', cheb(Mr, nb, 2)), ('node block + Cheb 2', cheb(Mr, nb, 3))]
        print(f"  e {e:5g}: " + " | ".join(f"{n} {s[2]:.2f} [{s[0]:.3f}, {s[1]:.2f}]"
                                          for n, s in ((n, stats(S, K@Mi@(M + A*H(G)@X@G))) for n, X in cand)))
    print("(3) X_1 positivity with the top b underestimated (uniform grid, diag lumping):")
    D = np.kron(np.array(LUMP['diag']), np.array(LUMP['diag']))
    ev = np.linalg.eigvals(np.diag(1/D)@M).real; a, b = ev.min(), ev.max()
    for fb in [1.0, 0.8, 0.6]:
        th, de = (a+fb*b)/2, (fb*b-a)/2; lam = np.linspace(a, b, 2001)
        p = (4*th-2*lam)/(2*th**2-de**2)
        print(f"  b_est = {fb:.1f} b: min p = {p.min():+.3f} -> X_1 {'SPD' if p.min() > 0 else 'INDEFINITE'}")
