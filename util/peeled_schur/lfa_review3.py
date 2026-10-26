"""Single-inverse X for T = M + A G* X G: diagonal vs node-block (4x4 Hermite block per node) vs
Chebyshev on node-block Jacobi; constant coefficients, with and without the resistive response (M + eK)."""
import numpy as np, sys
sys.path.insert(0, '.')
from lfa_mass_lumping import H, Me, Ke, kron
from lfa_review import setup, stats
A = 1e4; M, K, G, KZx, KZy, Mi = setup(A)
m1d = Me[:2,:2] + Me[2:,2:]; k1d = Ke[:2,:2] + Ke[2:,2:]
blkM = kron(m1d[None], m1d[None])[0]
blkK = kron(k1d[None], m1d[None])[0] + kron(m1d[None], k1d[None])[0]
def cheb(Mop, Bi, k):
    T = Bi@Mop; ev = np.linalg.eigvals(T).real; a, b = ev.min()*0.999, ev.max()*1.001
    th, de = (a+b)/2, (b-a)/2; sig = th/de; rho = 1/sig
    r = np.broadcast_to(np.eye(4), Mop.shape).copy(); d = r/th; x = 0*r
    for _ in range(k):
        x = x+d; r = r-T@d; rn = 1/(2*sig-rho); d = rn*rho*d+2*rn/de*r; rho = rn
    return x@Bi
for e in [0, 0.1, 1]:
    Mr = M + e*K; Xr = np.linalg.inv(Mr)
    S = K + A*H(G)@Mi@K@Xr@G
    dg = np.broadcast_to(np.diag(1/np.diag(blkM + e*blkK)), M.shape)
    nb = np.broadcast_to(np.linalg.inv(blkM + e*blkK), M.shape)
    row = []
    for nm, X in [('diag', dg), ('nodeblk', nb), ('nodeblk+cheb1', cheb(Mr, nb, 2)), ('nodeblk+cheb2', cheb(Mr, nb, 3))]:
        st = stats(S, K@Mi@(M + A*H(G)@X@G)); row.append(f"{nm}: {st[2]:6.2f} [{st[0]:.3f},{st[1]:.2f}]")
    print(f"e={e:4g}  " + " | ".join(row))
print("eig(nodeblk^-1 M):", np.round(np.sort(np.linalg.eigvals(np.linalg.solve(np.broadcast_to(blkM, M.shape), M)).real.ravel())[[0, -1]], 4))
