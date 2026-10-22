"""LFA of the 'peeled' (Tang-type) Schur preconditioner for the magnetic channel.
Exact:   S = K + A G* M^-1 K M^-1 G
Peeled:  P = K M^-1 T,  T = M + A G* X G,  X = M^-1 (exact) or a sparse approximation.
P^-1 = T^-1 M K^-1 : one isotropic Poisson-type solve, one multiply by M, one anisotropic single-inverse solve.
Requires lfa_mass_lumping.py in the same directory."""
import numpy as np
from lfa_mass_lumping import grid, ops, H, LUMP, cheb_X

def peeled(A, X='exact', k=0, aspect=1, kappa=0.0):
    hx, hy = 1/aspect, 1.0; tx, ty = grid(); M, K, G, _ = ops(tx, ty, hx, hy, kappa)
    Mi = np.linalg.inv(M); S = K+A*H(G)@Mi@K@Mi@G
    if X == 'exact': Xm = Mi
    else:
        D = np.kron(hx*np.array(LUMP[X]), hy*np.array(LUMP[X]))
        Xm = np.broadcast_to(np.diag(1/D), M.shape) if k == 0 else cheb_X(M, D, k)
    P = K@Mi@(M+A*H(G)@Xm@G)
    ev = np.linalg.eigvals(np.linalg.solve(P, S))
    w, V = np.linalg.eigh(S); Sh = V@(np.sqrt(w)[..., None]*H(V))
    T = Sh@np.linalg.solve(P, Sh)                       # S^1/2 P^-1 S^1/2 (energy norm)
    c = np.linalg.eigvalsh((T+H(T))/2).min(); C = np.linalg.svd(T, compute_uv=False).max()
    return abs(ev).max()/abs(ev).min(), np.abs(np.angle(ev)).max(), c, C

if __name__ == "__main__":
    print("kappa_eig, max|arg|, field of values in S-norm [min Re, max |.|]")
    for A in [1, 1e2, 1e4, 1e6]:
        print(f"exact masses A={A:7.0e}: " + "%.3f %.3f [%.3f, %.3f]" % peeled(A))
    for asp in [1, 8]:
        for kp in [0, np.pi/8, np.pi/2]:
            print(f"exact masses A=1e4 aspect={asp} shift={kp:.2f}: " + "%.3f %.3f [%.3f, %.3f]" % peeled(1e4, aspect=asp, kappa=kp))
    for k, rad in [(0, 0), (2, 1), (3, 2), (4, 3)]:
        print(f"X = Chebyshev-{k} on diag(M) (X radius {rad}, T stencil radius {2+rad}), A=1e6: kappa_eig = %.2f" % peeled(1e6, 'diag', k)[0])
