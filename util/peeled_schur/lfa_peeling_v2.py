"""LFA for peeling v2: resistive flux response moved out of the inverse, and additive-Schwarz mass inverse.
Model (h_y = 1, sigma = 1): S = K + A G* M^-1 K R^-1 G,  R = M + E K  (E = theta*eta/(sigma h^2)).
v1 (current arm):   P = K M^-1 (M + A G* X_R G),   X_R ~ R^-1   (inverse of a Helmholtz operator)
v2 (proposed):      P = K R^-1 N,  N = R + A G* X G,  X ~ M^-1  (inverse of a pure mass)
                    P^-1 = N^-1 R K^-1   -> R is multiplied, never inverted.
X options: diag(M)^-1, or element additive Schwarz X_AS = sum_e R_e^T (R_e M R_e^T)^-1 R_e.
Requires lfa_mass_lumping.py in the same directory."""
import numpy as np
from lfa_mass_lumping import grid, ops, H, LUMP, Me

M1 = {0: Me[2:, 2:]+Me[:2, :2], 1: Me[:2, 2:], -1: Me[2:, :2]}
def _Mblk(dx, dy): return np.kron(M1[dx], M1[dy]) if max(abs(dx), abs(dy)) <= 1 else np.zeros((4, 4))
def X_as_symbol(tx, ty, patch=((0, 0), (1, 0), (0, 1), (1, 1))):
    n = len(patch); B = np.zeros((4*n, 4*n))
    for a, pa in enumerate(patch):
        for b, pb in enumerate(patch): B[4*a:4*a+4, 4*b:4*b+4] = _Mblk(pb[0]-pa[0], pb[1]-pa[1])
    Bi = np.linalg.inv(B); X = 0
    for a, pa in enumerate(patch):
        for b, pb in enumerate(patch):
            X = X + Bi[4*a:4*a+4, 4*b:4*b+4]*np.exp(1j*((pb[0]-pa[0])*tx+(pb[1]-pa[1])*ty))[:, None, None]
    return X

def kappa(A, E, variant, kappa_shift=0.0):
    tx, ty = grid(64); M, K, G, _ = ops(tx, ty, 1.0, 1.0, kappa_shift)
    Mi = np.linalg.inv(M); R = M+E*K; Ri = np.linalg.inv(R)
    S = K+A*H(G)@Mi@K@Ri@G
    Dm = np.diag(1/np.kron(np.array(LUMP['diag']), np.array(LUMP['diag'])))
    if variant == 'v1 exact':  P = K@Mi@(M+A*H(G)@Ri@G)
    if variant == 'v1 diag(R)':
        P = K@Mi@(M+A*H(G)@np.diag(1/np.real(np.diagonal(R.mean(axis=0))))@G)
    if variant == 'v2 exact':  P = K@Ri@(R+A*H(G)@Mi@G)
    if variant == 'v2 diag(M)': P = K@Ri@(R+A*H(G)@Dm@G)
    if variant == 'v2 AS(M)':  P = K@Ri@(R+A*H(G)@X_as_symbol(tx, ty)@G)
    ev = np.linalg.eigvals(np.linalg.solve(P, S)); return abs(ev).max()/abs(ev).min()

if __name__ == "__main__":
    tx, ty = grid(64); M = ops(tx, ty)[0]
    ev = np.linalg.eigvals(X_as_symbol(tx, ty)@M).real
    print(f"element-AS: spectrum of X_AS M in [{ev.min():.3f}, {ev.max():.3f}], cond {ev.max()/ev.min():.2f} (diag: 49)")
    V = ['v1 exact', 'v1 diag(R)', 'v2 exact', 'v2 diag(M)', 'v2 AS(M)']
    print("kappa(P^-1 S) at A = 1e4;  E = grid resistive number")
    print("E      " + "  ".join(f"{v:>11s}" for v in V))
    for E in [0, 0.01, 0.1, 1, 10]:
        print(f"{E:<6g} " + "  ".join(f"{kappa(1e4, E, v):11.2f}" for v in V))
