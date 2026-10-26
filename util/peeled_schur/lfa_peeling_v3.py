"""LFA for peeling v3: viscosity moved into the single inverse (no sixth-order cross term in N).
Model (h = 1, sigma = 1):  S = K + A G* M^-1 K R^-1 G + nu K M^-1 K,   R = M + e K + hh K M^-1 K
current arm:  P = K Rt^-1 N,          N = Rt + A G* X G + nu sym(Rt X K)        (reach 5 with hyper)
v3:           P = K M^-1 Hn Rt^-1 N,  N = Rt + A G* Y G,  Hn = M + nu K,  Y ~ Hn^-1   (reach 3, PSD)
              P^-1 = N^-1 Rt Hn^-1 M K^-1
Rt = R with M^-1 in the hyper-resistive sandwich replaced by X (exact or element additive Schwarz).
Requires lfa_mass_lumping.py in the same directory."""
import numpy as np
from lfa_mass_lumping import grid, ops, H, Me, Ke

def _blocks(Ae): return {0: Ae[2:, 2:]+Ae[:2, :2], 1: Ae[:2, 2:], -1: Ae[2:, :2]}
M1, K1 = _blocks(Me), _blocks(Ke)
def _blk(dx, dy, nu):
    if max(abs(dx), abs(dy)) > 1: return np.zeros((4, 4))
    return np.kron(M1[dx], M1[dy])+nu*(np.kron(K1[dx], M1[dy])+np.kron(M1[dx], K1[dy]))
def AS_symbol(tx, ty, nu=0.0, patch=((0, 0), (1, 0), (0, 1), (1, 1))):
    """element additive Schwarz approximation of (M + nu K)^-1 (exact solves on element patches)"""
    n = len(patch); B = np.zeros((4*n, 4*n))
    for a, pa in enumerate(patch):
        for b, pb in enumerate(patch): B[4*a:4*a+4, 4*b:4*b+4] = _blk(pb[0]-pa[0], pb[1]-pa[1], nu)
    Bi = np.linalg.inv(B); X = 0
    for a, pa in enumerate(patch):
        for b, pb in enumerate(patch):
            X = X+Bi[4*a:4*a+4, 4*b:4*b+4]*np.exp(1j*((pb[0]-pa[0])*tx+(pb[1]-pa[1])*ty))[:, None, None]
    return X

def compare(A, e, hh, nu, approx, tx, ty, M, K, G):
    Mi = np.linalg.inv(M); R = M+e*K+hh*K@Mi@K
    S = K+A*H(G)@Mi@K@np.linalg.inv(R)@G+nu*K@Mi@K
    Hn = M+nu*K
    X, Y = (AS_symbol(tx, ty), AS_symbol(tx, ty, nu)) if approx else (Mi, np.linalg.inv(Hn))
    Rt = M+e*K+hh*K@X@K
    RXK = Rt@X@K
    Pcur = K@np.linalg.inv(Rt)@(Rt+A*H(G)@X@G+nu*(RXK+H(RXK))/2)
    Pv3 = K@Mi@Hn@np.linalg.inv(Rt)@(Rt+A*H(G)@Y@G)
    out = []
    for P in (Pcur, Pv3):
        ev = np.linalg.eigvals(np.linalg.solve(P, S)); out.append((abs(ev).min(), abs(ev).max()))
    return out

if __name__ == "__main__":
    tx, ty = grid(48); M, K, G, _ = ops(tx, ty)
    print("A=1e4; grid numbers e=th*eta/(sig h^2), hh=th*eta_h/(sig h^4), nu=th*nu/(sig h^2)")
    for approx in (False, True):
        print("exact inner inverses" if not approx else "element-AS for M^-1 (X) and for (M+nu K)^-1 (Y)")
        for e, hh in [(0, 0), (1, 0), (0.3, 3), (1, 30)]:
            for nu in [0, 0.3, 3]:
                (a, b), (c, d) = compare(1e4, e, hh, nu, approx, tx, ty, M, K, G)
                print(f"  e={e:<4g} hh={hh:<3g} nu={nu:<4g}  current: ratio {b/a:6.2f}   v3: [{c:.3f},{d:.3f}] ratio {d/c:6.2f}")
    for nu in [0, 0.3, 3, 30]:
        ev = np.linalg.eigvals(AS_symbol(tx, ty, nu)@(M+nu*K)).real
        print(f"element-AS on M + nu K, nu={nu:<4g}: spectrum [{ev.min():.3f},{ev.max():.3f}], cond {ev.max()/ev.min():.1f}")
