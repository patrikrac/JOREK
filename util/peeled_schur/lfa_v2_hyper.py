"""v2 with hyper-resistivity: S = K + A G* M^-1 K R^-1 G, R = M + E K + E4 K M^-1 K.
P_v2 = K Rh^-1 (Rh + A G* X G) with Rh = M + E K (design) or + E4 K X K (hyper sandwich)."""
import numpy as np
from lfa_mass_lumping import grid, ops, H
from lfa_peeling_v2 import X_as_symbol
tx, ty = grid(64); M, K, G, _ = ops(tx, ty); Mi = np.linalg.inv(M); Xa = X_as_symbol(tx, ty); A = 1e4
def kap(S, P):
    ev = np.linalg.eigvals(np.linalg.solve(P, S)); return abs(ev).min(), abs(ev).max()
for E in [0, 0.1, 1]:
    for E4 in [0, 0.01, 0.1, 1]:
        R = M + E*K + E4*K@Mi@K; S = K + A*H(G)@Mi@K@np.linalg.inv(R)@G
        row = []
        for nm, Rh, X in [('design exact', M+E*K, Mi), ('design AS', M+E*K, Xa),
                          ('+hyper exact', M+E*K+E4*K@Mi@K, Mi), ('+hyper AS', M+E*K+E4*K@Xa@K, Xa)]:
            lo, hi = kap(S, K@np.linalg.inv(Rh)@(Rh + A*H(G)@X@G)); row.append(f"{nm} {hi/lo:7.2f} [{lo:.3f},{hi:.2f}]")
        print(f"E {E:4g} E4 {E4:5g}: " + " | ".join(row))
