"""Resistive X with sparse approximations: X0 = diag(M)^-1 (design), X0r = diag(M + eK)^-1."""
import numpy as np, sys
sys.path.insert(0, '.')
from lfa_mass_lumping import H
from lfa_review import setup, stats
A = 1e4; M, K, G, KZx, KZy, Mi = setup(A)
dM = np.diag(M[0]).real; dK = np.diag(K[np.argmax(np.abs(K).sum(axis=(1, 2)))]).real
# diagonals of the 4x4 symbols are frequency independent for the self-coupling entries: take the theta = pi sample mean
from lfa_mass_lumping import Me, Ke
m1 = np.diag(Me[:2,:2] + Me[2:,2:]); k1 = np.diag(Ke[:2,:2] + Ke[2:,2:])
dM = np.kron(m1, m1); dK = np.kron(k1, m1) + np.kron(m1, k1)
print("diag M", dM, "diag K", dK)
for e in [0, 1e-2, 1e-1, 1, 10]:
    Xr = np.linalg.inv(M + e*K)
    S = K + A*H(G)@Mi@K@Xr@G
    X0 = np.broadcast_to(np.diag(1/dM), M.shape); X0r = np.broadcast_to(np.diag(1/(dM + e*dK)), M.shape)
    d = stats(S, K@Mi@(M + A*H(G)@X0@G)); r = stats(S, K@Mi@(M + A*H(G)@X0r@G))
    print(f"  e={e:6g}: X0 kappa {d[2]:8.2f} [{d[0]:.3f},{d[1]:.3f}] | X0r kappa {r[2]:8.2f} [{r[0]:.3f},{r[1]:.3f}]")
