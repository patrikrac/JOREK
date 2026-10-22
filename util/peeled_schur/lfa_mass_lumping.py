"""LFA: can the magnetic Schur term G* M^-1 K M^-1 G be replaced by G* X K X G with a sparse X ~ M^-1?
Bicubic Hermite (Bezier scaling, DOFs f, f_s/3 per direction), constant coefficients, periodic.
Model: S = K + A G* M^-1 K M^-1 G (grid Alfven number A, h_y = 1), Shat = K + A G* X K X G.
Validation: reproduces det M,K,G of Theorem 8 and lam_min = 1/(1+60A) of the continuum surrogate."""
import numpy as np
gx, gw = np.polynomial.legendre.leggauss(8); s = 0.5*(gx+1); w = 0.5*gw
N   = np.array([1-3*s**2+2*s**3, 3*s*(1-s)**2, 3*s**2-2*s**3, -3*s**2*(1-s)])
dN  = np.array([-6*s+6*s**2, 3*(1-s)**2-6*s*(1-s), 6*s-6*s**2, -6*s*(1-s)+3*s**2])
d2N = np.array([-6+12*s, -6*(1-s)-6*(1-2*s), 6-12*s, -6+18*s])
Me, Ke, Ge, He = (N*w)@N.T, (dN*w)@dN.T, (N*w)@dN.T, (d2N*w)@d2N.T
LUMP = {'diag': [26/35, 6/35], 'moment': [1, 3/5], 'typesum': [1, 3/70]}

def sym1(Ae, th):
    e = np.exp(1j*th)[..., None, None]
    return (Ae[2:, 2:]+Ae[:2, :2]) + Ae[:2, 2:]*e + Ae[2:, :2]/e
def kron(a, b):
    return np.einsum('...ij,...kl->...ikjl', a, b).reshape(a.shape[:-2]+(4, 4))
def H(a): return np.conj(np.swapaxes(a, -1, -2))
def grid(n=96):
    t = (np.arange(n)+0.5)/n*2*np.pi-np.pi
    tl = np.logspace(-4, -0.5, 25); t = np.unique(np.concatenate([t, tl, -tl]))
    TX, TY = np.meshgrid(t, t, indexing='ij'); return TX.ravel(), TY.ravel()
def ops(tx, ty, hx=1.0, hy=1.0, kappa=0.0):
    Mx, My = hx*sym1(Me, tx), hy*sym1(Me, ty)
    Kx, Ky = sym1(Ke, tx)/hx, sym1(Ke, ty)/hy
    M = kron(Mx, My); K = kron(Kx, My)+kron(Mx, Ky)
    G = kron(Mx, sym1(Ge, ty)) + 1j*kappa/hy*M          # derivative along rings (+ toroidal shift)
    WB = kron(Kx, Ky)+kron(Mx, sym1(He, ty)/hy**3)      # continuum int grad(d_y v).grad(d_y u)
    return M, K, G, WB
def lam(S, Sh):
    Li = np.linalg.inv(np.linalg.cholesky(Sh)); T = Li@S@H(Li)
    ev = np.linalg.eigvalsh((T+H(T))/2); return ev.min(), ev.max()
def cheb_X(M, D, k):
    Di = np.diag(1/D); T = Di@M; ev = np.linalg.eigvals(T).real
    a, b = ev.min()*0.999, ev.max()*1.001; th, de = (a+b)/2, (b-a)/2
    sig = th/de; rho = 1/sig; r = np.broadcast_to(np.eye(4), M.shape).copy(); d = r/th; x = 0*r
    for _ in range(k):
        x = x+d; r = r-T@d; rn = 1/(2*sig-rho); d = rn*rho*d+2*rn/de*r; rho = rn
    return x@Di
def kappa_of(A, X_kind, k=0, single=False, aspect=1, kappa=0.0):
    hx, hy = 1/aspect, 1.0; tx, ty = grid(); M, K, G, WB = ops(tx, ty, hx, hy, kappa)
    Mi = np.linalg.inv(M)
    if X_kind == 'cont':
        return lam(K+A*H(G)@Mi@K@Mi@G, K+A*WB)
    D = np.kron(hx*np.array(LUMP[X_kind]), hy*np.array(LUMP[X_kind]))
    X = np.broadcast_to(np.diag(1/D), M.shape) if k == 0 else cheb_X(M, D, k)
    if single: return lam(K+A*H(G)@Mi@G, K+A*H(G)@X@G)
    return lam(K+A*H(G)@Mi@K@Mi@G, K+A*H(G)@H(X)@K@X@G)

if __name__ == "__main__":
    for A in [1, 1e2, 1e4]:
        lo, hi = kappa_of(A, 'cont'); print(f"continuum A={A:g}: lam_min={lo:.3e} (1/(1+60A)={1/(1+60*A):.3e}) lam_max={hi:.4f}")
    for th in [0, np.pi]:
        for kd in LUMP:
            print(f"theta={th:.2f} eig(D^-1 M) {kd:8s}", np.round(np.sort(np.linalg.eigvals(np.diag(1/np.array(LUMP[kd]))@sym1(Me, np.array(th))).real), 4))
    for A in [1, 1e2, 1e4, 1e6]:
        row = {f"{kd}/k{k}": kappa_of(A, kd, k) for kd, k in [('diag', 0), ('moment', 0), ('typesum', 0), ('diag', 2), ('diag', 4)]}
        print(f"A={A:7.0e} " + " ".join(f"{n}: [{lo:.2e},{hi:.2e}]" for n, (lo, hi) in row.items()))
    print("single inverse, A=1e6:", {kd: round(np.divide(*kappa_of(1e6, kd, single=True)[::-1]), 1) for kd in LUMP})
