"""Commutation-based preconditioners for pair_w in the two-grid model, with a radial density
profile rho(x) (pedestal) in the u-row inertia K_rho and M_rho.
  u  : K_rho u - nu K w + a G j        w  : K u + M w
  psi: -a G u + M psi - (e M + hh K) j  j  : K psi + M j
P_T2 (exact commutation limit of the peel, no N):
  w*, j* = X r_w, X r_j;  r'_u = r_u + nu K w* - a G j*;  r'_psi = r_psi + (e M + hh K) j*
  z_u = M_rho K_rho^-1 r'_u;  (du, dpsi) = T2^-1 (z_u, r'_psi),  T2 = [[M_rho + nu K, -a G], [-a G, R]]
  dw = X (r_w - K du);  dj = X (r_j - K dpsi)
with every inner inverse exact or replaced by a (two-grid) multigrid cycle."""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla, scipy.linalg as la
import tg

def asm_np_w(Ae, n, wt):
    A = np.zeros((2*n+2, 2*n+2))
    for e in range(n): A[2*e:2*e+4, 2*e:2*e+4] += wt[e]*Ae
    return sp.csr_matrix(A)

def rho_prof(nx, ratio, width, x0=0.7):
    xc = (np.arange(nx)+0.5)/nx
    return 1/ratio + (1-1/ratio)*0.5*(1-np.tanh((xc-x0)/width))   # 1 in the core, 1/ratio outside

class Model:
    def __init__(self, nx, ny, asp, p, semi=True, ratio=1.0, width=0.05):
        hy = 1.0; hx = hy/asp; self.p = p; a = np.sqrt(p['A'])
        g = tg.Grid(nx, ny, hx, hy, p['kap']); self.g = g; n = g.n; self.n = n
        rho = rho_prof(nx, ratio, width); self.rho = rho
        Mx_r, Kx_r = hx*asm_np_w(tg.Me, nx, rho), asm_np_w(tg.Ke, nx, rho)/hx
        My, Ky = hy*tg.asm_per(tg.Me, ny), tg.asm_per(tg.Ke, ny)/hy
        S = lambda A: A.tocsr()[g.sel][:, g.sel]
        self.Mr = S(sp.kron(Mx_r, My)); self.Kr = S(sp.kron(Kx_r, My)+sp.kron(Mx_r, Ky))
        M, K, G = g.M, g.K, g.G; self.M, self.K, self.G = M, K, G
        Wu = sp.bmat([[self.Kr, -p['nu']*K, None, a*G], [K, M, None, None],
                      [-a*G, None, M, -(p['e']*M+p['hh']*K)], [None, None, K, M]], format='csr').astype(complex)
        self.Ds = tg.field_scale(Wu, n); self.Wu = Wu
        self.W = (self.Ds@Wu@self.Ds).tocsr()
        Px = tg.prol_np(nx//2)[tg.keep_x(nx)][:, tg.keep_x(nx//2)]
        Py = sp.identity(2*ny) if semi else tg.prol_per(ny//2)
        self.P1 = sp.kron(Px, Py).tocsr()
        self.rings1 = g.rings1
        self.rings4 = [np.concatenate([f*n+r for f in range(4)]) for r in g.rings1]
        self.rings2 = [np.concatenate([f*n+r for f in range(2)]) for r in g.rings1]
        Xex = sp.csr_matrix(np.linalg.inv(M.toarray()))
        self.R = (M + p['e']*K + p['hh']*(K@Xex@K)).tocsr()
        self.T2 = sp.bmat([[self.Mr+p['nu']*K, -a*G], [-a*G, self.R]], format='csr').astype(complex)

# ---- generic two-grid (V(0,k) or V(k1,k2)) for a matrix A with ring blocks ----------------------
class TwoGrid:
    def __init__(self, A, P1, nf, rings, smoother='jacobi', om=0.7, scale=True):
        n = A.shape[0]//nf
        self.Ds = tg.field_scale4(A, nf) if scale else sp.identity(A.shape[0])
        As = (self.Ds@A@self.Ds).tocsr(); self.A = As
        self.P = sp.block_diag([P1]*nf).tocsr()
        self.lu = sla.splu((self.P.T@As@self.P).tocsc().astype(complex))
        self.B = tg.BlockInv(As, rings); self.rings = rings; self.om = om; self.sm = smoother
        if smoother == 'zebra':
            self.Bc = [tg.BlockInv(As, rings[c::2]) for c in (0, 1)]
    def smooth(self, b, x, k):
        A = self.A
        for _ in range(k):
            if self.sm == 'jacobi':
                x = x + self.om*self.B(b - A@x)
            else:                                   # zebra: even rings, then odd rings (exact block GS)
                for c in (0, 1): x = x + self.Bc[c](b - A@x)
        return x
    def vcycle(self, b, k_pre=0, k_post=2):
        x = np.zeros_like(b)
        x = self.smooth(b, x, k_pre)
        x = x + self.P@self.lu.solve(self.P.T@(b - self.A@x))
        return self.smooth(b, x, k_post)
    def solve(self, b, k_pre=0, k_post=2, ncyc=1):
        """unscaled in, unscaled out"""
        d = self.Ds.diagonal(); bs = d*b; x = np.zeros_like(bs)
        for _ in range(ncyc): x = x + self.vcycle(bs - self.A@x, k_pre, k_post)
        return d*x
    def rho(self, k_post, k_pre=0):
        n = self.A.shape[0]; I = np.eye(n, dtype=complex)
        cols = np.column_stack([self.vcycle(self.A@I[:, i], k_pre, k_post) for i in range(n)])
        return max(abs(np.linalg.eigvals(I - cols)))

def field_scale4(W, nf):
    n = W.shape[0]//nf; d = abs(W.diagonal())
    return sp.diags(np.concatenate([np.full(n, 1/np.sqrt(d[f*n:(f+1)*n].mean())) for f in range(nf)]))
tg.field_scale4 = field_scale4

def make_PT2(m, T2solve, Ksolve, X, qk='rho'):
    """P_T2 on the scaled W (m.W), with inner solvers T2solve(v) ~ T2^-1 v, Ksolve(v) ~ K_rho^-1 v"""
    p = m.p; a = np.sqrt(p['A']); n = m.n; M, K, G = m.M, m.K, m.G; d = m.Ds.diagonal()
    B13 = -(p['e']*M+p['hh']*K)
    def P(rs):
        dd = d if rs.ndim == 1 else d[:, None]; r = rs/dd                   # W_s = D W D: unscaled residual D^-1 r_s
        ru, rw, rp, rj = np.split(r, 4)
        ws, js = X(rw), X(rj)
        ru2 = ru + p['nu']*(K@ws) - a*(G@js); rp2 = rp - B13@js
        y = T2solve(np.concatenate([(m.Mr if qk == 'rho' else M)@Ksolve(ru2), rp2])); du, dp = y[:n], y[n:]
        dw = X(rw - K@du); dj = X(rj - K@dp)
        return np.concatenate([du, dw, dp, dj])/dd
    return P

def fgmres(A, Pinv, b, rtol=0.1, maxit=60):
    beta = np.linalg.norm(b); V = [b/beta]; Hm = np.zeros((maxit+1, maxit), complex)
    for j in range(maxit):
        wv = A@Pinv(V[j])
        for i in range(j+1):
            Hm[i, j] = np.vdot(V[i], wv); wv = wv - Hm[i, j]*V[i]
        Hm[j+1, j] = np.linalg.norm(wv); V.append(wv/Hm[j+1, j])
        e1 = np.zeros(j+2, complex); e1[0] = beta
        y = np.linalg.lstsq(Hm[:j+2, :j+1], e1, rcond=None)[0]
        if np.linalg.norm(Hm[:j+2, :j+1]@y-e1)/beta < rtol: return j+1
    return maxit

def spec(A, Pinv):
    n = A.shape[0]; Ad = A.toarray().astype(complex)
    ev = np.linalg.eigvals(Pinv(Ad) if True else None)
    return ev
