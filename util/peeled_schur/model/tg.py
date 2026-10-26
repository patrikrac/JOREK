"""Two-grid analysis of pair_w: collective ring smoother vs distributive smoother.
Bicubic Hermite (Bezier DOFs, as lfa_mass_lumping); x = radial, nx cells, value DOFs Dirichlet
at both ends; y = rings, periodic, ny nodes. Fields (u, w, psi, j), positive convention,
sigma = 1, grid numbers:
  u  : K u - nu K w + a G j            = r_u      (a = sqrt(A), A grid Alfven number)
  w  : K u + M w                       = r_w
  psi: -a G u + M psi - (e M + hh K) j = r_psi
  j  : K psi + M j                     = r_j
G = Mx (x) Gy + i kap M (along the rings, toroidal shift kap).
W is block scaled per field (as production's pair_w block scale). Coarse: Galerkin P^T W P with
P = Hermite refinement (radial only on semi levels), exact coarse solve."""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla, scipy.linalg as la

gx, gw = np.polynomial.legendre.leggauss(8); s = 0.5*(gx+1); w8 = 0.5*gw
N  = np.array([1-3*s**2+2*s**3, 3*s*(1-s)**2, 3*s**2-2*s**3, -3*s**2*(1-s)])
dN = np.array([-6*s+6*s**2, 3*(1-s)**2-6*s*(1-s), 6*s-6*s**2, -6*s*(1-s)+3*s**2])
Me, Ke, Ge = (N*w8)@N.T, (dN*w8)@dN.T, (N*w8)@dN.T

def asm_per(Ae, n):
    A = np.zeros((2*n, 2*n))
    for e in range(n):
        idx = [2*e, 2*e+1, (2*e+2) % (2*n), (2*e+3) % (2*n)]
        A[np.ix_(idx, idx)] += Ae
    return sp.csr_matrix(A)

def asm_np(Ae, n):
    A = np.zeros((2*n+2, 2*n+2))
    for e in range(n): A[2*e:2*e+4, 2*e:2*e+4] += Ae
    return sp.csr_matrix(A)

_PM = np.array([[0.5, 0.375, 0.5, -0.375], np.array([-1.5, -0.75, 1.5, -0.75])/6])
def prol_per(nc):
    P = np.zeros((4*nc, 2*nc))
    for i in range(nc):
        P[4*i, 2*i] = 1; P[4*i+1, 2*i+1] = 0.5
        i1 = (i+1) % nc; c = [2*i, 2*i+1, 2*i1, 2*i1+1]
        P[4*i+2, c] = _PM[0]; P[4*i+3, c] = _PM[1]
    return sp.csr_matrix(P)

def prol_np(nc):
    P = np.zeros((4*nc+2, 2*nc+2))
    for i in range(nc+1):
        P[4*i, 2*i] = 1; P[4*i+1, 2*i+1] = 0.5
        if i < nc:
            c = [2*i, 2*i+1, 2*i+2, 2*i+3]; P[4*i+2, c] = _PM[0]; P[4*i+3, c] = _PM[1]
    return sp.csr_matrix(P)

def keep_x(nx):
    """1D radial DOFs kept: drop the value DOFs of the end nodes"""
    k = np.ones(2*nx+2, bool); k[0] = k[2*nx] = False; return np.where(k)[0]

class Grid:
    """one level: nx radial cells, ny ring nodes, scalar FE operators on kept DOFs"""
    def __init__(self, nx, ny, hx, hy, kap):
        self.nx, self.ny = nx, ny
        Mx, Kx = hx*asm_np(Me, nx), asm_np(Ke, nx)/hx
        My, Ky, Gy = hy*asm_per(Me, ny), asm_per(Ke, ny)/hy, asm_per(Ge, ny)
        kx = keep_x(nx); self.kx = kx
        sel = np.concatenate([i*2*ny+np.arange(2*ny) for i in kx])   # kept 2D DOFs
        self.sel = sel
        def S(A): return A.tocsr()[sel][:, sel]
        self.M = S(sp.kron(Mx, My)); self.K = S(sp.kron(Kx, My)+sp.kron(Mx, Ky))
        self.G = S(sp.kron(Mx, Gy)) + 1j*kap/hy*self.M
        self.n = len(sel)
        # ring of radial node ix: its kept x DOFs (x node ix: 1D DOFs 2ix, 2ix+1)
        pos = {d: q for q, d in enumerate(kx)}
        self.rings1 = [np.concatenate([pos[d]*2*ny+np.arange(2*ny) for d in (2*ix, 2*ix+1) if d in pos])
                       for ix in range(nx+1)]

def build_W(g, p):
    a = np.sqrt(p['A']); M, K, G = g.M, g.K, g.G
    return sp.bmat([[K, -p['nu']*K, None, a*G],
                    [K, M, None, None],
                    [-a*G, None, M, -(p['e']*M+p['hh']*K)],
                    [None, None, K, M]], format='csr').astype(complex)

def field_scale(W, n):
    """production-like block scaling: per field, s_f = 1/sqrt(mean |diag|)"""
    d = abs(W.diagonal()); s = np.concatenate([np.full(n, 1/np.sqrt(d[f*n:(f+1)*n].mean())) for f in range(4)])
    return sp.diags(s)

class BlockInv:
    def __init__(self, A, rings):
        A = A.tocsr(); self.r = rings
        self.lu = [la.lu_factor(A[np.ix_(r, r)].toarray()) for r in rings]
    def __call__(self, v):
        z = np.zeros_like(v)
        for r, f in zip(self.r, self.lu): z[r] = la.lu_solve(f, v[r])
        return z

def nodal_inv(g):
    """X ~ M^-1: inverse of M's nodal diagonal blocks (all DOFs of one 2D node)"""
    M = g.M.tocsr(); X = sp.lil_matrix(M.shape); ny = g.ny
    for r in g.rings1:
        for iy in range(ny):
            idx = r[[q for q in range(len(r)) if (r[q] % (2*ny))//2 == iy]]
            X[np.ix_(idx, idx)] = np.linalg.inv(M[np.ix_(idx, idx)].toarray())
    return X.tocsr()

def cheb_inv(M, D, k):
    """X = p_k(D^-1 M) D^-1: k-step Chebyshev for M with the nodal-block Jacobi D^-1 (sparse)"""
    T = (D@M).tocsr(); ev = np.linalg.eigvals(T.toarray()).real
    a, b = ev.min()*0.99, ev.max()*1.01; th, de = (a+b)/2, (b-a)/2
    sig = th/de; rho = 1/sig; I = sp.identity(M.shape[0], format='csr')
    r = I.copy(); d = r/th; x = 0*I
    for _ in range(k):
        x = x+d; r = r-T@d; rn = 1/(2*sig-rho); d = rn*rho*d+2*rn/de*r; rho = rn
    return (x@D).tocsr()

def eas_inv(g):
    """element additive Schwarz: sum_e R_e^T (M_ee)^-1 R_e over the 2D elements"""
    M = g.M.tocsr(); X = sp.lil_matrix(M.shape); ny = g.ny; nx = g.nx
    pos = {d: q for q, d in enumerate(g.kx)}
    X = np.zeros(M.shape)
    for ex in range(nx):
        for ey in range(ny):
            idx = [pos[dx]*2*ny + (2*((ey+jy) % ny)+dy) for dx in range(2*ex, 2*ex+4) if dx in pos
                   for jy in (0, 1) for dy in (0, 1)]
            idx = np.array(idx); X[np.ix_(idx, idx)] += np.linalg.inv(M[np.ix_(idx, idx)].toarray())
    return sp.csr_matrix(X)

def adi_inv(g):
    """X = M_rad^-1 D M_ring^-1 (radial line solves, nodal blocks, ring solves): exact for a tensor mass"""
    M = g.M.tocsr(); n = M.shape[0]; ny = g.ny
    ring = [r for r in g.rings1]
    Mr = sp.lil_matrix((n, n))
    for r in ring: Mr[np.ix_(r, r)] = M[np.ix_(r, r)]
    rad_lines = []
    for iy in range(ny):
        rad_lines.append(np.array([q for q in range(n) if (q % (2*ny))//2 == iy]))
    Ml = sp.lil_matrix((n, n))
    for l in rad_lines: Ml[np.ix_(l, l)] = M[np.ix_(l, l)]
    D = sp.csr_matrix(np.linalg.inv(nodal_inv(g).toarray()))
    Mri = np.linalg.inv(Mr.toarray()); Mli = np.linalg.inv(Ml.toarray())
    return sp.csr_matrix(Mli@D.toarray()@Mri)

class Problem:
    def __init__(self, nx, ny, asp, p, semi=True):
        hy = 1.0; hx = hy/asp; self.p = p
        self.g = Grid(nx, ny, hx, hy, p['kap']); g = self.g; n = g.n
        W = build_W(g, p); self.Ds = field_scale(W, n)
        self.Wu = W                                            # unscaled (smoothers' algebra)
        self.W = (self.Ds@W@self.Ds).tocsr()                   # scaled (what FGMRES sees)
        Px = prol_np(nx//2)[keep_x(nx)][:, keep_x(nx//2)]
        Py = sp.identity(2*ny) if semi else prol_per(ny//2)
        P1 = sp.kron(Px, Py).tocsr(); self.P = sp.block_diag([P1]*4).tocsr()
        self.lu = sla.splu((self.P.T@self.W@self.P).tocsc())
        self.rings4 = [np.concatenate([f*n+r for f in range(4)]) for r in g.rings1]

    def coll(self):
        B = BlockInv(self.W, self.rings4); return B

    variant = 'peel'
    def dist(self, xkind='nodal', ws=1.0, hyper=True, qkind='ring', tkind='ring', variant='peel', xop=None):
        self.variant = variant
        g, p = self.g, self.p; M, K, G = g.M, g.K, g.G; a = np.sqrt(p['A']); n = g.n
        if xkind == 'nodal': X = nodal_inv(g)
        elif xkind == 'exact': X = sp.csr_matrix(np.linalg.inv(M.toarray()))
        elif xkind == 'eas': X = eas_inv(g)
        elif xkind == 'adi': X = adi_inv(g)
        elif xkind.startswith('cheb'): X = cheb_inv(M, nodal_inv(g), int(xkind[4:]))
        Qi = BlockInv(K, g.rings1) if qkind == 'ring' else (lambda v, lu=sla.splu(K.tocsc().astype(complex)): lu.solve(v.astype(complex)))
        R = M + p['e']*K + (p['hh']*(K@X@K) if hyper else 0)
        if self.variant == 'cschur':
            Ci = BlockInv(self.Wu, self.rings4); z0 = None
            def T2i(v):
                zero = np.zeros_like(v[:n]); y = Ci(np.concatenate([v[:n], zero, v[n:], zero]))
                return np.concatenate([y[:n], y[2*n:3*n]])
            Qi = lambda v: v; Mq = sp.identity(n)
        elif self.variant == 'schur':
            Xo = X
            if xop == 'nodal': Xo = nodal_inv(g)
            elif xop == 'lump': Xo = sp.diags(1/np.asarray(abs(M).sum(axis=1)).ravel())
            elif xop == 'ring':
                Mr = sp.lil_matrix(M.shape)
                for r in g.rings1: Mr[np.ix_(r, r)] = M[np.ix_(r, r)]
                Xo = sp.csr_matrix(np.linalg.inv(Mr.toarray()))
            R = M + p['e']*K + p['hh']*(K@Xo@K)
            T2 = sp.bmat([[K+p['nu']*(K@Xo@K), -a*(G@Xo@K)], [-a*G, R]]).tocsr()
            Qi = lambda v: v; Mq = sp.identity(n)
        else:
            T2 = sp.bmat([[M+p['nu']*K, -a*G], [-a*G, R]]).tocsr(); Mq = M
        if self.variant == 'cschur': pass
        else: T2i = BlockInv(T2, [np.concatenate([r, n+r]) for r in g.rings1]) if tkind == 'ring' else \
            (lambda v, lu=sla.splu(T2.tocsc().astype(complex)): lu.solve(v.astype(complex)))
        B24, B23, B13 = -p['nu']*K, a*G, -(p['e']*M+p['hh']*K)
        Ds1 = self.Ds.diagonal()
        def P(rs):
            Ds = Ds1 if rs.ndim == 1 else Ds1[:, None]
            r = rs/Ds                                          # scaled residual -> unscaled (W_s = D W D)
            ru, rw, rp, rj = np.split(r, 4)
            ws_, js = X@rw, X@rj
            ru2 = ru - B24@ws_ - B23@js; rp2 = rp - B13@js
            d = ws*T2i(np.concatenate([Mq@Qi(ru2), rp2])); du, dp = d[:n], d[n:]
            dw = X@(rw - K@du); dj = X@(rj - K@dp)
            return np.concatenate([du, dw, dp, dj])/Ds         # unscaled update -> scaled
        return P

    # --- two-grid with level-0 smoothing after the coarse correction (V(0,k) as production)
    def tg_apply(self, Pinv, k, mode='gmres', om=0.8):
        W, Pm, lu = self.W, self.P, self.lu
        def apply(b):
            x = Pm@lu.solve(Pm.T@b)
            if mode == 'gmres': x = gmres_smooth(W, b, x, Pinv, k)
            else:
                for _ in range(k): x = x + om*Pinv(b - W@x)
            return x
        return apply

    def rho(self, Pinv, k, om):
        """spectral radius of the linear two-grid error operator (Richardson smoothing)"""
        n = self.W.shape[0]; app = self.tg_apply(Pinv, k, 'rich', om)
        E = sla.LinearOperator((n, n), matvec=lambda e: e - app(self.W@e), dtype=complex)
        v = sla.eigs(E, k=1, which='LM', return_eigenvectors=False, maxiter=3000, tol=1e-4)
        return abs(v[0])

def gmres_smooth(W, b, x, Pinv, k):
    r0 = b - W@x; beta = np.linalg.norm(r0)
    if beta == 0 or k == 0: return x
    V = [r0/beta]; Z = []; Hm = np.zeros((k+1, k), complex)
    for j in range(k):
        z = Pinv(V[j]); Z.append(z); wv = W@z
        for i in range(j+1):
            Hm[i, j] = np.vdot(V[i], wv); wv = wv - Hm[i, j]*V[i]
        Hm[j+1, j] = np.linalg.norm(wv); V.append(wv/max(Hm[j+1, j], 1e-300))
    e1 = np.zeros(k+1, complex); e1[0] = beta
    y = np.linalg.lstsq(Hm, e1, rcond=None)[0]
    return x + sum(yi*zi for yi, zi in zip(y, Z))

def fgmres_its(W, apply, b, rtol=0.1, maxit=40):
    beta = np.linalg.norm(b); V = [b/beta]; Hm = np.zeros((maxit+1, maxit), complex); rr = 1
    for j in range(maxit):
        wv = W@apply(V[j])
        for i in range(j+1):
            Hm[i, j] = np.vdot(V[i], wv); wv = wv - Hm[i, j]*V[i]
        Hm[j+1, j] = np.linalg.norm(wv); V.append(wv/Hm[j+1, j])
        e1 = np.zeros(j+2, complex); e1[0] = beta
        y = np.linalg.lstsq(Hm[:j+2, :j+1], e1, rcond=None)[0]
        rr = np.linalg.norm(Hm[:j+2, :j+1]@y-e1)/beta
        if rr < rtol: return j+1
    return maxit

def factor(pb, Pinv, k, nrhs=3, seed=0):
    """residual reduction of one V(0,k) application to random right-hand sides (scaled W)"""
    rng = np.random.default_rng(seed); app = pb.tg_apply(Pinv, k); n = pb.W.shape[0]; f = []
    for _ in range(nrhs):
        b = rng.standard_normal(n)+1j*rng.standard_normal(n)
        f.append(np.linalg.norm(b-pb.W@app(b))/np.linalg.norm(b))
    return max(f)
