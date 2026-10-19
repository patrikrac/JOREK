#!/usr/bin/env python3
"""Generate a synthetic flux-Schur family dump (same format as -sf_flux_dump).

Geometry: a 2-D slab annulus x in [x0, x0+Lx] (rings I = 0..nr-1), y in [0,1)
periodic (index J = 0..nt-1), bicubic Hermite elements with 4 DOFs per node
(f, hx f_x, hy f_y, hx hy f_xy: h-scaled like the LFA and like JOREK's Bezier
control points), exact 6x6 Gauss quadrature.  --physical-dofs uses unscaled
derivatives instead (badly conditioned mass, cond ~ 1/h^4: the plain l2 SPAI
of M_w then fails, see README).
Scalar index idx = (I*nt + J)*4 + d; row = idx*nslot + m.

Boundary: Dirichlet on the outer ring (DOFs f and the tangential f_y), natural
at the inner ring.  psi0 is chosen with psi0_y = 0 on the inner ring so the
bracket form stays skew on the constrained space (as in JOREK with the axis).

The forms mimic Table 1 of the design spec with plausible coefficients
(R = R0 + x cos 2 pi y, rho0(x), T0(x), psi0 slightly misaligned with the
rings, j0, a small flow u0).  Toroidal family n: slots (cos, sin), d/dphi ->
n [[0,1],[-1,0]] on the trial function.  Bracket [a,b] = a_x b_y - a_y b_x;
"d_Z" = sin(2 pi y) d_x + cos(2 pi y) d_y / (2 pi x) (polar-like).

The pressure surrogate W_c is the element-assembled compression stiffness
-(theta^2/varsigma) int 4 R rho0 T0 (1+gamma') dZ v dZ f (negative
semidefinite, the sign convention of B22 = -varsigma K_{R^3 rho0}).

All blocks and M_w carry JOREK's slot normalisation (--n-plane: n_plane on n=0,
n_plane/2 on cos/sin); Dirichlet rows stay identity.  meta.txt has bend_sign 1.

C_w is an l2 SPAI of the Jacobi-scaled D^-1/2 M_w D^-1/2 (mapped back; --spai-plain
for the unscaled one), M_w = M_{R^3 rho0}, on the Chebyshev node pattern of
radius r (periodic in J), all 4 DOFs per node, computed on the unconstrained
DOFs and symmetrised; Dirichlet rows/cols of Mw and Cw are identity.
"""
import os
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla

from petsc_bin import write_petsc_mat

# ---------------------------------------------------------------- 1-D Hermite
def H(s):
    return np.array([1 - 3 * s**2 + 2 * s**3, s - 2 * s**2 + s**3, 3 * s**2 - 2 * s**3, -s**2 + s**3])


def dH(s):
    return np.array([-6 * s + 6 * s**2, 1 - 4 * s + 3 * s**2, 6 * s - 6 * s**2, -2 * s + 3 * s**2])


class Grid:
    def __init__(self, nr, nt, x0, Lx, nq=6, physical=False):
        self.nr, self.nt, self.x0, self.Lx = nr, nt, x0, Lx
        self.physical = physical
        self.hx, self.hy = Lx / (nr - 1), 1.0 / nt
        self.nnode = nr * nt
        self.nindex = 4 * self.nnode
        xg, wg = np.polynomial.legendre.leggauss(nq)
        sq, wq = (xg + 1) / 2, wg / 2
        S, T = np.meshgrid(sq, sq, indexing="ij")
        self.S, self.T = S.ravel(), T.ravel()
        self.W = np.outer(wq, wq).ravel() * self.hx * self.hy
        Hs, dHs = H(self.S), dH(self.S)
        Ht, dHt = H(self.T), dH(self.T)
        nq2 = self.S.size
        phi = np.zeros((nq2, 16)); phx = np.zeros((nq2, 16)); phy = np.zeros((nq2, 16))
        hx, hy = self.hx, self.hy
        for cy in range(2):
            for cx in range(2):
                c = cx + 2 * cy
                for d in range(4):
                    dx, dy = d in (1, 3), d in (2, 3)
                    ax, ay = 2 * cx + dx, 2 * cy + dy
                    fx = hx if (dx and physical) else 1.0
                    fy = hy if (dy and physical) else 1.0
                    L = 4 * c + d
                    phi[:, L] = fx * Hs[ax] * fy * Ht[ay]
                    phx[:, L] = fx * dHs[ax] / hx * fy * Ht[ay]
                    phy[:, L] = fx * Hs[ax] * fy * dHt[ay] / hy
        self.phi, self.phx, self.phy = phi, phx, phy
        self.elems = []
        for I in range(nr - 1):
            for J in range(nt):
                nodes = [(I + cx) * nt + (J + cy) % nt for cy in range(2) for cx in range(2)]
                dofs = np.array([4 * nd + d for nd in nodes for d in range(4)])
                x = x0 + (I + self.S) * hx
                y = (J + self.T) * hy
                self.elems.append((dofs, x, y))

    def nodal(self, f, fx, fy, fxy):
        """Hermite DOF vector of an analytic field (callables of x, y)"""
        v = np.zeros(self.nindex)
        for I in range(self.nr):
            for J in range(self.nt):
                x, y = self.x0 + I * self.hx, J * self.hy
                b = 4 * (I * self.nt + J)
                sx, sy = (1.0, 1.0) if self.physical else (self.hx, self.hy)
                v[b:b + 4] = [f(x, y), sx * fx(x, y), sy * fy(x, y), sx * sy * fxy(x, y)]
        return v

    def assemble(self, kernel):
        """kernel(x, y, dofs) -> 16x16 local matrix (row = test, col = trial)"""
        rows, cols, vals = [], [], []
        for dofs, x, y in self.elems:
            E = kernel(x, y, dofs)
            rows.append(np.repeat(dofs, 16)); cols.append(np.tile(dofs, 16)); vals.append(E.ravel())
        A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(self.nindex, self.nindex)).tocsr()
        A.sum_duplicates()
        A.eliminate_zeros()
        return A

    # elementary forms ----------------------------------------------------
    def mass(self, w):
        return self.assemble(lambda x, y, d: self.phi.T @ ((w(x, y) * self.W)[:, None] * self.phi))

    def stiff(self, w):
        def k(x, y, d):
            c = (w(x, y) * self.W)[:, None]
            return self.phx.T @ (c * self.phx) + self.phy.T @ (c * self.phy)
        return self.assemble(k)

    def conv(self, bx, by, w=None):
        """int w v (b . grad f)"""
        def k(x, y, d):
            ww = self.W * (1.0 if w is None else w(x, y))
            df = bx(x, y, d)[:, None] * self.phx + by(x, y, d)[:, None] * self.phy
            return self.phi.T @ (ww[:, None] * df)
        return self.assemble(k)

    def conv2(self, ax, ay, bx, by, w=None):
        """int w (a . grad v)(b . grad f)"""
        def k(x, y, d):
            ww = self.W * (1.0 if w is None else w(x, y))
            dv = ax(x, y, d)[:, None] * self.phx + ay(x, y, d)[:, None] * self.phy
            df = bx(x, y, d)[:, None] * self.phx + by(x, y, d)[:, None] * self.phy
            return dv.T @ (ww[:, None] * df)
        return self.assemble(k)

    def conv_tv(self, ax, ay, w):
        """int w (a . grad v) f"""
        def k(x, y, d):
            dv = ax(x, y, d)[:, None] * self.phx + ay(x, y, d)[:, None] * self.phy
            return dv.T @ ((w(x, y) * self.W)[:, None] * self.phi)
        return self.assemble(k)


def spai(M, nodes_of, pattern_nodes, free):
    """l2 SPAI of SPD M on node patterns; rows/cols restricted to `free` DOFs."""
    n = M.shape[0]
    fmask = np.zeros(n, bool); fmask[free] = True
    Mc = M.tocsc()
    C = sp.lil_matrix((n, n))
    for k in free:
        J = np.array([4 * nd + d for nd in pattern_nodes[nodes_of[k]] for d in range(4)])
        J = J[fmask[J]]
        sub = Mc[:, J]
        I = np.unique(sub.indices)
        I = I[fmask[I]]
        A = sub[I, :].toarray()
        e = (I == k).astype(float)
        c, *_ = np.linalg.lstsq(A, e, rcond=None)
        for jj, cj in zip(J, c):
            C[jj, k] = cj
    C = sp.csr_matrix(C)
    C = 0.5 * (C + C.T)
    return C.tocsr()


def dirichlet_fix(A, D, diag):
    """zero rows and columns D; set diagonal to 1 if diag"""
    A = sp.csr_matrix(A).tolil()
    A[D, :] = 0
    A = A.tocsc().tolil()
    A[:, D] = 0
    A = sp.csr_matrix(A)
    if diag:
        A = A + sp.csr_matrix((np.ones(len(D)), (D, D)), shape=A.shape)
    A.eliminate_zeros()
    return A


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir")
    ap.add_argument("--nr", type=int, default=6, help="rings")
    ap.add_argument("--nt", type=int, default=12, help="poloidal nodes")
    ap.add_argument("--families", default="0,1", help="toroidal n per family, family f = index+1")
    ap.add_argument("--theta", type=float, default=10.0, help="theta*dt")
    ap.add_argument("--varsigma", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.05, help="T0 amplitude (pressure)")
    ap.add_argument("--eps", type=float, default=0.05, help="psi0 misalignment amplitude")
    ap.add_argument("--shear", type=float, default=1.0,
                    help="psi0 = -(shear x^2/2 + (1-shear) 0.45 x) + misalignment; 0 = uniform B_y")
    ap.add_argument("--rcos",type=float, default=1.0, help="R = R0 + rcos * x cos(2 pi y)")
    ap.add_argument("--rho-var", type=float, default=0.6, help="rho0 = 1 - rho_var s^2")
    ap.add_argument("--j0", choices=["equilibrium", "profile", "zero"], default="equilibrium",
                    help="j0 = R div(R^-1 grad psi0) | 1 - s^2 | 0")
    ap.add_argument("--u0", type=float, default=1e-2, help="flow amplitude")
    ap.add_argument("--nu", type=float, default=1e-6, help="viscosity (B24 = theta nu K_R)")
    ap.add_argument("--eta", type=float, default=1e-5, help="resistivity (hyper-resistivity 1e-3 eta)")
    ap.add_argument("--chipar", type=float, default=100.0, help="chi_par (D_par = 0.1 chi_par)")
    ap.add_argument("--rspai", type=int, default=2)
    ap.add_argument("--b65", type=float, default=1e-3,
                    help="B65 = b65 theta M (a non-physical rho->T coupling that only exercises the "
                         "block-triangular D2; 0.1 already opens a spurious B26 B66^-1 B65 B55^-1 B52 "
                         "channel that makes the n=0 S1 indefinite)")
    ap.add_argument("--spai-plain", dest="spai_jacobi", action="store_false",
                    help="plain l2 SPAI of Mw (default: SPAI of the Jacobi-scaled D^-1/2 Mw D^-1/2, "
                         "then mapped back, as in the Fortran)")
    ap.add_argument("--n-plane", type=int, default=8, help="FFT slot normalisation factor")
    ap.add_argument("--no-b65", action="store_true", help="do not write B65")
    ap.add_argument("--no-b16", action="store_true", help="do not write B16")
    ap.add_argument("--bend-form", type=int, choices=[0, 1, 2], default=0,
                    help="spai dumps: 0 B12^T Cw B12 (default), 2 Dpsi B12 (Du Cw Du) B12^T Dpsi "
                         "(meta bend_form + dirichlet_psi; form 1 is not generated)")
    ap.add_argument("--hpatch", choices=["off", "banded", "patch"], default="off",
                    help="write an hpatch-style dump (meta s1 hpatch, H.bin, Bb = Dpsi B12 H, no Cw.bin): "
                         "'banded' = SPAI-banded nonsymmetric H, 'patch' = the t5_hpatch patch construction")
    ap.add_argument("--hp-p", type=int, default=3, help="hpatch patch radius (meta p; 'patch' mode)")
    ap.add_argument("--hp-r", type=int, default=3, help="hpatch band radius (meta r)")
    ap.add_argument("--hp-g", type=int, default=1, help="hpatch group size (meta g; 1 = exact t5 gate)")
    ap.add_argument("--hp-vel", type=int, default=0, help="hpatch velocity block (meta vel)")
    ap.add_argument("--int64", action="store_true", help="write 64-bit PetscInt files")
    ap.add_argument("--physical-dofs", action="store_true", help="unscaled derivative DOFs")
    a = ap.parse_args()

    x0, Lx, R0, F0 = 0.2, 0.5, 3.0, 3.0
    th, vs = a.theta, a.varsigma
    eta, etah, nu = a.eta, 1e-3 * a.eta, a.nu
    Dpar, Dperp, chipar, chiperp, gp = 0.1 * a.chipar, 1e-3, a.chipar, 1e-3, 2.0 / 3.0
    tp = 2 * np.pi
    g = Grid(a.nr, a.nt, x0, Lx, physical=a.physical_dofs)
    s = lambda x: (x - x0) / Lx
    R = lambda x, y: R0 + a.rcos * x * np.cos(tp * y)
    rho0 = lambda x, y: 1 - a.rho_var * s(x)**2
    rho0x = lambda x, y: -2 * a.rho_var * s(x) / Lx
    T0 = lambda x, y: a.beta * (1 - 0.8 * s(x)**2)
    T0x = lambda x, y: -1.6 * a.beta * s(x) / Lx
    sZ = lambda x, y: np.sin(tp * y)
    cZ = lambda x, y: np.cos(tp * y) / (tp * x)
    zero = lambda x, y, d=None: np.zeros_like(x)

    # FE fields (Hermite interpolants of analytic fields), evaluated at quadrature points
    import sympy as sy
    X, Y = sy.symbols("x y")
    S_ = (X - x0) / Lx
    R_ = R0 + a.rcos * X * sy.cos(2 * sy.pi * Y)
    psi_ = -(a.shear * X**2 / 2 + (1 - a.shear) * 0.45 * X) + a.eps * Lx * S_ * (1 - S_) * sy.cos(2 * sy.pi * Y)
    if a.j0 == "equilibrium":          # j0 = R div(R^-1 grad psi0), consistent with B31/B33
        j_ = sy.simplify(R_ * (sy.diff(sy.diff(psi_, X) / R_, X) + sy.diff(sy.diff(psi_, Y) / R_, Y)))
    elif a.j0 == "profile":
        j_ = 1 - S_**2
    else:
        j_ = sy.Integer(0) * X
    u_ = a.u0 * S_ * (1 - S_) * sy.sin(2 * sy.pi * Y)

    def herm(e):
        fs = [sy.lambdify((X, Y), d, "numpy") for d in
              (e, sy.diff(e, X), sy.diff(e, Y), sy.diff(e, X, Y))]
        return g.nodal(*[(lambda F: (lambda x, y: float(F(x, y))))(F) for F in fs])
    psi_dofs, j_dofs, u_dofs = herm(psi_), herm(j_), herm(u_)
    fe = lambda v, which: (lambda x, y, d: {"f": g.phi, "x": g.phx, "y": g.phy}[which] @ v[d])
    psx, psy = fe(psi_dofs, "x"), fe(psi_dofs, "y")
    jx, jy = fe(j_dofs, "x"), fe(j_dofs, "y")
    j0v = fe(j_dofs, "f")
    ux, uy = fe(u_dofs, "x"), fe(u_dofs, "y")
    # bracket [f, b] = f_x b_y - f_y b_x  -> conv with (b_y, -b_x)
    neg = lambda F: (lambda x, y, d: -F(x, y, d))
    bpx, bpy = psy, neg(psx)                              # poloidal B.grad = [., psi0]
    an = lambda F: (lambda x, y, d: F(x, y))              # analytic -> kernel signature

    Mi = g.mass(lambda x, y: 1 / R(x, y)); Ki = g.stiff(lambda x, y: 1 / R(x, y))
    MR = g.mass(R); KR = g.stiff(R)
    w = lambda x, y: R(x, y)**3 * rho0(x, y)
    Mw_s = g.mass(w); Kw = g.stiff(w)
    M1 = g.mass(lambda x, y: np.ones_like(x)); K1 = g.stiff(lambda x, y: np.ones_like(x))
    MFR = g.mass(lambda x, y: F0 / R(x, y)); MFR2 = g.mass(lambda x, y: (F0 / R(x, y))**2)
    Gbr = g.conv(bpx, bpy)
    Adv = g.conv(uy, neg(ux))
    Brj = g.conv(jy, neg(jx))
    Brr = g.conv(zero, an(lambda x, y: -rho0x(x, y)))     # [f, rho0], rho0_y = 0
    BrT = g.conv(zero, an(lambda x, y: -T0x(x, y)))
    Kpp = g.conv2(bpx, bpy, bpx, bpy)
    Kpt = g.conv_tv(bpx, bpy, lambda x, y: F0 / R(x, y))          # int (b.grad v)(F0/R) f
    Xr = g.conv2(bpx, bpy, zero, an(lambda x, y: -rho0x(x, y)))   # int (b.grad v)[f,rho0]
    XT = g.conv2(bpx, bpy, zero, an(lambda x, y: -T0x(x, y)))
    Xtr = g.conv(zero, an(lambda x, y: -rho0x(x, y)), w=lambda x, y: F0 / R(x, y))
    XtT = g.conv(zero, an(lambda x, y: -T0x(x, y)), w=lambda x, y: F0 / R(x, y))
    Dz = lambda wf: g.conv(an(sZ), an(cZ), w=wf)
    B25_s = th * (Dz(lambda x, y: 2 * R(x, y) * T0(x, y)) + g.mass(lambda x, y: 2 * R(x, y) * sZ(x, y) * T0x(x, y)))
    B26_s = th * (Dz(lambda x, y: 2 * R(x, y) * rho0(x, y)) + g.mass(lambda x, y: 2 * R(x, y) * sZ(x, y) * rho0x(x, y)))
    B52_s = th * (Brr - Dz(lambda x, y: 2 * rho0(x, y)))
    B62_s = th * (BrT - Dz(lambda x, y: 2 * gp * T0(x, y)))
    Wc_s = -(th**2 / vs) * g.conv2(an(sZ), an(cZ), an(sZ), an(cZ),
                                   w=lambda x, y: 4 * R(x, y) * rho0(x, y) * T0(x, y) * (1 + gp))
    M16 = g.assemble(lambda x, y, d: g.phi.T @ ((0.5 * eta * j0v(x, y, d) / R(x, y) * g.W)[:, None] * g.phi))
    M63 = g.assemble(lambda x, y, d: g.phi.T @ ((2 * eta * j0v(x, y, d) * g.W)[:, None] * g.phi))

    # Dirichlet: outer ring, DOFs f and f_y
    I = a.nr - 1
    Dix = np.array([4 * (I * a.nt + J) + d for J in range(a.nt) for d in (0, 2)])

    # SPAI of Mw (scalar, per slot)
    nodes_of = np.arange(g.nindex) // 4
    pat = {}
    for Ii in range(a.nr):
        for J in range(a.nt):
            pat[Ii * a.nt + J] = [I2 * a.nt + (J2 % a.nt)
                                  for I2 in range(max(0, Ii - a.rspai), min(a.nr, Ii + a.rspai + 1))
                                  for J2 in range(J - a.rspai, J + a.rspai + 1)]
            pat[Ii * a.nt + J] = sorted(set(pat[Ii * a.nt + J]))
    free_ix = np.setdiff1d(np.arange(g.nindex), Dix)
    if a.spai_jacobi:      # SPAI of D^-1/2 Mw D^-1/2, D = diag(Mw): invariant to DOF scaling
        dh = 1.0 / np.sqrt(Mw_s.diagonal())
        Dh = sp.diags(dh)
        Cw_s = (Dh @ spai(Dh @ Mw_s @ Dh, nodes_of, pat, free_ix) @ Dh).tocsr()
    else:
        Cw_s = spai(Mw_s, nodes_of, pat, free_ix)
    Cw_s = Cw_s + sp.csr_matrix((np.ones(len(Dix)), (Dix, Dix)), shape=Cw_s.shape)

    fams = [int(v) for v in a.families.split(",")]
    for fi, n in enumerate(fams):
        fam = fi + 1
        nslot = 1 if n == 0 else 2
        Is = sp.identity(nslot, format="csr")
        Tm = sp.csr_matrix(np.array([[0.0, 1.0], [-1.0, 0.0]]) * n) if nslot == 2 else None
        kr = lambda A: sp.kron(A, Is, format="csr")
        tor = lambda A: sp.kron(A, Tm, format="csr") if Tm is not None else 0 * kr(A)
        torT = lambda A: sp.kron(A, Tm.T, format="csr") if Tm is not None else 0 * kr(A)
        G = kr(Gbr) + tor(MFR)
        Kpar = kr(Kpp) + tor(Kpt) + torT(Kpt.T) + n**2 * kr(MFR2)
        Flr = kr(Xr) + torT(Xtr)
        FlT = kr(XT) + torT(XtT)
        D = np.array([i * nslot + m for i in Dix for m in range(nslot)])
        nn = g.nindex * nslot
        B = {}
        B["B11"] = vs * kr(Mi) - th * kr(Adv)
        B["B12"] = th * G
        B["B13"] = -th * (eta * kr(Mi) + etah * kr(Ki))
        B["B16"] = -th * kr(M16)
        B["B21"] = -th * kr(Brj)
        B["B22"] = -vs * kr(Kw)
        B["B23"] = th * G
        B["B24"] = th * nu * kr(KR)
        B["B25"] = kr(B25_s); B["B26"] = kr(B26_s)
        B["B31"] = kr(Ki); B["B33"] = kr(Mi)
        B["B42"] = kr(KR); B["B44"] = kr(MR)
        B["B51"] = -th * Dpar * Flr
        B["B52"] = kr(B52_s)
        B["B55"] = vs * kr(M1) + th * (Dperp * kr(K1) + Dpar * Kpar)
        B["B61"] = -th * chipar * FlT
        B["B62"] = kr(B62_s)
        B["B63"] = -th * kr(M63)
        B["B65"] = a.b65 * th * kr(M1)
        B["B66"] = vs * kr(M1) + th * (chiperp * kr(K1) + chipar * Kpar)
        Wc = kr(Wc_s)
        # JOREK FFT slot normalisation: n_plane on the n=0 slot, n_plane/2 on cos/sin
        fn = a.n_plane if n == 0 else a.n_plane / 2.0
        diag = {"B11", "B22", "B33", "B44", "B55", "B66"}
        for k in B:
            B[k] = dirichlet_fix(fn * B[k], D, k in diag)    # Dirichlet rows stay identity
        Wc = dirichlet_fix(fn * Wc, D, False)
        IDs = sp.csr_matrix((np.ones(len(Dix)), (Dix, Dix)), shape=Cw_s.shape)
        Mw = kr(dirichlet_fix(fn * Mw_s, Dix, True))
        Cw = kr((Cw_s - IDs) / fn + IDs)                      # SPAI of the normalised Mw
        dmk = np.ones(nn); dmk[D] = 0.0                       # u / psi free mask (the same here)
        Dm_ = sp.diags(dmk)
        if a.bend_form == 2:    # Dpsi B12 (Du Cw Du) B12^T Dpsi / vs
            Bb = Dm_ @ B["B12"] @ Dm_ @ Cw @ Dm_ @ B["B12"].T @ Dm_ / vs
        else:
            Bb = (B["B12"].T @ Cw @ B["B12"]) / vs
        Z = sp.csr_matrix((nn, nn))
        PJ = sp.bmat([[B["B11"], B["B13"]], [B["B31"], B["B33"]]], format="csr")
        S1 = PJ + sp.bmat([[Bb, Z], [Z, Z]], format="csr")
        Dv0 = sp.bmat([[B["B22"], B["B24"]], [B["B42"], B["B44"]]], format="csr")
        Dv = Dv0 + sp.bmat([[Wc, Z], [Z, Z]], format="csr")
        od = os.path.join(a.outdir, "fam%d" % fam)
        os.makedirs(od, exist_ok=True)
        out = {k: B[k] for k in ["B12", "B16", "B21", "B23", "B25", "B26", "B51", "B52", "B61",
                                 "B62", "B63", "B65", "B55", "B66"]}
        if a.no_b65:
            out.pop("B65")
        if a.no_b16:
            out.pop("B16")
        out.update(Mw=Mw, Cw=Cw, Bb=Bb, PJ=PJ, S1=S1, Dv0=Dv0, Dv=Dv)
        hp = a.hpatch != "off"
        if hp:      # hpatch-style dump: H.bin, Bb = Dpsi B12 H, no Cw
            out.pop("Cw")
        for k, A in out.items():
            write_petsc_mat(os.path.join(od, k + ".bin"), A, int64=a.int64)
        for k in ["B65", "B16"] + (["Cw"] if hp else ["H"]):
            p = os.path.join(od, k + ".bin")
            if k not in out and os.path.exists(p):
                os.remove(p)

        def write_meta():
            with open(os.path.join(od, "meta.txt"), "w") as fh:
                fh.write("family %d\nnslot %d\nnindex %d\nvarsigma %.17g\ntheta_dt %.17g\n"
                         % (fam, nslot, g.nindex, vs, th))
                fh.write("dirichlet " + " ".join(str(i) for i in D) + "\n")
                fh.write("bend_sign 1\n")
                if hp:
                    fh.write("bend_form 0\ns1 hpatch\np %d\nr %d\ng %d\nvel %d\n" % (a.hp_p, a.hp_r, a.hp_g, a.hp_vel))
                else:
                    fh.write("bend_form %d\ns1 spai\n" % a.bend_form)
                if hp or a.bend_form == 2:
                    fh.write("dirichlet_psi " + " ".join(str(i) for i in D) + "\n")
                fh.write("# synthetic: n_tor %d %s\n" % (n, " ".join("%s=%s" % kv for kv in sorted(vars(a).items()))))
        write_meta()
        if hp:
            Dpm = sp.diags(dmk)
            if a.hpatch == "patch":     # the patch construction of t5_hpatch (g == 1 passes its gate)
                from petsc_bin import Family
                from t5_hpatch import rebuild_H
                Hm, _ = rebuild_H(Family(od), a.hp_p, a.hp_r, a.hp_g, a.hp_vel)
            else:                       # banded H on the SPAI pattern; the triu term makes Bb nonsymmetric
                Hm = (Dpm @ (Cw + 0.3 * sp.triu(Cw, 1)) @ B["B12"].T @ Dpm / vs).tocsr()
            Bb_h = (Dpm @ B["B12"] @ Hm).tocsr()
            S1_h = PJ + sp.bmat([[Bb_h, Z], [Z, Z]], format="csr")
            for k, A in (("H", Hm), ("Bb", Bb_h), ("S1", S1_h)):
                write_petsc_mat(os.path.join(od, k + ".bin"), A, int64=a.int64)
        print("wrote %s  (n=%d, nslot=%d, nindex=%d, %d Dirichlet rows)" % (od, n, nslot, g.nindex, len(D)))


if __name__ == "__main__":
    main()
