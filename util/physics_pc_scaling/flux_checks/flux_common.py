"""Dense block algebra for the flux-Schur checks (T1, T2/T3).

Unknown ordering of the full family operator: (psi, u, j, omega, rho, T),
field-major, each field in the scalar row order (idx*nslot + m).
Groups: y1 = (psi, j), v = (u, omega), y2 = (rho, T).

Blocks that are not in the dump are zero (B14, B32, B56, ... and any absent
file).  T1 therefore checks consistency against the block-assembled A.
"""
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

PSI, U, J, W, RHO, T = range(6)
FIELD = ["psi", "u", "j", "omega", "rho", "T"]
# (row field, col field) -> scalar block name
BLOCKMAP = {(PSI, PSI): "B11", (PSI, U): "B12", (PSI, J): "B13", (PSI, T): "B16",
            (U, PSI): "B21", (U, U): "B22", (U, J): "B23", (U, W): "B24", (U, RHO): "B25", (U, T): "B26",
            (J, PSI): "B31", (J, J): "B33",
            (W, U): "B42", (W, W): "B44",
            (RHO, PSI): "B51", (RHO, U): "B52", (RHO, RHO): "B55",
            (T, PSI): "B61", (T, U): "B62", (T, J): "B63", (T, RHO): "B65", (T, T): "B66"}


def assemble_A(fam, with_b16=True):
    """sparse 6n x 6n operator from the dumped blocks"""
    n = fam.n
    rows = [[None] * 6 for _ in range(6)]
    for (r, c), name in BLOCKMAP.items():
        if name == "B16" and not with_b16:
            continue
        if fam.has(name):
            rows[r][c] = fam.blk(name)
    for i in range(6):
        if rows[i][i] is None:
            rows[i][i] = sp.csr_matrix((n, n))
    return sp.bmat(rows, format="csr")


def group_perm(n):
    """index arrays of y1, v, y2 inside the (psi,u,j,omega,rho,T) vector"""
    f = lambda k: np.arange(k * n, (k + 1) * n)
    y1 = np.concatenate([f(PSI), f(J)])
    v = np.concatenate([f(U), f(W)])
    y2 = np.concatenate([f(RHO), f(T)])
    return y1, v, y2


def dense(A):
    return A.toarray() if sp.issparse(A) else np.asarray(A)


class LU:
    def __init__(self, A):
        self.lu = sla.lu_factor(dense(A))

    def __call__(self, b):
        return sla.lu_solve(self.lu, b)


class Blocks:
    """dense group blocks of A and the exact Schur quantities"""

    def __init__(self, fam):
        self.fam = fam
        n = self.n = fam.n
        b = lambda k: dense(fam.blk(k))
        Z = np.zeros((n, n))
        self.PJ = np.block([[b("B11"), b("B13")], [b("B31"), b("B33")]])
        self.Dv0 = np.block([[b("B22"), b("B24")], [b("B42"), b("B44")]])
        self.Dv = dense(fam.packed["Dv"]) if "Dv" in fam.packed else None
        self.D2 = np.block([[b("B55"), Z], [b("B65"), b("B66")]])     # block lower triangular
        self.U1 = np.block([[b("B12"), Z], [Z, Z]])
        self.L1 = np.block([[b("B21"), b("B23")], [Z, Z]])
        self.L2 = np.block([[b("B25"), b("B26")], [Z, Z]])
        self.U2 = np.block([[b("B52"), Z], [b("B62"), Z]])
        self.C21 = np.block([[b("B51"), Z], [b("B61"), b("B63")]])
        self.E16 = np.block([[Z, b("B16")], [Z, Z]])
        self.Mw = b("Mw") if fam.has("Mw") else None
        self.Cw = b("Cw") if fam.has("Cw") else None
        self.B12 = b("B12")
        self.S1_dump = dense(fam.packed["S1"]) if "S1" in fam.packed else None
        self.Bb = b("Bb") if fam.has("Bb") else None
        self.D2s = LU(self.D2)
        self.D2iU2 = self.D2s(self.U2)
        self.D2iC21 = self.D2s(self.C21)
        self.Dvh = self.Dv0 - self.L2 @ self.D2iU2               # exact \hat D_v
        self.L1h = self.L1 - self.L2 @ self.D2iC21               # exact \hat L_1

    def S1_exact(self, with_e16=False, L1=None):
        """S1 = P_J - U1 Dvh^{-1} L1h  (with_e16: the E16-consistent version)"""
        L1 = self.L1h if L1 is None else L1
        PJ, U1 = self.PJ, self.U1
        if with_e16:
            PJ = PJ - self.E16 @ self.D2iC21
            U1 = U1 - self.E16 @ self.D2iU2
        return PJ - U1 @ np.linalg.solve(self.Dvh, L1)

    def S1_bend(self, Cmid, form=None):
        """P_J + [[ (bend_sign/varsigma) D B12^T (D Cmid D) B12 D, 0 ], [0, 0]]
        (form 1: B12 (D Cmid D) B12^T), D = diag(free mask)"""
        n = self.n
        m = self.fam.meta
        form = m["bend_form"] if form is None else form
        d = np.zeros(n)
        d[self.fam.free()] = 1.0
        B = d[:, None] * self.B12 * d[None, :]
        C = d[:, None] * Cmid * d[None, :]
        Bb = (B @ C @ B.T) if form == 1 else (B.T @ C @ B)
        S = self.PJ.copy()
        S[:n, :n] += m["bend_sign"] / self.fam.varsigma * Bb
        return S

    def Mw_inv(self):
        return np.linalg.inv(self.Mw)


class Sweep:
    """z = P^{-1} x, spec section 4, with switchable surrogates.

    dv    : 'exact' (Dvh = Dv0 - L2 D2^{-1} U2) | 'dumped' (Dv = Dv0 + W_c) | 'dv0'
    s1    : 'exact' (P_J - U1 Dvh^{-1} L1h) | 'schur_l1' (P_J - U1 Dvh^{-1} L1)
            | 'mwinv' (P_J + B12^T Mw^{-1} B12 / varsigma) | 'cw' (same with Cw)
            | 'dumped' (S1.bin)
    step6 : 'L1'  : r_u - B21 psi - B23 j        (spec section 4 as written)
            'L1h' : r_u - B21 psi - B23 j + [L2 D2^{-1} C21 y1]_u   (exact L1h)
    b16   : 'off' | 'rhs' (step 4: - B16 T*, T* = B66^{-1}... spec)
            | 'exact' (step 4: - E16 y2* - (U1 - E16 D2^{-1}U2) v*, and with s1='exact'
              the E16-consistent S1; makes the sweep equal to A^{-1})
    """

    def __init__(self, blk, dv="exact", s1="exact", step6="L1h", b16="off"):
        self.b = blk
        self.opts = dict(dv=dv, s1=s1, step6=step6, b16=b16)
        self.nD2 = 2 if step6 == "L1" else 3     # number of D2 (B55 + B66) solves per application
        if dv == "exact":
            Dt = blk.Dvh
        elif dv == "dumped":
            if blk.Dv is None:
                raise ValueError("Dv.bin not in dump")
            Dt = blk.Dv
        elif dv == "dv0":
            Dt = blk.Dv0
        else:
            raise ValueError(dv)
        if s1 == "exact":
            St = blk.S1_exact(with_e16=(b16 == "exact"))
        elif s1 == "schur_l1":
            St = blk.S1_exact(L1=blk.L1)
        elif s1 == "mwinv":
            St = blk.S1_bend(blk.Mw_inv())
        elif s1 == "cw":
            St = blk.S1_bend(blk.Cw)
        elif s1 == "dumped":
            if blk.S1_dump is None:
                raise ValueError("S1.bin not in dump")
            St = blk.S1_dump
        else:
            raise ValueError(s1)
        self.Dts = LU(Dt)
        self.Sts = LU(St)
        self.y1, self.v, self.y2 = group_perm(blk.n)

    def __call__(self, X):
        b, o = self.b, self.opts
        x1, xv, x2 = X[self.y1], X[self.v], X[self.y2]
        y2s = b.D2s(x2)                                   # 1: y2 predictor
        rv = xv - b.L2 @ y2s                              # 2: velocity rhs
        vs = self.Dts(rv)                                 # 3: velocity predictor
        r1 = x1 - b.U1 @ vs                               # 4: flux rhs
        if o["b16"] == "rhs":
            r1 = r1 - b.E16 @ y2s
        elif o["b16"] == "exact":
            r1 = r1 - b.E16 @ (y2s - b.D2s(b.U2 @ vs))
        y1 = self.Sts(r1)                                 # 5: flux Schur solve
        r6 = rv - b.L1 @ y1                               # 6: velocity re-solve
        if o["step6"] == "L1h":
            r6 = r6 + b.L2 @ b.D2s(b.C21 @ y1)
        v = self.Dts(r6)
        y2 = b.D2s(x2 - b.C21 @ y1 - b.U2 @ v)            # 7: y2 re-solve
        Z = np.empty_like(X)
        Z[self.y1], Z[self.v], Z[self.y2] = y1, v, y2
        return Z


def spectrum_stats(PA):
    """eigenvalue statistics of P^{-1} A (dense)"""
    e = np.linalg.eigvals(PA)
    N = PA.shape[0]
    dev = PA - np.eye(N)
    return dict(re_min=e.real.min(), re_max=e.real.max(), im_max=np.abs(e.imag).max(),
                rho=np.abs(1 - e).max(), err_max=np.abs(dev).max(),
                err_fro=np.linalg.norm(dev) / np.sqrt(N), eig=e)
