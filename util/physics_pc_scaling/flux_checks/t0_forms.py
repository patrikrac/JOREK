#!/usr/bin/env python3
"""T0 form checks on one family dump (spec section 9, T0).

All checks are made on the unconstrained rows/cols (Dirichlet indices of
meta.txt removed).  G := B12 / theta_dt.

  G skew          ||G + G^T||_F / ||G||_F                 (and the same for B23)
  B12 = B23       ||B12 - B23||_F / ||B12||_F             (||B12 + B23|| reported: sign flip)
  B31 symmetric   ||B31 - B31^T||_F / ||B31||_F
  B33 SPD         symmetric and lambda_min > 0
  Bb  sym PSD     symmetric, lambda_min >= -tol * lambda_max
  Bb formula      ||Bb - B12^T Cw B12 / varsigma||_F / ||Bb||_F
  Bb Dirichlet    max |Bb| on Dirichlet rows/cols (must be 0 so S1 keeps identity rows)
  S1 = PJ + Bb    relative difference
  Cw              symmetric, lambda_min > 0, sigma(Cw Mw) > 0 and
                  ||I - Cw Mw||_Mw = max|1 - sigma(Cw Mw)| < tol (the Euclidean
                  ||I - Cw Mw||_2 is reported as info: CwMw is non-normal, so it
                  is large whenever Mw is badly conditioned)
  Mw              symmetric, lambda_min > 0
  Dv - Dv0        only in the (u,u) block, W_c symmetric (info)
"""
import argparse
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from petsc_bin import Family


def fro(A):
    return sp.linalg.norm(A) if sp.issparse(A) else np.linalg.norm(A)


def rel(A, B):
    nb = fro(B)
    return fro(A) / nb if nb > 0 else (0.0 if fro(A) == 0 else np.inf)


def eig_range(A, max_dense):
    """(min, max) eigenvalue of a symmetric matrix (symmetrised)"""
    A = 0.5 * (A + A.T)
    if A.shape[0] <= max_dense:
        e = sla.eigvalsh(A.toarray() if sp.issparse(A) else A)
        return e[0], e[-1]
    lo = spla.eigsh(A, k=1, which="SA", return_eigenvectors=False)[0]
    hi = spla.eigsh(A, k=1, which="LA", return_eigenvectors=False)[0]
    return lo, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("famdir")
    ap.add_argument("--int64", action="store_true")
    ap.add_argument("--tol-skew", type=float, default=1e-10, help="G skew / B12=B23 / symmetry tolerance")
    ap.add_argument("--tol-psd", type=float, default=1e-10, help="relative negative-eigenvalue tolerance")
    ap.add_argument("--tol-formula", type=float, default=1e-10, help="Bb / S1 assembly tolerance")
    ap.add_argument("--tol-spai", type=float, default=0.5,
                    help="max ||I - Cw Mw||_Mw = max|1 - sigma(Cw Mw)| for PASS (requires sigma > 0)")
    ap.add_argument("--max-dense", type=int, default=6000)
    a = ap.parse_args()
    f = Family(a.famdir, int64=True if a.int64 else None)
    fr = f.free()
    R = lambda A: A[fr][:, fr]
    th, vs = f.theta, f.varsigma
    res = []

    def chk(name, val, ok, info=""):
        res.append((name, val, "PASS" if ok else "FAIL", info))

    def inf(name, val, info=""):
        res.append((name, val, "info", info))

    def na(name, info="", tag="hpatch"):
        res.append((name, np.nan, "n/a (%s)" % tag, info))

    print("family %s  nslot %d  nindex %d  n %d  free %d  varsigma %g  theta_dt %g"
          % (f.meta["family"], f.nslot, f.nindex, f.n, fr.size, vs, th))
    print("absent blocks (zero): %s" % (" ".join(f.absent) or "-"))
    B12, B23 = R(f.blk("B12")), R(f.blk("B23"))
    G = B12 / th
    chk("G=B12/theta skew  ||G+G^T||/||G||", rel(G + G.T, G), rel(G + G.T, G) < a.tol_skew)
    chk("B23 skew  ||B23+B23^T||/||B23||", rel(B23 + B23.T, B23), rel(B23 + B23.T, B23) < a.tol_skew)
    d, s = rel(B12 - B23, B12), rel(B12 + B23, B12)
    chk("B12 = B23  ||B12-B23||/||B12||", d, d < a.tol_skew,
        "||B12+B23||/||B12|| = %.3e%s" % (s, "  (SIGN FLIP)" if s < a.tol_skew else ""))
    B31, B33 = R(f.blk("B31")), R(f.blk("B33"))
    chk("B31 symmetric", rel(B31 - B31.T, B31), rel(B31 - B31.T, B31) < a.tol_skew)
    lo, hi = eig_range(B33, a.max_dense)
    sy = rel(B33 - B33.T, B33)
    chk("B33 SPD  lambda_min", lo, sy < a.tol_skew and lo > 0, "lambda_max %.3e, asym %.1e" % (hi, sy))
    hp = f.hform                        # hpatch or splitm: H.bin, no Cw
    form = f.meta["bend_form"]
    print("s1 %s  bend_form %d  #dirichlet(u) %d  #dirichlet_psi %d"
          % (f.s1, form, len(f.dirichlet), len(f.dirichlet_psi)))

    def bb_dirichlet_and_s1(Bb_full, D):
        """Bb zero on the Dirichlet rows/cols D; S1 = PJ + [[Bb,0],[0,0]]"""
        if D.size:
            Bc = sp.csr_matrix(Bb_full)
            m = max(abs(Bc[D]).max() if Bc[D].nnz else 0.0, abs(Bc[:, D]).max() if Bc[:, D].nnz else 0.0)
            chk("Bb zero on psi Dirichlet rows/cols", m, m <= a.tol_formula * max(abs(Bb_full).max(), 1e-300))
        if "S1" in f.packed and "PJ" in f.packed:
            n = f.n
            Z = sp.csr_matrix((n, n))
            S1f = f.packed["PJ"] + sp.bmat([[Bb_full, Z], [Z, Z]])
            e = rel(f.packed["S1"] - S1f, f.packed["S1"])
            chk("S1 = PJ + [[Bb,0],[0,0]]", e, e < a.tol_formula)

    if hp:
        # hpatch: Bb = Dpsi B12 H is nonsymmetric; no Cw; symmetric-PSD and Cw formula gates do not apply
        na("Bb symmetric PSD", "hpatch: Bb is nonsymmetric by construction")
        na("Bb = (bend_sign/vs) B12^T Cw B12 formulas", "hpatch: no Cw")
        na("Cw SPD / SPAI quality", "hpatch: no Cw")
        if not (f.has("H") and f.has("Bb")):
            chk("H.bin and Bb.bin present", 0.0, False, "absent: %s" % " ".join(f.absent))
        else:
            H, Bb_full = f.blk("H"), f.blk("Bb")
            dp = np.zeros(f.n); dp[f.free_psi()] = 1.0
            Bf = sp.diags(dp) @ f.blk("B12") @ H
            e = rel(Bf - Bb_full, Bb_full)
            chk("Bb = Dpsi B12 H", e, e < a.tol_formula)
            bb_dirichlet_and_s1(Bb_full, f.dirichlet_psi)
            Du = f.dirichlet
            if Du.size:
                Hc = sp.csr_matrix(H)
                m = abs(Hc[Du]).max() if Hc[Du].nnz else 0.0
                chk("H zero on u Dirichlet rows", m, m <= a.tol_formula * max(abs(Hc).max(), 1e-300))
            # info: H sparsity, asymmetry and definiteness of the bending term
            Hc = sp.csc_matrix(H)
            cnt = np.diff(Hc.indptr)
            cnt = cnt[cnt > 0]          # empty columns = psi Dirichlet columns
            inf("H nnz per nonempty column (min)", float(cnt.min()) if cnt.size else 0.0,
                "mean %.1f, max %d, total nnz %d (p %s r %s g %s vel %s)"
                % (cnt.mean() if cnt.size else 0.0, cnt.max() if cnt.size else 0, Hc.nnz, f.meta.get("p", "?"),
                   f.meta.get("r", "?"), f.meta.get("g", "?"), f.meta.get("vel", "?")))
            frp = f.free_psi()
            Bp = sp.csr_matrix(Bb_full)[frp][:, frp]
            inf("Bb asymmetry  ||Bb-Bb^T||/||Bb||", rel(Bp - Bp.T, Bp) if fro(Bp) > 0 else 0.0)
            if frp.size <= a.max_dense and fro(Bp) > 0:
                lo, hi = eig_range(Bp, a.max_dense)
                inf("min eig of sym(Bb)", lo, "max %.3e (sym part, free psi)" % hi)
                # psi Schur of S1 (free psi, j eliminated), Jacobi-whitened
                PJ = f.packed.get("PJ")
                if PJ is not None:
                    n = f.n
                    B11, B13 = PJ[:n, :n], PJ[:n, n:]
                    B31, B33 = PJ[n:, :n], PJ[n:, n:]
                    fj = f.free()
                    idx = frp
                    X = sla.solve(B33[fj][:, fj].toarray(), B31[fj][:, idx].toarray())
                    Sp = (B11[idx][:, idx] + Bp).toarray() - B13[idx][:, fj].toarray() @ X
                    dg = np.sqrt(np.abs(np.diag(Sp)))
                    dg[dg == 0] = 1.0
                    W = Sp / dg[:, None] / dg[None, :]
                    ev = sla.eigvalsh(0.5 * (W + W.T))
                    inf("min eig of whitened psi-Schur sym part", ev[0],
                        "max %.3e; D^-1/2 (B11 + Bb - B13 B33^-1 B31) D^-1/2, D = |diag|" % ev[-1])
            else:
                na("min eig of sym(Bb) / whitened psi-Schur", "skipped: %d free psi > --max-dense" % frp.size, tag="skip")
    elif f.has("Bb"):
        Bb_full = f.blk("Bb")
        frp = f.free_psi() if form == 2 else fr
        Bb = Bb_full[frp][:, frp]
        sgn = 1.0 if f.meta["bend_sign"] >= 0 else -1.0
        lo, hi = eig_range(sgn * Bb, a.max_dense)
        sy = rel(Bb - Bb.T, Bb)
        chk("sign(bend_sign)*Bb symmetric PSD  lmin/lmax", lo / hi if hi > 0 else lo,
            sy < a.tol_skew and lo >= -a.tol_psd * abs(hi), "lambda [%.3e, %.3e], asym %.1e" % (lo, hi, sy))
        if f.has("Cw"):
            dm = np.zeros(f.n); dm[fr] = 1.0
            Dm = sp.diags(dm)
            dpm = np.zeros(f.n); dpm[f.free_psi()] = 1.0
            Dp = sp.diags(dpm)
            B = Dm @ f.blk("B12") @ Dm
            C = Dm @ f.blk("Cw") @ Dm
            sg = f.meta["bend_sign"] / vs
            F0 = sg * (B.T @ C @ B)          # form 0 (default)
            F1 = sg * (B @ C @ B.T)          # form 1 (-sf_flux_bend_form 1)
            F2 = sg * (Dp @ f.blk("B12") @ C @ f.blk("B12").T @ Dp)    # form 2
            e0, e1, e2 = (rel(F - Bb_full, Bb_full) for F in (F0, F1, F2))
            ed = rel(F0 - F1, F0)
            chk("Bb = (bend_sign/vs) D B12^T(DCwD)B12 D (form 0)", e0, e0 < a.tol_formula or form != 0,
                "bend_sign %g, meta bend_form %d%s" % (f.meta["bend_sign"], form,
                                                      "" if form == 0 else " (not the selected form)"))
            chk("Bb = (bend_sign/vs) D B12(DCwD)B12^T D (form 1)", e1, e1 < a.tol_formula or form != 1,
                "" if form == 1 else "(not the selected form)")
            chk("Bb = (bend_sign/vs) Dpsi B12(DuCwDu)B12^T Dpsi (form 2)", e2, e2 < a.tol_formula or form != 2,
                "" if form == 2 else "(not the selected form)")
            inf("||B12^T C B12 - B12 C B12^T|| / ||B12^T C B12||", ed,
                "zero iff B12 skew on the free space; form difference")
            if sg < 0:
                res.append(("bend_sign/varsigma < 0: bending term negative", sg, "info",
                            "PSD check above then applies to -Bb"))
        bb_dirichlet_and_s1(Bb_full, f.dirichlet_psi if form == 2 else f.dirichlet)
    if f.has("Mw"):
        Mw = R(f.blk("Mw"))
        lo, hi = eig_range(Mw, a.max_dense)
        sy = rel(Mw - Mw.T, Mw)
        chk("Mw SPD  lambda_min", lo, sy < a.tol_skew and lo > 0, "lambda_max %.3e, asym %.1e" % (hi, sy))
    if f.has("Cw"):
        Cw = R(f.blk("Cw"))
        lo, hi = eig_range(Cw, a.max_dense)
        sy = rel(Cw - Cw.T, Cw)
        chk("Cw symmetric PD  lambda_min", lo, sy < a.tol_skew and lo > 0, "lambda_max %.3e, asym %.1e" % (hi, sy))
        if f.has("Mw") and Mw.shape[0] <= a.max_dense:
            Md, Cd = Mw.toarray(), Cw.toarray()
            L = np.linalg.cholesky(0.5 * (Md + Md.T))
            e = sla.eigvalsh(L.T @ (0.5 * (Cd + Cd.T)) @ L)       # sigma(Cw Mw)
            E = np.eye(Md.shape[0]) - Cd @ Md
            nM = max(abs(1 - e[0]), abs(1 - e[-1]))               # ||I - Cw Mw|| in the Mw norm
            chk("SPAI  ||I - Cw Mw||_Mw = max|1-sigma(CwMw)|", nM, e[0] > 0 and nM < a.tol_spai,
                "sigma(Cw Mw) in [%.4f, %.4f]; Euclid: ||I-CwMw||_2 %.3e, ||.||_F/sqrt(n) %.3e, max|.| %.3e"
                % (e[0], e[-1], np.linalg.norm(E, 2), np.linalg.norm(E) / np.sqrt(E.shape[0]), abs(E).max()))
    if "Dv" in f.packed and "Dv0" in f.packed:
        n = f.n
        Dd = f.packed["Dv"] - f.packed["Dv0"]
        off = rel(Dd[n:, :], Dd) + rel(Dd[:n, n:], Dd) if fro(Dd) > 0 else 0.0
        Wc = R(Dd[:n, :n])
        lo, hi = eig_range(Wc, a.max_dense)
        inf("Dv-Dv0 outside (u,u)", off)
        inf("W_c asym / eig range", rel(Wc - Wc.T, Wc) if fro(Wc) > 0 else 0.0,
            "lambda [%.3e, %.3e] (B22 sign convention: W_c should be <= 0 if B22 < 0)" % (lo, hi))
        lo22, hi22 = eig_range(R(f.blk("B22")), a.max_dense)
        inf("B22 eig range", lo22, "lambda [%.3e, %.3e]" % (lo22, hi22))

    w = max(len(r[0]) for r in res)
    print("\n%-*s  %12s  %-12s  %s" % (w, "check", "value", "result", "info"))
    print("-" * (w + 46))
    for name, val, st, info in res:
        print("%-*s  %12.4e  %-12s  %s" % (w, name, val, st, info))
    nf = sum(r[2] == "FAIL" for r in res)
    print("\nT0 %s (%d FAIL)" % ("PASS" if nf == 0 else "FAIL", nf))
    raise SystemExit(0 if nf == 0 else 1)


if __name__ == "__main__":
    main()
