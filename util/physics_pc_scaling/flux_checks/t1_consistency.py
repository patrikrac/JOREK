#!/usr/bin/env python3
"""T1 algebraic consistency of the flux-Schur sweep (spec sections 2, 4, 9).

The full family operator A is block-assembled from the dump in the order
(psi, u, j, omega, rho, T); blocks that are not dumped are zero, so T1 checks
consistency against the block-assembled A (not against the JOREK matrix).
A0 = A with B16 zeroed.  D2 = [[B55, 0], [B65, B66]] (B65 if present).

Exact elimination (y2 | v | y1), E16 neglected:
  1  y2* = D2^{-1} x2
  2  r_v = x_v - L2 y2*
  3  v*  = Dvh^{-1} r_v                       Dvh = Dv0 - L2 D2^{-1} U2
  4  r_1 = x_1 - U1 v*
  5  y1  = S1^{-1} r_1                        S1  = P_J - U1 Dvh^{-1} L1h
  6  v   = Dvh^{-1} (r_v - L1h y1)            L1h = L1  - L2 D2^{-1} C21
  7  y2  = D2^{-1} (x2 - C21 y1 - U2 v)
Step 6 of the spec uses L1 (r_u - B21 psi - B23 j) instead of L1h.  The term
L2 D2^{-1} C21 y1 (curvature force of the flux-driven rho/T change) is then
missing from the velocity re-solve and the sweep is NOT A0^{-1}, even with all
components exact.  Option step6=L1h restores it (one extra B55 + B66 solve).

Runs (gate: 'EXACT' and 'EXACT+E16' must satisfy max|P^{-1}A0 - I| <= --tol, or be
within --ref-factor of the round-off of a dense LU solve of the Ruiz-equilibrated A0; 1e-12
absolute is only reachable for well-conditioned A, the j / omega rows inherit
cond(M_{1/R}), cond(M_R) ~ 1e5 of the Hermite mass):
  EXACT              all exact, step6 = L1h, B16 off                    vs A0
  EXACT+E16          all exact incl. the E16-consistent S1 / step-4 rhs  vs A
  SPEC-AS-WRITTEN    all exact, step6 = L1 (spec section 4)              vs A0
then one surrogate at a time (on top of EXACT), and the production combination.
"""
import argparse
import time
import numpy as np
import scipy.linalg as sla

from petsc_bin import Family
from flux_common import Blocks, Sweep, assemble_A, spectrum_stats


RUNS = [
    # name, target, options, gate
    ("EXACT", "A0", dict(dv="exact", s1="exact", step6="L1h", b16="off"), True),
    ("EXACT+E16", "A", dict(dv="exact", s1="exact", step6="L1h", b16="exact"), True),
    ("SPEC-AS-WRITTEN (exact parts, step6=L1)", "A0", dict(dv="exact", s1="exact", step6="L1", b16="off"), False),
    ("Dv~ = Dv (dumped, W_c)", "A0", dict(dv="dumped", s1="exact", step6="L1h", b16="off"), False),
    ("Dv~ = Dv0 (no W)", "A0", dict(dv="dv0", s1="exact", step6="L1h", b16="off"), False),
    ("L1~ = L1 (Schur + step 6)", "A0", dict(dv="exact", s1="schur_l1", step6="L1", b16="off"), False),
    ("S1~ = PJ + B12^T Mw^-1 B12/vs", "A0", dict(dv="exact", s1="mwinv", step6="L1h", b16="off"), False),
    ("S1~ = S1.bin (Cw)", "A0", dict(dv="exact", s1="dumped", step6="L1h", b16="off"), False),
    ("B16 off  (vs A)", "A", dict(dv="exact", s1="exact", step6="L1h", b16="off"), False),
    ("B16 rhs, step 4 as spec (vs A)", "A", dict(dv="exact", s1="exact", step6="L1h", b16="rhs"), False),
    ("PRODUCTION (Dv, S1.bin, step6=L1, B16 rhs)", "A", dict(dv="dumped", s1="dumped", step6="L1", b16="rhs"), False),
    ("PRODUCTION + step6=L1h", "A", dict(dv="dumped", s1="dumped", step6="L1h", b16="rhs"), False),
    ("PRODUCTION with Mw^-1 instead of Cw", "A", dict(dv="dumped", s1="mwinv", step6="L1", b16="rhs"), False),
    # hpatch dumps (H.bin): S1~ = PJ + Dpsi B12 H rebuilt from H.bin (n/a without H.bin)
    ("S1~ = PJ + B12 H (hpatch, H.bin)", "A0", dict(dv="exact", s1="hpatch", step6="L1h", b16="off"), False),
    ("PRODUCTION hpatch (Dv, PJ+B12 H, step6=L1)", "A", dict(dv="dumped", s1="hpatch", step6="L1", b16="rhs"), False),
    ("PRODUCTION hpatch + step6=L1h", "A", dict(dv="dumped", s1="hpatch", step6="L1h", b16="rhs"), False),
]


def ruiz(A, it=20):
    """row / column scalings with max-norm equilibration (Ruiz)"""
    dr, dc = np.ones(A.shape[0]), np.ones(A.shape[1])
    for _ in range(it):
        B = abs(dr[:, None] * A * dc[None, :])
        r, c = B.max(1), B.max(0)
        r[r == 0] = 1.0
        c[c == 0] = 1.0
        dr /= np.sqrt(r)
        dc /= np.sqrt(c)
    return dr, dc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("famdir")
    ap.add_argument("--int64", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-12, help="gate on max|P^{-1}A0 - I|")
    ap.add_argument("--ref-factor", type=float, default=10.0,
                    help="also PASS if max|P^{-1}A0 - I| <= ref_factor * max|LU(A0)^{-1}A0 - I| with the "
                         "LU of the Ruiz-equilibrated A0 (round-off of a direct solve; the Hermite mass "
                         "blocks B33/B44 have cond ~1e5)")
    ap.add_argument("--runs", default="", help="comma-separated run indices (default all)")
    ap.add_argument("--no-eig", action="store_true", help="skip eigenvalues (errors only)")
    a = ap.parse_args()
    f = Family(a.famdir, int64=True if a.int64 else None)
    N = 6 * f.n
    print("family %s  nslot %d  n %d  N=6n=%d  varsigma %g  theta_dt %g  s1 %s  bend_form %d"
          % (f.meta["family"], f.nslot, f.n, N, f.varsigma, f.theta, f.s1, f.meta["bend_form"]))
    if f.hform:
        print("%s dump: Mw / Cw based runs are n/a where the file is missing; 'S1.bin' runs use the %s S1"
              % (f.s1, f.s1))
    print("absent blocks (zero): %s;  not dumped (zero by contract): B14 B15 B32 B34 B35 B36 B41 B43 "
          "B45 B46 B53 B54 B56 B64" % (" ".join(f.absent) or "-"))
    t0 = time.time()
    A = assemble_A(f, with_b16=True).toarray()
    A0 = assemble_A(f, with_b16=False).toarray()
    blk = Blocks(f)
    print("||B16||_F/||A||_F = %.3e;  cond-ish: ||A||_max %.2e" % (
        np.linalg.norm(blk.E16) / np.linalg.norm(A), abs(A).max()))
    rng = np.random.default_rng(0)
    xr = rng.standard_normal((N, 4))
    # round-off reference: the same check for a direct dense LU solve of the
    # Ruiz-equilibrated A0 / A (plain LU of the unscaled A is far less accurate)
    ref, dcs = {}, {}
    for nm, At in (("A0", A0), ("A", A)):
        dr, dc = ruiz(At)
        Ah = dr[:, None] * At * dc[None, :]
        X = dc[:, None] * sla.lu_solve(sla.lu_factor(Ah), dr[:, None] * At)
        ref[nm] = abs(X - np.eye(N)).max()
        dcs[nm] = dc
        if nm == "A0":
            print("cond_2(A0) %.2e   cond_2(Ruiz-scaled A0) %.2e" % (np.linalg.cond(At), np.linalg.cond(Ah)))
    print("round-off reference max|LU_ruiz(A)^-1 A - I|:  A0 %.2e   A %.2e" % (ref["A0"], ref["A"]))
    print("gate (EXACT runs): max|P^-1 A - I| <= max(tol=%.0e, %g * reference)" % (a.tol, a.ref_factor))
    runs = [(nm.replace("(Cw)", "(%s)" % f.s1) if f.hform else nm, t, o, g_) for nm, t, o, g_ in RUNS]
    sel = range(len(runs)) if not a.runs else [int(i) for i in a.runs.split(",")]
    hdr = "%-44s %-3s %10s %10s %10s %10s %10s %10s %10s %9s" % (
        "run", "vs", "max|PA-I|", "|E^2|/|E|", "res(rand)", "Re min", "Re max", "max|Im|", "rho(I-PA)",
        "#D2 solve")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    nfail = 0
    for i in sel:
        name, tgt, opt, gate = runs[i]
        try:
            P = Sweep(blk, **opt)
        except ValueError as e:
            print("%-44s n/a (%s)" % (name, e))
            continue
        At = A0 if tgt == "A0" else A
        PA = P(At)
        z = P(xr)
        res = np.linalg.norm(At @ z - xr) / np.linalg.norm(xr)
        if a.no_eig:
            dev = abs(PA - np.eye(N)).max()
            st = dict(err_max=dev, re_min=np.nan, re_max=np.nan, im_max=np.nan, rho=np.nan)
        else:
            st = spectrum_stats(PA)
        E = PA - np.eye(N)
        e2 = abs(E @ E).max() / max(abs(E).max(), 1e-300)
        dc = dcs[tgt]
        esc = abs(E * dc[None, :] / dc[:, None]).max()      # max|Dc^-1 E Dc|, same spectrum
        flag = ""
        if gate:
            ok = st["err_max"] <= max(a.tol, a.ref_factor * ref[tgt])
            flag = "  PASS" if ok else "  FAIL"
            if ok and st["err_max"] > a.tol:
                flag += " (round-off: %.1fx LU ref; scaled %.1e)" % (st["err_max"] / ref[tgt], esc)
            nfail += not ok
        print("%-44s %-3s %10.2e %10.2e %10.2e %10.5f %10.5f %10.2e %10.2e %9d%s" % (
            name, tgt, st["err_max"], e2, res, st["re_min"], st["re_max"], st["im_max"], st["rho"], P.nD2, flag))
    print("\n(E = P^{-1}A - I;  |E^2|/|E| ~ 0 with |E| = O(1): E nilpotent of index 2, i.e. all eigenvalues 1 but"
          "\n FGMRES needs 2 iterations;  res(rand) = ||A z - x|| / ||x|| for z = P^{-1} x, 4 random x;"
          "\n #D2 solve = B55+B66 pairs per application)")
    print("elapsed %.1f s" % (time.time() - t0))
    print("T1 %s" % ("PASS" if nfail == 0 else "FAIL"))
    raise SystemExit(0 if nfail == 0 else 1)


if __name__ == "__main__":
    main()
