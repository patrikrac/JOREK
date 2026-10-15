#!/usr/bin/env python3
"""T2 / T3: spectrum of S1~^{-1} S1 on one family dump (spec sections 3.3, 9).

  S1 (exact)  = P_J - U1 Dsel^{-1} Lsel,  by default Dsel = Dvh = Dv0 - L2 D2^{-1} U2
                and Lsel = L1h = L1 - L2 D2^{-1} C21 (the true flux Schur complement)
  T2 : S1~ = P_J + [[ B12^T Mw^{-1} B12 / varsigma, 0 ], [0, 0]]   (commutation only)
  T3 : S1~ = P_J + [[ B12^T Cw     B12 / varsigma, 0 ], [0, 0]]   (commutation + SPAI)
  T3d: S1~ = S1.bin (as dumped; equals T3 if the dump is consistent)
  SPAI only: sigma( S1~(Cw)^{-1} S1~(Mw^{-1}) )   (isolates the SPAI error)

Dirichlet rows/cols (psi and j) are removed (they only add eigenvalues 1).
Diagnostic switches to localise a defect: --dv dv0 (no pressure channel in
the exact Schur complement), --no-kink (B21 dropped from the exact Schur
complement), --l1 plain (C21 path dropped).
Expected (spec): T2 near [0.95, 1.05] (investigate outside [0.8, 1.25]);
T3 close to the LFA bound [0.72, 1.37].
"""
import argparse
import os
import numpy as np

from petsc_bin import Family
from flux_common import Blocks


def stats(e):
    return e.real.min(), e.real.max(), np.abs(e.imag).max()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("famdir")
    ap.add_argument("--int64", action="store_true")
    ap.add_argument("--dv", choices=["exact", "dv0", "dumped"], default="exact",
                    help="velocity block inside the exact Schur complement")
    ap.add_argument("--l1", choices=["exact", "plain"], default="exact")
    ap.add_argument("--no-kink", action="store_true", help="drop B21 from the exact Schur complement")
    ap.add_argument("--plot", action="store_true", help="write a PNG to --plotdir")
    ap.add_argument("--plotdir", default="/shares/users/code/racpat/plots")
    ap.add_argument("--tag", default="", help="suffix of the plot file name")
    a = ap.parse_args()
    f = Family(a.famdir, int64=True if a.int64 else None)
    b = Blocks(f)
    n = f.n
    Dsel = {"exact": b.Dvh, "dv0": b.Dv0, "dumped": b.Dv}[a.dv]
    if Dsel is None:
        raise SystemExit("Dv.bin not in dump")
    Lsel = (b.L1h if a.l1 == "exact" else b.L1).copy()
    if a.no_kink:
        Lsel[:n, :n] -= b.L1[:n, :n]
    S1 = b.PJ - b.U1 @ np.linalg.solve(Dsel, Lsel)
    fr = f.free()
    idx = np.concatenate([fr, n + fr])                     # free psi and j
    R = lambda X: X[np.ix_(idx, idx)]
    print("family %s  nslot %d  n %d  2n_free %d  varsigma %g  theta_dt %g   exact S1: dv=%s l1=%s kink=%s"
          % (f.meta["family"], f.nslot, n, idx.size, f.varsigma, f.theta, a.dv, a.l1, not a.no_kink))
    cases = []
    if b.Mw is not None:
        cases.append(("T2  S1~(Mw^-1)^-1 S1", b.S1_bend(b.Mw_inv()), S1))
    if b.Cw is not None:
        cases.append(("T3  S1~(Cw)^-1 S1", b.S1_bend(b.Cw), S1))
    if b.S1_dump is not None:
        cases.append(("T3d S1.bin^-1 S1", b.S1_dump, S1))
    if b.Mw is not None and b.Cw is not None:
        cases.append(("SPAI S1~(Cw)^-1 S1~(Mw^-1)", b.S1_bend(b.Cw), b.S1_bend(b.Mw_inv())))
    print("\n%-30s %10s %10s %10s %8s" % ("case", "Re min", "Re max", "max|Im|", "#Re<0"))
    print("-" * 72)
    eigs = {}
    for name, St, Sx in cases:
        e = np.linalg.eigvals(np.linalg.solve(R(St), R(Sx)))
        eigs[name] = e
        lo, hi, im = stats(e)
        print("%-30s %10.5f %10.5f %10.3e %8d" % (name, lo, hi, im, (e.real < 0).sum()))
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for name, e in eigs.items():
            ax[0].hist(e.real, bins=80, histtype="step", label=name)
            ax[1].plot(e.real, e.imag, ".", ms=3, label=name)
        ax[0].set_xlabel("Re lambda"); ax[0].set_ylabel("count"); ax[0].set_yscale("log"); ax[0].legend(fontsize=7)
        ax[1].set_xlabel("Re lambda"); ax[1].set_ylabel("Im lambda"); ax[1].axvline(0, color="k", lw=0.5)
        fig.suptitle("%s  family %s  theta_dt %g" % (os.path.abspath(a.famdir), f.meta["family"], f.theta), fontsize=8)
        fig.tight_layout()
        os.makedirs(a.plotdir, exist_ok=True)
        p = os.path.join(a.plotdir, "flux_t2t3_fam%s%s.png" % (f.meta["family"], a.tag))
        fig.savefig(p, dpi=130)
        print("\nplot: %s" % p)


if __name__ == "__main__":
    main()
