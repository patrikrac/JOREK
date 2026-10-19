# Flux-Schur preconditioner: offline verification (spec T0 to T3)

These are offline Python checks for the flux-Schur arm of the SF physics preconditioner
(design: `/shares/users/code/racpat/misc/flux_schur_design.md`, sections 1-4 and 9). They read
the per-family matrix dump that the Fortran code writes with `-sf_flux_dump <dir>`. They can
also read a synthetic dump in the same format.

Python: `/shares/users/code/racpat/venvs/jorek-plot/bin/python` (numpy, scipy, sympy, matplotlib).
Run every script from this directory, or put the directory on `PYTHONPATH`.

**Which dumps.** `-sf_flux_dump` needs the per-family mode split (`-sf_flux_mode_split 1`: the
`sf_*_flux_ms*` arms; the global path writes no dump). The dump depends on `-sf_flux_s1`:
- `spai`: the dump contains Cw and the symmetric Bb = (1/varsigma) B12^T Cw B12 (or the form
  selected by `-sf_flux_bend_form`). All checks below apply (arm `sf_lu_flux_ms_spai`).
- `hpatch` (default): the dump contains H.bin and Bb = Dpsi B12 H, which is nonsymmetric, and no
  Cw.bin (Mw.bin is still written). meta.txt has `s1 hpatch`, `p r g vel` and `dirichlet_psi`.
  A dump without an `s1` key is read as `spai` (old dumps).
- `splitm`: as `hpatch` (H.bin, Bb = Dpsi B12 H, no Cw.bin), meta `s1 splitm` with `q` as well.
  Every script treats it like `hpatch` (`Family.hform`); t5 rebuilds the split-M H.

What each script does per variant:

| script | `spai` | `hpatch` |
|---|---|---|
| `t0_forms.py` | Bb symmetric PSD, Bb = formula (form 0 / 1 / 2, the selected one gated), Cw SPD, SPAI quality | the Bb-PSD, Cw-formula and Cw/SPAI gates are reported `n/a (hpatch)`. Gated instead: Bb = Dpsi B12 H, S1 = PJ + Bb, Bb zero on psi Dirichlet rows/cols, H zero on u Dirichlet rows. Info: nnz per H column, Bb asymmetry, min eig of sym(Bb) and of the Jacobi-whitened psi-Schur (skipped above `--max-dense`) |
| `t1_consistency.py` | all runs; `S1.bin` = SPAI S1 | Mw-based runs as before, Cw-based ones `n/a`; extra runs `S1~ = PJ + B12 H`, `PRODUCTION hpatch`, `+ step6=L1h` (`S1.bin` equals it) |
| `t2_t3_spectrum.py` | T2 (Mw^-1), T3 (Cw), T3d, SPAI | T3 / SPAI `n/a (no Cw.bin)`; T2 (Mw^-1) still printed; new `T2h` = sigma(S~^-1 S1) with S~ = PJ + Dpsi B12 H from H.bin; T3d |
| `t5_hpatch.py` | not applicable | rebuilds H from the dumped blocks and compares with H.bin / Bb.bin (see below) |

## Files

| file | purpose |
|---|---|
| `petsc_bin.py` | PETSc binary Mat reader and writer, `Family` / `load_family(dir)`. `python petsc_bin.py <famdir>` lists a dump |
| `flux_common.py` | dense block algebra: assembling A, the exact Schur quantities, and the switchable section 4 sweep (`Sweep`) |
| `t0_forms.py` | T0 form checks (PASS/FAIL table) |
| `t1_consistency.py` | T1 algebraic consistency and sigma(P^-1 A) per surrogate |
| `t2_t3_spectrum.py` | T2 / T3: sigma(S1~^-1 S1) with Mw^-1 and with Cw, optional plot |
| `t5_hpatch.py` | T5: independent rebuild of H from an hpatch dump, compared with H.bin / Bb.bin |
| `make_synthetic.py` | writes a synthetic family dump (bicubic Hermite slab annulus) to validate the machinery; `--bend-form 2`, `--hpatch banded` / `--hpatch patch` write the new variants |

## Dump contract (what the reader expects)

- One directory per family, `<dir>/fam<f>/`. f = 1 is n = 0 with 1 slot; f >= 2 has 2 slots
  (cos, sin). Each family is written by one rank, so every matrix is a full sequential matrix.
- PETSc native binary from MatView: big-endian, classid 1211216, then M, N, nnz, the row
  lengths, the column indices and float64 values.
  - The layout (classid width and PetscInt width) is auto-detected from the file size, so 64-bit
    PETSc builds are read without a flag. `--int64` forces 64-bit indices.
  - Complex PETSc builds are not supported.
- Scalar-field blocks: row = idx*nslot + m. The files are `B12 B16 B21 B23 B25 B26 B51 B52 B61
  B62 B63 B65 B55 B66 Mw Cw H Bb` (Cw only with `spai`, H only with `hpatch`).
  - An absent file counts as a zero block.
  - Mw and Cw may also be given at size nindex; they are then expanded with kron(., I_nslot).
- Two-field packed blocks (field-major): `PJ = [[B11,B13],[B31,B33]]`, `S1 = PJ + [[Bb,0],[0,0]]`,
  `Dv0 = [[B22,B24],[B42,B44]]`, `Dv = Dv0 + [[W_c,0],[0,0]]`.
- `meta.txt` holds `key value` lines: `family`, `nslot`, `nindex`, `varsigma`, `theta_dt`, and
  `dirichlet` followed by 0-based scalar row indices (row = idx*nslot + m).
  - The code also writes `bend_sign`, `bend_form`, and `s1 hpatch` (then `p`, `r`, `g`, `vel`)
    or `s1 spai`. `dirichlet_psi` (psi's Dirichlet rows) is written for `hpatch` and for
    `bend_form 2`.
  - The optional `dirichlet_index` key gives index-level entries, which are expanded over the slots.
  - `bend_sign <real>` (default 1) and `bend_form 0|1|2` (default 0) are read. `s1` defaults to
    `spai`; `p r g vel` are ints; `dirichlet_psi` defaults to `dirichlet`. Any other unknown
    key is kept and ignored.
  - Lines starting with `#` are ignored.
- All blocks, Mw included, carry JOREK's FFT slot normalisation: n_plane on the n = 0 slot and
  n_plane/2 on the cos/sin slots. Every check is invariant to this factor.
- `-sf_flux_s1 spai`: Bb = (bend_sign/varsigma) D B12^T (D Cw D) B12 D, with D = diag(free mask).
  With `-sf_flux_bend_form 1` it is B12 (D Cw D) B12^T instead. With `-sf_flux_bend_form 2` it is
  (bend_sign/varsigma) Dpsi B12 (Du Cw Du) B12^T Dpsi, Dpsi from `dirichlet_psi`, Du from the
  `dirichlet` (u) mask.
- `-sf_flux_s1 hpatch`: Bb = Dpsi B12 H (nonsymmetric, zero on psi's Dirichlet rows); H.bin
  (u rows x psi columns, scalar layout) is written as well, and Cw is not. S1 = PJ + [[Bb,0],[0,0]].
- Full family operator, unknown order (psi, u, j, omega, rho, T): A is assembled from these blocks
  only, and every other block is zero (B14, B32, B56, ...). **T1 checks consistency against this
  block-assembled A, not against the full JOREK matrix.**
  - D2 = [[B55, 0], [B65, B66]] is block lower triangular when B65 is present.

## Running

```
P=/shares/users/code/racpat/venvs/jorek-plot/bin/python
$P make_synthetic.py /path/syn                 # fam1 (n=0), fam2 (n=1); 6 rings x 12 nodes
$P t0_forms.py        /path/syn/fam2           # exit 0 = PASS
$P t1_consistency.py  /path/syn/fam2           # ~3 min for n=1 at 6x12 (N = 3456, dense eig)
$P t2_t3_spectrum.py  /path/syn/fam2 --plot    # PNG into /shares/users/code/racpat/plots/
$P make_synthetic.py /path/synh --hpatch patch # hpatch-style dump (H from the t5 construction, g = 1)
$P t5_hpatch.py      /path/synh/fam2           # exit 0 = PASS
```

**T5 (`t5_hpatch.py`, hpatch dumps).** It rebuilds H with dense patch solves from the dumped
PJ, Dv, B21, B23 and B12 only, following the definition in `mod_petsc_pc_sf_hpatch.f90`: index
graph from B33's structure, free rows = rows with an off-diagonal entry (psi, j from PJ; u, omega
from Dv), patch radius p, band radius r, X = B33^-1 B31, R = B23 X - B21, Y = Dv11^-1 R (`vel 0`)
or the u part of Dv^-1 [R; 0] (`vel 1`), H cut to node distance <= r, Bb = Dpsi B12 H.
- For the exact gate dump with `-sf_flux_hp_group 1` (mode split arm, one rank per family). Then
  ||H_py - H.bin||/||H.bin|| and ||Bb_py - Bb.bin||/||Bb.bin|| must be < `--tol` (1e-10).
- For g > 1 the greedy grouping here may break ties differently from the Fortran one, so the
  differences are only reported (info, exit 0).
- Both variants also get sigma(S~^-1 S1) (S1 the exact flux Schur complement, free psi/j; skipped
  above `--max-dense` rows or with `--no-sigma`).
- Split-M dumps (`s1 splitm`): additionally Q = nodes within q of the group, H_T = -(1/varsigma)
  Mw[Qu,Qu]^-1 B23[Qu,cols] cut to q, the velocity rhs R - Dv[Pu,Qu] H_T (omega rows -Dv[Pw,Qu] H_T
  with `vel 1`), H = H_T + (Y within r). The Fortran solves M_w and B33 per toroidal slot when
  slot-diagonal (off-slot entries below 1e-12 of the row's largest), which is the same to round-off.
  Cross-check of `rebuild_H(..., q=4)` on the 2026-10-16 11x16 dump (an `s1 spai` dump, so not via
  the t5 command line; g = 1 / 4): sigma max|lam - 1| = 0.053 / 0.054 (fam1), 0.251 / 0.248 (fam2),
  against 0.065 / 0.250 for hpatch.

## What PASS means

**T0 (spec T0, gate).** All checks are made on the unconstrained rows/cols.
- G = B12/theta_dt is skew: ||G+G^T||_F/||G||_F < `--tol-skew`. The same check is made for B23.
- B12 = B23. ||B12+B23|| is printed as well, so a sign flip is visible.
- B31 is symmetric and B33 is SPD.
- sign(bend_sign)·Bb is symmetric PSD, and Bb is zero on the Dirichlet rows and columns.
  Otherwise S1~ loses its identity rows.
- Bb is compared against both bend forms. Only the form named by `bend_form` must match. The
  form difference ||B12^T C B12 - B12 C B12^T|| / ||B12^T C B12|| is printed as info; it is zero
  exactly when B12 is skew on the free space.
- S1 = PJ + Bb.
- Mw is SPD and Cw is symmetric PD.
- SPAI quality: sigma(Cw Mw) > 0 and ||I - Cw Mw||_Mw = max|1 - sigma(Cw Mw)| < `--tol-spai`.
  - The Euclidean ||I - Cw Mw||_2 is only printed as info. Cw Mw is non-normal, so that norm is
    large whenever Mw is badly conditioned.
- W_c is reported for information only. With B22 = -varsigma K < 0, it should be negative semidefinite.

**T1 (spec T1, gate).** `EXACT` is the elimination of spec section 2 with exact components. It
must reproduce A0^-1, where A0 is A with B16 = 0. `EXACT+E16` is the E16-consistent version, and
it must reproduce A^-1.
- Gate: max|P^-1 A - I| <= max(`--tol`=1e-12, `--ref-factor` x the round-off of a dense LU of
  the Ruiz-equilibrated A).
- The absolute 1e-12 is not reachable on Hermite discretizations, because B33 = M_{1/R} and
  B44 = M_R have cond ~1e5. The error sits in the j and omega rows. The scaled error
  max|Dc^-1 E Dc| is printed next to the result and is ~1e-11.
- The other rows are reported, not gated:
  - Re range and max|Im| of sigma(P^-1 A), and rho(I - P^-1 A);
  - |E^2|/|E| with E = P^-1 A - I. A value ~0 means E is nilpotent: every eigenvalue is 1, but
    FGMRES needs 2 iterations.

T1 run list:

| run | what it shows |
|---|---|
| `SPEC-AS-WRITTEN` | section 4 as written, with exact components. Not A0^-1 (see below) |
| `Dv~ = Dv` / `Dv0` | the W_c surrogate / no pressure channel |
| `L1~ = L1` | the C21 path dropped consistently, in both S1 and step 6 (spec section 3.2) |
| `S1~ = Mw^-1` / `S1.bin` | the commutation alone (T2) and with the SPAI (T3), inside the full sweep |
| `B16 off` / `B16 rhs` | the E16 treatments, against A |
| `PRODUCTION` | Dv, S1.bin, step 6 as spec, B16 rhs. `+ step6=L1h` adds the restored term |

**T2 / T3 (spec T2, T3).** These scripts report the Re range and max|Im| of sigma(S1~^-1 S1).
- S1 is the true flux Schur complement P_J - U1 Dvh^-1 L1h.
- S1~ is built with Mw^-1 (T2) or with Cw (T3).
- `SPAI` compares S1~(Cw) against S1~(Mw^-1) and isolates the SPAI error.
- Spec expectations: T2 near [0.95, 1.05] (investigate outside [0.8, 1.25]); T3 close to
  [0.72, 1.37].
- `--dv dv0`, `--no-kink` and `--l1 plain` remove the pressure channel, B21 or the C21 path
  from the true S1, which helps to locate a defect.

## Finding: the section 4 sweep is not A^-1 even with exact components

Step 6 re-solves the velocity with `r_u - B21 psi - B23 j`, i.e. with L1. Exact block elimination
(section 2) needs L1h = L1 - L2 D2^-1 C21 there:

    v = Dvh^-1 ( r_v - L1 y1 + L2 D2^-1 C21 y1 )
      = Dvh^-1 ( x_v - L1 y1 - L2 D2^-1 (x2 - C21 y1) )

Section 3.2 says the omitted path is "restored in the final re-solve of y2 (step 7)". This is not
the case:
- Step 7 puts C21 y1 into y2.
- The force L2 D2^-1 C21 y1 that the flux-driven rho/T change exerts on the velocity never enters v.
- Steps 1-5 still give the exact y1, so E = P^-1 A0 - I maps only y1 into (v, y2) and E^2 = 0.
  All eigenvalues are 1 (sigma cannot show the defect), max|E| = O(10^2), and FGMRES takes 2
  iterations instead of 1.

The consistent version (`step6=L1h` in `Sweep`) needs a third B55 + B66 solve pair:

    d2 = D2^-1 (C21 y1);  v = Dv~^-1 (r_v - L1 y1 + L2 d2);  y2 = D2^-1 (x2 - C21 y1 - U2 v)

With B16 the exact version also changes step 4 and S1. The rhs is r1 = x1 - B12 u* - B16 T**,
where T** is the T-component of D2^-1 (x2 - U2 v*), not B66^-1 x_T. S1 gets the extra terms
-E16 D2^-1 C21 and +E16 D2^-1 U2 Dvh^-1 L1h. If B65 is non-zero, D2 is lower triangular, and the
T solves need -B65 rho.

## Synthetic dump (make_synthetic.py)

- **Geometry.** A slab annulus x in [0.2, 0.7], periodic in y, with genuine bicubic Hermite
  element matrices (exact Gauss 6x6).
  - DOFs are h-scaled (f, h f_x, h f_y, h^2 f_xy), as in the LFA.
  - Dirichlet (f, f_y) on the outer ring; the inner ring is natural.
- **Coefficients.** psi0 has shear and a misalignment `--eps`; j0 = R div(R^-1 grad psi0); R
  varies poloidally, rho0 and T0 radially, plus a small flow.
- **Blocks.** They have the zero structure of the spec. W_c is the assembled compression
  stiffness.
  - Cw is the l2 SPAI (r = 2) of the Jacobi-scaled Mw, mapped back, as in the Fortran.
    `--spai-plain` gives the unscaled SPAI.
  - All blocks carry the slot normalisation (`--n-plane 8`), and meta.txt has `bend_sign 1`.
- **New variants.** `--bend-form 2` writes the form-2 Bb with `dirichlet_psi`. `--hpatch banded`
  writes an hpatch-style dump (meta `s1 hpatch`, `--hp-p/-r/-g/-vel`, no Cw.bin) with a banded
  nonsymmetric H on the SPAI pattern (T5 must then FAIL). `--hpatch patch` uses the T5
  construction itself (a plumbing test of the new code paths, not an independent check of the Fortran).
- **Not a force-balanced equilibrium.** The curvature operator is fake. Use it to validate the
  machinery, not the physics.
- **Calibration against the LFA.** With `--beta 0 --j0 zero --u0 0 --rcos 0 --rho-var 0 --eps 0
  --shear 0 --nu 0 --eta 0` (constant coefficients, aligned), T2 = [0.988, 1.012] and
  T3 = [0.80, 1.39], against the LFA values [0.988, 1.01] and [0.72, 1.37].
- **SPAI scaling.** With `--physical-dofs --spai-plain` (unscaled derivative DOFs, plain SPAI),
  Cw is indefinite. The Jacobi-scaled SPAI fixes this and also improves the h-scaled case:
  sigma(Cw Mw) goes from [0.67, 1.25] to [0.81, 1.15].
