# Peeled Schur preconditioner for pair_w: review and prototype

The arm now defaults to v2 (`pc_design_fix/peeled_schur_preconditioner_v2.md`,
section "v2" below). v1 stays selectable with `-sf_peel_v 1`; its review follows the
v2 section.

## v2 (2026-10-22)

**Verdict.** Two corrections to the v2 design were needed, both now in the arm:
the hyper-resistive term in R_hat, and the resistive factor on the viscous term.
With them, gates M1, M2a and M2b pass. **M3 fails at 41x32:** at tstep 10, v2
takes 2.3x production's pair_w iterations and 2.3-3.7x its wall time. The cost
is N's stencil and setup, not X_AS. Production remains not endorsed; v2 remains
the prototype.

### Corrections to the v2 design
1. **Hyper-resistivity belongs in R_hat.** v2 puts only eta in R_hat. With
   eta_num, S's psi response is (opz + theta dt eta k^2 + theta dt eta_num R k^4) /
   R. In LFA (`lfa_v2_hyper.py`), leaving the k^4 term out gives kappa 70 at
   E_4 = theta dt eta_num / (opz h^4) = 0.01 and 673 at 0.1. The arm adds it as
   theta dt K_w X K_w with w^2 = eta_num rho0 R^3, which restores kappa ~ 8.8
   with X_AS.
   - On the case (eta_num 3e-10, tstep 10), exact X, viscosity off: with the
     sandwich S P^-1 is in [0.57, 1.97]; without it the ratio is 37.8
     (lambda_min 0.052).
2. **The "O(theta^2 eta nu k^6)" cross term is not small.**
   - In P = Q (opz R_hat)^-1 N, the design's opz theta dt K_nu is divided by
     R_hat's resistive factor (opz + theta dt eta k^2 + ...) / opz.
   - Here that factor reaches thousands at the edge: eta_T ~ T^-1.5, and
     diag(theta dt K_eta) / diag(opz M_rho) runs up to 2666.
   - So on viscosity-dominated modes (k_par ~ 0) P fell short of S by up to
     ~600x. With exact X and exact inner solves, lambda_max was 589 at tstep 10
     and 73 at tstep 1. It disappears with visco = 0.
   - The arm uses theta dt sym(R_hat X_AS(M_rho) K_nu) instead (`visc 2`).
     X_AS(M_rho) is rebuilt per PC rebuild, because rho0 evolves.
   - The hyper part must be in it too: `visc 3`, which leaves it out, takes
     29-30 pair_w its.

### Gates (41x32, eta 2e-5 with eta_T(T), eta_num 3e-10, exact inner solves)

| gate | result | criterion | |
|---|---|---|---|
| M1 | cond(X_AS B_33) = 8.96, [0.79, 7.1] (LFA 8.38; diag 474); N_c = 13 >= 7.1. Extreme modes: Hermite cross-derivative DOFs, top in the plasma (R 3.54, Z 0.46), bottom on the boundary | go <= 15 | pass |
| M2a, exact X | tstep 1: [1.00, 1.31], 10 its to 1e-6; tstep 10: [0.60, 2.00], 16 its | match v1 exact ([1.03, 1.21], [0.54, 1.82]) | pass |
| M2b, X_AS | tstep 1: ratio 5.45 (23 its); tstep 10: [0.14, 1.27], ratio 8.9 (32 its) | spread <= 10 at tstep 10 | pass, barely |
| M2c, tstep 10, exact X | no curvature or kink: [0.76, 2.00]; no W at all: [0.90, 2.02]; also visco = 0: [0.90, 1.99] | attribution | low end = curvature / kink / compression (physics); top ~ 2 = the commutation itself |
| M3 | see below | cheaper than production at tstep >= 1 | **fail** |

The colouring bound holds but is loose (13 vs 7.1): min(N_c, 1.25 lambda_Lanczos)
picks the Lanczos estimate, which is the heuristic again.

### M3: 41x32, 3 steps, np 1 x 4 threads, GMG inner solves

| tstep | production: outer / pair_w its / wall | v2: outer / pair_w its / wall | v1: pair_w its / wall |
|---|---|---|---|
| 0.1 | 4 3 3 / 1.3 / 9.6 s | 5 5 5 / 1.4 / 29.5 s | 1.0 / 9.4 s |
| 1 | 4 3 4 / 1.25-2 / 9.6 s | 7 7 7 / 2.9 / 34 s | 3.1 / 21 s |
| 10 | 6 6 6 / 2.0 / 10.9 s | 9 8 8 / 4.6 / 40 s | 14.6 / 66 s |

`-log_view` at tstep 10 (3 steps):
- **pair_w setup:** 19.7 s vs 1.5 s.
  - R_hat and N products: 13.1 s.
  - N's Galerkin hierarchy: 6.4 s.
  - X_AS for B_33: 0.09 s.
- **pair_w solves:** 12.6 s vs 1.6 s.
- **Per V-cycle:** N 46 ms, (psi, j) V(0,10) 21 ms (two per iteration), Q 9 ms,
  against production's whole pair_w V-cycle at 43 ms.
- **N's stencil:** 1470 nonzeros per row (radius 5: the viscous sandwich over
  R_hat's hyper part), production pair_w about 430.

Levers tried at tstep 10:

| lever | pair_w its | wall |
|---|---|---|
| lower-triangular variant | 4.7 | 37.7 s |
| hyper-resistivity left out of R_hat and K_nu (inconsistent with S, but cheaper) | 5.6 | 25.6 s (N 619 nonzeros per row) |

Not done (the design gates step 3 behind M3): Chebyshev/X_AS for A_aa in place
of the two (psi, j) V-cycles, which would save at most ~40% of the solve time.
Even then the iteration count stays 2.3x production's.

### What would change it
- **A cheaper viscous term.** One that keeps the resistive factor without
  widening N, e.g. lumped weights where the resistive factor is large.
  Viscosity is a small term, but it is the one that dominates on k_par ~ 0 modes.
- **121x48 and production-sized cases.** These are needed only if production's
  pair_w counts grow with the mesh while v2's do not; at 41x32 both are flat in
  tstep 0.1-10.


## v1 review (first round)

Review of `pc_design_22_10_26/peeled_schur_preconditioner.md` (2026-10-22), on
branch `feature/sf-peeled-schur` (off `numerics_develop_pbpc` 322d65c81).

**Verdict.**
- (a) Prototype: implemented here as a separate pair_w arm
  (`physics_pc_sf_pair_w = "peel"`, model199, global path). It is endorsed as an
  evaluation vehicle, not as a pair_w replacement candidate.
- (b) Production: **not endorsed.** The design's sparse X fails on JOREK meshes
  at tstep >= 1-10. Even with the exact X, pair_w needs more iterations than
  production, and each iteration costs more.

### What is in the branch

| file | change |
|---|---|
| `solvers/mod_petsc_pc_sf_peel.f90` | the arm: FGMRES on pair_w, block-LDU preconditioner, P = Q M_rho^-1 T, Stage-0/1 diagnostics |
| `models/model199/mod_pc_elt_matrix_force_fft.f90` | `pc_elt_matrix_peel_fft`: opz K_rho, W_comp, M_rho, K_nu (stubs in 180/183/600) |
| `matrix/construct_pc_matrix_mod.f90` | `construct_peel_matrices` (`pack_rows` moved to module level) |
| `solvers/mod_petsc_pc_physics_element.f90`, `_ctx.f90` | the four operators assembled with W when pair_w = peel |
| `solvers/mod_petsc_pc_gmg.f90` | 8 hierarchy instances (6 Q, 7 T, 8 the arm's (psi, j)) |
| `solvers/mod_petsc_pc_sf_solver.f90` | `SF_PEEL`; `ncyc`: a fixed number of V-cycles (Richardson, a linear operator) |
| `solvers/mod_petsc_pc_sf.f90` | backend "peel" for pair_w only: no block scaling, peel_build / peel_solve / peel_report |
| `util/peeled_schur/lfa_review.py` | the LFA the design's scripts leave out (with copies of both design scripts) |
| `util/peeled_schur/lfa_peeling_v2.py`, `lfa_v2_hyper.py` | the v2 design's LFA; v2 with hyper-resistivity |

The arm reuses the pair_w/pair_psi GMG (rings + radial semi-coarsening for T,
zebra rings for Q and (psi, j)), `pack_pair_aij` / `make_field_block_scale`, and
the Chebyshev mass inverse for B_44. Every other SF path is unchanged (the
production pair_w gives identical counts).

Options (`PETSC_OPTIONS`, all `-sf_peel_*`):
- `x 0|1`, `xres 0|1`: the choice of X.
- `wcomp w`, `visc 0|1`: the W_comp and K_nu terms.
- `q|t|aa lu|gmg`, `qcyc|tcyc|pjcyc n`: the inner solvers and their V-cycle counts.
- `tri 1`: lower block-triangular variant.
- `check 1`: Stage 0.
- `spec n`, `spec_every 1`, `spec_drop bits`: the Stage-1 spectrum and term attribution.
- `texact 1`: T with the exact X.

## Findings

Case: model199 shaped pcbench, 41x32, n_tor 3 (n = 0, 6), eta 2e-5,
eta_num 3e-10, restart from the suite equilibrium, np 1 x 4 threads.

### Correct in the design
- The block elimination, A_aa = diag(B_44, [[opz B_33, B_13], [B_31, B_33]]),
  and the sign of S are right. In model199, B_12 and B_23 are the same form
  (-theta dt [psi0, .] + theta dt F0/R d_phi), so B_23 = -B_12^T in the interior.
  Measured: |B_23 + B_12^T| / |B_23| = 1.5e-6 (eliminated Dirichlet rows).
- The LFA numbers reproduce exactly: kappa 1.025 with exact masses; 48.7, 12.4
  and 6.2 for X_0, X_1 and degree 2; FoV [0.53, 1.58] at a phase shift of pi/2.
- The commutation step holds on this case, density pedestal included. With the
  exact X and exact inner solves, S P^-1 is in [1.03, 1.21] at tstep 1 and
  [0.54, 1.82] at tstep 10 (11 / 19 GMRES its to 1e-6).

### Wrong or missing

1. **X is the crux, and the design's X does not hold on JOREK meshes.** The
   kappa bound uses the uniform-grid spectrum of D^-1 B_33, [0.0625, 3.06],
   cond 49. On the 41x32 mesh, Lanczos gives [0.0084, 3.99], **cond 474**.
   Gershgorin's top is 13.5. "kappa <= cond(D^-1 B_33)^2" bounds only the T
   pencil T_X vs T_exact, not the non-normal product P^-1 S.
2. **The resistive response is not optional.** "Exact" S_B assumes eta = 0.
   With B_13 = -theta dt (eta M_{1/R} + eta_num K), the psi row's response is
   (opz M + theta dt eta K + ...)^-1. The design's X ~ B_33^-1 overweights
   bending once theta dt eta k_h^2 / opz ~ 0.01-0.1. In LFA, X_0 goes from
   kappa 48 to 64 (e = 1) and to 610 (e = 10).
3. **Cross terms of the peeled product.** Q M^-1 T contains W_comp M^-1 (bending
   + viscosity), which S does not have. LFA: with W_comp in Q, lambda_min =
   1/(1 + Pc), Pc = W_comp / (opz K_rho). With W_comp left out of P instead,
   lambda_max = 1 + Pc on the k_par ~ 0 modes. Neither gives "a few outliers".
   Measured diag ratio: mean 0.07 at tstep 10, 0.65 at tstep 30 (max 2.4). It
   grows as tstep^2 and with beta, so at production tstep Pc >> 1. The LFA
   scripts carry no W_comp at all.
4. **Ordering of the commuted factors.** K_rho M_rho^-1 G^T vs
   G^T M_{1/R}^-1 K_{1/R} differ at first order by grad ln(rho0 R^4) / k: a
   density-pedestal term the design's Stage-1 list omits. It was not dominant
   on this case (the exact-X spectrum above).
5. **X_1 can be indefinite.** It needs the true top b of D^-1 B_33. A Lanczos b
   that is 20% low makes p(lambda) < 0 on the top of the spectrum, so T is
   indefinite. Use a safe upper bound; Gershgorin is safe but 3x too wide here.
6. **The omitted indefinite terms set the large-tstep limit.** At tstep 30, the
   exact X still leaves eigenvalues -0.034 and 0.049. Dropping W and the kink
   term from S gives [0.82, 2.9], so this is the design's own instability
   limit (theta dt gamma ~ opz). Production pair_w also needs 16 its there.
7. **Boundary and axis.** Eliminated Dirichlet rows and fix_axis_nodes' penalty
   rows are decoupled in pair_w (192 u rows here). The arm solves them from
   pair_w's diagonal and makes Q, M_rho and T the identity there. X is zero
   where B_33's diagonal vanishes (zero-size axis DOFs). Untested: open field
   lines (X-point / SOL, where [psi0, .] is not skew because psi0 is not
   constant on the wall), sheath BCs, free boundary.
8. **Cost statement.** Per outer iteration the arm needs:
   - two (psi, j) V(0,10) cycles (A_aa twice);
   - two Chebyshev mass solves;
   - one Q and one T V-cycle;
   - three more Galerkin hierarchies, with T at stencil radius 2.

   Per rebuild it needs four element-assembled operators and the B_12^T X B_12
   triple product. "Only scalar multigrids, 8 or fewer unknowns per node"
   leaves out the (psi, j) pair.
9. **The validation plan cannot separate the error sources as written.**
   - Stage 1 does not fix the inner solves or X. Stage 2 looks only at outer
     counts and wall time.
   - Stage 0.3 (a periodic constant-coefficient test) has no JOREK grid.
   - The arm adds a ladder: exact inner solves (`q/t/aa lu`), then the exact X
     (`texact`), then terms removed from S (`spec_drop`), plus the Stage-0
     weights (Pc, the resistive change of X, cond(D^-1 B_33)).
   - Multigrid vs exact inner solves gave the same spectrum (ratio 7.3 vs 7.8
     at tstep 1). On this case the inner solvers are not the problem.

### Measurements

S P^-1 with exact Q, T, A_aa (`spec 100`), ratio = max|lambda| / min|lambda|:

| tstep | X_0 (design) | X_1 (design) | X_0 + resistive (arm default) | exact X | exact X, no W / kink |
|---|---|---|---|---|---|
| 0.1 | | | 1.49 (7 its) | | |
| 1 | 93 (72 its) | 62 (58) | 7.8 (26) | 1.18 (11) | |
| 10 | | 1093 (kappa 474 given) | 124 (64) | 3.3 (19) | 2.2 (18) |
| 30 | | | 1464, one lambda < 0 | 93, lambda = -0.034 | 3.6 (22) |

At tstep 10, removing W or the kink term from S leaves max|lambda| ~ 40-44:
the outliers come from X, not from the omitted terms.

Time steps, 3 steps each (outer its per step / mean pair_w its / wall):

| tstep | production pair_w (GMG) | peel (arm default, GMG inner) | peel, exact X (tstep 1 / 10 only, MUMPS) |
|---|---|---|---|
| 0.1 | 4 3 3 / 1.3 / 8.1 s | 5 4 4 / 1.0 / 9.4 s | |
| 1 | 4 3 4 / 1.25-2 / 8.6 s | 7 7 7 / 3.1 / 21 s | 5 / 1.8 (1 step, 25 s) |
| 10 | 6 6 6 / 2.0 / 10.3 s | 9 9 9 / 14.6 / 66 s | 8 / 3.0 (1 step, 80 s) |
| 30 | 24 19 23 / 16 / 89 s | 28 27 29 / 30 (cap) / 345 s | |

Wall times are indicative only: another session's runs shared the machine.
np 2 reproduces np 1 exactly (7 7 7, pair_w 3.14).

## What would change the verdict

1. **A sparse X with the exact X's spectrum on real meshes,** resistive response
   included. Target: `texact`'s 1.2 / 3.3 at tstep 1 / 10. This is open:
   - node-block inverses equal the diagonal on uniform grids;
   - Chebyshev on the resistive operator widens T's stencil to radius 4 and
     still needs the real-mesh spectral bounds.
2. **Even then, pair_w would need fewer iterations than production.** The
   exact-X arm took 3.0 vs 2.0 at tstep 10, so per-iteration cost would have to
   drop about 2x below the pair_w V(0,6) cycle. The two (psi, j) V(0,10) cycles
   alone make that unlikely.
3. **Before any production work:** X-point / open-field-line boundaries,
   model600 (different viscosity, v_par), the mode split, and Pc >> 1 cases
   (Q's anisotropic dZ^2 term under the zebra smoother).

## Reproduce

```
# build: util/config.sh model=199 n_tor=3 n_period=6 n_plane=4
PETSC_OPTIONS="-sf_peel_check 1 -sf_peel_spec 100 -sf_peel_q lu -sf_peel_t lu -sf_peel_aa lu [-sf_peel_texact 1]" \
  mpirun -n 1 ./jorek_model199 < in      # in: physics_pc_sf_pair_w = "peel", tstep_n = 1.d1
python lfa_review.py                     # (1) compression, (2) resistive X, (3) X_1 positivity
```

## Follow-up 2026-10-23 to 2026-10-26: v3, commutation, the W surrogate

**v3** (`-sf_peel_v 3`, design `numerics/v3`, LFA `lfa_peeling_v3.py`): cheaper N
(nnz 619 vs 1470, setup 2.0 vs 6.4 s) but the same verdict as v2. At tstep 10,
41x32 it takes 9 8 8 outer and 5.4-5.9 pair_w its, against production's 6 6 6
and 2.0 (`results/seq.txt`, `v3*.txt`, `b1.txt`, `b2.txt`, `sweep_cyc.txt`).

**Distributive smoother** (pair_w GMG; two-grid model in `model/`): rejected.
- The commutation itself is exact in the model (spectrum [0.99, 1.01]).
- The local Hermite mass inverses are not accurate enough: nodal-block Jacobi
  gives eig(X M) in [0.06, 3.1], and both it and element additive Schwarz
  diverge.
- The only variant that converges eliminates omega/j with a near-exact X. Even
  that is only mixed against the collective smoother.

**Commutation as an outer preconditioner** (`model/t2.py`, `outer.py`;
`-sf_peel_qv 1`):
- In the model it is robust only when it commutes through the unweighted
  vorticity Laplacian, z_u = B_44 B_42^-1 r. Under a 10:1 density pedestal,
  the rho-weighted version gives |E| 100-2000, against 0.5-4.6 unweighted.
- In JOREK, even the exact limit gives ratio 2.80 (v3 3.05) and 4 GMRES its
  (`results/qv.txt`). Not competitive.

**pair_w is not what limits the outer count; the W surrogate is.**
- Production outer its at tstep 10 stay 6 6 6 with pair_w at rtol 1e-1 to 1e-6.
- Option `-sf_w_scale s` (`mod_petsc_pc_sf_mixed`) weights W in pair_w's u block.
- Env `JOREK_W_RHO` / `JOREK_W_T` (model199) weight W's density and temperature
  halves.
- Low beta (this case; `results/wdef.txt`, `ablate.txt`, `wsign.txt`,
  `wsplit.txt`):

  | tstep | W 1 (production) | W 0 | W 0.5 | W 2 | W -1 |
  |---|---|---|---|---|---|
  | 1 | 3 3 4 | 3 3 4 | 3 3 4 | 3 3 4 | |
  | 10 | 6 6 6 (KSPSolve 2.84 s) | 5 5 5 (2.33 s) | 6 5 5 | 8 8 8 | 7 7 6 |
  | 30 | 21 21 25 (51.8 s) | 7 7 8 (3.4 s) | 10 9 11 | 24 22 25 | 64 52 60 |

  Both halves of W hurt at tstep 30: rho only gives 11 10 12, T only 11 10 11.
  Neither W setting converges at tstep 100.

- High beta (tok, m600_balloon t045, 65x96 n_tor 9, PC rebuilt every step;
  mean outer its for W 1 / 0.5 / 0):

  | snapshot | tstep 1 | tstep 3 | tstep 5 |
  |---|---|---|---|
  | s1000 | 4 / 4 / 4 | 6.4 / 6.0 / 6.7 | 9.1 / 8.2 / 9.4 |
  | s1410 (nonlinear) | 4 / 4 / 4 | 8.7 / 8.3 / 8.6 | 17.0 / 14.8 / 15.9 |

  With the PC built once and reused, W 0 matches W 1, including the stale-PC
  climb at s1410 (14 -> 138 its at tstep 5).

**Conclusion.** Keep W. At low beta it carries little physics, and its error,
~(theta dt)^2 because it is the continuum form of the pressure Schur term
without the discrete projections, dominates. At high beta the physics it
carries matters about as much as its error. W 0.5 never did worse than W 1,
but it is only a tuning knob. The next step would be explicit pressure rows in
pair_w, the discrete counterpart of the explicit psi rows.

Tools: `model/` (two-grid model: `tg.py` smoothers, `rho.py` dense rho(S^k C),
`t2.py` P_T2, `outer.py`, `run.py`); `harness/` (41x32 local runs: `run.sh`
<dir> <np> key=val, `one.sh`, `wsplit.sh`, `sweep_cyc.sh`, `base_stripped.in`;
tok `*.slurm`); `results/`.
