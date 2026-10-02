# Physics-PC MPI scaling study

These scripts run strong- and weak-scaling series of the SFM2 physics
preconditioner on a SLURM cluster, and compare it against JOREK's default PETSc
preconditioner. The test case is the committed benchmark
`namelist/model199/intear_island_demo`: a 2/1 tearing mode with a tstep ramp of
0.1, 1 and 10, three steps each.

| file | role |
|---|---|
| `job_study.slurm` | generic jobscript template: copy it to `job_study.local.slurm` (git-ignored) and add the machine's partition, account and modules there |
| `pc_study.sh` | runs a strong, weak, nonlinear or probe series one case after another inside one allocation |
| `pc_case.sh` | runs one case: writes the namelist, launches, profiles, records `case.meta` |
| `mknml.py` | namelist = base + arm flags + mesh + ramp (+ `key=value` overrides) |
| `collect.py` | turns a study directory into `results.tsv`, and each case's log into `steps.tsv` |
| `plot_motivation.py` | the motivation figures from a strong, a weak and a nonlinear study (see below) |

## Build requirements

- `jorek_model199` built with `n_tor = 3` in `models/mod_settings.f90`. These
  settings are compile-time; the namelist cannot change them.
- `n_period` decides which harmonics exist: 6 gives n = 0, 6 and 1 gives
  n = 0, 1.
  - All the local reference numbers, and the whole serial study, used the
    committed `n_period = 6`.
  - The 2/1 tearing physics of `intear_island_demo` needs `n_period = 1`.
  - For the solver study either works, since the operator structure is the
    same, but the toroidal couplings differ, so never mix the two in one
    comparison.
  - `collect.py` reads both values from each log into the `n_tor` and
    `n_period` columns.
- A PETSc build with MUMPS (`USE_PETSC`). Use the same compiler and MPI stack
  at run time as at build time.
- Python 3 (standard library only).

## Quick start

```bash
cd util/physics_pc_scaling
cp job_study.slurm job_study.local.slurm       # machine specifics go here, not into git
export JOREK_BIN=/path/to/jorek/jorek_model199
PCS_DRYRUN=1 PCS_MAXNP=64 PCS_OMP=16 ./pc_study.sh strong   # list the cases
sbatch --export=ALL,MODE=strong job_study.local.slurm
sbatch --export=ALL,MODE=weak   job_study.local.slurm
```

**Hybrid MPI+OpenMP.** JOREK is always run hybrid. Every case uses `np` MPI
ranks with `PCS_OMP` OpenMP threads each; the default comes from the job's
`--cpus-per-task`. Case directories are named
`<arm>_<mesh>_np<ranks>x<threads>`, so 8×16 and 16×16 runs of the same mesh
can share one study directory. Choose `--ntasks-per-node` × `--cpus-per-task`
= the node's physical cores, for example 8 × 16 or 16 × 16.

Each series runs inside one allocation, so request the largest rank count in
the series; cases needing more ranks than the allocation has are skipped.
Finished cases are skipped on resubmission, so a timed-out job can simply be
submitted again. `results.tsv` is rewritten after every case.

## Arms (`PCS_ARMS`)

| arm | preconditioner |
|---|---|
| `sfm2_gmg` | SFM2 without refactorisations: C¹ GMG on pair_w (matrix-free fine operator with the exact mass), eta-Schur + GMG on pair_psi, GMG on ρ/T. Only the constraint masses are factored, once per run. Every hierarchy uses the stage-D13 axis treatment: rings 0–3 of every level in one axis block solved by MUMPS (`physics_pc_gmg_axis_rings = 3`), and coarse levels without the boundary ring's Dirichlet DOFs (`physics_pc_gmg_bnd_drop = 1`). The ring count is fixed on purpose: the automatic choice (`-1`, all rings with r·Δθ/Δr < 1) grows like n_tht/2π rings, so its factor grows faster than N (438 MB at 49×64, tens of GB at 641×256). k = 3 was the fastest setting on both benchmarks. |
| `sfm2_gmg_mass` | `sfm2_gmg` with the exact-mass solves done by a fixed-degree Chebyshev iteration preconditioned by additive Schwarz (one subdomain per rank, overlap 1, local ICC(0)) instead of MUMPS with a centralized RHS (`physics_pc_mass_solver = 2`). Iteration counts are identical; the point is that its cost per rank falls with the rank count while the MUMPS solve's rises (161×64: `PhysPC_MjSolve` 90 s at np 1, 312 s at np 32). On few ranks it is SLOWER than MUMPS. |
| `sfm2_gmg_smop` | `sfm2_gmg` with the fine GMG smoother on the assembled operator, the exact matrix-free operator only for residuals (`physics_pc_gmg_smooth_op = 1`): about 4.8× fewer exact-mass solves, at +16% pair_w V-cycles and +1..2 outer iterations. |
| `sfm2_gmg_q` | both of the above. |
| `sfm2_gmg_d12` | `sfm2_gmg` without the axis treatment (the configuration of commit `3cafa9f46`). Opt-in only: the comparison is already measured (161×64, np 16: 473 s against 365 s), so it is not in the default arms. |
| `sfm2_lu` | SFM2 with MUMPS LU inner solves (the reference for approximation quality); refactored at every PC rebuild |
| `jorek` | JOREK's default: fieldsplit per toroidal harmonic + MUMPS |
| `jorek_fresh` | `jorek` with the PC rebuilt at every step (`iter_precon = 0`): separates the loss of the mode coupling from a stale factorisation |
| `sfm2_lu_hs0`, `sfm2_gmg_hs0` | `sfm2_lu` / `sfm2_gmg` keeping the cross-\|n\| entries (`physics_pc_harm_split = 0`). They are zeros at equilibrium but O(1) in the saturated island, where the `harm_split = 1` arms stall. |

## Series

| series | default | knob |
|---|---|---|
| strong | 161×64 (741k DOFs) at np = 1, 2, 4, …, 64 MPI ranks | `PCS_MESH`, `PCS_NPS` |
| weak | 81×32 at 1, 161×64 at 4, 321×128 at 16 (about 186k DOFs per rank). Extend with `641x256:64` only on a build with larger `n_nodes_max`. | `PCS_WEAK` |
| nonlinear | one trajectory per arm on 41×16, np 1: tstep 1, 10, 100 (10 steps each), then 200 steps at 1000, through the island's linear growth into saturation; restart files every 10 steps | `PCS_NL_MESH`, `PCS_NL_NP`, `PCS_NL_N` |

Problem sizes: 41×16 = 47k, 81×32 = 186k, 161×64 = 741k, 321×128 = 2.96M,
481×192 = 6.6M, 641×256 = 11.8M DOFs (72 DOFs per node: 6 variables × 4 C¹
Hermite DOFs × n_tor).

**Mesh sizes are limited at compile time.** `n_nodes_max` and
`n_elements_max` in `models/mod_settings.f90` are 60001 in the committed
settings, which covers up to 321×128 (41k nodes). Above that JOREK stops in
the grid generation ("hard-coded parameter n_nodes_max is too small"), so
481×192 (92k nodes) and 641×256 (164k nodes) — the last point of the default
weak series — need their own build with larger values. The node list is
allocated at `n_nodes_max` on every rank, so raise it only just above the
mesh you run and keep a separate binary for the big cases. `pc_case.sh` warns
when a case needs more nodes than `PCS_NODES_MAX` (default 60001). Keep at least about 20k DOFs, and a few flux-surface
rings, per MPI rank. Below that, the rank-local line smoother degenerates and
communication dominates, so use 321×128 for strong scaling beyond about 64
ranks. The partition, and therefore the smoother, depends only on the rank
count, not on the thread count.

**Two grids scale together.** The equilibrium is computed on the initial
polar grid (`n_radial`, `n_pol`) and only then aligned to the flux-surface
grid (`n_flux`, `n_tht`). `mknml.py` scales both, keeping the base namelist's
ratio — `n_radial = n_flux + 10`, `n_pol = n_tht` — so 41×16 still reproduces
the committed namelist exactly. Scaling only `n_flux`/`n_tht` leaves the
equilibrium on a 51×16 grid and every larger mesh then fails in the check
after the equilibrium computation. Override with `PCS_N_RADIAL`/`PCS_N_POL`.

The GMG arm needs n_flux − 1 divisible by 4 and n_tht divisible by 8 (at least
3 levels); `mknml.py` refuses other meshes. The `sfm2_lu` arm at np = 1 on
161×64 and above needs a lot of memory (7.4 GB at 81×32, and LU fill grows
faster than the DOF count). Run it on whole nodes (`--mem=0`), or leave it out
of the large cases.

Two further knobs are off by default and have no arm, because they need a
case the study does not cover (see `docs/physics_pc/workstream_D_matrix_free.md`
§14):

- `physics_pc_gmg_axis_split = 1` solves the axis block as one LU per |n|
  group, group k on rank mod(k, np), instead of one LU of all slots on rank 0.
  The counts are identical. At n_tor = 3 it only cuts the peak by 20% and the
  gather/scatter costs more than that, so it needs a larger n_tor.
- `physics_pc_gmg_axis_droptol = 1.d-4` drops the entries below
  tol·sqrt(|a_ii a_jj|) from the axis block before its LU: −47% factor entries
  and −23% `GMG_AxSolve` at 41×16, with unchanged counts.

## The production SF path (`sf_gmg`, `sf_gmg_w`, `sf_gmg_wpj`, `sf_lu`, `sf_lu_w`, `sf_lu_wpj`, `sf_jorek`, `sf_direct`)

The split-field preconditioner of `mod_petsc_pc_sf*` (workstream H) runs on the
reference physics case `namelist/model199/inxflow_shaped_pcbench`, not on the
island demo. Its composed force operator is valid up to tstep ~1 (at tstep 10
even the all-LU SF path diverges), so these arms start from a fresh
equilibrium and ramp 1e-3, 1e-2, 1e-1, 1 (2, 2, 3, 3 steps). The initial grid
keeps the case's own ratio, `n_radial = 2 n_flux - 1`, `n_pol = 2 n_tht`, so
161×64 needs 41k nodes and 321×128 needs 164k: a build with a larger
`n_nodes_max` (`pc_case.sh` warns).

| arm | blocks |
|---|---|
| `sf_gmg` | S_uu = `schur` (the default): every block on its C¹ GMG, pair_psi split (ψ, j) and pair_w on zebra lines, ρ / T on radial lines |
| `sf_gmg_w` | the same with S_uu = `w`, B₂₂ + W assembled |
| `sf_gmg_wpj` | the same with pair_w mixed (u, ω, ψ, j), ring smoother |
| `sf_gmg_wpj_eq16` | `sf_gmg_wpj` with the Eq. (16) corrector (a second pair_psi solve) instead of the default Eq. (17) |
| `sf_lu` / `sf_lu_w` / `sf_lu_wpj` | every block by MUMPS LU: the exact references, for approximation quality (`sf_lu_wj`: the lumped-mass form; `sf_lugw_wpj`: LU except pair_w) |
| `sf_gmg_wpj_hc1` / `_hcall`, `sf_lu_wpj_hc1` / `_hcall` | `sf_gmg_wpj` / `sf_lu_wpj` keeping the cross-\|n\| couplings of the \|n\| groups at most 1 apart / all (`physics_pc_sf_harm_couple` = 1 / -1, fixed for the run) |
| `sf_jorek` | JOREK's default PC on the same case and ramp |
| `sf_direct` | full-system direct solve: one MUMPS LU of the whole coupled Jacobian (`-jorek_pc_full_lu`, in-core), refactorised every step (`iter_precon = 0`); FGMRES only checks it |

**The wpj corrector** (`physics_pc_sf_corrector`, 2026-10-02). Step 3 of the
sweep needs M⁻¹ U δv. Chacon's Eq. (16) applies the full M⁻¹ (a second pair_psi
solve); Eq. (17) replaces it by the surrogate P_SF is built on. On `wpj` that
surrogate is pair_w's own small-flow ψ row, so the correction is the ψ, j part
of pair_w's solution, which the sweep used to discard: no extra solve. `B₁₆ T*`
goes onto pair_w's ψ-row right-hand side (2 drops it). The default (−1) picks
Eq. (17) on `wpj` and Eq. (16) elsewhere. 41×32, np 1 × 4, ramp 1e-3 … 10:
identical outer counts (53 in 13 steps) with Eq. (16), (17) and (17) without
B₁₆; pair_psi solves 106 → 53, PC apply 7.55 → 6.49 s. The tables below
predate it (Eq. 16).

**The two pair_w operators** (`physics_pc_sf_suu`, the only namelist entry
that picks the method; the rest of the sweep is shared):

- `schur`: S_uu is the ψ-channel Schur complement of the Jacobian, applied
  matrix-free, with pair_psi's own solver inside (one V-cycle per pair_w
  iteration; its LU on `sf_lu`). Its multigrid keeps B₂₂ + W for the Galerkin
  chain and runs level 0 on the channel with a diagonal ψ-row inverse and a
  Chebyshev constraint mass. The outer count is flat in the mesh.
- `w`: S_uu = B₂₂ + W, assembled. No mass inverse and no nested solve, so
  the cheapest per outer iteration, but W lacks the discrete M_j⁻¹ projection
  and the outer count grows with the mesh.
- `wpj` / `wj` (mixed, `mod_petsc_pc_sf_mixed`): pair_w assembled with the
  ψ channel back in explicit fields instead of composed into W. `wpj` packs
  (u, ω, ψ, j) with the small-flow ψ row (opz M_ψ, B₁₂, B₁₃) and keeps only
  W's curvature term; `wj` packs (u, ω, j) with ψ eliminated by the
  node-lumped ψ mass and keeps W's kink and curvature. See below.

**The mixed pair_w** (2026-09-29). W composes the ψ channel in the continuum,
so it misses the M_ψ⁻¹ and M_j⁻¹ projections and the resistive damping of the
ψ response, all growing with dt: `w` needs 37 its per step at tstep 1 and
diverges at 10. Keeping ψ and j explicit restores all three with every block
sparse and C¹-stencil (175 nnz/row, no products). Its multigrid needs two
things the other pairs do not: the smoother blocks are flux-surface RINGS
(`SF_GMG_SMOOTHER_RINGS`, block Jacobi in the level-0 GMRES), because at large
dt the ψ–u coupling θdt·B∥ runs along the field lines; and a harmonic's cos
and sin slots share each block (`SF_GMG_HARM_PAIR_MIXED`), because its
toroidal part θdt F₀/R ∂φ maps cos to sin. Without the pairing pair_w stalls
at a 1e-4 reduction at tstep 1 with any smoother; with radial or ring zebra
it stalls at tstep 10.

Outer its per step, np 1 × 4, ramp 1e-3 / 1e-2 / 1e-1 / 1 / 10 (2/2/3/3/3
steps), pair_w V-cycles per solve in parentheses:

| arm | mesh | tstep 1e-1 | tstep 1 | tstep 10 |
|---|---|---|---|---|
| `sf_lu` (schur, exact) | 41×32 | 4/3/3 | 6/6/7 | 15/13/12 |
| `sf_lu_w` | 41×32 | 6/5/5 | 37/39/37 | diverges |
| `sf_lu_wpj` | 41×32 | 2/2/2 | 3/3/4 | 6/6/6 |
| `sf_gmg` (schur) | 41×32 | 4/4/4 | 7/7/7 (4.2) | 388/314/357 (30, capped) |
| `sf_gmg_w` | 41×32 | 6/5/6 | 38/39/37 | diverges |
| `sf_gmg_wpj` | 41×32 | 4/4/4 (1.3) | 4/5/4 (2.0) | 14/9/12 (2.1) |
| `sf_lu` | 121×48 | 4/4/4 | 6/7/7 | 20/17/17 |
| `sf_lu_wpj` | 121×48 | 2/2/3 | 3/3/4 | 6/6/6 |
| `sf_gmg_wpj` | 121×48 | 4/4/5 (2.6) | 5/5/5 (3.7) | 9/18/10 (5.0) |

- `wpj` beats the exact schur arm because it keeps W's curvature term, a
  pressure channel the schur arm does not have (2–3× at tstep 10).
- The whole 41×32 ramp: `sf_gmg_wpj` 41.5 s, `sf_gmg` 2198 s (its tstep 10).
- With full 2:1 coarsening pair_w's V-cycles grow with the mesh (2.0 → 3.7
  at tstep 1; to 1e-8: 8–10 → 20–29 FGMRES its): ring blocks leave the
  radial coupling to the coarse grid. Alternating rings with radial-line
  blocks (smoother 9, multiplicative) is WORSE (121×48: no convergence at
  tstep 1): at large dt radial-line Jacobi amplifies the poloidal ψ–u
  coupling instead of smoothing it. The fix is radial semi-coarsening, below.
- np > 1: JOREK partitions ring by ring, so a rank boundary cuts some rings
  (3 on level 0 at 41×32 np 4); cut ring blocks lose the poloidal coupling
  and pair_w hits its cap at tstep 10. `SF_GMG_RING_OVERLAP` completes a cut
  ring from the neighbours (restricted additive Schwarz along J, 7.4% ghost
  rows): back to the np 1 counts (8–10 its to 1e-8). The walk runs until
  every ring is whole, on every level (coarse rings lie 2^g fine rings apart,
  so their owners rarely hold a whole neighbouring ring); a ring still cut
  prints a `GMG WARNING`.
- The rings are ordered 0, nc−1, 1, nc−2, … so the periodic wrap stays in a
  narrow band: 121×48 np 4, `GMG_Lines` 103 → 20 s. Completed rings keep
  that order (ghosts sorted in with the own rows); before, every cut ring was
  factored dense (at 321×128, 4096 rows per cos/sin block).

np 4 × 1, outer its per step (pair_w V-cycles), wall s:

| arm | mesh | tstep 1 | tstep 10 | wall, ramp to tstep 1 |
|---|---|---|---|---|
| `sf_gmg_wpj` | 41×32 | 4/4/4 (2.0) | 9/9/14 (2.2) | |
| `sf_gmg_wpj` | 121×48 | 6/5/5 (3.7) | 11/14/8 (5.2) | 107.9 |
| `sf_gmg` (schur) | 121×48 | 7/7/7 (4.1) | | 85.5 |
| `sf_gmg_w` | 121×48 | 56/58/55 | | 115.2 |

**The mixed pair_w's V-cycle: radial semi-coarsening** (`SF_GMG_SEMI_R_MIXED`,
`-sf_gmg1_semi_r k`). On these grids the cells are long in θ (median
r·dθ/dr 3.75 at 41×32, 7.6 at 81×32, 121×48 and 161×64), so the elliptic
couplings are strongest radially; the ring blocks do not smooth them and a
2:1 grid in both directions cannot take what they leave. The first levels
therefore coarsen in I only (J kept), until the median aspect is 2 (−1 =
that count from the grid: 1 level at 41×32, 2 at 81×32 – 161×64); line
relaxation with semi-coarsening across the lines is the standard robust
pairing (Schaffer; Trottenberg et al. §5.1). Semi-coarsened hierarchies may
use up to 8 levels (full ones keep 6, as all tuned arms ran).

pair_w to 1e-8, np 4, tstep 0.1 / 1 / 10 (`diag` runs): ρ per V-cycle, V-cycle
seconds per decade of reduction.

| 121×48 | ρ | s/decade |
|---|---|---|
| full coarsening, V(0,6), rings (was the default) | 0.450 | 0.70 |
| GMRES / damping 0.7 / 8⁄9 on levels ≥ 1, nsc 2 / 8, nlev 3 | 0.45–0.47 | 0.64–0.89 |
| two-grid, exact coarse solve (nlev 2) | 0.461 | – |
| V(0,4) / V(0,10) / V(2,4) | 0.62 / 0.30 / 0.59 | 0.90 / 0.66 / 1.16 |
| zebra rings, V(0,6) / V(0,10) | 0.33 / 0.19 | 0.54 / 0.57 |
| semi-coarsening, 1 radial level | 0.124 | 0.36 |
| **semi-coarsening, 2 radial levels (auto)** | **0.045** | **0.25** |

The two-grid cycle converging at the V-cycle's rate is the diagnosis: the
coarse-level smoothing is irrelevant, the fine smoother and the coarse space
do not fit. At 41×32: full 0.112, auto (1 level) 0.043; 161×64 full 0.51.
Semi-coarsening with zebra rings stalled at tstep 10 (np 4; smoother 8 has no
ring completion). np 8 gives the np 4 rates at both meshes.

Production ramp, np 4 × 1 (outer its per step, pair_w V-cycles):

| mesh | cycle | tstep 1e-1 | tstep 1 | tstep 10 | outer its | KSPSolve | wall s |
|---|---|---|---|---|---|---|---|
| 81×32 | full | 4/4/4 (2.7) | 5/5/5 (3.5) | 8/8/9 (4.5) | 65 | 31.7 | 55.6 |
| 81×32 | auto (2) | 3/3/3 (1.3) | 4/4/4 (2.0) | 7/7/7 (2.0) | 53 | 20.4 | 48.4 |
| 121×48 | full | 4/4/5 (2.6) | 6/5/5 (3.7) | 9/10/8 (5.6) | 69 | 88.5 | 144.4 |
| 121×48 | auto (2) | 3/3/3 (1.4) | 5/5/5 (2.0) | 7/7/7 (2.0) | 55 | 85.0 | 191.3 |

121×48 is a poor mesh for this hierarchy: 120 = 8·15 stops the radial
coarsening at 16 surfaces, leaving a 16×24 coarsest level (17k rows, LU
factorised at each of the 20 rebuilds: 39 s). The study meshes have n_flux −
1 = 5·2^k and end at 6×8 (1.9k rows at 81×32).
- A `wpj` variant with the ψ row B₁₁ (the flow kept) was identical to `wpj` on
  this case: in its linear phase B₁₁ = opz M_ψ to 5 digits, so the small-flow
  assumption is untested. It and the `wj` ablations below have been removed.
- The (u, ω, j) forms, i.e. parabolized with ψ eliminated by the lumped mass,
  are limited by that lumping (κ ≈ 58 for bicubic Hermite): the discrete-kink ablation
  12–17 its at tstep 1; with W's continuum kink (`wj`) they diverge at 10.

Both pairs run asymmetric V-cycles, all level-0 smoothing after the coarse
correction (`SF_PJ_*`, `SF_W_*` in `mod_petsc_pc_sf_solver.f90`): pair_psi
V(0,12) on `schur` and V(0,10) on `w` with 2 coarse steps per side, pair_w
V(0,6). From a ψ-row right-hand side (the corrector's, and every nested solve)
the coarse correction leaves a large j-row residual at the edge that only
post-smoothing removes; with it every pair_psi solve converges in one cycle.
The setup line `GMG smoother 7: V(a,b) on level 0, c steps per side below`
shows the shape each hierarchy runs. pair_w on both arms and pair_psi on
`w` run smoother 8 (`SF_GMG_SMOOTHER_ZEBRA_RINGS`): GMGPolar's circle/radial split, a zebra over
the rings between the axis block and the switch ring I_s ≈ n_tht/2π, zebra
radial lines outside. Its setup line adds `GMG hybrid smoother: ring blocks on
fine rings I < I_s, radial lines outside`. At 321×128 np 64 it cut pair_w from
3.17 to 2.23 V-cycles (the exact solve of rings 0..I_s−1 reaches 2.16) and the
solve from 89 to 78 s (three runs); on the schur arm's pair_w it is neutral
within that arm's ±10% run-to-run spread (five runs each). At 161×64
(I_s = 10) it is neutral. ρ and T keep radial
line Jacobi (smoother 8 there: T 2.0 → 8.8 V-cycles).

Laptop reference (M4 Mac mini, np 4 × 1 thread, shaped pcbench, tstep 1,
3 steps; outer its summed, wall in s; "before" = the configurations of
commit `d28fafa86` / `e0b9f7855`):

| mesh | `sf_gmg` (schur) | schur before | `sf_gmg_w` | w before |
|---|---|---|---|---|
| 41×32 | 21 / 13.8 | 60 / 21.5* | 112 / 15.8 | 112 / 21.3* |
| 61×48 | 20 / 30.2 | 52 / 47.7* | 164 / 44.8 | 164 / 64.1* |
| 81×32 | 19 / 21.3 | 53 / 37.7* | 116 / 30.3 | 116 / 41.9* |
| 121×48 | 21 / 58.6 | 73 / 109.3 | 166 / 92.0 | 166 / 109.5 |
| 161×64 | 22 / 148.3 | 104 / 348.9 | 201 / 258.6 † | 203 / 285.0 † |

\* np 1 × 4. † back to back; at 161×64 the laptop runs near its memory
limit and the outer Jacobian matvec varies between runs (the previous `w`
configuration took 227 s on another day). The exact references at 41×32:
`sf_lu` 20 its, `sf_lu_w` 112 — `w`'s count is its operator's, not its
solvers'.

The multigrid configuration is compiled in (`mod_petsc_pc_sf_solver.f90`),
so one binary is one configuration. On the first build the log prints what
the parallel parts actually do; check it before reading any timing:

- `GMG line overlap <k> node(s): ... ghost rows on level 0 (x% of the rows)`:
  the overlapping line segments across rank boundaries. They keep the
  V-cycle counts flat in np (41×64: pair_psi 2.0 at np 1; without overlap
  5.3–6.5 at np 8, with overlap 2 2.0–2.2).
- `GMG<k>: Richardson smoothing on levels >= 1 (scale s, no reductions)`:
  the coarse levels smooth with Richardson on the same zebra / line blocks
  (`SF_GMG_RICH_FROM`; scale 1 for zebra, 0.8 for the ρ / T lines), not
  GMRES, whose two allreduces per step over all ranks made the coarse
  smoothing flat in np (`t_GMG*_SmoothC` in `prof.txt`). Every hierarchy
  prints it except the schur arm's pair_w (GMG1) and pair_psi (GMG2): pair_w's
  V-cycle needs GMRES there, and pair_psi also runs nested in the S_uu shell.
- `GMG<k>: level matvecs on the OpenMP block kernel (bs 3, ... levels)`: the
  SF operators, every GMG level, the prolongations and the SFM2 coupling
  blocks multiply with `solvers/jorek_blockmv_attach.c` (`SF_BLOCKMV = 1`),
  which threads over the rank's OpenMP threads and reads each harmonic
  block's column indices once. It is gated against PETSc's `MatMult` on first
  attach and shows up as `PC_BlockMV` in `prof.txt`. The comparison binary is
  `SF_BLOCKMV = 0`: PETSc's kernel, one thread per rank, except that the
  fine-level GMG operators become AIJMKL on a PETSc with MKL sparse
  (`SF: GMG operators converted to AIJMKL (threaded SpMV)`). The laptop cannot
  judge the kernel: its memory bandwidth saturates at 2 threads
  (81×32 np 1×4: MatMult 23.5 → 20.2 s, KSP 41.9 → 38.3 s; np 2×2 even;
  one thread per rank ~10% slower). Compare the two binaries on the cluster
  over the thread sweep.
- `GMG<k> level types (A/P)`: the PETSc type of every level operator and
  prolongation (`MatPtAP` decides the coarse ones).
- `... n J-sectors (... rows, reduced system ...)` and `J-sector solve
  backward error e -> in use`: the axis blocks' J-sector solve (see below).
  `-> LU used` means the exactness gate (row-wise backward error 1e-8)
  refused the sector solve for that level; only then is the sequential LU
  of the whole block built (`[Mem] ... axis block level g LU`).

**The constraint mass on `schur`** (decided on the laptop, to confirm on the
cluster). The level-0 operator of pair_w's multigrid applies B_33⁻¹ in every
smoother step and V-cycle residual. A MUMPS factor with a centralized RHS grew
with np on the cluster, so it is a Chebyshev iteration on additive Schwarz
(`mod_petsc_pc_mass_cheb`: overlap 1, local ICC(0) on an ND ordering), with
the degree taken from the κ measured on the first build. The log prints:

- `pair_w Dh channel: mass B_33: Chebyshev(<d>) + ASM(ovl 1, ICC(0) ND), ..., kappa
  <k>, ..., B-norm err <e> (bound <b>)`. On the laptop κ is 3.25 at np 1 and
  4.7 at np 4, and d = 3. κ should stay bounded in np, since overlap only
  raises λ_max. A κ that keeps growing with np, and d with it, is the thing
  to watch for.

`t_PhysPC_MjSolve` / `n_PhysPC_MjSolve` is the time per mass solve.

Laptop audit that chose it (shaped pcbench, tstep 1, 3 steps; outer its
summed; MjSolve in ms per call; at the time the mass sat in pair_w's Krylov
operator too, and MUMPS was a run-time option):

| variant | 61×48 np 1×4 | 61×48 np 4×1 | 121×48 np 1×4 |
|---|---|---|---|
| MUMPS factor | 52 its, 2.3 ms | 52 its, 2.8 ms | 72 its, 4.2 ms |
| Chebyshev(3) + ASM/ICC(0) ND | 52 its, 3.8 ms | 53 its, **1.9 ms** | 72 its, 7.8 ms |
| Chebyshev(3) + ASM/ICC(0) natural (κ 18 → 34) | 50 its | **77 its** (degree 5: 55) | 70 its |
| diagonal Qi only | 126 its (pair_w 25 its/solve) | | |
| Chebyshev(8) + point Jacobi (κ 552) | 98 its | | |
| Chebyshev + rank-local FSAI (κ 21) | 57 (deg 6), **130** (deg 3) | 57 (deg 4) | |
| Chebyshev + hypre ParaSails (κ 39) | 53 (deg 12), **218** (deg 4) | | |

- At np 1 each Chebyshev call costs more than a MUMPS one, because its ICC(0)
  triangular solves run on one thread. The KSP time is still equal or lower
  (121×48: 81 s against 90 s, 81×32: 25 s against 22 s), and the cost per
  call falls with np.
- Below its degree floor a Chebyshev mass fails abruptly, not gradually,
  which is why the degree follows κ.
- Rejected too: a cheaper mass (Chebyshev(1) or Qi) only in the smoother's
  operator, which gave fewer mass solves per V-cycle but more V-cycles:
  +4..6 outer its and a higher KSP time at np 4.

**The axis block** (decided). Every GMG level has an exact axis block (rings
0..3, all J). Its size depends on n_tht only, so a sequential LU on the ranks
that own it (rank 0, and from 161×64 at np ≈ 43 on also its neighbours, each
repeating the whole LU) is the strong-scaling critical path. The first
cluster runs settled it: `SF_GMG_AXIS_SECTORS = -1` (the default) solves it
over J-sector ranks, exact and gated on its backward error on the first build. On
the laptop the sector solve loses about as much to synchronisation as it
saves; that is expected there and no reason to switch it off. A binary with
`SF_GMG_AXIS_SECTORS = 0` restores the sequential LU for comparison
(`t_GMG<k>_AxSolve` against `t_GMG<k>_Lines`).

**One thing only the cluster can decide.** It is a compile-time constant in
`mod_petsc_pc_sf_solver.f90`; for experiments the PETSc options below
override it without a new binary. Give such cases a `PCS_TAG`.

Runtime overrides of the V-cycle (`PCS_PETSC_OPTS`), for scaling
experiments only: `-sf_gmg_<name> v` sets every hierarchy,
`-sf_gmg<k>_<name> v` hierarchy k (1 pair_w, 2 pair_psi, 3 ρ, 4 T), with
`<name>` one of `line_overlap`, `axis_sectors`, `axis_rings`, `rich_from` (first Richardson level; 0 =
all, a level count above the hierarchy's = none), `rich_omega`, `pre0`,
`post0`, `nsc` (coarse steps per side) and `nlev` (cap on the hierarchy
depth). The setup lines print what is in effect.

- **The line overlap.** Overlap 2 kept the V-cycles flat to np 8 at 41×64
  (~5 rings per rank), but its ghost fraction (150% at 161×64 np 64) made
  `t_GMG<k>_Lines` flat in np on the cluster. `SF_GMG_LINE_OVERLAP = 1` since
  2026-09-28 (np 64 × 8, 2 nodes: w arm solve −17% at 161×64, −20% at
  321×128, outer its +1–2%; overlap 0 does not converge). The schur arm's
  pair_w and pair_psi keep overlap 2 and GMRES coarse smoothing
  (`SF_GMG_LINE_OVERLAP_SCHUR_W`, `SF_GMG_RICH_NONE`), and its pair_w axis
  block the sequential LU (`SF_GMG_AXIS_SECTORS_SCHUR_W = 0`).

Local check (laptop, 41×32, np 2, `PCS_NSTEP_N=1,1,2,2`): all three arms
finish, and `sf_gmg`'s outer counts match `sf_lu`'s (3 3 6 5 38 39 against
2 2 6 5 37 39); `sf_jorek` needs 1–3.

```bash
PCS_ROOT=$PWD/sf_strong_161 PCS_MESH=161x64  PCS_NPS="1 2 4 8 16 32 64" \
  PCS_ARMS="sf_jorek sf_lu sf_gmg sf_gmg_w" ./pc_study.sh strong
PCS_ROOT=$PWD/sf_strong_321 PCS_MESH=321x128 PCS_NPS="8 16 32 64 128" \
  PCS_ARMS="sf_jorek sf_gmg sf_gmg_w"       ./pc_study.sh strong
PCS_ROOT=$PWD/sf_weak PCS_WEAK="81x32:1 161x64:4 321x128:16" \
  PCS_ARMS="sf_jorek sf_gmg sf_gmg_w"       ./pc_study.sh weak
```

Columns: `sf_pj_mean`, `sf_w_mean`, `sf_rho_mean`, `sf_T_mean` are the mean
inner iterations (V-cycles) per solve, per time step; they, and `outer_its`,
must stay flat in np. The time per part is in `t_GMG<k>_Lines` (line
smoothing including the overlap scatter), `t_GMG<k>_AxSolve` (axis blocks),
`t_GMG_Prolong` / `t_GMG_SmSetup` / `t_PhysPC_Extract` (setup), and
`t_MatMult`; hierarchy k = 1 pair_w, 2 pair_psi, 3 ρ, 4 T.

## Motivation figures (`plot_motivation.py`)

Five figures for the case for a scalable physics PC. Each comes from one
series; a series is one study directory:

| figure | claim | series (`pc_study.sh …`) | arms |
|---|---|---|---|
| `F1_lu_strong` | JOREK's LU PC does not strong-scale: factorisation and triangular-solve time and efficiency vs cores | `strong` | `jorek` |
| `F1_lu_weak` | at fixed DOFs per rank the LU build, apply and memory per rank grow with N; SFM2-GMG's stay flatter | `weak` | `jorek sfm2_lu sfm2_gmg` |
| `F2_nonlinear_lu` | JOREK's iterations rise ~10× once the island goes nonlinear, also when the PC is rebuilt every step | `nonlinear` | `jorek jorek_fresh` |
| `F3_nonlinear_all` | SFM2 keeps its count through the nonlinear phase, absolute and relative to its own linear-phase count | `nonlinear` | `jorek jorek_fresh sfm2_lu sfm2_lu_hs0 sfm2_gmg_hs0` |
| `F4_components` | each SFM2 part (D_ρ/D_T, pair_psi, pair_w, the Schur assembly), LU inner solves vs scalable ones: time per outer iteration vs cores, and parallel efficiency | `strong` | `sfm2_lu sfm2_gmg sfm2_gmg_q` (+ `jorek` as the whole-system LU reference) |
| `F5_mode_coupling` | why: the n=0↔n=1 blocks go from ~0 to O(1) as the island grows | `probes` (one step from each restart of the `nonlinear` run) | `sfm2_lu_hs0` (the A8 report needs harm_split = 0) |

```bash
export JOREK_BIN=/path/to/jorek_model199          # n_tor = 3, n_period = 1 build
PCS_ROOT=$PWD/pc_scaling_strong PCS_ARMS="jorek sfm2_lu sfm2_gmg sfm2_gmg_q" ./pc_study.sh strong
PCS_ROOT=$PWD/pc_scaling_weak   PCS_ARMS="jorek sfm2_lu sfm2_gmg"            ./pc_study.sh weak
PCS_ROOT=$PWD/pc_scaling_nl     PCS_NL_MESH=41x16                            ./pc_study.sh nonlinear
# F5 (and F3's probe points): one step at tstep 1000 from each restart of the
# jorek trajectory. sfm2_gmg_hs0 is usually too slow for a whole trajectory,
# so probe it here instead.
PCS_ROOT=$PWD/pc_scaling_coupling PCS_REF=$PWD/pc_scaling_nl/jorek_41x16_np1x1 \
  PCS_PROBE_ARMS="sfm2_lu_hs0 sfm2_gmg_hs0" PCS_PROBE_NP=1 ./pc_study.sh probes
python3 -m venv .venv && .venv/bin/pip install matplotlib numpy
.venv/bin/python plot_motivation.py --strong pc_scaling_strong --weak pc_scaling_weak \
    --nonlinear pc_scaling_nl --coupling pc_scaling_coupling --out figures
```

On a 64-rank allocation, with the committed `mod_settings.f90` (meshes up to
321×128):

```bash
# strong, two meshes: the second is the one that separates the arms
PCS_ROOT=$PWD/pc_scaling_strong_161  PCS_MESH=161x64  PCS_NPS="1 2 4 8 16 32"        ./pc_study.sh strong
PCS_ROOT=$PWD/pc_scaling_strong_321  PCS_MESH=321x128 PCS_NPS="4 8 16 32 64" \
  PCS_ARMS="jorek sfm2_gmg sfm2_gmg_q"                                               ./pc_study.sh strong
PCS_ROOT=$PWD/pc_scaling_weak        ./pc_study.sh weak      # 81x32:1 161x64:4 321x128:16
PCS_ROOT=$PWD/pc_scaling_nl PCS_NL_MESH=161x64 PCS_NL_NP=16 \
  PCS_ARMS="jorek jorek_fresh sfm2_lu sfm2_lu_hs0" ./pc_study.sh nonlinear
```

- 161×64 stops at np 32: at np 64 it is only 11.6k DOFs per rank, below the
  ~20k where the rank-local line smoother degenerates.
- `sfm2_lu` is left out of the 321×128 series: its LU fill grows faster than
  N (7.4 GB already at 81×32), and it is not needed there — `sfm2_lu` at
  161×64 is what F4 compares against.
- The nonlinear and probe series only produce iteration counts, so their rank
  count is a matter of turnaround, not of the result. 81×32 gives the same
  figures for a fraction of the time.

**What the nonlinear series actually runs.** Unlike `strong` and `weak`,
which take 9 time steps (the 0.1/1/10 ramp, 3 steps each), `nonlinear` is a
whole simulation per arm: the 1/10/100 ramp with 10 steps each, then
`PCS_NL_N` (200) steps at tstep 1000 — about 230 steps from scratch, through
the linear growth into the saturated island. Scaled from the laptop, at
161×64 on 16 ranks expect roughly 20–40 min for `jorek`, ~1.5× that for
`jorek_fresh` (it refactorises every step), 1–2 h for `sfm2_lu_hs0`, and
6 h or more for `sfm2_gmg_hs0` — which is why the command above leaves that
arm to the `probes` series.

- `sfm2_lu` (harm_split = 1) is EXPECTED to end as `status=noconv` partway
  through: it stalls at 400 iterations as the island goes nonlinear. That is
  the result, not a broken run, and F3 marks it.
- Afterwards check that the island really saturated:
  `cut -f1,2,10 <root>/jorek_<mesh>_np<np>x<omp>/steps.tsv | tail` — the
  `wmag_nlast` column must flatten. If it is still rising, raise `PCS_NL_N`
  and resubmit with `PCS_FORCE=1` (a finished case is skipped otherwise).

**Things the figures do not hide:**
- **The mode coupling decides the nonlinear phase.** With `harm_split = 1`
  SFM2 drops the cross-|n| blocks, the same approximation JOREK's
  per-harmonic PC makes. It then stalls at 400 its in the saturated island at
  tstep 1000 (41×16, both `sfm2_lu` and `sfm2_gmg`). Keeping the blocks
  (`*_hs0`) gives a flat 71 (LU) / 55 (GMG) its per step there, against
  103–105 in the linear phase. Every nonlinear-phase claim needs the `_hs0`
  arms; `harm_split = 1` is only for linear-phase timing.
- **SFM2 needs more iterations than JOREK's PC in absolute terms**: ~70 against
  14–17 in saturation, and ~105 against 1–2 in the linear phase. F3 shows
  both panels. The case for SFM2 is its flat count and its parallel parts,
  not its count today.
- **The whole SFM2 PC is still slower than JOREK's.** F4 compares the parts'
  scaling, not their absolute cost.
- The nonlinear degradation shows at large tstep. At tstep 10, JOREK's PC
  needs 2–3 its even in the saturated island, and 4–8 at tstep 100.
- With n_tor = 3 only n = 0 ↔ 1 is coupled. Production runs with more
  harmonics drop more coupling.
- On the laptop, np > 4 uses the efficiency cores, so the laptop scaling
  numbers only show that the pipeline works.

## Output columns (`results.tsv`)

- `outer_its`: FGMRES iterations per time step. They must stay flat with np;
  only round-off changes them.
- `pw_cycles_mean`: mean pair_w GMG V-cycles per solve, per step. At np > 1
  the radial-line smoother is cut into per-rank segments, so a rise with np is
  expected. JOREK distributes the rows ring by ring, so each rank owns a band
  of flux surfaces and every radial line gets one block-Jacobi cut per rank
  boundary. On 41×16 the D13 axis treatment saves 35% / 20% / 10% of the
  cycles at np = 1 / 2 / 4, because these cuts grow with np. Compare
  `sfm2_gmg` against `sfm2_gmg_d12` at equal np.
- `pp_its_mean` / `rt_its_mean`: inner iterations of pair_psi and ρ/T.
- `wall_s`: PETSc total time. `setup_s_sum` / `solve_s_sum` are the PC build
  and the Krylov solves, summed over steps.
- `t_<event>`: `-log_view` times (max over ranks, all stages). The ones to watch:
  - `PhysPC_SolveW`, `GMG_VCycle`: the pair_w multigrid.
  - `PhysPC_MjSolve`: the constraint-mass solve inside the matrix-free pair_w
    operators. On the `sfm2_*` arms it is MUMPS with a centralized RHS, which
    stopped scaling on the laptop and on the cluster. On `sf_gmg` it is the
    Chebyshev solve in the level-0 operator of pair_w's multigrid (see the
    SF path section); `sf_gmg_w` has none.
  - `GMG_Coarse`, `GMG_AxSolve`: the coarsest level and the axis blocks. Both
    are exact LUs solved redundantly on the few ranks that own their rows
    (rank 0 for the axis), so they involve no collective over all ranks. A
    growing `GMG_AxSolve` shows the extra work that rank 0 carries.
  - `GMG_Lines`: the radial-line block solves of the smoother (OpenMP).
- `mem_max_total_GB` / `mem_max_rank_GB`: `-memory_view` peak RSS, summed over
  ranks / largest rank.
- `rebuilds`: PC builds (the first setup plus every refactorisation).
- `n_<event>`: the call count of each event, for times per call.
- `status`: `ok`, `noconv` (JOREK aborted after 400 its; it still exits 0),
  `error`, `incomplete`.

Every case directory also gets a `steps.tsv`, one row per time step: tstep,
time, outer its, rebuild flag, step/setup/solve seconds, and W_mag/W_kin of
the first and last harmonic. An aborted step is kept as the last row, with a
negative reason.

## Things to know

- **MUMPS centralized RHS.** PETSc's parallel default for MUMPS (distributed
  right-hand side, ICNTL(20) = 10) corrupted the heap on the development build,
  so `petsc_initialize` now defaults to `-mat_mumps_icntl_20 0`. That setting
  gathers every right-hand side to one rank, which is what makes
  `PhysPC_MjSolve` scale badly. On a MUMPS build that handles the distributed
  RHS correctly, try `PCS_PETSC_OPTS="-mat_mumps_icntl_20 10 ..."`; see
  `job_study.slurm` for the prefixes. Command-line values override the default.
- **Do not add `-log_view_memory`.** PETSc 3.24.6 aborts with it on this path
  (unbalanced log event in `MatGetBrowsOfAoCols_MPIAIJ`).
- **Threads.** `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS`
  are set to `PCS_OMP`, with `OMP_PLACES=cores` and `OMP_PROC_BIND=close`
  unless already set.
  - What uses the threads: JOREK's matrix construction, MUMPS through a
    threaded BLAS, and in the physics PC the GMG smoother's block solves
    (`GMG_Lines`) and block factorisations.
  - What doesn't: PETSc's sparse kernels (MatMult, PtAP, the Krylov vector
    work) run on one thread per rank, and so do the shell matvecs. With 8 × 16
    per node, those use 8 of the 128 cores.
  - Keep this in mind when comparing arms. The `t_*` event times show how
    much of each case runs single-threaded.

## Local reference: 8-core MacBook Air (4 performance + 4 efficiency cores)

`sfm2_gmg` as of commit `3cafa9f46` (now the `sfm2_gmg_d12` arm) on 81×32
(186k DOFs), 1 thread per rank:

| np | wall [s] | pair_w solve [s] | MjSolve [s] | outer its (tstep 10) | pair_w cycles (tstep 10) |
|---|---|---|---|---|---|
| 1 | 252 | 114 | 19 | 37 38 39 | 5.5 5.5 6.1 |
| 2 | 211 | 121 | 43 | 37 38 40 | 6.1 6.2 6.9 |
| 4 | 224 | 136 | 57 | 37 38 40 | 6.3 6.3 6.7 |
| 8 | 352 | 249 | 136 | 37 38 40 | 8.8 8.1 8.3 |

At np = 8 the efficiency cores are in use. The PC setup (block extraction,
products, PtAP) roughly halves from np = 1 to np = 2, but the solve does not.

With the D13 axis treatment (`sfm2_gmg` now), the serial runs on the same
laptop give, at tstep 10:

| mesh | `sfm2_gmg_d12` wall / pair_w cycles | `sfm2_gmg` wall / pair_w cycles |
|---|---|---|
| 81×32 | 245 s / 5.4–6.1 | 203 s / 2.6–2.7 (automatic axis block) |
| 121×48 | 706 s / 7.4–8.3 | 477 s / 3.2–3.7 (k = 3) |

Outer iterations are unchanged to within +4. On the ballooning corner
(`inxflow600_circ_pcbench`, 49×64, tstep 10) the pair_w cycles fall from
20–24 to 10–11 and the wall time from 1337 s to 992 s; JOREK's default PC
needs 273 s there.

### Motivation figures on the same laptop (2026-09-15, `n_period = 1` build)

`plot_motivation.py` on 81×32 (strong), `41x16:1 57x24:2 81x32:4` (weak) and
the 41×16 nonlinear run:

- **Nonlinear phase, tstep 1000:**
  - JOREK's PC goes from 1–3 to 16–18 its, and still 13–15 when it is rebuilt every step.
  - `sfm2_lu_hs0` falls from 105 to 66–72; `sfm2_gmg_hs0` probes from 85 to 52–63.
  - `sfm2_lu` (harm_split = 1) stalls at 400 its at step 105.
- **Strong scaling:**
  - JOREK's LU factorisation stays at ~9 s per rebuild at np 1, 2 and 4.
  - Among the SFM2 parts, only the Schur assembly scales (~60% at 4).
  - pair_w's GMG smoother scales, but its exact-mass MUMPS solve (8 → 29 s) and rank-0 axis LU (5 → 12 s) grow with np.
- **Weak scaling:** JOREK's PC build grows 6.5× over 4× the DOFs at fixed DOFs per rank; SFM2-GMG's grows 2.1×.

Four P-cores sharing one memory bus cannot show strong scaling of sparse
kernels. Only the cluster can confirm or refute claim 4.
