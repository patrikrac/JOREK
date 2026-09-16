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
| weak | 81×32 at 1, 161×64 at 4, 321×128 at 16, 641×256 at 64 (about 186k DOFs per rank) | `PCS_WEAK` |
| nonlinear | one trajectory per arm on 41×16, np 1: tstep 1, 10, 100 (10 steps each), then 200 steps at 1000, through the island's linear growth into saturation; restart files every 10 steps | `PCS_NL_MESH`, `PCS_NL_NP`, `PCS_NL_N` |

Problem sizes: 41×16 = 47k, 81×32 = 186k, 161×64 = 741k, 321×128 = 2.96M,
641×256 = 11.8M DOFs. Keep at least about 20k DOFs, and a few flux-surface
rings, per MPI rank. Below that, the rank-local line smoother degenerates and
communication dominates, so use 321×128 for strong scaling beyond about 64
ranks. The partition, and therefore the smoother, depends only on the rank
count, not on the thread count.

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

On the cluster: strong at 161×64 (and 321×128 beyond ~64 ranks), weak with
the default `PCS_WEAK`, and the nonlinear series at 161×64 on 16 ranks
(`PCS_NL_MESH=161x64 PCS_NL_NP=16`). The nonlinear figures only plot
iteration counts, so the rank count there does not matter.

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
  - `PhysPC_MjSolve`: the exact-mass MUMPS solve inside every matrix-free
    pair_w matvec. **This is the component that stopped scaling on the
    8-core laptop.**
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
