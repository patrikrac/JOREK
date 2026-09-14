# Physics-PC MPI scaling study

These scripts run strong- and weak-scaling series of the SFM2 physics
preconditioner on a SLURM cluster, and compare it against JOREK's default PETSc
preconditioner. The test case is the committed benchmark
`namelist/model199/intear_island_demo`: a 2/1 tearing mode with a tstep ramp of
0.1, 1 and 10, three steps each.

| file | role |
|---|---|
| `job_study.slurm` | generic jobscript template: copy it to `job_study.local.slurm` (git-ignored) and add the machine's partition, account and modules there |
| `pc_study.sh` | runs a strong or weak series one case after another inside one allocation |
| `pc_case.sh` | runs one case: writes the namelist, launches, profiles, records `case.meta` |
| `mknml.py` | namelist = base + arm flags + mesh + ramp (+ `key=value` overrides) |
| `collect.py` | turns a study directory into `results.tsv` |

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
| `sfm2_gmg_d12` | `sfm2_gmg` without the axis treatment (the configuration of commit `3cafa9f46`), for comparison. |
| `sfm2_lu` | SFM2 with MUMPS LU inner solves (the reference for approximation quality); refactored at every PC rebuild |
| `jorek` | JOREK's default: fieldsplit per toroidal harmonic + MUMPS |

## Series

| series | default | knob |
|---|---|---|
| strong | 161×64 (741k DOFs) at np = 1, 2, 4, …, 64 MPI ranks | `PCS_MESH`, `PCS_NPS` |
| weak | 81×32 at 1, 161×64 at 4, 321×128 at 16, 641×256 at 64 (about 186k DOFs per rank) | `PCS_WEAK` |

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
  - `GMG_Coarse`: the coarsest-level MUMPS solve. Try
    `-gmg1_coarse_pc_type redundant` if it grows with np.
- `mem_max_total_GB` / `mem_max_rank_GB`: `-memory_view` peak RSS, summed over
  ranks / largest rank.

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
  - What uses the threads: JOREK's matrix construction, and MUMPS through a
    threaded BLAS.
  - What doesn't: the physics PC has no OpenMP regions, and PETSc's sparse
    kernels (MatMult, PtAP) run on one thread per rank. With 8 × 16 per node,
    the GMG, the shell matvecs and the Krylov work use 8 of the 128 cores.
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
