# Physics-PC MPI scaling study

These scripts run strong- and weak-scaling series of the SFM2 physics
preconditioner on a SLURM cluster, and compare it against JOREK's default PETSc
preconditioner. The test case is the committed benchmark
`namelist/model199/intear_island_demo`: a 2/1 tearing mode with a tstep ramp of
0.1, 1 and 10, three steps each.

| file | role |
|---|---|
| `job_study.slurm` | example jobscript: adapt the `#SBATCH` lines and modules, then `sbatch` it |
| `pc_study.sh` | runs a strong or weak series one case after another inside one allocation |
| `pc_case.sh` | runs one case: writes the namelist, launches, profiles, records `case.meta` |
| `mknml.py` | namelist = base + arm flags + mesh + ramp (+ `key=value` overrides) |
| `collect.py` | turns a study directory into `results.tsv` |

## Build requirements

- `jorek_model199` built with `n_tor = 3` and `n_period = 1` in
  `models/mod_settings.f90`. These are compile-time settings: a wrong binary
  runs the axisymmetric problem and still exits 0.
- A PETSc build with MUMPS (`USE_PETSC`). Use the same compiler and MPI stack
  at run time as at build time.
- Python 3 (standard library only).

## Quick start

```bash
export JOREK_BIN=/path/to/jorek/jorek_model199
PCS_DRYRUN=1 PCS_MAXNP=64 util/physics_pc_scaling/pc_study.sh strong   # list the cases
sbatch --export=ALL,MODE=strong util/physics_pc_scaling/job_study.slurm
sbatch --export=ALL,MODE=weak   util/physics_pc_scaling/job_study.slurm
```

Each series runs inside one allocation, so request the largest rank count in
the series; cases needing more ranks than the allocation has are skipped.
Finished cases are skipped on resubmission, so a timed-out job can simply be
submitted again. `results.tsv` is rewritten after every case.

## Arms (`PCS_ARMS`)

| arm | preconditioner |
|---|---|
| `sfm2_gmg` | SFM2 without refactorisations: C¹ GMG on pair_w (matrix-free fine operator with the exact mass), eta-Schur + GMG on pair_psi, GMG on ρ/T. Only the constraint masses are factored, once per run. |
| `sfm2_lu` | SFM2 with MUMPS LU inner solves (the reference for approximation quality); refactored at every PC rebuild |
| `jorek` | JOREK's default: fieldsplit per toroidal harmonic + MUMPS |

## Series

| series | default | knob |
|---|---|---|
| strong | 161×64 at np = 1, 2, 4, …, 256 | `PCS_MESH`, `PCS_NPS` |
| weak | 81×32 at 1, 161×64 at 4, 321×128 at 16, 641×256 at 64 (about 47k DOFs per rank) | `PCS_WEAK` |

The GMG arm needs n_flux − 1 divisible by 4 and n_tht divisible by 8 (at least
3 levels); `mknml.py` refuses other meshes. The `sfm2_lu` arm at np = 1 on
161×64 and above needs a lot of memory (7.4 GB at 81×32, and LU fill grows
faster than the DOF count). Run it on whole nodes (`--mem=0`), or leave it out
of the large cases.

## Output columns (`results.tsv`)

- `outer_its`: FGMRES iterations per time step. They must stay flat with np;
  only round-off changes them.
- `pw_cycles_mean`: mean pair_w GMG V-cycles per solve, per step. At np > 1
  the radial-line smoother is cut into per-rank segments, so a slow rise with
  np is expected.
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
- **Threads.** `OMP_NUM_THREADS=1` is exported per case. The physics PC has no
  OpenMP, so use one rank per physical core.

## Local reference: 8-core MacBook Air (4 performance + 4 efficiency cores)

`sfm2_gmg` on 81×32 (186k DOFs), commit `3cafa9f46`:

| np | wall [s] | pair_w solve [s] | MjSolve [s] | outer its (tstep 10) | pair_w cycles (tstep 10) |
|---|---|---|---|---|---|
| 1 | 252 | 114 | 19 | 37 38 39 | 5.5 5.5 6.1 |
| 2 | 211 | 121 | 43 | 37 38 40 | 6.1 6.2 6.9 |
| 4 | 224 | 136 | 57 | 37 38 40 | 6.3 6.3 6.7 |
| 8 | 352 | 249 | 136 | 37 38 40 | 8.8 8.1 8.3 |

At np = 8 the efficiency cores are in use. The PC setup (block extraction,
products, PtAP) roughly halves from np = 1 to np = 2, but the solve does not.
