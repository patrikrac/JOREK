#!/usr/bin/env python3
"""Write a JOREK namelist for one physics-PC scaling case.

    mknml.py <arm> <n_flux> <n_tht> <out_file> [key=value ...]
    mknml.py nodes <arm> <n_flux> <n_tht>     # nodes the case needs (n_nodes_max check)
    mknml.py petscopts <arm>                  # PETSc options the arm adds (pc_case.sh)

Starts from the arm's base namelist -- namelist/model199/intear_island_demo
(the committed benchmark), or inxflow_shaped_pcbench for the sf_* arms --
applies the arm's physics-PC flags, the mesh, and a short tstep ramp, then any
extra key=value overrides. Existing keys are replaced in place; new keys are
inserted before the closing '/'. A new key that is not a physics_pc_* flag is
refused, so a typo fails loudly instead of being ignored by the namelist read.

Environment:
  PCS_BASE     base namelist (default: the arm's, see above)
  PCS_TSTEP_N  tstep ramp   (default: 1.d-1,1.d0,1.d1; sf_* arms 1.d-3,1.d-2,1.d-1,1.d0)
  PCS_NSTEP_N  steps per tstep (default: 3,3,3; sf_* arms 2,2,3,3)
  PCS_NOUT     restart/field output every N steps (default: 1000)
  PCS_N_RADIAL / PCS_N_POL   initial equilibrium grid (default: n_flux+10, n_tht;
               sf_* arms 2 n_flux - 1, 2 n_tht, the shaped case's own ratio)
  PCS_RESTART  if set, the case restarts (restart = .t.) from that file
  PCS_DROP_KEYS  comma list of base-namelist keys to comment out (binary from
               a branch that lacks them, e.g. numerics_develop)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))

ARMS = {
    # JOREK's default PETSc preconditioner (fieldsplit per harmonic + MUMPS)
    'jorek': {'use_physics_pc': '.f.'},
    # ... rebuilt at EVERY step (iter_precon = 0), so a rise of its in the
    # nonlinear phase is the per-harmonic PC losing the mode coupling, not a
    # stale factorisation
    'jorek_fresh': {'use_physics_pc': '.f.', 'iter_precon': '0'},
}

# The physics PC (use_physics_pc, mod_petsc_pc_sf*) on the reference physics
# case, inxflow_shaped_pcbench, ramped from a fresh equilibrium up to tstep 1.
# Every block has two backends: gmg (production) and lu (the exact
# reference). The multigrid configuration is fixed in mod_petsc_pc_sf_solver
# (smoothers, V-cycle shapes, line overlap, axis block), not in the namelist.
# The namelist defaults are the production configuration (wpj + mode split +
# gmg everywhere); the arms below set every SF flag they depend on, so each
# arm means the same whatever the defaults. The arms without an S_uu suffix
# run "schur", the _w arms "w" = B_22 + W; the arms without _ms run the
# global (not mode-split) operators. The production arm is sf_gmg_wpj_ms.
SF_BASE = 'inxflow_shaped_pcbench'
SF_COMMON = {
    'use_physics_pc': '.t.',
    'eliminate_boundary_dofs': '.t.',
    'physics_pc_sf_suu': '"schur"',
    'physics_pc_sf_mode_split': '.f.',
    'physics_pc_sf_rtol': '1.d-1',
}
ARMS['sf_gmg'] = dict(SF_COMMON, **{
    'physics_pc_sf_pair_psi': '"gmg"', 'physics_pc_sf_pair_w': '"gmg"',
    'physics_pc_sf_rho': '"gmg"', 'physics_pc_sf_T': '"gmg"',
})
ARMS['sf_lu'] = dict(SF_COMMON, **{
    'physics_pc_sf_pair_psi': '"lu"', 'physics_pc_sf_pair_w': '"lu"',
    'physics_pc_sf_rho': '"lu"', 'physics_pc_sf_T': '"lu"',
})
ARMS['sf_gmg_w'] = dict(ARMS['sf_gmg'], physics_pc_sf_suu='"w"')
ARMS['sf_lu_w'] = dict(ARMS['sf_lu'], physics_pc_sf_suu='"w"')
# pair_w mixed (mod_petsc_pc_sf_mixed): B_22 + W with W's psi-channel terms
# taken out and the channel put back through explicit j (wj: psi by the lumped
# mass) or psi and j (wpj: the small-flow psi row). LU on pair_w only, as the
# gate before any multigrid.
for _v in ('wj', 'wpj'):
    ARMS['sf_lu_' + _v] = dict(ARMS['sf_lu'], physics_pc_sf_suu='"%s"' % _v)
# ... and on the multigrid: pair_w alone (sf_lugw_*: everything else on its
# LU, isolating the V-cycle) and the whole path (sf_gmg_*)
ARMS['sf_lugw_wpj'] = dict(ARMS['sf_lu_wpj'], physics_pc_sf_pair_w='"gmg"')
ARMS['sf_gmg_wpj'] = dict(ARMS['sf_gmg'], physics_pc_sf_suu='"wpj"')
# Cross-|n| couplings kept in the SF blocks (physics_pc_sf_harm_couple):
# hc1 = |n| groups at most 1 apart, hcall = all; fixed for the run. The LU arm
# with all couplings is the exact-block floor of the coupled path.
for _b, _k in (('hc1', '1'), ('hcall', '-1')):
    ARMS['sf_gmg_wpj_' + _b] = dict(ARMS['sf_gmg_wpj'], physics_pc_sf_harm_couple=_k,
                                    physics_pc_sf_cross_weights='.t.')
    ARMS['sf_lu_wpj_' + _b] = dict(ARMS['sf_lu_wpj'], physics_pc_sf_harm_couple=_k,
                                   physics_pc_sf_cross_weights='.t.')
# The wpj corrector (physics_pc_sf_corrector): the default (auto) is Chacon
# Eq. (17), (psi, j) read off pair_w; this arm keeps Eq. (16), the second
# pair_psi solve, for the A/B comparison.
ARMS['sf_gmg_wpj_eq16'] = dict(ARMS['sf_gmg_wpj'], physics_pc_sf_corrector='0')
# JOREK's default PC on the same case and ramp
ARMS['sf_jorek'] = {'use_physics_pc': '.f.'}
# Full-system direct solve: one MUMPS LU of the whole coupled Jacobian,
# refactorised every step (iter_precon = 0), so FGMRES only checks it (1 it).
ARMS['sf_direct'] = {'use_physics_pc': '.f.', 'iter_precon': '0'}

# SF arms rebuilt at EVERY step (iter_precon = 0), like jorek_fresh: the
# like-for-like comparison of per-step setup + solve against JOREK's PC.
# The cross-|n| weight report is a diagnostic pass over A: off in these timing arms.
ARMS['sf_gmg_wpj_fresh'] = dict(ARMS['sf_gmg_wpj'], iter_precon='0',
                                physics_pc_sf_cross_weights='.f.')
ARMS['sf_gmg_wpj_hcall_fresh'] = dict(ARMS['sf_gmg_wpj_hcall'], iter_precon='0',
                                      physics_pc_sf_cross_weights='.f.')
# ... and the SF solvers on one |n| family per rank, the families concurrent
# (physics_pc_sf_mode_split; np >= (n_tor+1)/2). With a cross-|n| band (_hc1,
# _hcall: the same band on every operator, W included) the block solves become
# FGMRES over all families, the family solvers as block-Jacobi preconditioner.
ARMS['sf_gmg_wpj_ms'] = dict(ARMS['sf_gmg_wpj'], physics_pc_sf_mode_split='.t.')
ARMS['sf_gmg_wpj_fresh_ms'] = dict(ARMS['sf_gmg_wpj_fresh'], physics_pc_sf_mode_split='.t.')
for _b in ('hc1', 'hcall'):
    ARMS['sf_gmg_wpj_ms_' + _b] = dict(ARMS['sf_gmg_wpj_' + _b], physics_pc_sf_mode_split='.t.',
                                       physics_pc_sf_cross_weights='.f.')
    ARMS['sf_gmg_wpj_fresh_ms_' + _b] = dict(ARMS['sf_gmg_wpj_ms_' + _b], iter_precon='0')

# JOREK's PC with each mode family on its own sub-communicator, so the family
# blocks are factorised and solved concurrently instead of one after another
# (-jorek_pc_mode_split, branch numerics_develop only; needs np >= n_mode_families).
ARMS['jorek_fresh_ms'] = dict(ARMS['jorek_fresh'])

# PETSc options an arm needs besides its namelist (`mknml.py petscopts <arm>`).
ARM_PETSC_OPTS = {'sf_direct': '-jorek_pc_full_lu',
                  'jorek_fresh_ms': '-jorek_pc_mode_split'}


def is_sf(arm):
    return arm.startswith('sf_')


def grids(arm, n_flux, n_tht):
    """(n_radial, n_pol) of the initial equilibrium grid for this arm."""
    if is_sf(arm):
        return (int(os.environ.get('PCS_N_RADIAL', 2 * n_flux - 1)),
                int(os.environ.get('PCS_N_POL', 2 * n_tht)))
    return (int(os.environ.get('PCS_N_RADIAL', n_flux + 10)),
            int(os.environ.get('PCS_N_POL', n_tht)))


def gmg_levels(n_flux, n_tht):
    """Number of GMG levels the C1 hierarchy will build (mod_petsc_pc_gmg)."""
    ni, nj, nlev = n_flux, n_tht, 1
    while nlev < 6 and (ni - 1) % 2 == 0 and nj % 2 == 0 and nj >= 4:
        ni, nj, nlev = (ni - 1) // 2 + 1, nj // 2, nlev + 1
    return nlev


def main(argv):
    if len(argv) == 5 and argv[1] == 'nodes':
        nr, npol = grids(argv[2], int(argv[3]), int(argv[4]))
        print(max(nr * npol, int(argv[3]) * int(argv[4])))
        return
    if len(argv) == 3 and argv[1] == 'petscopts':
        print(ARM_PETSC_OPTS.get(argv[2], ''))
        return
    if len(argv) < 5:
        sys.exit(__doc__)
    arm, n_flux, n_tht, out = argv[1], int(argv[2]), int(argv[3]), argv[4]
    if arm not in ARMS:
        sys.exit('unknown arm %r; choose from %s' % (arm, ', '.join(sorted(ARMS))))
    if arm.startswith('sf_gmg') and gmg_levels(n_flux, n_tht) < 3:
        sys.exit('mesh %dx%d gives fewer than 3 GMG levels: use (n_flux-1) divisible '
                 'by 4 or more powers of 2 and n_tht divisible by 8 (e.g. 81x32, '
                 '161x64, 321x128)' % (n_flux, n_tht))
    if arm.startswith('sf_gmg') and gmg_levels(n_flux, n_tht) < 4:
        # a shallow hierarchy leaves a large coarse level, solved by one
        # sequential LU on every rank that owns part of it (61x96 stopped at
        # 3 levels with an 8646-row coarse LU). Full depth needs
        # n_flux = 2^L k + 1 and n_tht = 2^L m.
        print('mknml.py: WARNING: %dx%d gives only %d GMG levels; prefer n_flux = 2^L k + 1, '
              'n_tht = 2^L m with L >= 3' % (n_flux, n_tht, gmg_levels(n_flux, n_tht)),
              file=sys.stderr)

    ov = dict(ARMS[arm])
    ov['n_flux'] = str(n_flux)
    ov['n_tht'] = str(n_tht)
    # The equilibrium is computed on the INITIAL polar grid (n_radial, n_pol)
    # and only then aligned to the flux-surface grid (n_flux, n_tht). Scaling
    # only the latter leaves the equilibrium on the base namelist's 51x16, and
    # the post-equilibrium check fails on every larger mesh. Keep the base
    # namelist's ratio: n_radial = n_flux + 10, n_pol = n_tht.
    nr, npol = grids(arm, n_flux, n_tht)
    ov['n_radial'] = str(nr)
    ov['n_pol'] = str(npol)
    ov['tstep_n'] = os.environ.get('PCS_TSTEP_N', '1.d-3,1.d-2,1.d-1,1.d0' if is_sf(arm) else '1.d-1,1.d0,1.d1')
    ov['nstep_n'] = os.environ.get('PCS_NSTEP_N', '2,2,3,3' if is_sf(arm) else '3,3,3')
    ov['nout'] = os.environ.get('PCS_NOUT', '1000')   # 1000: no field output during timing
    if os.environ.get('PCS_RESTART'):         # pc_case.sh copied the restart file in
        ov['restart'] = '.t.'
    for a in argv[5:]:
        k, v = a.split('=', 1)
        ov[k.strip()] = v.strip()

    base = os.environ.get('PCS_BASE', os.path.join(REPO, 'namelist', 'model199',
                                                   SF_BASE if is_sf(arm) else 'intear_island_demo'))
    lines = open(base).read().split('\n')
    # keys a binary from another branch does not know (its namelist read would
    # abort): commented out, so they must be at their defaults there
    drop = [k.strip() for k in os.environ.get('PCS_DROP_KEYS', '').split(',') if k.strip()]
    out_lines = []
    for line in lines:
        st = line.strip()
        if st.startswith('!') or '=' not in st:
            out_lines.append(line)
            continue
        k = st.split('=')[0].strip()
        if k in drop:                         # ... and the arm's value for it
            out_lines.append('!' + line + '   ! PCS_DROP_KEYS')
            ov.pop(k, None)
            continue
        if k in ov:
            out_lines.append(' %s = %s' % (k, ov.pop(k)))
        else:
            out_lines.append(line)
    if ov:
        bad = [k for k in ov if not k.startswith('physics_pc')]
        if bad:
            sys.exit('keys not in the base namelist (typo?): %s' % bad)
        last = max(i for i, l in enumerate(out_lines) if l.strip() == '/')
        for k, v in ov.items():
            out_lines.insert(last, ' %s = %s' % (k, v))
    with open(out, 'w') as f:
        f.write('\n'.join(out_lines))


if __name__ == '__main__':
    main(sys.argv)
