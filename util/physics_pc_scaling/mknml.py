#!/usr/bin/env python3
"""Write a JOREK namelist for one physics-PC scaling case.

    mknml.py <arm> <n_flux> <n_tht> <out_file> [key=value ...]
    mknml.py nodes <arm> <n_flux> <n_tht>     # nodes the case needs (n_nodes_max check)

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
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))

# Physics-PC flags common to every SFM2 arm (serial study, docs/physics_pc
# workstream D, recommended configuration of 2026-09-14).
SFM2_COMMON = {
    'use_physics_pc': '.t.',
    'physics_pc_multi_step': '.t.',
    'physics_pc_wave_schur': '.t.',
    'physics_pc_schur_channels': '1',
    'physics_pc_schur_variant': '"SFM2"',
    'physics_pc_schur_massinv': '7',          # FSAI-0; 2 doubles the pair_w V-cycles
    'physics_pc_schur_pairinv': '2',
    'physics_pc_pair_scale': '1',
    'physics_pc_harm_split': '1',
    'physics_pc_lean_setup': '3',
    'physics_pc_mass_split': '1',
    'physics_pc_pair_rtol': '1.d-1',
    # PC reuse: JOREK rebuilds when its_n + its_{n-1} > 2*iter_precon. The
    # base namelists' 22 suits the direct-solver default PC (1-5 its); SFM2
    # needs ~40-50 at tstep 10 and would rebuild every step, although reusing
    # the PC within a tstep block left every iteration count unchanged
    # (81x32, np 4: 2 rebuilds instead of 4). A tstep change still rebuilds.
    'iter_precon': '60',
}

ARMS = {
    # JOREK's default PETSc preconditioner (fieldsplit per harmonic + MUMPS)
    'jorek': {'use_physics_pc': '.f.'},
    # SFM2 with direct (MUMPS LU) inner solves: the approximation-quality
    # reference; factorisations are redone at every PC rebuild
    'sfm2_lu': dict(SFM2_COMMON, **{
        'physics_pc_w_gmg': '0',
        'physics_pc_suu_shell': '0',
        'physics_pc_psi_schur': '0',
        'physics_pc_rhot_gmg': '0',
    }),
    # SFM2, factorisation-free: C1 GMG on pair_w (matrix-free fine operator with
    # the exact mass), eta-Schur + GMG on pair_psi, GMG on rho/T. Every
    # hierarchy uses the stage-D13 axis treatment: rings 0..3 of every level
    # in one axis block (solved by MUMPS) and coarse levels without the
    # Dirichlet u/b DOFs of the boundary ring. A FIXED ring count, not the
    # automatic rings-below-r*dtheta/dr=1 (-1): that block holds ~n_tht/(2 pi)
    # rings, so its factor grows faster than N (438 MB at 49x64, tens of GB at
    # 641x256), while k = 3 was the fastest setting on both benchmarks.
    'sfm2_gmg': dict(SFM2_COMMON, **{
        'physics_pc_w_gmg': '2',
        'physics_pc_suu_shell': '2',
        'physics_pc_gmg_smoother': '5',
        'physics_pc_psi_schur': '3',
        'physics_pc_psi_outer': '10',
        'physics_pc_rhot_gmg': '10',
        'physics_pc_rhot_gmg_smoother': '5',
        'physics_pc_gmg_axis_rings': '3',
        'physics_pc_gmg_bnd_drop': '1',
    }),
}
# The same without the axis treatment (stage D12, commit 3cafa9f46): radial
# lines with the axis ring alone as a block, Dirichlet DOFs kept on the coarse
# levels. The comparison arm for the D13 changes.
# Stage Q (2026-09-15): the two components that did not scale on the first
# cluster series, each on its own and together, on top of sfm2_gmg.
#   mass  = the exact-mass solves by Chebyshev + additive Schwarz instead of
#           MUMPS with a centralized RHS (PhysPC_MjSolve: 90 s at np 1, 312 s
#           at np 32 on 161x64). Iteration counts are unchanged; on the laptop
#           at np 4 it is SLOWER than MUMPS, which is the point of the test.
#   smop  = the fine GMG smoother on the assembled operator, so the exact
#           operator is applied only for residuals: 4.8x fewer mass solves,
#           +16% pair_w V-cycles and +1..2 outer its (41x16 / 81x32, np 4).
ARMS['sfm2_gmg_mass'] = dict(ARMS['sfm2_gmg'], **{
    'physics_pc_mass_solver': '2',
})
ARMS['sfm2_gmg_smop'] = dict(ARMS['sfm2_gmg'], **{
    'physics_pc_gmg_smooth_op': '1',
})
ARMS['sfm2_gmg_q'] = dict(ARMS['sfm2_gmg'], **{
    'physics_pc_mass_solver': '2',
    'physics_pc_gmg_smooth_op': '1',
})
# Motivation study (nonlinear phase, `pc_study.sh nonlinear`):
#   jorek_fresh  = JOREK's default PC rebuilt at EVERY step (iter_precon = 0),
#                  so a rise of its in the nonlinear phase is the per-harmonic
#                  PC losing the mode coupling, not a stale factorisation.
#   sfm2_*_hs0   = keeping the cross-|n| entries (harm_split = 0). They are
#                  exact zeros at equilibrium but NOT once the island is
#                  nonlinear: at saturation (41x16, tstep 1000) ||cross||/||A||
#                  is 0.98 in B_24 and 0.32 in B_21, and with harm_split = 1
#                  both sfm2_lu and sfm2_gmg stall at 400 its while sfm2_lu_hs0
#                  converges in a flat 71 its per step.
ARMS['jorek_fresh'] = dict(ARMS['jorek'], **{'iter_precon': '0'})
ARMS['sfm2_lu_hs0'] = dict(ARMS['sfm2_lu'], **{'physics_pc_harm_split': '0'})
ARMS['sfm2_gmg_hs0'] = dict(ARMS['sfm2_gmg'], **{'physics_pc_harm_split': '0'})
ARMS['sfm2_gmg_d12'] = dict(ARMS['sfm2_gmg'], **{
    'physics_pc_gmg_axis_rings': '0',
    'physics_pc_gmg_bnd_drop': '0',
})


# The production SF path (mod_petsc_pc_sf*, workstream H) on the reference
# physics case, inxflow_shaped_pcbench. Its composed force operator W is valid
# up to tstep ~ 1 (Chacon 2025 Eq. 17-19; at tstep 10 even the all-LU SF path
# diverges), so these arms ramp from a fresh equilibrium up to tstep 1 only.
# Every block has two backends: gmg (production) and lu (the exact
# reference). The multigrid configuration is fixed in mod_petsc_pc_sf_solver
# (smoothers, line overlap, axis block), not in the namelist.
SF_BASE = 'inxflow_shaped_pcbench'
SF_COMMON = {
    'use_physics_pc': '.t.',
    'physics_pc_monolithic': '.f.',
    'physics_pc_reduced_pde': '.f.',
    'eliminate_boundary_dofs': '.t.',
    'physics_pc_force_operator': '1',
    'physics_pc_sf': '.t.',
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
# JOREK's default PC on the same case and ramp
ARMS['sf_jorek'] = {'use_physics_pc': '.f.'}


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
    if len(argv) < 5:
        sys.exit(__doc__)
    arm, n_flux, n_tht, out = argv[1], int(argv[2]), int(argv[3]), argv[4]
    if arm not in ARMS:
        sys.exit('unknown arm %r; choose from %s' % (arm, ', '.join(sorted(ARMS))))
    if (arm.startswith('sfm2_gmg') or arm == 'sf_gmg') and gmg_levels(n_flux, n_tht) < 3:
        sys.exit('mesh %dx%d gives fewer than 3 GMG levels: use (n_flux-1) divisible '
                 'by 4 or more powers of 2 and n_tht divisible by 8 (e.g. 81x32, '
                 '161x64, 321x128)' % (n_flux, n_tht))
    if arm == 'sf_gmg' and gmg_levels(n_flux, n_tht) < 4:
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
    out_lines = []
    for line in lines:
        st = line.strip()
        if st.startswith('!') or '=' not in st:
            out_lines.append(line)
            continue
        k = st.split('=')[0].strip()
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
