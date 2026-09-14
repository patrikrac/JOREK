#!/usr/bin/env python3
"""Write a JOREK namelist for one physics-PC scaling case.

    mknml.py <arm> <n_flux> <n_tht> <out_file> [key=value ...]

Starts from namelist/model199/intear_island_demo (the committed benchmark),
applies the arm's physics-PC flags, the mesh, and a short tstep ramp, then any
extra key=value overrides. Existing keys are replaced in place; new keys are
inserted before the closing '/'. A new key that is not a physics_pc_* flag is
refused, so a typo fails loudly instead of being ignored by the namelist read.

Environment:
  PCS_BASE     base namelist (default: <repo>/namelist/model199/intear_island_demo)
  PCS_TSTEP_N  tstep ramp   (default: 1.d-1,1.d0,1.d1)
  PCS_NSTEP_N  steps per tstep (default: 3,3,3)
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
    # the exact mass), eta-Schur + GMG on pair_psi, GMG on rho/T
    'sfm2_gmg': dict(SFM2_COMMON, **{
        'physics_pc_w_gmg': '2',
        'physics_pc_suu_shell': '2',
        'physics_pc_gmg_smoother': '5',
        'physics_pc_psi_schur': '3',
        'physics_pc_psi_outer': '10',
        'physics_pc_rhot_gmg': '10',
        'physics_pc_rhot_gmg_smoother': '5',
    }),
}


def gmg_levels(n_flux, n_tht):
    """Number of GMG levels the C1 hierarchy will build (mod_petsc_pc_gmg)."""
    ni, nj, nlev = n_flux, n_tht, 1
    while nlev < 6 and (ni - 1) % 2 == 0 and nj % 2 == 0 and nj >= 4:
        ni, nj, nlev = (ni - 1) // 2 + 1, nj // 2, nlev + 1
    return nlev


def main(argv):
    if len(argv) < 5:
        sys.exit(__doc__)
    arm, n_flux, n_tht, out = argv[1], int(argv[2]), int(argv[3]), argv[4]
    if arm not in ARMS:
        sys.exit('unknown arm %r; choose from %s' % (arm, ', '.join(sorted(ARMS))))
    if arm.startswith('sfm2_gmg') and gmg_levels(n_flux, n_tht) < 3:
        sys.exit('mesh %dx%d gives fewer than 3 GMG levels: use (n_flux-1) divisible '
                 'by 4 or more powers of 2 and n_tht divisible by 8 (e.g. 81x32, '
                 '161x64, 321x128)' % (n_flux, n_tht))

    ov = dict(ARMS[arm])
    ov['n_flux'] = str(n_flux)
    ov['n_tht'] = str(n_tht)
    ov['tstep_n'] = os.environ.get('PCS_TSTEP_N', '1.d-1,1.d0,1.d1')
    ov['nstep_n'] = os.environ.get('PCS_NSTEP_N', '3,3,3')
    ov['nout'] = '1000'                       # no field output during timing
    for a in argv[5:]:
        k, v = a.split('=', 1)
        ov[k.strip()] = v.strip()

    base = os.environ.get('PCS_BASE', os.path.join(REPO, 'namelist', 'model199',
                                                   'intear_island_demo'))
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
