#!/usr/bin/env python3
"""Collect physics-PC scaling cases into one TSV table.

    collect.py <root> [<root> ...] > results.tsv

Every directory below a root that holds a `case.meta` (written by pc_case.sh)
and a `log` becomes one row. Times of the -log_view events are the MAX over
ranks, summed over logging stages. Missing quantities are left empty, so a
crashed or unfinished case still shows up (status column).
"""
import os
import re
import sys

EVENTS = [
    'PhysPC_Extract', 'PhysPC_BuildSuu', 'PhysPC_FactPJ', 'PhysPC_FactW',
    'PhysPC_FactRhoT', 'GMG_PtAP', 'PhysPC_Apply', 'PhysPC_SolvePJ', 'PhysPC_SolveW',
    'PhysPC_SolveRhoT', 'PhysPC_ShellMult', 'PhysPC_MjSolve', 'PhysPC_PsiPC',
    'GMG_VCycle', 'GMG_Smooth0', 'GMG_Lines', 'GMG_AxSolve', 'GMG_Coarse', 'GMG2_VCycle', 'GMG3_VCycle',
    'GMG4_VCycle', 'MatMult', 'KSPSolve',
]
COLS = (['case', 'arm', 'n_flux', 'n_tht', 'np', 'omp', 'cores', 'nodes', 'n_tor', 'n_period',
         'status', 'ndof', 'wall_s',
         'step_s_sum', 'setup_s_sum', 'solve_s_sum', 'outer_its', 'outer_sum',
         'pw_cycles_mean', 'pp_its_mean', 'rt_its_mean', 'mem_max_total_GB',
         'mem_max_rank_GB'] + ['t_' + e for e in EVENTS])

RE_FLOAT = r'([-+]?\d+(?:\.\d*)?(?:[eEdD][-+]?\d+)?)'


def fnum(s):
    return float(s.replace('D', 'E').replace('d', 'e'))


def parse_meta(path):
    meta = {}
    for line in open(path):
        if '=' in line:
            k, v = line.strip().split('=', 1)
            meta[k] = v
    return meta


def parse_log(path, row):
    txt = open(path, errors='replace').read()
    outer = [int(m) for m in re.findall(r'\[PETSc\] outer iterations:\s+(\d+)', txt)]
    row['outer_its'] = ' '.join(map(str, outer))
    row['outer_sum'] = sum(outer) if outer else ''

    def means(tag):
        v = re.findall(r'\[Physics PC\] ' + tag + r' inner its: solves = \d+, sum = \d+, mean = \s*'
                       + RE_FLOAT, txt)
        return ' '.join('%.2f' % fnum(x) for x in v)
    row['pw_cycles_mean'] = means('pair_w')
    row['pp_its_mean'] = means('pair_psi')
    row['rt_its_mean'] = means('rho/T')

    def tsum(pat):
        v = re.findall(pat + r'\s*' + RE_FLOAT, txt)
        return '%.2f' % sum(fnum(x) for x in v) if v else ''
    row['step_s_sum'] = tsum(r'Elapsed time ITERATION :')
    row['setup_s_sum'] = tsum(r'\[PETSc\] Elapsed time in solver setup :')
    row['solve_s_sum'] = tsum(r'\[PETSc\] Elapsed time in solve :')
    # compile-time toroidal settings, from the log (the namelist cannot set them)
    for k in ('n_tor', 'n_period'):
        m = re.search(r'^\s*' + k + r'\s*=\s*(\d+)', txt, re.M)
        row[k] = m.group(1) if m else ''
    m = re.search(r'create_matrix: BAIJ (\d+)x\d+', txt)
    row['ndof'] = m.group(1) if m else ''
    # -memory_view: "Maximum (over computational time) process memory: total X max Y min Z"
    m = re.search(r'Maximum \(over computational time\) process memory:\s+total\s+'
                  + RE_FLOAT + r'\s+max\s+' + RE_FLOAT, txt)
    if m:
        row['mem_max_total_GB'] = '%.3f' % (fnum(m.group(1)) / 1e9)
        row['mem_max_rank_GB'] = '%.3f' % (fnum(m.group(2)) / 1e9)
    if 'PETSC ERROR' in txt or 'FATAL' in txt:
        row['status'] = 'error'
    elif re.search(r'Elapsed time ITERATION', txt) and outer:
        row['status'] = 'ok'
    else:
        row['status'] = 'incomplete'


def parse_prof(path, row):
    tot = {}
    wall = None
    for line in open(path, errors='replace'):
        f = line.split()
        if not f:
            continue
        if line.startswith('Time (sec):') and len(f) >= 3:
            wall = fnum(f[2])
        if f[0] in EVENTS and len(f) > 4:
            try:
                tot[f[0]] = tot.get(f[0], 0.0) + fnum(f[3])
            except ValueError:
                pass
    if wall is not None:
        row['wall_s'] = '%.2f' % wall
    for e in EVENTS:
        if e in tot:
            row['t_' + e] = '%.3f' % tot[e]


def main(roots):
    print('\t'.join(COLS))
    rows = []
    for root in roots:
        for d, _, files in os.walk(root):
            if 'case.meta' not in files:
                continue
            meta = parse_meta(os.path.join(d, 'case.meta'))
            row = {c: '' for c in COLS}
            row.update({k: meta.get(k, '') for k in ('arm', 'n_flux', 'n_tht', 'np', 'omp', 'nodes')})
            if row['np'] and row['omp']:
                row['cores'] = int(row['np']) * int(row['omp'])
            row['case'] = os.path.basename(d)
            row['status'] = 'no-log'
            if 'log' in files:
                parse_log(os.path.join(d, 'log'), row)
            if 'prof.txt' in files:
                parse_prof(os.path.join(d, 'prof.txt'), row)
            if not row['wall_s'] and meta.get('wall_s'):
                row['wall_s'] = meta['wall_s']
            rows.append(row)
    rows.sort(key=lambda r: (r['arm'], int(r['n_flux'] or 0), int(r['omp'] or 0), int(r['np'] or 0)))
    for r in rows:
        print('\t'.join(str(r[c]) for c in COLS))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
