#!/usr/bin/env python3
"""Motivation figures for the scalable SFM2 preconditioner.

    plot_motivation.py [--strong ROOT] [--weak ROOT] [--nonlinear ROOT]
                       [--coupling ROOT] [--out DIR]

Each ROOT is a study directory written by pc_study.sh (collect.py is re-run on
it, so results.tsv and every case's steps.tsv are current). Figures are written
as PDF and PNG to DIR (default: <first root>/figures):

  F1_lu_strong      JOREK's LU PC under strong scaling: factorisation and
                    triangular-solve time, parallel efficiency   (--strong)
  F1_lu_weak        LU vs SFM2-GMG at fixed DOFs per rank: time per rebuild,
                    time per PC application, memory per rank     (--weak)
  F2_nonlinear_lu   outer iterations of JOREK's PC through the island's linear
                    growth and saturation, with/without PC reuse (--nonlinear)
  F3_nonlinear_all  the same for every arm, absolute and normalised by each
                    arm's linear-phase count                     (--nonlinear)
  F5_mode_coupling  ||cross-|n| part|| / ||block|| of the SFM2 blocks through
                    the trajectory, next to JOREK's iterations   (--coupling)
                    (the --coupling root holds one-step probes from the
                    nonlinear run's restarts; its sfm2_gmg_hs0 probes are
                    also drawn as points in F3)
  F4_components     strong scaling of each SFM2 part, LU inner solves vs the
                    scalable ones, and their parallel efficiency (--strong)
  F4b_pair_w        what pair_w's cost is made of vs cores: the GMG smoother,
                    and the exact-mass and axis direct solves    (--strong)

Needs matplotlib (python3 -m venv .venv && .venv/bin/pip install matplotlib).
"""
import argparse
import csv
import math
import os
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# --- style: one look for every figure ---------------------------------------
SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
# Colour follows the arm in every figure (categorical slots in fixed order).
ARM_STYLE = {
    'sfm2_gmg':    dict(color='#2a78d6', marker='o', ls='-',  label='SFM2, scalable inner solves'),
    'jorek':       dict(color='#eb6834', marker='s', ls='-',  label='JOREK LU PC (reused)'),
    'sfm2_lu':     dict(color='#1baf7a', marker='^', ls='-',  label='SFM2, LU inner solves'),
    'jorek_fresh': dict(color='#e34948', marker='D', ls='--', label='JOREK LU PC (rebuilt every step)'),
    'sfm2_gmg_q':  dict(color='#4a3aa7', marker='v', ls='-',  label='SFM2, scalable + Schwarz mass'),
    'sfm2_lu_hs0': dict(color='#1baf7a', marker='^', ls='-',  label='SFM2, LU inner solves'),
    'sfm2_gmg_hs0': dict(color='#2a78d6', marker='o', ls='-', label='SFM2, scalable inner solves'),
}
# In the nonlinear phase the cross-|n| blocks matter: the harm_split = 0 arms
# are "SFM2" there, and the harm_split = 1 arm is the variant that drops the
# mode coupling (the same approximation JOREK's per-harmonic PC makes).
NL_STYLE = dict(ARM_STYLE, **{
    'sfm2_lu':  dict(color='#1baf7a', marker='^', ls=':', label='SFM2, LU, mode coupling dropped'),
    'sfm2_gmg': dict(color='#2a78d6', marker='o', ls=':', label='SFM2, scalable, mode coupling dropped'),
})
COMPONENTS = [  # (title, setup events, solve events)
    ('D$_\\rho$, D$_T$', ['PhysPC_FactRhoT'], ['PhysPC_SolveRhoT']),
    ('pair_psi', ['PhysPC_FactPJ'], ['PhysPC_SolvePJ']),
    ('pair_w', ['PhysPC_FactW'], ['PhysPC_SolveW']),
    ('Schur assembly (shared)', ['PhysPC_Extract', 'PhysPC_BuildSuu'], []),
]

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'lines.linewidth': 2.0, 'lines.markersize': 6,
    'legend.frameon': False, 'legend.fontsize': 9,
    'text.color': INK,
})


def style(arm):
    return ARM_STYLE.get(arm, dict(color=MUTED, marker='x', ls=':', label=arm))


# --- data ---------------------------------------------------------------------
def load(root, ok_only=True):
    """Rows of <root>/results.tsv after re-running collect.py. The nonlinear
    figures keep unfinished runs too: a run that stalls is part of the result."""
    tsv = os.path.join(root, 'results.tsv')
    with open(tsv, 'w') as fh:
        subprocess.run([sys.executable, os.path.join(HERE, 'collect.py'), root], stdout=fh, check=True)
    rows = list(csv.DictReader(open(tsv), delimiter='\t'))
    for r in rows:
        r['_dir'] = find_case(root, r['case'])
    return [r for r in rows if r['_dir'] and (r['status'] == 'ok' or not ok_only)]


def find_case(root, case):
    for d, _, files in os.walk(root):
        if os.path.basename(d) == case and 'case.meta' in files:
            return d
    return None


def f(r, key, default=0.0):
    v = r.get(key, '')
    return float(v) if v not in ('', None) else default


def ev(r, events):
    return sum(f(r, 't_' + e) for e in events)


def steps(r):
    p = os.path.join(r['_dir'], 'steps.tsv')
    if not os.path.exists(p):
        return []
    out = []
    for s in csv.DictReader(open(p), delimiter='\t'):
        for k in list(s):
            try:
                s[k] = float(s[k])
            except (ValueError, TypeError):
                pass
        out.append(s)
    return out


def by_arm(rows, arms=None):
    d = {}
    for r in rows:
        if arms is None or r['arm'] in arms:
            d.setdefault(r['arm'], []).append(r)
    return d


def cores(r):
    return int(r['np']) * int(r['omp'] or 1)


# --- helpers --------------------------------------------------------------------
def log2_axis(ax, xs):
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_locator(FixedLocator(sorted(set(xs))))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: '%d' % v))


def ideal(ax, xs, t0, x0):
    xx = sorted(set(xs))
    ax.plot(xx, [t0 * x0 / x for x in xx], ls=':', lw=1.2, color=MUTED, zorder=1)


def footnote(fig, rows, extra=''):
    r = rows[0]
    host = open(os.path.join(r['_dir'], 'case.meta')).read() if r.get('_dir') else ''
    h = [l.split('=', 1)[1] for l in host.splitlines() if l.startswith('host=')]
    g = [l.split('=', 1)[1] for l in host.splitlines() if l.startswith('git=')]
    txt = 'intear_island_demo, n_tor=%s, n_period=%s, %s thread(s)/rank; host %s, commit %s. %s' % (
        r['n_tor'], r['n_period'], r['omp'], h[0] if h else '?', g[0] if g else '?', extra)
    fig.text(0.01, 0.005, txt, fontsize=7.5, color=MUTED, ha='left', va='bottom')


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, name + '.' + ext), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('[plot] ' + os.path.join(out, name) + '.{pdf,png}')


def end_label(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords='offset points', va='center',
                fontsize=8.5, color=INK2)


# --- F1: LU strong scaling --------------------------------------------------------
def fig_lu_strong(rows, out):
    lu = sorted(by_arm(rows, ['jorek']).get('jorek', []), key=cores)
    if len(lu) < 2:
        print('[plot] F1_lu_strong: needs >= 2 jorek cases in the strong root'); return
    xs = [cores(r) for r in lu]
    series = [
        ('factorisation, per rebuild',
         [(f(r, 't_MatLUFactorSym') + f(r, 't_MatLUFactorNum')) / max(f(r, 'rebuilds', 1), 1) for r in lu],
         '#eb6834', 's'),
        ('triangular solves, per iteration',
         [f(r, 't_MatSolve') / max(f(r, 'outer_sum', 1), 1) for r in lu], '#4a3aa7', 'o'),
    ]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2))
    for name, ys, c, m in series:
        a.plot(xs, ys, color=c, marker=m, label=name)
        ideal(a, xs, ys[0], xs[0])
        b.plot(xs, [ys[0] * xs[0] / (y * x) for x, y in zip(xs, ys)], color=c, marker=m, label=name)
    a.set_yscale('log'); log2_axis(a, xs); log2_axis(b, xs)
    a.set_xlabel('cores'); a.set_ylabel('time [s]')
    a.set_title('JOREK LU preconditioner, %sx%s mesh (%s DOFs)' % (lu[0]['n_flux'], lu[0]['n_tht'], lu[0]['ndof']))
    a.plot([], [], ls=':', color=MUTED, lw=1.2, label='ideal')
    a.legend(loc='best')
    b.axhline(1.0, ls=':', color=MUTED, lw=1.2)
    b.set_ylim(0, 1.1); b.set_xlabel('cores'); b.set_ylabel('parallel efficiency')
    b.set_title('Parallel efficiency')
    footnote(fig, lu)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out, 'F1_lu_strong')


# --- F1: weak scaling -----------------------------------------------------------------
def fig_lu_weak(rows, out):
    arms = by_arm(rows, ['jorek', 'sfm2_lu', 'sfm2_gmg'])
    if 'jorek' not in arms or len(arms['jorek']) < 2:
        print('[plot] F1_lu_weak: needs >= 2 jorek cases in the weak root'); return

    def setup(r):   # one PC build, all ranks busy
        if r['arm'] == 'jorek':
            v = f(r, 't_MatLUFactorSym') + f(r, 't_MatLUFactorNum')
        else:
            v = ev(r, [e for _, s, _ in COMPONENTS for e in s])
        return v / max(f(r, 'rebuilds', 1), 1)

    def apply(r):   # one outer iteration's preconditioner application
        if r['arm'] == 'jorek':
            v = f(r, 't_MatSolve')
        else:
            v = ev(r, [e for _, _, s in COMPONENTS for e in s])
        return v / max(f(r, 'outer_sum', 1), 1)

    panels = [('PC build (per rebuild)', setup, 'time [s]'),
              ('PC application (per outer iteration)', apply, 'time [s]'),
              ('Peak memory per rank', lambda r: f(r, 'mem_max_rank_GB'), 'memory [GB]')]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (title, fn, yl) in zip(axs, panels):
        for arm in ('jorek', 'sfm2_lu', 'sfm2_gmg'):
            rs = sorted(arms.get(arm, []), key=lambda r: int(r['ndof'] or 0))
            if not rs:
                continue
            xs = [int(r['ndof']) for r in rs]
            ys = [fn(r) for r in rs]
            st = style(arm)
            ax.plot(xs, ys, color=st['color'], marker=st['marker'], ls=st['ls'], label=st['label'])
            for x, y, r in zip(xs, ys, rs):
                if arm == 'jorek':
                    ax.annotate('np %s' % r['np'], (x, y), xytext=(0, 7), textcoords='offset points',
                                ha='center', fontsize=7.5, color=MUTED)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('DOFs, with DOFs per rank fixed (flat = ideal)'); ax.set_ylabel(yl); ax.set_title(title)
    axs[0].legend(loc='upper left')
    footnote(fig, arms['jorek'], 'Memory: -memory_view peak RSS; JOREK\'s MUMPS runs out-of-core (ICNTL(22)=1).')
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out, 'F1_lu_weak')


# --- nonlinear phase ---------------------------------------------------------------------
def phases(st):
    """(first step of the nonlinear phase, first saturated step) from W_mag(n=1)."""
    w = [s['wmag_nlast'] for s in st if isinstance(s['wmag_nlast'], float)]
    if not w:
        return None, None
    wmax = max(w)
    nl = next((s['step'] for s in st if isinstance(s['wmag_nlast'], float)
               and s['wmag_nlast'] > 1e-3 * wmax), None)
    sat = next((s['step'] for s in st if isinstance(s['wmag_nlast'], float)
                and s['wmag_nlast'] > 0.5 * wmax), None)
    return nl, sat


def shade(ax, nl, sat, last, label=False):
    if nl is None:
        return
    ax.axvspan(nl, sat or last, color='#eeedea', zorder=0, lw=0)
    if sat:
        ax.axvspan(sat, last, color='#e1e0d9', zorder=0, lw=0)
    if label:   # inside the axes, along the top edge
        x_lo = ax.get_xlim()[0]
        spans = [(x_lo, nl, 'linear'), (nl, sat or last, 'nonlinear')]
        if sat:
            spans.append((sat, last, 'saturated'))
        for x0, x1, t in spans:
            ax.text((x0 + x1) / 2, 0.98, t, ha='center', va='top', fontsize=8.5, color=INK2,
                    transform=ax.get_xaxis_transform())


def tstep_marks(ax, st):
    prev = None
    for s in st:
        if s['tstep'] != prev:
            ax.axvline(s['step'] - 0.5, color=MUTED, lw=0.8, ls='--', zorder=1)
            prev = s['tstep']


def tstep_labels(ax, st):
    prev = None
    for s in st:
        if s['tstep'] != prev:
            ax.annotate('Δt=%g' % s['tstep'], (s['step'] - 0.5, 0.03), xycoords=('data', 'axes fraction'),
                        xytext=(-2, 0), textcoords='offset points', fontsize=7.5, color=MUTED,
                        rotation=90, ha='right', va='bottom',
                        bbox=dict(fc=SURFACE, ec='none', pad=0.6))
            prev = s['tstep']


def fig_nonlinear_lu(rows, out):
    arms = {a: rs[0] for a, rs in by_arm(rows, ['jorek', 'jorek_fresh']).items()}
    if 'jorek' not in arms:
        print('[plot] F2_nonlinear_lu: needs a jorek case in the nonlinear root'); return
    ref = steps(arms['jorek'])
    nl, sat = phases(ref)
    last = ref[-1]['step']
    fig, (a, b, c) = plt.subplots(3, 1, figsize=(10, 8), sharex=True,
                                  gridspec_kw=dict(height_ratios=[1, 1.6, 1]))
    a.plot([s['step'] for s in ref], [s['wmag_nlast'] for s in ref], color=INK2, lw=1.6)
    a.set_yscale('log'); a.set_ylabel('W$_{mag}$(n=1)')
    a.set_title('The 2/1 island: linear growth to saturation')
    for arm in ('jorek', 'jorek_fresh'):
        if arm not in arms:
            continue
        st = steps(arms[arm]); sty = style(arm)
        xs = [s['step'] for s in st]
        b.plot(xs, [s['outer_its'] for s in st], color=sty['color'], ls=sty['ls'], lw=1.8,
               label=sty['label'], drawstyle='steps-mid')
        if arm == 'jorek':
            rb = [s['step'] for s in st if s['rebuild'] == 1]
            b.plot(rb, [0.3] * len(rb), ls='none', marker='|', ms=10, mew=1.5, color=sty['color'],
                   label='PC rebuild (reused run)', clip_on=False)
        cum, acc = [], 0.0
        for s in st:
            acc += (s['setup_s'] or 0) + (s['solve_s'] or 0)
            cum.append(acc)
        c.plot(xs, cum, color=sty['color'], ls=sty['ls'], label=sty['label'])
    b.set_ylim(bottom=0); b.set_ylabel('outer FGMRES iterations')
    b.set_title('Iterations per time step')
    b.legend(loc='upper left')
    c.set_ylabel('cumulative solver time [s]'); c.set_xlabel('time step')
    c.set_title('Linear-solver cost (PC build + Krylov solve)')
    for ax in (a, b, c):
        tstep_marks(ax, ref)
        shade(ax, nl, sat, last)
    tstep_labels(a, ref)
    shade(a, nl, sat, last, label=True)
    footnote(fig, list(arms.values()), 'Shading: W_mag(n=1) above 1e-3 (nonlinear) and 0.5 (saturated) of its maximum.')
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, out, 'F2_nonlinear_lu')


def probe_points(prob_rows, arm):
    """(step, outer its) of the one-step probes of one arm, sorted."""
    pts = []
    for r in prob_rows:
        if r['arm'] == arm:
            st = steps(r)
            if st:
                pts.append((st[0]['step'], st[0]['outer_its']))
    return sorted(pts)


def fig_nonlinear_all(rows, out, prob_rows=()):
    arms = {a: rs[0] for a, rs in by_arm(rows).items()}
    if 'jorek' not in arms:
        print('[plot] F3_nonlinear_all: needs a jorek case in the nonlinear root'); return
    ref = steps(arms['jorek'])
    nl, sat = phases(ref)
    last = ref[-1]['step']
    big = max(s['tstep'] for s in ref)
    order = [a for a in ('jorek', 'jorek_fresh', 'sfm2_lu', 'sfm2_gmg', 'sfm2_lu_hs0', 'sfm2_gmg_hs0')
             if a in arms]
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 7.6), sharex=True)
    for arm in order:
        st = [s for s in steps(arms[arm]) if s['tstep'] == big]
        if not st:
            continue
        sty = NL_STYLE.get(arm, style(arm))
        xs = [s['step'] for s in st]
        its = [s['outer_its'] for s in st]
        lin = [s['outer_its'] for s in st if nl is None or s['step'] < nl]
        base = sum(lin) / len(lin) if lin else its[0]
        a.plot(xs, its, color=sty['color'], ls=sty['ls'], lw=1.8, label=sty['label'], drawstyle='steps-mid')
        b.plot(xs, [i / base for i in its], color=sty['color'], ls=sty['ls'], lw=1.8,
               label=sty['label'], drawstyle='steps-mid')
        stalled = isinstance(st[-1]['reason'], float) and st[-1]['reason'] < 0
        end_label(b, xs[-1], its[-1] / base,
                  'stalled (%d its)' % its[-1] if stalled else '%.1f×' % (its[-1] / base), sty['color'])
        if len(st) < sum(1 for s in ref if s['tstep'] == big):
            a.plot(xs[-1], its[-1], marker='x', ms=9, mew=2, color=sty['color'])
    # arms too expensive for a whole trajectory: one step from every
    # restart of the reference trajectory, drawn as points
    for arm in ('sfm2_gmg_hs0',):
        pts = probe_points(prob_rows, arm)
        if arm in arms or not pts:
            continue
        sty = NL_STYLE.get(arm, style(arm))
        lin = [i for x, i in pts if nl is None or x < nl]
        base = sum(lin) / len(lin) if lin else pts[0][1]
        nps = sorted({r['np'] for r in prob_rows if r['arm'] == arm})
        a.plot([x for x, _ in pts], [i for _, i in pts], ls='none', marker=sty['marker'], ms=7,
               color=sty['color'],
               label=sty['label'] + ' (1 step from each state, np %s)' % '/'.join(nps))
        b.plot([x for x, _ in pts], [i / base for _, i in pts], ls='none', marker=sty['marker'], ms=7,
               color=sty['color'])
    a.set_yscale('log'); a.set_ylabel('outer iterations')
    a.set_title('Outer iterations per step at Δt = %g' % big)
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.025))
    b.axhline(1.0, ls=':', color=MUTED, lw=1.2)
    b.set_ylabel('iterations / linear-phase mean'); b.set_xlabel('time step')
    b.set_title('Relative to each preconditioner\'s own linear-phase count')
    for ax in (a, b):
        shade(ax, nl, sat, last)
    shade(a, nl, sat, last, label=True)
    footnote(fig, list(arms.values()), 'x: JOREK aborted the run (400 outer its without convergence).')
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, out, 'F3_nonlinear_all')


# --- F4: component strong scaling -----------------------------------------------------------
def fig_components(rows, out):
    arms = by_arm(rows, ['sfm2_lu', 'sfm2_gmg', 'sfm2_gmg_q'])
    if not arms:
        print('[plot] F4_components: needs sfm2 cases in the strong root'); return

    def comp(r, setup_ev, solve_ev):  # component seconds per outer iteration (setup amortised)
        return (ev(r, setup_ev) + ev(r, solve_ev)) / max(f(r, 'outer_sum', 1), 1)

    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.15])
    axs = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    bar = fig.add_subplot(gs[:, 2])
    effs = {}   # (component, arm) -> efficiency at the largest core count
    maxc = {}
    allx = []
    for ax, (title, s_ev, v_ev) in zip(axs, COMPONENTS):
        for arm in ('sfm2_lu', 'sfm2_gmg', 'sfm2_gmg_q'):
            rs = sorted(arms.get(arm, []), key=cores)
            if len(rs) < 1:
                continue
            xs = [cores(r) for r in rs]; allx += xs
            ys = [comp(r, s_ev, v_ev) for r in rs]
            if ys[0] <= 0:
                continue
            sty = style(arm)
            ax.plot(xs, ys, color=sty['color'], marker=sty['marker'], ls=sty['ls'], label=sty['label'])
            ideal(ax, xs, ys[0], xs[0])
            effs[(title, arm)] = ys[0] * xs[0] / (ys[-1] * xs[-1])
            maxc[arm] = xs[-1]
        ax.set_yscale('log'); log2_axis(ax, allx or [1])
        ax.set_title(title); ax.set_xlabel('cores'); ax.set_ylabel('s per outer iteration')
    axs[0].plot([], [], ls=':', color=MUTED, lw=1.2, label='ideal')
    axs[0].legend(loc='lower left', fontsize=8)

    # whole-system LU reference: JOREK's PC at the same mesh
    ref = sorted(by_arm(rows, ['jorek']).get('jorek', []), key=cores)
    if len(ref) >= 2:
        t = [(f(r, 't_MatLUFactorSym') + f(r, 't_MatLUFactorNum') + f(r, 't_MatSolve'))
             / max(f(r, 'outer_sum', 1), 1) for r in ref]
        effs[('JOREK LU (whole system)', 'jorek')] = t[0] * cores(ref[0]) / (t[-1] * cores(ref[-1]))
        maxc['jorek'] = cores(ref[-1])

    labels = [c[0] for c in COMPONENTS] + ['JOREK LU (whole system)']
    arms_b = [a for a in ('jorek', 'sfm2_lu', 'sfm2_gmg', 'sfm2_gmg_q') if a in maxc]
    h = 0.8 / max(len(arms_b), 1)
    for k, arm in enumerate(arms_b):
        ys = [i - 0.4 + h * (k + 0.5) for i in range(len(labels))]
        vals = [effs.get((l, arm), float('nan')) for l in labels]
        sty = style(arm)
        bar.barh(ys, vals, height=h * 0.85, color=sty['color'], label=sty['label'])
        for y, v in zip(ys, vals):
            if not math.isnan(v):
                bar.text(v + 0.01, y, '%.0f%%' % (100 * v), va='center', fontsize=8, color=INK2)
    bar.set_yticks(range(len(labels))); bar.set_yticklabels(labels)
    bar.invert_yaxis(); bar.set_xlim(0, 1.15); bar.axvline(1.0, ls=':', color=MUTED, lw=1.2)
    bar.set_xlabel('parallel efficiency at %d cores' % max(maxc.values()))
    bar.set_title('Parallel efficiency'); bar.grid(axis='y', visible=False)
    bar.legend(loc='lower right', fontsize=8)
    r = next(iter(arms.values()))[0]
    footnote(fig, [r], 'Mesh %sx%s (%s DOFs); each part\'s setup is amortised over the run.'
             % (r['n_flux'], r['n_tht'], r['ndof']))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out, 'F4_components')


# --- F5: mode coupling ------------------------------------------------------------------------
# Blocks are not arms: a neutral ramp with distinct markers, so no block shares
# an arm's colour in the panel below.
A8_BLOCKS = [('B_24', '#0b0b0b', 'o'), ('B_21', '#52514e', 's'), ('B_11', '#898781', '^'),
             ('S_W', '#b5b3ab', 'D')]


def a8(path):
    """{block: ||cross||_F/||A||_F} from the physics PC's first-build A8 report."""
    out = {}
    for line in open(path, errors='replace'):
        if 'A8 harmonic coupling' in line:
            name = line.split('A8 harmonic coupling')[1].split(':')[0].strip()
            v = line.split('||cross||_F/||A||_F =')[1].split(',')[0]
            out[name] = float(v)
    return out


def fig_coupling(prob_rows, nl_rows, out):
    pts = []
    for r in prob_rows:
        if r['arm'] != 'sfm2_lu_hs0':
            continue
        st = steps(r)
        if not st:
            continue
        pts.append((st[0]['step'] - 1, a8(os.path.join(r['_dir'], 'log')), st[0]['outer_its']))
    if not pts:
        print('[plot] F5_mode_coupling: no probe cases'); return
    pts.sort(key=lambda p: p[0])
    ref = next((r for r in nl_rows if r['arm'] == 'jorek'), None)
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for blk, c, m in A8_BLOCKS:
        xs = [p[0] for p in pts if p[1].get(blk, 0) > 0]
        ys = [p[1][blk] for p in pts if p[1].get(blk, 0) > 0]
        if xs:
            a.plot(xs, ys, color=c, marker=m, lw=1.6, label=blk)
            end_label(a, xs[-1], ys[-1], blk, c)
    a.set_yscale('log'); a.set_ylabel('||cross-|n| part|| / ||block||')
    a.set_title('Coupling between toroidal harmonics n=0 and n=1 in the PC blocks')
    a.legend(loc='lower right', ncol=2)
    sty = style('sfm2_lu_hs0')
    b.plot([p[0] for p in pts], [p[2] for p in pts], color=sty['color'], marker=sty['marker'],
           label='SFM2 (LU), coupling kept: 1 step from each state')
    if ref:
        st = [s for s in steps(ref) if s['tstep'] == 1000]
        b.plot([s['step'] for s in st], [s['outer_its'] for s in st], color=style('jorek')['color'],
               lw=1.6, drawstyle='steps-mid', label=style('jorek')['label'])
        nl, sat = phases(steps(ref))
        for ax in (a, b):
            shade(ax, nl, sat, steps(ref)[-1]['step'])
        shade(a, nl, sat, steps(ref)[-1]['step'], label=True)
    b.set_yscale('log'); b.set_ylabel('outer iterations'); b.set_xlabel('time step (Δt = 1000)')
    b.set_title('Outer iterations')
    b.legend(loc='center left')
    footnote(fig, prob_rows, 'A8 report of the physics PC, harm_split = 0.')
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, out, 'F5_mode_coupling')


# --- F4b: inside pair_w ---------------------------------------------------------------------
# Stacked parts of the pair_w cost (setup + solve). GMG_Lines is the multigrid's
# own smoother work; MjSolve and GMG_AxSolve are the two direct solves that
# remain inside it (the exact constraint mass and the axis block).
PW_PARTS = [('GMG line smoother', ['GMG_Lines'], '#008300'),
            ('exact-mass solve', ['PhysPC_MjSolve'], '#4a3aa7'),
            ('axis LU (rank 0)', ['GMG_AxSolve'], '#e87ba4')]


def fig_pair_w(rows, out):
    arms = [a for a in ('sfm2_gmg', 'sfm2_gmg_q') if a in by_arm(rows)]
    if not arms:
        print('[plot] F4b_pair_w: needs sfm2_gmg cases in the strong root'); return
    fig, axs = plt.subplots(1, len(arms), figsize=(5.2 * len(arms), 4.4), sharey=True, squeeze=False)
    for ax, arm in zip(axs[0], arms):
        rs = sorted(by_arm(rows)[arm], key=cores)
        xs = list(range(len(rs)))
        bottom = [0.0] * len(rs)
        per_it = [max(f(r, 'outer_sum', 1), 1) for r in rs]
        total = [ev(r, ['PhysPC_FactW', 'PhysPC_SolveW']) / n for r, n in zip(rs, per_it)]
        for name, evs, c in PW_PARTS:
            ys = [ev(r, evs) / n for r, n in zip(rs, per_it)]
            ax.bar(xs, ys, bottom=bottom, color=c, width=0.6, label=name, edgecolor=SURFACE, lw=1.5)
            bottom = [b + y for b, y in zip(bottom, ys)]
        rest = [max(t - b, 0) for t, b in zip(total, bottom)]
        ax.bar(xs, rest, bottom=bottom, color='#d6d4cc', width=0.6, label='rest (V-cycle, shell matvecs)',
               edgecolor=SURFACE, lw=1.5)
        for x, t in zip(xs, total):
            ax.text(x, t, '%.2f s' % t, ha='center', va='bottom', fontsize=8.5, color=INK2)
        ax.set_xticks(xs); ax.set_xticklabels([str(cores(r)) for r in rs])
        ax.set_xlabel('cores'); ax.set_title(style(arm)['label'])
        ax.grid(axis='x', visible=False)
    axs[0][0].set_ylabel('pair_w seconds per outer iteration')
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc='center left', bbox_to_anchor=(1.0, 0.55))
    footnote(fig, rows, 'Setup amortised over the run.')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out, 'F4b_pair_w')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--strong'); p.add_argument('--weak'); p.add_argument('--nonlinear')
    p.add_argument('--coupling')
    p.add_argument('--out')
    a = p.parse_args()
    roots = [x for x in (a.strong, a.weak, a.nonlinear, a.coupling) if x]
    if not roots:
        p.error('give at least one of --strong / --weak / --nonlinear')
    out = a.out or os.path.join(roots[0], 'figures')
    if a.strong:
        rows = load(a.strong)
        fig_lu_strong(rows, out)
        fig_components(rows, out)
        fig_pair_w(rows, out)
    if a.weak:
        fig_lu_weak(load(a.weak), out)
    if a.nonlinear:
        rows = load(a.nonlinear, ok_only=False)
        probes = load(a.coupling) if a.coupling else []
        fig_nonlinear_lu(rows, out)
        fig_nonlinear_all(rows, out, probes)
        if probes:
            fig_coupling(probes, rows, out)
    elif a.coupling:
        fig_coupling(load(a.coupling), [], out)


if __name__ == '__main__':
    main()
