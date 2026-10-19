#!/usr/bin/env python3
"""T5: independent rebuild of the hpatch velocity response H from an hpatch dump.

The Fortran side (mod_petsc_pc_sf_hpatch, -sf_flux_s1 hpatch) builds
  S1~ = [[B11 + Dpsi B12 H, B13], [B31, B33]]
from local patches.  This script repeats the construction from the dumped blocks
only (dense patch solves) and compares it with the dumped H.bin / Bb.bin:

  index graph   edges = stored structure of B33 (index = row // nslot), symmetrised
                and closed; nodes = indices with identical closed neighbourhoods
  free masks    a row is free if it has an off-diagonal entry: psi / j from PJ,
                u / omega from Dv (the other rows are eliminated Dirichlet rows)
  groups        meta g == 1: single nodes.  g > 1: greedy aggregation (seed = lowest
                unassigned node, then the g-1 unassigned neighbour nodes with the most
                common neighbours; tie-break: lower node id, which need not be the
                Fortran one)
  patch P       nodes within graph distance <= p of the group (all slots)
  X = B33[P,P]^-1 B31[P,cols]      cols = free psi DOFs of the group's indices
  R = B23[Pu,Pj] X - B21[Pu,cols]
  Y = Dv11[Pu,Pu]^-1 R                          (meta vel 0)
    = u part of Dv[Pu+Pw]^-1 [R; 0]             (meta vel 1)
  H[i,col] = Y[i,col] if dist(node(i), node(col)) <= r;   Bb = Dpsi B12 H

Split-M dumps (meta s1 splitm, q; -sf_flux_s1 splitm): Q = nodes within q of the group,
  H_T = -(1/varsigma) Mw[Qu,Qu]^-1 B23[Qu,cols], zero where dist > q
  R  <- R - Dv11[Pu,Qu] H_T   (vel 1: the omega rows' rhs -Dv[Pw,Qu] H_T)
  H = H_T + (Y where dist <= r), rows Qu within q.

Gate (g == 1 only; dump with -sf_flux_hp_group 1): ||H_py - H|| / ||H|| and
||Bb_py - Bb|| / ||Bb|| < --tol (default 1e-10).  For g > 1 the grouping may differ in
tie-breaking from the Fortran heuristic, so the differences are reported as info only.
Info for both H (dumped / rebuilt): sigma(S~^-1 S1) with S~ = P_J + [[Bb,0],[0,0]] and
S1 the exact flux Schur complement (as T2); free psi and j rows only; skipped if larger
than --max-dense or with --no-sigma.
"""
import argparse
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from petsc_bin import Family


def free_mask(M):
    """rows of M with at least one nonzero off-diagonal entry"""
    C = sp.coo_matrix(M)
    k = (C.row != C.col) & (C.data != 0)
    return np.bincount(C.row[k], minlength=M.shape[0]) > 0


def index_graph(B33, ns, nidx):
    """closed symmetrised index graph from B33's stored structure; -> csr pattern (nidx x nidx)"""
    C = sp.coo_matrix(B33)
    r, c = C.row // ns, C.col // ns
    G = sp.coo_matrix((np.ones(r.size), (r, c)), shape=(nidx, nidx)).tocsr()
    G = G + G.T + sp.identity(nidx, format="csr")
    G.data[:] = 1.0
    G.sort_indices()
    return G


def node_classes(G):
    """node id of every index (identical closed neighbourhood -> same node) and the node graph"""
    key = {}
    node_of = np.empty(G.shape[0], dtype=np.int64)
    for i in range(G.shape[0]):
        k = G.indices[G.indptr[i]:G.indptr[i + 1]].tobytes()
        node_of[i] = key.setdefault(k, len(key))
    nn = len(key)
    C = sp.coo_matrix(G)
    Gn = sp.coo_matrix((np.ones(C.nnz), (node_of[C.row], node_of[C.col])), shape=(nn, nn)).tocsr()
    Gn.data[:] = 1.0
    return node_of, Gn


def greedy_groups(Gn, g):
    """seed (lowest unassigned node) + the g-1 unassigned neighbours with most common neighbours"""
    nn = Gn.shape[0]
    if g <= 1:
        return [[k] for k in range(nn)]
    un = np.ones(nn, bool)
    groups = []
    for s in range(nn):
        if not un[s]:
            continue
        ns_ = Gn.indices[Gn.indptr[s]:Gn.indptr[s + 1]]
        nb = [int(k) for k in ns_ if un[k] and k != s]
        cn = [len(np.intersect1d(Gn.indices[Gn.indptr[k]:Gn.indptr[k + 1]], ns_)) for k in nb]
        nb = [k for _, k in sorted(zip([-c for c in cn], nb))][:g - 1]
        grp = [s] + nb
        un[grp] = False
        groups.append(grp)
    return groups


def rebuild_H(f, p, r, g, vel, q=0):
    """-> (H as csr n x n, info dict); H = u rows x psi columns.
    q > 0: split-M, H = H_T + cut_r(Y), H_T = -(1/varsigma) Mw[Q]^-1 B23[Q, cols] cut to q,
    Y the patch solve of R - Dv[:, Q] H_T (Q = nodes within q of the group)"""
    n, ns, nidx = f.n, f.nslot, f.nindex
    PJ, Dv = f.packed["PJ"], f.packed["Dv"]
    hasP, hasD = free_mask(PJ), free_mask(Dv)
    psi_f, j_f, u_f, w_f = hasP[:n], hasP[n:], hasD[:n], hasD[n:]
    B31, B33 = PJ[n:, :n].tocsr(), PJ[n:, n:].tocsr()
    B23, B21 = f.blk("B23").tocsr(), f.blk("B21").tocsr()
    Dv11 = Dv[:n, :n].tocsr()
    Dv = Dv.tocsr()
    Mw = f.blk("Mw").tocsr() if q else None
    G = index_graph(B33, ns, nidx)
    node_of, Gn = node_classes(G)
    nn = Gn.shape[0]
    idx_of_node = [[] for _ in range(nn)]
    for i, k in enumerate(node_of):
        idx_of_node[k].append(i)
    slots = np.arange(ns)
    dofs_of_node = [(np.array(v)[:, None] * ns + slots[None, :]).ravel() for v in idx_of_node]
    groups = greedy_groups(Gn, g)
    rows, cols_, vals = [], [], []
    psz = []
    for grp in groups:
        gd = np.concatenate([dofs_of_node[k] for k in grp])
        cols = gd[psi_f[gd]]
        if cols.size == 0:
            continue
        D = dijkstra(Gn, directed=False, unweighted=True, indices=grp, limit=max(p, r, q) + 0.5)
        dmin = D.min(0)
        Pn = np.nonzero(dmin <= p)[0]
        Pd = np.concatenate([dofs_of_node[k] for k in Pn])
        Pj, Pu, Pw = Pd[j_f[Pd]], Pd[u_f[Pd]], Pd[w_f[Pd]]
        if Pu.size == 0:
            continue
        gpos = {k: i for i, k in enumerate(grp)}
        cq = np.array([gpos[node_of[c // ns]] for c in cols])
        X = sla.solve(B33[Pj][:, Pj].toarray(), B31[Pj][:, cols].toarray())
        R = B23[Pu][:, Pj] @ X - B21[Pu][:, cols].toarray()
        Rw = np.zeros((Pw.size, R.shape[1]))
        if q:       # split-M: Tang's part on Q, the remainder into the patch solve
            Qn = np.nonzero(dmin <= q)[0]
            Qd = np.concatenate([dofs_of_node[k] for k in Qn])
            Qu = Qd[u_f[Qd]]
            HT = -sla.solve(Mw[Qu][:, Qu].toarray(), B23[Qu][:, cols].toarray()) / f.varsigma
            dq = D[cq[None, :], node_of[Qu // ns][:, None]]
            HT[dq > q] = 0.0
            R = R - Dv11[Pu][:, Qu] @ HT
            Rw = -(Dv[n + Pw][:, Qu] @ HT)
        if vel == 0:
            Y = sla.solve(Dv11[Pu][:, Pu].toarray(), R)
        else:
            Pv = np.concatenate([Pu, n + Pw])
            Y = sla.solve(Dv[Pv][:, Pv].toarray(), np.vstack([R, Rw]))[:Pu.size]
        un_ = node_of[Pu // ns]
        keep = D[cq[None, :], un_[:, None]] <= r
        if q:
            Hq = HT.copy()
            pos = {d: i for i, d in enumerate(Qu)}
            iu = np.array([pos[d] for d in Pu])
            Hq[iu] += np.where(keep, Y, 0.0)
            ii, jj = np.nonzero(dq <= q)
            rows.append(Qu[ii]); cols_.append(cols[jj]); vals.append(Hq[ii, jj])
        else:
            ii, jj = np.nonzero(keep)
            rows.append(Pu[ii]); cols_.append(cols[jj]); vals.append(Y[ii, jj])
        psz.append(Pd.size)
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols_))), shape=(n, n))
    return H, dict(nodes=nn, ngroups=len(groups), patch=(min(psz), int(np.mean(psz)), max(psz)))


def sigma_range(St, Sx):
    e = np.linalg.eigvals(np.linalg.solve(St, Sx))
    return e.real.min(), e.real.max(), np.abs(e.imag).max()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("famdir")
    ap.add_argument("--int64", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-10, help="relative H / Bb error gate (g == 1)")
    ap.add_argument("--p", type=int, default=None, help="patch radius (default: meta p, else 3)")
    ap.add_argument("--r", type=int, default=None, help="band radius (default: meta r, else 3)")
    ap.add_argument("--g", type=int, default=None, help="nodes per group (default: meta g, else 4)")
    ap.add_argument("--vel", type=int, default=None, help="0 / 1 (default: meta vel, else 0)")
    ap.add_argument("--q", type=int, default=None, help="splitm H_T radius (default: meta q, else p + 1)")
    ap.add_argument("--max-dense", type=int, default=6000, help="max free 2n for the sigma(S~^-1 S1) info")
    ap.add_argument("--no-sigma", action="store_true")
    a = ap.parse_args()
    f = Family(a.famdir, int64=True if a.int64 else None)
    if not f.hform:
        print("meta s1 = %s: not an hpatch / splitm dump (nothing to rebuild)" % f.s1)
        raise SystemExit(0)
    for k in ("H", "Bb") + (("Mw",) if f.s1 == "splitm" else ()):
        if not f.has(k):
            raise SystemExit("%s.bin not in dump" % k)
    for k in ("PJ", "Dv"):
        if k not in f.packed:
            raise SystemExit("%s.bin not in dump" % k)
    m = f.meta
    p = a.p if a.p is not None else m.get("p", 3)
    r = a.r if a.r is not None else m.get("r", 3)
    g = a.g if a.g is not None else m.get("g", 4)
    vel = a.vel if a.vel is not None else m.get("vel", 0)
    q = (a.q if a.q is not None else m.get("q", p + 1)) if f.s1 == "splitm" else 0
    n = f.n
    print("family %s  nslot %d  nindex %d  n %d  s1 %s  p %d r %d g %d vel %d q %d"
          % (m["family"], f.nslot, f.nindex, n, f.s1, p, r, g, vel, q))
    H, info = rebuild_H(f, p, r, g, vel, q)
    print("index nodes %d, groups %d, patch DOFs min/mean/max %s" % (info["nodes"], info["ngroups"], info["patch"]))
    dp = np.zeros(n); dp[f.free_psi()] = 1.0
    Bb_py = sp.diags(dp) @ f.blk("B12") @ H
    Hd, Bbd = f.blk("H"), f.blk("Bb")
    nrm = lambda A: sp.linalg.norm(A)
    eH = nrm(H - Hd) / max(nrm(Hd), 1e-300)
    eB = nrm(Bb_py - Bbd) / max(nrm(Bbd), 1e-300)
    print("\n||H_py - H.bin||/||H.bin||   = %.3e   (nnz py %d, dumped %d)" % (eH, H.nnz, Hd.nnz))
    print("||Bb_py - Bb.bin||/||Bb.bin|| = %.3e" % eB)

    if not a.no_sigma:
        PJ = f.packed["PJ"]
        hasP = free_mask(PJ)
        fp, fj = np.nonzero(hasP[:n])[0], np.nonzero(hasP[n:])[0]
        idx = np.concatenate([fp, n + fj])
        if idx.size <= a.max_dense:
            from flux_common import Blocks
            blk = Blocks(f)
            S1 = blk.S1_exact()
            Z = sp.csr_matrix((n, n))
            PJd = PJ.toarray()
            R_ = lambda X: X[np.ix_(idx, idx)]
            print("\nsigma(S~^-1 S1), S1 exact flux Schur, free psi/j (%d rows):" % idx.size)
            for name, Bb in (("dumped H", Bbd), ("rebuilt H", Bb_py)):
                St = PJd + sp.bmat([[Bb, Z], [Z, Z]]).toarray()
                lo, hi, im = sigma_range(R_(St), R_(S1))
                print("  %-10s Re [%.5f, %.5f]  max|Im| %.3e" % (name, lo, hi, im))
        else:
            print("\nsigma(S~^-1 S1) skipped: %d free rows > --max-dense" % idx.size)

    if g == 1:
        ok = eH < a.tol and eB < a.tol
        print("\nT5 %s (g == 1 gate, tol %.0e)" % ("PASS" if ok else "FAIL", a.tol))
        raise SystemExit(0 if ok else 1)
    print("\nT5 info only (g = %d > 1: grouping may differ from the Fortran heuristic; "
          "dump with -sf_flux_hp_group 1 for the exact gate)" % g)


if __name__ == "__main__":
    main()
