#!/usr/bin/env python3
"""Reader / writer for PETSc binary matrices (MatView on a binary viewer) and
the flux-Schur family dump (``-sf_flux_dump <dir>``).

File layout of one matrix (PETSc native binary, big-endian):

    classid  (int32 or PetscInt)   1211216 (MAT_FILE_CLASSID)
    M, N, nnz                      PetscInt
    row lengths [M]                PetscInt
    column indices [nnz]           PetscInt
    values [nnz]                   float64 (real PETSc build)

PetscInt is int32 by default and int64 with ``--with-64-bit-indices``.  In a
64-bit build PETSc writes the classid as a PetscInt as well; the reader accepts
both an int32 and an int64 classid and either index width.  By default the
layout is auto-detected from the file size (``int64=None``); pass
``int64=True/False`` to force the index width.

Family dump layout (one directory per toroidal family, ``<dir>/fam<f>/``):

    scalar-field matrices  (row = idx*nslot + m)  B12 B16 B21 B23 B25 B26 B51
                           B52 B61 B62 B63 B65 B55 B66 Mw Cw H Bb
                           (Cw only with ``s1 spai``, H only with ``s1 hpatch``;
                           H = u rows x psi cols)
    two-field packed       (field-major)          PJ S1 Dv0 Dv
    meta.txt               key value lines; ``dirichlet i0 i1 ...``; unknown keys are kept
                           (bend_sign default 1, bend_form default 0, 1 or 2,
                           s1 default "spai" or "hpatch" with p r g vel,
                           dirichlet_psi default = dirichlet)

All blocks, Mw included, carry JOREK's FFT slot normalisation (n_plane on the
n = 0 slot, n_plane/2 on cos/sin slots); the checks are invariant to it.

Blocks that are absent are returned as ``None``; use ``Family.blk`` to get a
zero matrix of the right shape instead.
"""
import os
import sys
import numpy as np
import scipy.sparse as sp

MAT_FILE_CLASSID = 1211216

SCALAR_BLOCKS = ["B12", "B16", "B21", "B23", "B25", "B26", "B51", "B52",
                 "B61", "B62", "B63", "B65", "B55", "B66", "Mw", "Cw", "H", "Bb"]
PACKED_BLOCKS = {"PJ":  (("B11", "B13"), ("B31", "B33")),
                 "Dv0": (("B22", "B24"), ("B42", "B44"))}
PACKED_ONLY = ["S1", "Dv"]          # kept packed (2n x 2n)


def _layouts(nbytes):
    """candidate (classid width, index width) pairs"""
    return [(4, 4), (8, 8), (4, 8), (8, 4)]


def read_petsc_mat(path, int64=None, verbose=False):
    """Read one PETSc binary Mat -> scipy.sparse.csr_matrix (float64).

    int64 : None (auto-detect), True (64-bit PetscInt) or False (32-bit).
    """
    with open(path, "rb") as fh:
        buf = fh.read()
    nbytes = len(buf)
    if nbytes < 16:
        raise ValueError("%s: file too short (%d bytes)" % (path, nbytes))
    cands = []
    for cw, iw in _layouts(nbytes):
        if int64 is not None and iw != (8 if int64 else 4):
            continue
        cdt = ">i4" if cw == 4 else ">i8"
        idt = ">i4" if iw == 4 else ">i8"
        try:
            cid = int(np.frombuffer(buf, cdt, 1, 0)[0])
        except ValueError:
            continue
        if cid != MAT_FILE_CLASSID:
            continue
        off = cw
        if off + 3 * iw > nbytes:
            continue
        M, N, nnz = (int(v) for v in np.frombuffer(buf, idt, 3, off))
        if M < 0 or N < 0 or nnz < 0:
            continue
        need = off + 3 * iw + M * iw + nnz * iw + nnz * 8
        if need > nbytes:
            continue
        cands.append((need == nbytes, cw, iw, M, N, nnz, need))
    if not cands:
        raise ValueError("%s: not a PETSc binary Mat (classid %d not found or "
                         "inconsistent sizes; int64=%s)" % (path, MAT_FILE_CLASSID, int64))
    # prefer an exact file-size match, then 32-bit
    cands.sort(key=lambda c: (not c[0], c[2], c[1]))
    exact, cw, iw, M, N, nnz, need = cands[0]
    idt = ">i4" if iw == 4 else ">i8"
    off = cw + 3 * iw
    rowlen = np.frombuffer(buf, idt, M, off).astype(np.int64); off += M * iw
    cols = np.frombuffer(buf, idt, nnz, off).astype(np.int64); off += nnz * iw
    vals = np.frombuffer(buf, ">f8", nnz, off).astype(np.float64)
    if rowlen.sum() != nnz:
        raise ValueError("%s: sum(row lengths)=%d != nnz=%d" % (path, rowlen.sum(), nnz))
    if nnz and (cols.min() < 0 or cols.max() >= N):
        raise ValueError("%s: column index out of range" % path)
    if not exact:
        print("WARNING %s: %d trailing bytes after the first Mat ignored"
              % (path, nbytes - need), file=sys.stderr)
    if verbose:
        print("%s: %dx%d nnz=%d classid=int%d idx=int%d" % (path, M, N, nnz, 8 * cw, 8 * iw))
    indptr = np.concatenate([[0], np.cumsum(rowlen)])
    A = sp.csr_matrix((vals, cols, indptr), shape=(M, N))
    A.sum_duplicates()
    return A


def write_petsc_mat(path, A, int64=False):
    """Write a scipy sparse matrix in PETSc binary Mat format (big-endian)."""
    A = sp.csr_matrix(A)
    A.sort_indices()
    idt = ">i8" if int64 else ">i4"
    M, N = A.shape
    rowlen = np.diff(A.indptr)
    with open(path, "wb") as fh:
        fh.write(np.array([MAT_FILE_CLASSID], idt).tobytes())
        fh.write(np.array([M, N, A.nnz], idt).tobytes())
        fh.write(rowlen.astype(idt).tobytes())
        fh.write(A.indices.astype(idt).tobytes())
        fh.write(A.data.astype(">f8").tobytes())


def read_meta(path):
    meta = {}
    with open(path) as fh:
        for line in fh:
            tok = line.split()
            if not tok or tok[0].startswith("#"):
                continue
            key, vals = tok[0], tok[1:]
            if key in ("dirichlet", "dirichlet_index", "dirichlet_psi"):
                meta.setdefault(key, [])
                meta[key].extend(int(v) for v in vals)
            elif len(vals) == 1:
                v = vals[0]
                try:
                    meta[key] = int(v)
                except ValueError:
                    try:
                        meta[key] = float(v)
                    except ValueError:
                        meta[key] = v
            else:
                meta[key] = vals
    for k in ("family", "nslot", "nindex"):
        if k not in meta:
            raise ValueError("%s: missing key '%s'" % (path, k))
    meta.setdefault("varsigma", 1.0)
    meta.setdefault("theta_dt", 1.0)
    meta.setdefault("bend_sign", 1.0)     # Bb = (bend_sign/varsigma) D B12^T (D Cw D) B12 D
    meta.setdefault("bend_form", 0)       # 1: Bb = B12 (D Cw D) B12^T; 2: Dpsi B12 (Du Cw Du) B12^T Dpsi
    meta.setdefault("s1", "spai")         # absent (old dumps) = spai; "hpatch" / "splitm" (H.bin, Bb = B12 H)
    meta["s1"] = str(meta["s1"])
    for k in ("p", "r", "g", "vel", "q"):  # hpatch / splitm parameters (q: splitm only)
        if k in meta:
            meta[k] = int(meta[k])
    meta["bend_form"] = int(meta["bend_form"])    # 0, 1 or 2
    meta["bend_sign"] = float(meta["bend_sign"])
    meta["varsigma"] = float(meta["varsigma"])
    meta["theta_dt"] = float(meta["theta_dt"])
    n = meta["nindex"] * meta["nslot"]
    d = set(meta.get("dirichlet", []))
    # optional index-level list (expanded over all slots)
    for i in meta.get("dirichlet_index", []):
        d.update(i * meta["nslot"] + m for m in range(meta["nslot"]))
    d = np.array(sorted(d), dtype=np.int64)
    if d.size and (d.min() < 0 or d.max() >= n):
        raise ValueError("%s: dirichlet index out of range [0,%d)" % (path, n))
    meta["dirichlet"] = d
    # psi's Dirichlet rows (hpatch, bend_form 2); default: the same as the (u) list above
    if "dirichlet_psi" in meta:
        dp = np.array(sorted(set(int(i) for i in meta["dirichlet_psi"])), dtype=np.int64)
        if dp.size and (dp.min() < 0 or dp.max() >= n):
            raise ValueError("%s: dirichlet_psi index out of range [0,%d)" % (path, n))
    else:
        dp = d.copy()
    meta["dirichlet_psi"] = dp
    return meta


class Family:
    """All blocks of one toroidal family.

    n      : scalar-field size nindex*nslot
    blocks : dict name -> csr (scalar n x n), packed blocks split into B11..B44
    S1, Dv : packed 2n x 2n (or None)
    """

    def __init__(self, famdir, int64=None, verbose=False):
        self.dir = famdir
        self.meta = read_meta(os.path.join(famdir, "meta.txt"))
        self.nslot = self.meta["nslot"]
        self.nindex = self.meta["nindex"]
        self.n = self.nindex * self.nslot
        self.varsigma = self.meta["varsigma"]
        self.theta = self.meta["theta_dt"]
        self.dirichlet = self.meta["dirichlet"]
        self.dirichlet_psi = self.meta["dirichlet_psi"]
        self.s1 = self.meta["s1"]                 # "spai" (default), "hpatch" or "splitm"
        self.hform = self.s1 in ("hpatch", "splitm")   # H.bin, Bb = Dpsi B12 H, no Cw.bin
        self.blocks = {}
        self.packed = {}
        self.absent = []
        n = self.n

        def rd(name):
            p = os.path.join(famdir, name + ".bin")
            if not os.path.exists(p):
                return None
            return read_petsc_mat(p, int64=int64, verbose=verbose)

        for name in SCALAR_BLOCKS:
            A = rd(name)
            if A is None:
                self.absent.append(name)
                continue
            if name in ("Mw", "Cw") and A.shape == (self.nindex, self.nindex) and self.nslot > 1:
                A = sp.kron(A, sp.identity(self.nslot), format="csr")   # slot-independent
            if A.shape != (n, n):
                raise ValueError("%s: shape %s, expected %s" % (name, A.shape, (n, n)))
            self.blocks[name] = A
        for name, sub in PACKED_BLOCKS.items():
            A = rd(name)
            if A is None:
                self.absent.append(name)
                continue
            if A.shape != (2 * n, 2 * n):
                raise ValueError("%s: shape %s, expected %s" % (name, A.shape, (2 * n, 2 * n)))
            self.packed[name] = A
            for a in range(2):
                for b in range(2):
                    self.blocks[sub[a][b]] = A[a * n:(a + 1) * n, b * n:(b + 1) * n].tocsr()
        for name in PACKED_ONLY:
            A = rd(name)
            if A is None:
                self.absent.append(name)
                continue
            if A.shape != (2 * n, 2 * n):
                raise ValueError("%s: shape %s, expected %s" % (name, A.shape, (2 * n, 2 * n)))
            self.packed[name] = A
        if "Dv" in self.packed and "Dv0" in self.packed:
            self.blocks["Wc"] = (self.packed["Dv"] - self.packed["Dv0"])[:n, :n].tocsr()

    def has(self, name):
        return name in self.blocks or name in self.packed

    def blk(self, name):
        """scalar block or a zero matrix if absent"""
        if name in self.blocks:
            return self.blocks[name]
        return sp.csr_matrix((self.n, self.n))

    def free(self):
        """unconstrained scalar row indices"""
        mask = np.ones(self.n, bool)
        mask[self.dirichlet] = False
        return np.nonzero(mask)[0]

    def free_psi(self):
        """unconstrained psi row indices (dirichlet_psi; = free() if the key is absent)"""
        mask = np.ones(self.n, bool)
        mask[self.dirichlet_psi] = False
        return np.nonzero(mask)[0]


def load_family(famdir, int64=None, verbose=False):
    """-> (dict of blocks, meta).  See Family for the object interface."""
    f = Family(famdir, int64=int64, verbose=verbose)
    d = dict(f.blocks)
    d.update(f.packed)
    return d, f.meta


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="list the contents of a family dump")
    ap.add_argument("famdir")
    ap.add_argument("--int64", action="store_true", help="force 64-bit PetscInt")
    a = ap.parse_args()
    f = Family(a.famdir, int64=True if a.int64 else None, verbose=True)
    print("family %s nslot %d nindex %d n %d varsigma %g theta_dt %g  #dirichlet %d"
          % (f.meta["family"], f.nslot, f.nindex, f.n, f.varsigma, f.theta, len(f.dirichlet)))
    print("s1 %s  bend_form %d  #dirichlet_psi %d" % (f.s1, f.meta["bend_form"], len(f.dirichlet_psi)))
    print("absent (treated as zero):", " ".join(f.absent) or "-")
