# Third-party licenses

JOREK is distributed under the GNU Lesser General Public License v3.0
(see [COPYING](COPYING) and [COPYING.LESSER](COPYING.LESSER)).

A small number of individual routines originate from third parties and carry
their own licensing terms. They are compatible with the JOREK license and are
listed here so that their copyright and permission notices are preserved, as
required by their respective authors.

---

## comelp — complete elliptic integrals

- **Location:** `models/mod_plasma_response.f90`
- **Authors:** Shanjie Zhang and Jianming Jin
- **Reference:** *Computation of Special Functions*, Wiley, 1996
  (ISBN 0-471-11963-6).

This project includes the routine `comelp`, copyright © Shanjie Zhang and
Jianming Jin. Used with permission under the authors' license:

> This routine is copyrighted by Shanjie Zhang and Jianming Jin. However, they
> give permission to incorporate this routine into a user program provided that
> the copyright is acknowledged.

---

## coicsr — in-place COO→CSR conversion

- **Location:** `matrix/mod_coicsr.f90`
- **Author:** Yousef Saad, University of Minnesota
- **Origin:** SPARSKIT2
- **Copyright:** © 2005 Regents of the University of Minnesota
- **License:** GNU Lesser General Public License, version 2.1 or (at your
  option) any later version — compatible with JOREK's LGPL-3.0.
- **Source:** https://www-users.cse.umn.edu/~saad/software/SPARSKIT/

The `coicsr` routine is taken from SPARSKIT2. The `coicsr_cmplx`, `coicsr2`
and `coicsr2_cmplx` routines in the same file are JOREK modifications of the
original.
