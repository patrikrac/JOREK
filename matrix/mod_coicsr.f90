! SPDX-License-Identifier: LGPL-2.1-or-later
!
! Derived from SPARSKIT2 by Yousef Saad (coicsr).
! Copyright (C) 2005 Regents of the University of Minnesota.
! LGPL v2.1 or (at your option) any later version, hence compatible with
! JOREK's LGPL-3.0-or-later. The coicsr_cmplx / coicsr2 / coicsr2_cmplx
! variants are JOREK modifications of the original routine.
! Source: https://www-users.cse.umn.edu/~saad/software/SPARSKIT/
! See THIRD_PARTY_LICENSES.md.
!
      module mod_coicsr
      use mod_integer_types
      implicit none
      contains
!------------------------------------------------------------------------
! routine from SPARSKIT2
!------------------------------------------------------------------------
      subroutine coicsr (n,nnz,job,a,ja,ia,iwk)
      use mod_integer_types
      integer               :: job 
      integer(kind=int_all) :: n, nnz, ia(nnz),ja(nnz),iwk(n+1) 
      real*8                :: a(*)
!------------------------------------------------------------------------
! IN-PLACE coo-csr conversion routine.
!------------------------------------------------------------------------
! this subroutine converts a matrix stored in coordinate format into
! the csr format. The conversion is done in place in that the arrays
! a,ja,ia of the result are overwritten onto the original arrays.
!------------------------------------------------------------------------
! on entry:
!---------
! n     = integer. row dimension of A.
! nnz   = integer. number of nonzero elements in A.
! job   = integer. Job indicator. when job=1, the real values in a are
!         filled. Otherwise a is not touched and the structure of the
!         array only (i.e. ja, ia)  is obtained.
! a     = real array of size nnz (number of nonzero elements in A)
!         containing the nonzero elements
! ja    = integer array of length nnz containing the column positions
!         of the corresponding elements in a.
! ia    = integer array of length nnz containing the row positions
!         of the corresponding elements in a.
! iwk   = integer work array of length n+1
! on return:
!----------
! a
! ja
! ia    = contains the compressed sparse row data structure for the
!         resulting matrix.
! Note:
!-------
!         the entries of the output matrix are not sorted (the column
!         indices in each are not in increasing order) use coocsr
!         if you want them sorted.
!----------------------------------------------------------------------c
!  Coded by Y. Saad, Sep. 26 1989                                      c
!----------------------------------------------------------------------c
!#ifndef USE_COMPLEX_PRECOND
      real*8                ::   t,tnext
!#else
!      double complex  ::   t,tnext
!#endif
      logical               ::   values
      integer(kind=int_all) ::   i,j,k, init, ipos, inext, jnext
!-----------------------------------------------------------------------
      values = (job .eq. 1)
! find pointer array for resulting matrix.
      do i=1,n+1
         iwk(i) = 0
      enddo
      do k=1,nnz
         i = ia(k)
         iwk(i+1) = iwk(i+1)+1
      enddo
!------------------------------------------------------------------------
      iwk(1) = 1
      do i=2,n
         iwk(i) = iwk(i-1) + iwk(i)
      enddo
!
!     loop for a cycle in chasing process.
!
      init = 1
      k = 0
 5    if (values) t = a(init)
      i = ia(init)
      j = ja(init)
      ia(init) = -1
!------------------------------------------------------------------------
 6    k = k+1
!-------------------- current row number is i.  determine  where to go.
      ipos = iwk(i)
!-------------------- save the chased element.
      if (values) tnext = a(ipos)
      inext = ia(ipos)
      jnext = ja(ipos)
!-------------------- then occupy its location.
      if (values) a(ipos)  = t
      ja(ipos) = j
!     update pointer information for next element to come in row i.
      iwk(i) = ipos+1
!     determine  next element to be chased,
      if (ia(ipos) .lt. 0) goto 65
      t = tnext
      i = inext
      j = jnext
      ia(ipos) = -1
      if (k .lt. nnz) goto 6
      goto 70
 65   init = init+1
      if (init .gt. nnz) goto 70
      if (ia(init) .lt. 0) goto 65
!     restart chasing --
      goto 5
 70   do i=1,n
         ia(i+1) = iwk(i)
      enddo
      ia(1) = 1
      return
!----------------- end of coicsr ----------------------------------------
!------------------------------------------------------------------------
      end subroutine coicsr
!------------------------------------------------------------------------
! routine for complex matrix
!------------------------------------------------------------------------
      subroutine coicsr_cmplx (n,nnz,job,a,ja,ia,iwk)
      use mod_integer_types
     
      ! --- Routine parameters
      integer(kind=int_all), intent(inout) :: n, nnz 
      integer,               intent(in)    :: job 
      integer(kind=int_all), intent(inout) :: ia(nnz),ja(nnz),iwk(n+1) 
      double complex,        intent(inout) :: a(*) 
 
      ! --- Local variables
      double complex                :: t,tnext
      logical                       :: values
      integer(kind=int_all)         :: i,j,k, init, ipos, inext, jnext
!-----------------------------------------------------------------------
      values = (job .eq. 1)
! find pointer array for resulting matrix.
      do i=1,n+1
         iwk(i) = 0
      enddo
      do k=1,nnz
         i = ia(k)
         iwk(i+1) = iwk(i+1)+1
      enddo
!------------------------------------------------------------------------
      iwk(1) = 1
      do i=2,n
         iwk(i) = iwk(i-1) + iwk(i)
      enddo
!
!     loop for a cycle in chasing process.
!
      init = 1
      k = 0
 5    if (values) t = a(init)
      i = ia(init)
      j = ja(init)
      ia(init) = -1
!------------------------------------------------------------------------
 6    k = k+1
!-------------------- current row number is i.  determine  where to go.
      ipos = iwk(i)
!-------------------- save the chased element.
      if (values) tnext = a(ipos)
      inext = ia(ipos)
      jnext = ja(ipos)
!-------------------- then occupy its location.
      if (values) a(ipos)  = t
      ja(ipos) = j
!     update pointer information for next element to come in row i.
      iwk(i) = ipos+1
!     determine  next element to be chased,
      if (ia(ipos) .lt. 0) goto 65
      t = tnext
      i = inext
      j = jnext
      ia(ipos) = -1
      if (k .lt. nnz) goto 6
      goto 70
 65   init = init+1
      if (init .gt. nnz) goto 70
      if (ia(init) .lt. 0) goto 65
!     restart chasing --
      goto 5
 70   do i=1,n
         ia(i+1) = iwk(i)
      enddo
      ia(1) = 1
      return
!----------------- end of coicsr ----------------------------------------
!------------------------------------------------------------------------
      end subroutine coicsr_cmplx

      subroutine coicsr2 (n,nnz,a,ja,ia,ndof,iwk)
      use mod_integer_types
      implicit none
      integer(kind=int_all) :: n, nnz, ia(nnz),ja(nnz),iwk(n+1)
      real*8                :: a(*)
!------------------------------------------------------------------------
! IN-PLACE coo-csr conversion routine.(with added option ndof)
!------------------------------------------------------------------------
! this subroutine converts a matrix stored in coordinate format into
! the csr format. The conversion is done in place in that the arrays
! a,ja,ia of the result are overwritten onto the original arrays.
!------------------------------------------------------------------------
! on entry:
!---------
! n     = integer. row dimension of A.
! nnz   = integer. number of nonzero elements in A.
! a     = real array of size nnz (number of nonzero elements in A)
!         containing the nonzero elements
! ja    = integer array of length nnz containing the column positions
!         of the corresponding elements in a.
! ia    = integer array of length nnz containing the row positions
!         of the corresponding elements in a.
! ndof  = the number of degrees of freedom
! iwk   = integer work array of length n+1
! on return:
!----------
! a
! ja
! ia    = contains the compressed sparse row data structure for the
!         resulting matrix.
! Note:
!-------
!         the entries of the output matrix are not sorted (the column
!         indices in each are not in increasing order) use coocsr
!         if you want them sorted.
!----------------------------------------------------------------------c
!  Coded by Y. Saad, Sep. 26 1989                                      c
!----------------------------------------------------------------------c
      integer(kind=int_all) ::   i,j,k, init, ipos, inext, jnext, ndof, ndof2, id, jd
      real*8                ::   t(ndof*ndof),tnext(ndof*ndof)
!-----------------------------------------------------------------------
      ndof2  = ndof*ndof
      
! find pointer array for resulting matrix.
      do i=1,n+1
         iwk(i) = 0
      enddo
      do k=1,nnz
         i = ia(k)
         iwk(i+1) = iwk(i+1)+1
      enddo
!------------------------------------------------------------------------
      iwk(1) = 1
      do i=2,n
         iwk(i) = iwk(i-1) + iwk(i)
      enddo
!
!     loop for a cycle in chasing process.
!
      init = 1
      k = 0
 5    t = a((init-1)*ndof2+1:init*ndof2)
      i = ia(init)
      j = ja(init)
      ia(init) = -1
!------------------------------------------------------------------------
 6    k = k+1
!-------------------- current row number is i.  determine  where to go.
      ipos = iwk(i)
!-------------------- save the chased element.
      tnext = a((ipos-1)*ndof2+1:ipos*ndof2)
      inext = ia(ipos)
      jnext = ja(ipos)
!-------------------- then occupy its location.
      do id=1,ndof
        do jd=1,ndof
	  a((ipos-1)*ndof2+(id-1)*ndof + jd) = t((jd-1)*ndof + id)
	enddo
      enddo
      ja(ipos) = j
!     update pointer information for next element to come in row i.
      iwk(i) = ipos+1
!     determine  next element to be chased,
      if (ia(ipos) .lt. 0) goto 65
      t = tnext
      i = inext
      j = jnext
      ia(ipos) = -1
      if (k .lt. nnz) goto 6
      goto 70
 65   init = init+1
      if (init .gt. nnz) goto 70
      if (ia(init) .lt. 0) goto 65
!     restart chasing --
      goto 5
 70   do i=1,n
         ia(i+1) = iwk(i)
      enddo
      ia(1) = 1
      return
!----------------- end of coicsr2 ----------------------------------------
!------------------------------------------------------------------------
      end subroutine coicsr2

      !> Routine for complex matrix block
      subroutine coicsr2_cmplx (n,nnz,a,ja,ia,ndof,iwk)
      use mod_integer_types
      implicit none

      ! --- Routine parameters
      integer(kind=int_all), intent(inout) :: n, nnz, ia(nnz),ja(nnz),iwk(n+1)
      double complex,        intent(inout) :: a(*)
      integer(kind=int_all), intent(inout) :: ndof

      ! --- Local variables
      integer(kind=int_all)         :: i,j,k, init, ipos, inext, jnext, ndof2, id, jd
      double complex                :: t(ndof*ndof),tnext(ndof*ndof)
!-----------------------------------------------------------------------
      ndof2  = ndof*ndof
      
! find pointer array for resulting matrix.
      do i=1,n+1
         iwk(i) = 0
      enddo
      do k=1,nnz
         i = ia(k)
         iwk(i+1) = iwk(i+1)+1
      enddo
!------------------------------------------------------------------------
      iwk(1) = 1
      do i=2,n
         iwk(i) = iwk(i-1) + iwk(i)
      enddo
!
!     loop for a cycle in chasing process.
!
      init = 1
      k = 0
 5    t = a((init-1)*ndof2+1:init*ndof2)
      i = ia(init)
      j = ja(init)
      ia(init) = -1
!------------------------------------------------------------------------
 6    k = k+1
!-------------------- current row number is i.  determine  where to go.
      ipos = iwk(i)
!-------------------- save the chased element.
      tnext = a((ipos-1)*ndof2+1:ipos*ndof2)
      inext = ia(ipos)
      jnext = ja(ipos)
!-------------------- then occupy its location.
      do id=1,ndof
        do jd=1,ndof
	  a((ipos-1)*ndof2+(id-1)*ndof + jd) = t((jd-1)*ndof + id)
	enddo
      enddo
      ja(ipos) = j
!     update pointer information for next element to come in row i.
      iwk(i) = ipos+1
!     determine  next element to be chased,
      if (ia(ipos) .lt. 0) goto 65
      t = tnext
      i = inext
      j = jnext
      ia(ipos) = -1
      if (k .lt. nnz) goto 6
      goto 70
 65   init = init+1
      if (init .gt. nnz) goto 70
      if (ia(init) .lt. 0) goto 65
!     restart chasing --
      goto 5
 70   do i=1,n
         ia(i+1) = iwk(i)
      enddo
      ia(1) = 1
      return
!----------------- end of coicsr2 ----------------------------------------
!------------------------------------------------------------------------
      end subroutine coicsr2_cmplx

      end module mod_coicsr
