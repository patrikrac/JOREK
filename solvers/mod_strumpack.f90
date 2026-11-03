!> New strumpack module to be used with core version
module mod_strumpack
#ifdef USE_STRUMPACK
  use iso_c_binding
  use mod_integer_types
  use mpi_mod

  implicit none

  type type_STRUMPACK_SOLVER
    type(c_ptr) :: sscp ! STRUMPACK sparse solver c pointer
    !type(c_ptr) :: distr
    integer(kind=C_INT_ALL), pointer :: distr(:)
    logical                          :: initialized = .false.
    logical                          :: analyzed = .false.
    logical                          :: equilibrium = .false.
    integer                          :: comm = 0
    logical                          :: projection  = .false.
  end type type_STRUMPACK_SOLVER

  private
  public :: type_STRUMPACK_SOLVER, &
            strumpack_init, strumpack_set_mat, strumpack_analyze, &
            strumpack_factorize, strumpack_solve, strumpack_finalize, &
            spk_delete_factors

  interface
    subroutine spk() bind(C)
      use iso_c_binding
    end subroutine spk

    subroutine spk_init(sscp,iparm,comm) bind(C)
      use iso_c_binding
      implicit none

      type(c_ptr), intent(inout) :: sscp
      type(c_ptr)         :: iparm
      integer, intent(in) :: comm
    end subroutine spk_init

    subroutine spk_set_mat(n,dist,irn,jcn,val,sscp,comm,upd) bind(C)
      use iso_c_binding
      use mod_integer_types
      implicit none

      integer(kind=C_INT_ALL), intent(in) :: n
      integer, intent(in) :: comm
      type(c_ptr) :: irn, jcn, val
      type(c_ptr) :: dist
      !integer(kind=C_INT_ALL), dimension(:), pointer, intent(in) :: dist
      type(c_ptr), intent(inout) :: sscp
      logical :: upd
    end subroutine spk_set_mat

    subroutine spk_reord(sscp,comm) bind(C)
      use iso_c_binding
      implicit none

      type(c_ptr), intent(inout) :: sscp
      integer, intent(in) :: comm
    end subroutine spk_reord

    subroutine spk_fact(sscp,comm) bind(C)
      use iso_c_binding
      implicit none

      type(c_ptr), intent(inout) :: sscp
      integer, intent(in) :: comm
    end subroutine spk_fact

    subroutine spk_solve(n,dist,rhs,sscp,comm) bind(C)
      use iso_c_binding
      use mod_integer_types
      implicit none

      integer(kind=C_INT_ALL), intent(in) :: n
      type(c_ptr), intent(inout) :: sscp, rhs
      type(c_ptr) :: dist
      !integer(kind=C_INT_ALL), dimension(:), pointer, intent(in) :: dist
      integer, intent(in) :: comm
    end subroutine spk_solve

    subroutine spk_solve_multiple(n,nrhs,dist,rhs,sscp,comm) bind(C)
      use iso_c_binding
      use mod_integer_types
      implicit none

      integer(kind=C_INT_ALL), intent(in) :: n, nrhs
      type(c_ptr), intent(inout) :: sscp, rhs
      type(c_ptr) :: dist
      !integer(kind=C_INT_ALL), dimension(:), pointer, intent(in) :: dist
      integer, intent(in) :: comm
    end subroutine spk_solve_multiple


    subroutine spk_delete_factors(sscp) bind(C)
      use iso_c_binding
      implicit none

      type(c_ptr), intent(inout) :: sscp
    end subroutine spk_delete_factors

    subroutine spk_finalize(sscp,comm) bind(C)
      use iso_c_binding
      implicit none

      type(c_ptr), intent(inout) :: sscp
      integer, intent(in) :: comm
    end subroutine spk_finalize

  end interface

  contains
    
    subroutine strumpack_init(spss,comm)
      use phys_module, only: strumpack_matching
      implicit none

      integer comm, ierr, n_mpi
      type(type_STRUMPACK_SOLVER) spss
      type(c_ptr) iparm_c
      integer(kind=C_INT), target :: iparm(5)=0
      
      spss%comm = comm
      
      ! set STRUMPACK solver method to direct (1), gmres (2) or refine (3)
      iparm(1) = 1
      ! set STRUMPACK matching strategy
      if (strumpack_matching) then
        iparm(3) = 1
        ! set reordering to Metis (1)
        iparm(2) = 1
      else
        iparm(3) = 0
        ! set reordering to ParMetis (2) or PTScotch (3)
        iparm(2) = 2
      endif

      call spk_init(spss%sscp, c_loc(iparm), spss%comm)
      
      call MPI_COMM_SIZE(spss%comm, n_mpi, ierr)
      allocate(spss%distr(n_mpi+1))
      
      call MPI_Barrier(spss%comm,ierr)

      return
    end subroutine strumpack_init  
    
    
    subroutine strumpack_set_mat(spss,a_mat)
      use, intrinsic :: iso_c_binding
      use sorting_module, only : remove_duplicates, convert2csr, convert_sorting
      use mod_integer_types
      use data_structure, only: type_SP_MATRIX

      implicit none

      type int_buff
        integer(kind=C_INT_ALL), allocatable :: buff(:)
      end type 
    
      type double_buff
        real(kind=C_DOUBLE), allocatable :: buff(:)
      end type 

      
      type(type_STRUMPACK_SOLVER)       :: spss
      type(type_SP_MATRIX)              :: a_mat

      integer ierr
      integer(kind=C_INT_ALL), dimension(:), pointer :: irn_d, jcn_d
      real(kind=C_DOUBLE),  dimension(:), pointer :: val_d

      type(int_buff),     dimension(:), allocatable :: buff_irn_d, buff_jcn_d
      type(double_buff),  dimension(:), allocatable :: buff_val_d


      integer(kind=C_INT_ALL), allocatable, target :: distr(:)
      
      integer :: rank, n_mpi
      integer(kind=C_INT_ALL) :: nnz_d, tmp_nnz_d, n_d, i, j, imin, imax, ranks_i, request


      integer(kind=int_all), dimension(:), pointer :: myelm
      logical :: upd, dflag, eql
      type(c_ptr) :: irn_c, jcn_c, val_c, dist_c

      upd = spss%analyzed
      dflag = a_mat%row_distributed
      eql = spss%equilibrium

      call MPI_COMM_RANK(spss%comm, rank, ierr)
      call MPI_COMM_SIZE(spss%comm, n_mpi, ierr)
      if (eql .or. spss%projection) then
        if (rank.eq.0) call remove_duplicates(a_mat%ng,a_mat%nnz,a_mat%irn,a_mat%jcn,a_mat%val)
        call MPI_Bcast(a_mat%ng,       1, MPI_INTEGER, 0, spss%comm, ierr)
        call MPI_Bcast(a_mat%nnz,      1, MPI_INTEGER, 0, spss%comm, ierr)
        call MPI_Bcast(a_mat%indexing, 1, MPI_INTEGER, 0, spss%comm, ierr)
        call MPI_Bcast(a_mat%block_size, 1, MPI_INTEGER, 0, spss%comm, ierr)
        call MPI_Bcast(a_mat%row_distributed, 1, MPI_INTEGER, 0, spss%comm, ierr)
      endif
      
      allocate(distr(n_mpi+1))

      if ((.not. dflag).and.(n_mpi.gt.1)) then
        ! distribute rows between n_mpi
        call distribute_rows(a_mat,n_mpi,distr)
        if (rank.eq.0) write(*,*) "Matrix is not row-distributed. Distributing now."

        allocate(buff_irn_d(n_mpi), buff_jcn_d(n_mpi), buff_val_d(n_mpi))

        n_d = distr(rank+2) - distr(rank+1) ! number of local rows

        allocate(myelm(a_mat%nnz))
        if (rank.eq.0) then
          do ranks_i=0, n_mpi-1
            myelm = 0
            j = 1
            do i=1, a_mat%nnz
              if ((a_mat%irn(i) >= distr(ranks_i+1)).and.(a_mat%irn(i) <= (distr(ranks_i+2)-1))) then
                myelm(j) = i
                j = j + 1
              endif
            enddo
            tmp_nnz_d = j - 1
            if (ranks_i.ne.0) then  
              allocate(buff_irn_d(ranks_i+1)%buff(tmp_nnz_d), buff_jcn_d(ranks_i+1)%buff(tmp_nnz_d), buff_val_d(ranks_i+1)%buff(tmp_nnz_d))
              do i = 1, tmp_nnz_d
                buff_irn_d(ranks_i+1)%buff(i) = a_mat%irn(myelm(i)) - distr(ranks_i+1) + a_mat%indexing    ! irn starts from index
                buff_jcn_d(ranks_i+1)%buff(i) = a_mat%jcn(myelm(i))                                        ! jcn remains the same
                buff_val_d(ranks_i+1)%buff(i) = a_mat%val(myelm(i))
              enddo

              call MPI_Send(tmp_nnz_d, 1, MPI_INTEGER, ranks_i, 0, spss%comm, ierr)

              call MPI_Send(buff_irn_d(ranks_i+1)%buff, tmp_nnz_d, MPI_INTEGER,          ranks_i, 1, spss%comm, ierr)
              call MPI_Send(buff_jcn_d(ranks_i+1)%buff, tmp_nnz_d, MPI_INTEGER,          ranks_i, 2, spss%comm, ierr)
              call MPI_Send(buff_val_d(ranks_i+1)%buff, tmp_nnz_d, MPI_DOUBLE_PRECISION, ranks_i, 3, spss%comm, ierr)
            else 
              nnz_d = tmp_nnz_d

              allocate(irn_d(nnz_d), jcn_d(nnz_d), val_d(nnz_d))
  
              do i = 1, nnz_d
                irn_d(i) = a_mat%irn(myelm(i)) - distr(ranks_i+1) + a_mat%indexing    ! irn starts from index
                jcn_d(i) = a_mat%jcn(myelm(i))                                        ! jcn remains the same
                val_d(i) = a_mat%val(myelm(i))
              enddo
    
            endif
          enddo
        else 
          call MPI_Recv(nnz_d, 1, MPI_INTEGER, 0, 0, spss%comm, MPI_STATUS_IGNORE, ierr)
          allocate(irn_d(nnz_d), jcn_d(nnz_d), val_d(nnz_d))

          call MPI_Recv(irn_d, nnz_d, MPI_INTEGER,          0, 1, spss%comm, MPI_STATUS_IGNORE, ierr)
          call MPI_Recv(jcn_d, nnz_d, MPI_INTEGER,          0, 2, spss%comm, MPI_STATUS_IGNORE, ierr)
          call MPI_Recv(val_d, nnz_d, MPI_DOUBLE_PRECISION, 0, 3, spss%comm, MPI_STATUS_IGNORE, ierr)

        end if

        if (rank .eq. 0) deallocate(a_mat%irn, a_mat%jcn, a_mat%val)

        a_mat%irn => irn_d
        a_mat%jcn => jcn_d
        a_mat%val => val_d
        a_mat%nnz =  nnz_d
        a_mat%ng  =  n_d
        a_mat%row_distributed = .true.

        deallocate(myelm)

      elseif (dflag.and.(n_mpi.gt.1)) then
        ! get row distribution from irn in case of pre-distributed matrix
        if (allocated(distr)) deallocate(distr)
        allocate(distr(n_mpi+1))
        
        distr(1:n_mpi+1) = 0
        imin = minval(a_mat%irn(1:a_mat%nnz))
        imax = maxval(a_mat%irn(1:a_mat%nnz))
                
        distr(rank+1) = imin

        if (rank.eq.(n_mpi-1)) distr(rank+2) = imax + 1
        call MPI_Allreduce(MPI_IN_PLACE,distr,n_mpi+1,MPI_INTEGER_ALL,MPI_SUM,spss%comm,ierr)

        ! check for consistency
        ierr = 0
        if ((distr(1).ne.1)) ierr = 1
        do i = 2, n_mpi+1
          if (.not.(distr(i)>distr(i-1))) ierr = 1
        enddo

        if (ierr.ne.0) then
          write(*,*) "Error in harmonic matrix distribution"
          call exit(2)
        endif

        n_d = distr(rank+2) - distr(rank+1)
        nnz_d = a_mat%nnz
        a_mat%irn(1:nnz_d) = a_mat%irn(1:nnz_d) - imin + a_mat%indexing ! irn starts with indx
        
      elseif (n_mpi.eq.1) then
        call distribute_rows(a_mat,1,distr)
        n_d = a_mat%ng
        nnz_d = a_mat%nnz
        write(*,*) "n_d, nnz_d", n_d, nnz_d

      endif
      
      if (eql) then
        call remove_duplicates(a_mat%ng,a_mat%nnz,a_mat%irn,a_mat%jcn,a_mat%val)
        n_d = a_mat%ng
        nnz_d = a_mat%nnz
      endif
      
      if (a_mat%indexing.eq.1) then
        a_mat%irn(1:nnz_d) = a_mat%irn(1:nnz_d) - a_mat%indexing;
        a_mat%jcn(1:nnz_d) = a_mat%jcn(1:nnz_d) - a_mat%indexing;
        distr(1:n_mpi+1) = distr(1:n_mpi+1) - a_mat%indexing
        a_mat%indexing = 0
      endif
      
#if (!defined(USEMKL))
      call convert_sorting(nnz=nnz_d, irn=a_mat%irn, jcn=a_mat%jcn, val=a_mat%val, block_size=a_mat%block_size)
#endif

      irn_c = c_loc(a_mat%irn); jcn_c = c_loc(a_mat%jcn); val_c = c_loc(a_mat%val); dist_c = c_loc(distr)
      
#if (defined(USEMKL))
      call convert2csr(a_mat%indexing, n_d, a_mat%ng, nnz_d, irn_c, jcn_c, val_c)
#endif

      !spss%distr = c_loc(distr)
      
      spss%distr(1:n_mpi+1) = distr(1:n_mpi+1)
      deallocate(distr)
      dist_c = c_loc(spss%distr)
      
      call spk_set_mat(n_d, dist_c, irn_c, jcn_c, val_c, spss%sscp, spss%comm,upd)

      call MPI_Barrier(spss%comm,ierr)


      return
    end subroutine strumpack_set_mat
    
    subroutine strumpack_analyze(spss)
      use data_structure, only: type_SP_MATRIX

      implicit none
      
      type(type_STRUMPACK_SOLVER)       :: spss
      integer ierr

      call spk_reord(spss%sscp,spss%comm)

      call MPI_Barrier(spss%comm,ierr)
      spss%analyzed = .true.

      return
    end subroutine strumpack_analyze 
   
    subroutine strumpack_factorize(spss)
      use data_structure, only: type_SP_MATRIX

      implicit none
      
      type(type_STRUMPACK_SOLVER)       :: spss
      integer ierr

      call spk_fact(spss%sscp,spss%comm)
      call MPI_Barrier(spss%comm,ierr)

      return
    end subroutine strumpack_factorize
   
    subroutine strumpack_solve(spss, rhs_vec)
      use data_structure, only: type_SP_MATRIX, type_RHS
      use, intrinsic :: iso_c_binding

      implicit none
      
      type(type_STRUMPACK_SOLVER)   :: spss
      type(type_RHS)                :: rhs_vec
      type(c_ptr)                   :: rhs_c
      type(c_ptr)                   :: dist_c
      integer :: ierr
      integer(kind=C_INT_ALL), allocatable, target :: dist(:)
      integer :: i, n_mpi, rank
      
      call MPI_COMM_SIZE(spss%comm, n_mpi, ierr)
      call MPI_COMM_RANK(spss%comm, rank, ierr)

      if (spss%projection) then
        if (rank.ne.0) then
          if (associated(rhs_vec%val)) deallocate(rhs_vec%val)
          allocate (rhs_vec%val(rhs_vec%n*rhs_vec%nrhs))
        endif
        call MPI_Bcast(rhs_vec%val, rhs_vec%n*rhs_vec%nrhs, MPI_DOUBLE_PRECISION, 0, spss%comm, ierr)
      endif

      rhs_c = c_loc(rhs_vec%val);
      dist_c = c_loc(spss%distr)

      if (spss%projection) then
        call spk_solve_multiple(rhs_vec%n, rhs_vec%nrhs, dist_c, rhs_c, spss%sscp, spss%comm)
      else
        call spk_solve         (rhs_vec%n, dist_c, rhs_c, spss%sscp, spss%comm)
      endif

      call MPI_Barrier(spss%comm,ierr)

      return
    end subroutine strumpack_solve

!> Finalize strumpack solver instance
    subroutine strumpack_finalize(spss)
      implicit none

      type(type_STRUMPACK_SOLVER)   :: spss

      if (associated(spss%distr)) then
        deallocate(spss%distr); spss%distr => null()
      endif

      call spk_finalize(spss%sscp, spss%comm)

      spss%initialized = .false.
      spss%analyzed    = .false.
      spss%equilibrium = .false.
      spss%comm        = 0

      return
    end subroutine strumpack_finalize
   
!> Distribute rows between members of MPI group   
    subroutine distribute_rows(a_mat,n_mpi,distr)
      
      use, intrinsic :: iso_c_binding
      use data_structure, only: type_SP_MATRIX
      use mod_integer_types
      implicit none

      type(type_SP_MATRIX)                 :: a_mat
      integer, intent(in)                  :: n_mpi
      
      integer(kind=C_INT_ALL), allocatable :: nr(:)
      integer(kind=C_INT_ALL), allocatable :: distr(:)
      integer :: ierr, i
    
      
      allocate(nr(n_mpi))

      nr = a_mat%block_size*((a_mat%ng/a_mat%block_size)/n_mpi)
      nr(n_mpi) = nr(n_mpi) + (a_mat%ng - sum(nr))

      distr(1) = 1
      do i=1, n_mpi
        distr(i+1)= distr(i) + nr(i)
      enddo

      deallocate(nr)

      return
    end subroutine distribute_rows

#endif
end module mod_strumpack

