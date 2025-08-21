module mod_mumps
#ifdef USE_MUMPS
  include "dmumps_struc.h"
  
  type type_MUMPS_SOLVER

    integer                                      :: comm = 0
    real(kind=8), dimension(:), pointer          :: solution_scaling => Null()    !< matrix column scaling to be applied to solution vector
    logical                                      :: initialized = .false.
    logical                                      :: analyzed    = .false.
    logical                                      :: equilibrium = .false.    
    logical                                      :: scaled      = .false.
    logical                                      :: refine      = .false.
    logical                                      :: projection  = .false.
    
    type(DMUMPS_STRUC)  :: mumps_par
    integer             :: mumps_ordering
    logical             :: use_BLR_compression
    real(kind=8)        :: epsilon_BLR
  
  end type type_MUMPS_SOLVER
  
  private
  public :: type_MUMPS_SOLVER, mumps_initialize, mumps_analyze, mumps_factorize, mumps_solve, mumps_solve_multiple, mumps_finalize
  
  contains
  
  subroutine mumps_initialize(mmss,comm)
    use phys_module, only: mumps_ordering, epsilon_BLR, use_BLR_compression
    implicit none
    
    type(type_MUMPS_SOLVER)            :: mmss
    integer :: comm
    
    mmss%comm = comm ! normal communicator, doesnt work wor MPI_COMM_SELF
    
    mmss%mumps_par%COMM = mmss%comm                   ! Define a communicator for mumps
  
    mmss%mumps_par%JOB = -1
    mmss%mumps_par%SYM = 0
    mmss%mumps_par%PAR = 1
    mmss%mumps_par%icntl(13) = -1
  
    call  DMUMPS(mmss%mumps_par)                        ! Initialize an instance of mumps
    
    mmss%mumps_par%icntl(2)  = -1                       ! Verbosity levels
    mmss%mumps_par%icntl(3)  = -1
    mmss%mumps_par%icntl(4)  = 6
  
    if (.not. mmss%projection) then
      mmss%mumps_par%icntl(14) = 80                       ! memory working space increase for projection
    else 
      mmss%mumps_par%icntl(14) = 20
    endif
    !mmss%mumps_par%icntl(15) = 0                       ! memory balance only, 1: flops      
  
    mmss%initialized = .true.
    
  end subroutine mumps_initialize
  
  subroutine mumps_analyze(mmss,a_mat)
  
    use data_structure, only: type_SP_MATRIX
    implicit none
    
    type(type_MUMPS_SOLVER) :: mmss
    type(type_SP_MATRIX)    :: a_mat
    
    mmss%mumps_par%JOB = 1                                  ! Analysis, only needed when grid has changed

    if (.not. mmss%projection) then
      mmss%mumps_par%A_loc   => a_mat%val
      mmss%mumps_par%irn_loc => a_mat%irn
      mmss%mumps_par%jcn_loc => a_mat%jcn

      mmss%mumps_par%n      = a_mat%ng
      mmss%mumps_par%nz_loc = a_mat%nnz

      mmss%mumps_par%icntl(7)  = mmss%mumps_ordering               ! ordering option (7:automatic, 3:Scotch, 4:PORD, 5:METIS), default: 7
      mmss%mumps_par%icntl(8)  = 7                            ! row and column scaling  7: automatic scaling
      mmss%mumps_par%icntl(18) = 3
      mmss%mumps_par%icntl(14) = 50                           ! MAXS
      
      if (mmss%use_BLR_compression) then
        mmss%mumps_par%icntl(35) = 1                          ! Block-low-rank (BLR) compression. 0: off (default), 1: automatic, 2: factorisation and solution, 3: only factorisation
        mmss%mumps_par%cntl(7)   = mmss%epsilon_BLR                ! Accuracy of BLR approximation
      endif
      
    else 
      mmss%mumps_par%A       => a_mat%val
      mmss%mumps_par%irn     => a_mat%irn
      mmss%mumps_par%jcn     => a_mat%jcn

      mmss%mumps_par%n      = a_mat%ng
      mmss%mumps_par%nz     = a_mat%nnz

      mmss%mumps_par%icntl(2)  = 6 ! print diagnostics, statistics and warnings to stderr
      mmss%mumps_par%icntl(4)  = 1 ! print errors(1), debug(2), much(3)
      mmss%mumps_par%icntl(5)  = 0 ! assembled form
      mmss%mumps_par%icntl(7)  = 7 ! compute symmetric permutation (PORD or SCOTCH autoselect)
      mmss%mumps_par%icntl(8)  = 8 ! scaling
      mmss%mumps_par%icntl(14) = 80 ! memory relaxation parameter
      mmss%mumps_par%icntl(18) = 0 ! centralized input matrix (i.e. only on cpu 0)
    endif
    
    call DMUMPS(mmss%mumps_par)

    mmss%analyzed = .true.
  
  end subroutine mumps_analyze
  
  subroutine mumps_factorize(mmss,a_mat)
  
    use data_structure, only: type_SP_MATRIX
    implicit none
    
    type(type_MUMPS_SOLVER) :: mmss
    type(type_SP_MATRIX)    :: a_mat
    
    if (.not. mmss%projection) then
      mmss%mumps_par%A_loc   => a_mat%val
      mmss%mumps_par%irn_loc => a_mat%irn
      mmss%mumps_par%jcn_loc => a_mat%jcn
    else 
      mmss%mumps_par%A       => a_mat%val
      mmss%mumps_par%irn     => a_mat%irn
      mmss%mumps_par%jcn     => a_mat%jcn
    endif
      
    mmss%mumps_par%JOB = 2                                   ! factorisation
    call DMUMPS(mmss%mumps_par)
    
    mmss%analyzed = .true.
    
    return
      
  end subroutine mumps_factorize
  
  subroutine mumps_solve(mmss,rhs_vec)
    use mod_integer_types
    use mpi_mod
    use data_structure, only: type_RHS
    
    implicit none
    
    type(type_MUMPS_SOLVER) :: mmss
    type(type_RHS)          :: rhs_vec
    integer(kind=int_all)   :: k
    integer                 :: ierr
    
    mmss%mumps_par%rhs => rhs_vec%val
    
    mmss%mumps_par%JOB = 3                                   ! Solve

    call DMUMPS(mmss%mumps_par)
    
    call MPI_Bcast(mmss%mumps_par%rhs,mmss%mumps_par%n,MPI_DOUBLE_PRECISION,0,mmss%mumps_par%comm,ierr)    
    
    if (mmss%scaled) then
      do k=1,mmss%mumps_par%n
        rhs_vec%val(k) =  mmss%mumps_par%rhs(k)/mmss%solution_scaling(k)
      enddo
    endif

    return
    
  end subroutine mumps_solve

  subroutine mumps_solve_multiple (mmss, rhs_vec)
    use mod_integer_types
    use mpi_mod
    use data_structure, only: type_RHS
    use, intrinsic :: ieee_exceptions

    type(type_MUMPS_SOLVER) :: mmss
    type(type_RHS)          :: rhs_vec
    logical :: halt(size(IEEE_USUAL,1))

    mmss%mumps_par%rhs => rhs_vec%val
    mmss%mumps_par%nrhs = rhs_vec%nrhs
    mmss%mumps_par%lrhs = mmss%mumps_par%n

    ! Compute the solution of Ax=B (B = RHSes)
    mmss%mumps_par%JOB = 3
    mmss%mumps_par%icntl(21) = 0 ! solution is available only on host
    mmss%mumps_par%icntl(4)  = 0 !1 ! print only errors == 1

    ! Disable floating point exceptions in MUMPS
    ! some of the MKL routines make these exceptions on some vectorized
    ! calculations but then don't use the result for a speed increase.
    ! To allow running our code with -fpe0 we need to temporarily disable the
    ! checks, otherwise it'll crash here.
    call ieee_get_halting_mode(IEEE_USUAL, halt)
    call ieee_set_halting_mode(IEEE_USUAL, [.false., .false., .false.])
    call DMUMPS(mmss%mumps_par)
    call ieee_set_halting_mode(IEEE_USUAL, halt)

  end subroutine mumps_solve_multiple

  subroutine mumps_finalize(mmss)
    implicit none
    
    type(type_MUMPS_SOLVER) :: mmss  

    mmss%mumps_par%JOB = -2
    call DMUMPS(mmss%mumps_par)
    
    mmss%comm = 0
    if (associated(mmss%solution_scaling)) deallocate(mmss%solution_scaling)
    mmss%solution_scaling => Null()
    mmss%initialized = .false.
    mmss%analyzed    = .false.
    mmss%equilibrium = .false.    
    mmss%scaled      = .false.
    mmss%refine      = .false.    
    
    return
    
  end subroutine mumps_finalize
  
#endif
end module mod_mumps
