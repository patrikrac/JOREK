module mod_solve_sparse_projection
    use mod_sparse_data
  
    private
    public :: solve_sparse_projection_system
  
    contains
  
  !> solve Ax = rhs in the projetion step
  subroutine solve_sparse_projection_system(a_mat, rhs_vec, solver)
    use mod_integer_types
    use mod_clock
    use data_structure, only: type_SP_MATRIX, type_RHS
    use mod_sparse_data, only: type_SP_SOLVER, mumps, pastix, strumpack
#ifdef USE_STRUMPACK
      use mod_strumpack, only: spk_delete_factors
#endif
    use matio_module, only: save_mat_h5

    implicit none

    type(type_SP_SOLVER)          :: solver
    type(type_SP_MATRIX)          :: a_mat
    type(type_RHS)                :: rhs_vec

    integer                  :: my_id, n_mpi, ierr
    ! type(clcktype)           :: t_itstart, t0, t1, t2, t3
    ! real*8                   :: tsecond
    integer(kind=int_all)    :: i
    logical                  :: verbose = .false.
    integer                  :: tag = -1   !< tag for log file output
    character(len=10)        :: fname


    external :: solve_mumps_all, solve_pastix_all, solve_strumpack_all

    call MPI_COMM_SIZE(a_mat%comm, n_mpi, ierr)
    call MPI_COMM_RANK(a_mat%comm, my_id, ierr)

    verbose = solver%verbose.and.(my_id.eq.0)
#ifdef SAVEMATRIX
    write(fname,'(A5,I2.2,A3)') "matA_",my_id,".h5"
    call save_mat_h5_ext(fname, a_mat%ng, a_mat%ng, a_mat%nnz, &
                        a_mat%irn, a_mat%jcn, a_mat%val, rhs=rhs_vec%val, &
                        ind_min=a_mat%index_min(my_id+1),ind_max=a_mat%index_max(my_id+1), &
                        block_size=a_mat%block_size)
#endif

    if (.not.solver%iterative) then 

      if (verbose) tag = 0
      if (verbose) write(*,*) "Solving projection using direct solver"

      if (solver%library.eq.mumps) then
#ifdef USE_MUMPS
          if (verbose) write(*,*) "Using MUMPS solver"
          solver%mmss%equilibrium = solver%equilibrium
          solver%mmss%projection  = solver%projection
          call solve_mumps_all(solver%mmss, a_mat, rhs_vec, solver%solve_only, tag)
#endif
      elseif (solver%library.eq.strumpack) then
#ifdef USE_STRUMPACK
          if (verbose) write(*,*) "Using STRUMPACK solver"
          solver%spss%equilibrium = solver%equilibrium
          solver%spss%projection  = solver%projection
          call solve_strumpack_all(solver%spss, a_mat, rhs_vec, solver%solve_only, tag)
#endif
      elseif (solver%library.eq.pastix) then
#if (defined USE_PASTIX) || (defined USE_PASTIX6)
          if (verbose) write(*,*) "Using PaStiX solver"
          solver%ptss%equilibrium = solver%equilibrium
          solver%ptss%projection  = solver%projection
          solver%ptss%refine = .false.
          call solve_pastix_all(solver%ptss, a_mat, rhs_vec, solver%solve_only, tag)
#endif
      endif

      solver%solve_only = .true.
      solver%step_success = .true.

    elseif (solver%iterative) then

      if (verbose) write(*,*) "Solving projection system using iterative solver"

      if (solver%verbose) tag = my_id

      write(*,*) "Iterative solver not implemented for projection system"
      stop
    endif

  end subroutine solve_sparse_projection_system


end module mod_solve_sparse_projection
