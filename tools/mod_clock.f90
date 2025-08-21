! module dedicated to time measurement
module mod_clock

  use mpi_mod
  
  implicit none

  public 
  TYPE :: clcktype
     integer(8) :: bip
  END TYPE Clcktype

  !-> number of period per seconds
  integer(8) :: Nb_periods_sec 
  !-> maximum value of the clock counter   
  integer(8) :: nb_periods_max

  character(len=*), parameter :: FMT_TIMING = "(I3,A40,F20.7)"

  !******************************
contains
  !******************************

  !--------------------------------- 
  ! bip used to compute the CPU time 
  !---------------------------------
  subroutine clck_time(p_time)
    TYPE(clcktype), intent(out) :: p_time
    
    call system_clock(count=p_time%bip)
  end subroutine clck_time

  !--------------------------------- 
  ! bip used to compute the CPU time 
  !---------------------------------
  subroutine clck_time_barrier(p_time)
    TYPE(clcktype), intent(out) :: p_time
    integer ierr
#ifdef TIMER_BARRIER
    call MPI_Barrier(MPI_COMM_WORLD,ierr)
#endif
    call system_clock(count=p_time%bip)
  end subroutine clck_time_barrier

  !--------------------------------- 
  ! compute difference between 2 bips 
  !---------------------------------
  subroutine clck_diff(p_time1,p_time2,p_difftime)
    TYPE(clcktype), intent(in)    :: p_time1, p_time2
    real(8)      , intent(inout) :: p_difftime

    integer(8) :: Nb_periods
    
    nb_periods = p_time2%bip-p_time1%bip
    if (p_time2%bip < p_time1%bip) then
       nb_periods = nb_periods + nb_periods_max
    endif
    if (nb_periods_sec == 0) then
       p_difftime = p_difftime+0.0
    else
        p_difftime = p_difftime+real(nb_periods)/real(nb_periods_sec)
    endif
  end subroutine clck_diff

  !--------------------------------- 
  ! compute difference between 2 bips 
  !---------------------------------
  subroutine clck_ldiff(p_time1,p_time2,p_difftime)
    TYPE(clcktype), intent(in)    :: p_time1, p_time2
    real(8)      , intent(inout) :: p_difftime

    integer(8) :: nb_periods
    
    nb_periods = p_time2%bip-p_time1%bip
    if (p_time2%bip < p_time1%bip) then
       nb_periods = nb_periods + nb_periods_max
    endif
    ! if the number of periods is null, the time difference is null
    if (nb_periods_sec == 0) then
       p_difftime = 0.0
    else
        p_difftime = real(nb_periods)/real(nb_periods_sec)
    endif
  end subroutine clck_ldiff


  !---------------------------------------
  ! initialization of the time variables
  !--------------------------------------- 
  subroutine clck_init
    call system_clock(count_rate=nb_periods_sec, &
      count_max=nb_periods_max)
  end subroutine clck_init

end module mod_clock
