!> Module containing the particle side of recombination (fluid density --> neutral particle weight)
module mod_particle_recomb
  use mpi
  use mod_random_seed 
  use mod_atomic_elements
  use mod_interp
  use mod_particle_io
  use particle_tracer
  !$ use omp_lib

  implicit none

  contains

  subroutine do_1particle_recombination(element_list,node_list,target_group,jorek_stepper,rng,tstep_fluid_si)
    use mod_jorek_timestepping !< gives us access to sim?
    use mod_integrate_recomb, only : integrate_recombination
    use phys_module, only: use_manual_random_seed, central_density, central_mass, sqrt_mu0_over_rho0
  
    implicit none
  
    type(pcg32_rng), dimension(:), intent(inout)  :: rng
    type(jorek_timestep_action),target      :: jorek_stepper
    TYPE (type_node_list),     intent(in)   :: node_list
    TYPE (type_element_list),    intent(in)   :: element_list
    integer, intent(in)               :: target_group
    real*8, intent(in)              :: tstep_fluid_si
    
    !internal variables
    type (type_element)         :: element
    logical, allocatable, dimension(:) :: is_free
    integer, allocatable, dimension(:) :: i_free
    integer       :: Nrec_part, particles_per_element
    real*8        :: total_rec,total_rec_all ,total_volume,total_volume_all
    real*8        :: total_Erec_neutral,total_Erec_neutral_all, total_Erec_rad,total_Erec_rad_all
    real*8, dimension(n_plane) :: total_rec_nplane, total_volume_nplane, total_rec_nplane_all, total_volume_nplane_all
    integer       :: n_free, i, j, k,ielm,ife, i_rng, ierr, mp
    real*8        :: s, t,R, Z, phi_plane, delta_phi, phi_ran(2)
  
    !debug rec
    real*8                :: sanity_rec_local,total_sanity_rec
    !rec variables
    real*8, dimension(:,:), allocatable  :: rec_rate_local , rec_v_R, rec_v_Z, rec_v_phi 
    real*8, dimension(:,:), allocatable  :: volume_check, energy_neutrals, energy_radiation
  
    !Call mod_integrate_recombination
    call integrate_recombination(sim%my_id,sim%n_mpi, rec_rate_local, rec_v_R, rec_v_Z, rec_v_phi,volume_check, energy_neutrals, energy_radiation)

    sanity_rec_local = 0.d0
    !calculate total recombination per mpi proces
    total_volume = sum(sum( volume_check, DIM = 1 ), DIM=1)
    total_Erec_neutral = sum(sum( energy_neutrals, DIM = 1 ), DIM=1)
    total_Erec_rad = sum(sum( energy_radiation, DIM = 1 ), DIM=1)
    total_rec = sum(sum( rec_rate_local, DIM = 1 ), DIM=1)
    total_rec_nplane = sum( rec_rate_local, DIM = 1 )
    total_volume_nplane = sum( volume_check, DIM = 1 )
    ! total recombination
    call MPI_REDUCE(total_rec, total_rec_all, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    call MPI_REDUCE(total_volume, total_volume_all, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    call MPI_REDUCE(total_rec_nplane, total_rec_nplane_all, n_plane, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    call MPI_REDUCE(total_volume_nplane, total_volume_nplane_all, n_plane, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    call MPI_REDUCE(total_Erec_neutral, total_Erec_neutral_all, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    call MPI_REDUCE(total_Erec_rad, total_Erec_rad_all, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    if (sim%my_id .eq. 0) then
      write(*,'(A30,2E16.8)') 'total recombination weight : ' , sim%time,total_rec_all* central_density* 1.d20 
      write(*,*) 'total energy to recombined neutrals [J] : ' , total_Erec_neutral_all *1.5d0 / MU_ZERO
      write(*,*) 'total energy lost to Prb [J]: ' , total_Erec_rad_all *1.5d0 / MU_ZERO
      write(*,*) 'total volume : ' , total_volume_all
      write(*,'(A30,2E16.8)') 'Recombination rate  [#/s] : ', sim%time, total_rec_all* central_density* 1.d20 /tstep_fluid_si
      write(*,*) 'total power to recombined neutrals [MW]: ' , total_Erec_neutral_all *1.5d0 / MU_ZERO/tstep_fluid_si /1.d6
      write(*,*) 'total power lost to Prb [MW]: ' , total_Erec_rad_all *1.5d0 / MU_ZERO/tstep_fluid_si /1.d6
      write(*,'(A15,6E14.6)') 'TOTAL RECOMB: ',sim%time, total_rec_all* central_density* 1.d20 , total_Erec_neutral_all *1.5d0 / MU_ZERO, total_Erec_neutral_all *1.5d0 / MU_ZERO/tstep_fluid_si /1.d6, &
                  total_Erec_rad_all *1.5d0 / MU_ZERO, total_Erec_rad_all *1.5d0 / MU_ZERO/tstep_fluid_si /1.d6
      if(n_plane > 1) then
        write(*,'(A,100E16.8)') 'total recombination weight per plane : ' , total_rec_nplane_all* central_density* 1.d20 
        write(*,'(A,100E16.8)') 'total volume per plane: ' , total_volume_nplane_all
      endif
    endif
    !Nrec_part amount of particles needed for this amount of recombination
    Nrec_part = int( max(sim%groups(target_group)%n_particles * 1.d-2 ,total_rec/1.d14 ) )!< assumed average weight per particle (not necesarily the actual weight, as that depends on Srec)
    !< limited to 1% of the total initialized particles
  
  
    !============== Finding free particles !< make into a function?
    !> # is_free > n_elements * particles_per_element 
    if(use_manual_random_seed) then
      !$ call omp_set_schedule(omp_sched_static,100)
    else
      !$ call omp_set_schedule(omp_sched_dynamic,100)
    end if
  
    allocate(is_free(size(sim%groups(target_group)%particles,1))) 
    !$omp parallel do default(none) shared(sim, n_free, i_free, is_free, target_group) &
    !$omp private(j) schedule(runtime)
      do j=1,size(sim%groups(target_group)%particles,1) 
        is_free(j) = sim%groups(target_group)%particles(j)%i_elm .le. 0  !< array T/F is particle is free
      end do
    !$omp end parallel do
    !$omp barrier
    n_free = count(is_free)
    allocate(i_free(n_free))
    k = 1
    do j=1,size(is_free,1)
      if (is_free(j)) then
        i_free(k) = j !< i_free(k) has index of free particle in  sim%groups(target_group)%particles(j)
        k = k+1
      end if
    end do
    ! ==================
    delta_phi     = 2.d0 * PI / real(n_plane,8) / real(n_period,8)

    ! loop over all elements
    k = 0 !< first free particle
    particles_per_element = 1  
    select type (particles => sim%groups(target_group)%particles)
    type is (particle_kinetic_leapfrog)
      if(use_manual_random_seed) then
        !$ call omp_set_schedule(omp_sched_static,10)
      else
        !$ call omp_set_schedule(omp_sched_dynamic,10)
      end if
      !omp
#ifdef __GFORTRAN__
      !$omp parallel do default(shared) & ! workaround for Error: �__vtab_mod_pcg32_rng_Pcg32_rng� not specified in enclosing �parallel�
#else
      !$omp parallel do default(shared) &
      !$omp shared(sim,jorek_stepper, element_list, node_list, target_group, rec_v_R,rec_v_Z,rec_v_phi, &
      !$omp i_free,rng,rec_rate_local, delta_phi, &
      !$omp CENTRAL_DENSITY, CENTRAL_MASS,sqrt_mu0_over_rho0,particles_per_element ) &
#endif
      !$omp schedule(runtime)    &
      !$omp private(ife,ielm,k,i,element,s,t,R, Z , &
      !$omp phi_ran, i_rng,phi_plane) &
      !$omp reduction(+:sanity_rec_local)
        do ife = 1, size(rec_rate_local,1) ! loop over all local elements
  
          !$ i_rng = omp_get_thread_num()+1

          ! --- Get element
          ielm  = (sim%my_id+1) + sim%n_mpi*(ife - 1)
          element = element_list%element(ielm)
          
          do mp = 1, n_plane
            if (isnan(rec_v_R(ife,mp)) .or. isnan(rec_v_Z(ife,mp)) .or. isnan(rec_v_phi(ife,mp))) CYCLE !NaN check
            if (rec_rate_local(ife,mp) * central_density * 1.d20 .le. (1.d3/n_plane)) CYCLE
    
            phi_plane     = delta_phi * (mp-1)
            
            k = ife +(mp-1)* size(rec_rate_local,1)!< every OMP thread gets different values
            !< every MPI process has it's own list of i_free.


            ! initialise particle in the element with Position, Weight, Energy, Momentum
            do i = 1, particles_per_element
              k = k *i !< update free particle index ! at begin of loop as k is initialized at k =0
              particles(i_free(k))%weight = rec_rate_local(ife,mp) / real(particles_per_element,8)* central_density* 1.d20 !< rec_rate = in jorek units?
              particles(i_free(k))%i_elm  = ielm  !x, i_elm, st
              particles(i_free(k))%q    = 0
                
              sanity_rec_local = sanity_rec_local + particles(i_free(k))%weight
                
              call rng(i_rng)%next(phi_ran)
              !< sample random st combination
              particles(i_free(k))%st(1) = 0.5d0
              particles(i_free(k))%st(2) = 0.5d0
              
              s = particles(i_free(k))%st(1)
              t = particles(i_free(k))%st(2)
  
              !> uses i_elm and s,t to give us R,Z
              call interp_RZ(node_list,element_list,ielm,s,t,R,Z)
              particles(i_free(k))%x(1:2)  = [R, Z]
              particles(i_free(k))%x(3)    = phi_plane  + int(phi_ran(1)*n_period) * 2.d0*PI/real(n_period,8) + delta_phi*(phi_ran(2)-0.5d0)  !  randomly add particle to one of n_period toroidal wedges.
              
              particles(i_free(k))%v(1)  = rec_v_R(ife,mp)   / (particles(i_free(k))%weight * CENTRAL_MASS * ATOMIC_MASS_UNIT )/ sqrt_mu0_over_rho0 !m/s
              particles(i_free(k))%v(2)  = rec_v_Z(ife,mp)   / (particles(i_free(k))%weight * CENTRAL_MASS * ATOMIC_MASS_UNIT )/ sqrt_mu0_over_rho0
              particles(i_free(k))%v(3)  = rec_v_phi(ife,mp) / (particles(i_free(k))%weight * CENTRAL_MASS * ATOMIC_MASS_UNIT )/ sqrt_mu0_over_rho0
              !< v = momentum fluid lost to recombination / (mass of superparticle)
            end do ! parts_per_element
        
          end do !mp = 1, n_plane
        enddo   !ife 
      !$omp end parallel do
    end select
    !end omp
  
    call MPI_REDUCE(sanity_rec_local, total_sanity_rec, 1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    if (sim%my_id .eq. 0) then
      write(*,*) 'SANITY recombination weight : ' , total_sanity_rec 
      write(*,*) 'SANITY Recombination rate  [#/s] : ' , total_sanity_rec /tstep_fluid_si
    endif  
  
  end subroutine !do_1particle_recombination

end module mod_particle_recomb
