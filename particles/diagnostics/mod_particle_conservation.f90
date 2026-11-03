!> functions for checking the conservation of particles in kinetic monte carlo simulations

module mod_particle_conservation
  use particle_tracer
  
  implicit none

  contains
  
subroutine conservation_checks(sim)
  implicit none
  
  class(particle_sim), target, intent(in) :: sim

  integer :: i, j, ierr

  real*8  :: v_kin_temp, E(3), B(3), psi, U, B_norm(3)

  real*8  :: density_tot, density_in, density_out,  pressure, pressure_in, pressure_out
  real*8  :: mom_par_tot, mom_par_in, mom_par_out, kin_par_tot, kin_par_out, kin_par_in
  real*8, dimension(n_var) :: varminout, varmaxout
  real*8  :: particles_remaining, momentum_remaining, energy_remaining, all_particles, all_momentum, all_energy
  integer :: superparticles_remaining, all_superparticles, closest_iteration!, part_i_save,part_n_save

  call Integrals_3D(sim%my_id, sim%fields%node_list, sim%fields%element_list, density_tot, density_in, density_out, &
  pressure, pressure_in, pressure_out, kin_par_tot, kin_par_in, kin_par_out, mom_par_tot, mom_par_in, mom_par_out,varminout,varmaxout)

  do i=1, size(sim%groups)
    particles_remaining = 0.d0
    momentum_remaining  = 0.d0
    energy_remaining  = 0.d0
    superparticles_remaining = 0 !< just for debugging. Use count_action in actual simulation

    select type (particles => sim%groups(i)%particles)
      type is (particle_kinetic_leapfrog)
#ifdef __GFORTRAN__
      !$omp parallel do default(shared) & ! workaround for Error: �__vtab_mod_pcg32_rng_Pcg32_rng� not specified in enclosing �parallel�
#else
      !$omp parallel do default(none) &
      !$omp shared(sim) &
#endif
      !$omp reduction(+:particles_remaining, momentum_remaining, energy_remaining,superparticles_remaining) &
      !$omp private(j, E, B, psi, U, B_norm)
      do j=1,size(particles,1)

        if (particles(j)%i_elm .le. 0) cycle

        call sim%fields%calc_EBpsiU(sim%time , particles(j)%i_elm, particles(j)%st, particles(j)%x(3), E, B, psi, U)
        B_norm = B/norm2(B)

        particles_remaining = particles_remaining + particles(j)%weight
        momentum_remaining  = momentum_remaining  + particles(j)%weight * dot_product(B_norm,particles(j)%v) *sim%groups(1)%mass * ATOMIC_MASS_UNIT
        energy_remaining  = energy_remaining  + particles(j)%weight * dot_product(particles(j)%v,particles(j)%v) *sim%groups(1)%mass * ATOMIC_MASS_UNIT /2.d0
        superparticles_remaining = superparticles_remaining + 1

      enddo !j
      !omp end parallel do
    end select

    call MPI_REDUCE(particles_remaining, all_particles,     1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    call MPI_REDUCE(momentum_remaining,  all_momentum,      1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    call MPI_REDUCE(energy_remaining,  all_energy,      1, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    call MPI_REDUCE(superparticles_remaining,all_superparticles,1, MPI_INTEGER,      MPI_SUM, 0, MPI_COMM_WORLD, ierr) 

    if (sim%my_id .eq. 0) then
      write(*,'(A,A3,A)') " ---- Conservation Checks for Group: ", sim%groups(i)%id, " ---- "
      write(*,'(A,3e16.8)') 'REMAINING (START) : ',all_particles, all_momentum, all_energy

      write(*,'(A,126e16.8)') ' TOTAL1 : ',sim%time,density_tot+all_particles/1.d20, density_tot, all_particles/1.d20, &
              mom_par_tot+all_momentum, mom_par_tot, all_momentum, &
              pressure+kin_par_tot+all_energy, pressure, all_energy, kin_par_tot

      write(*,'(A,I13,A,E9.2,A,F17.10,A)') 'Superparticles in use ',all_superparticles,' of ', sim%groups(i)%n_particles, '| in use :', &
        real(all_superparticles)/sim%groups(i)%n_particles*100.d0,'%'

      if ( all_superparticles .gt. 0 ) then
        write(*,'(A,2E16.8)') 'Average weight of particles ',(all_particles)/all_superparticles
      endif !real(count(

      write(*,*) ! line break

    endif !(sim%my_id .eq. 0)
  enddo ! size(sim%groups)

  if (sim%my_id .eq. 0) write(*,*) " ---- End of conservation checks ---- "
  
end subroutine conservation_checks
  
end module mod_particle_conservation