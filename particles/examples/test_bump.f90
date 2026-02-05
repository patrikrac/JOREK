!> Test program for reproducing bump-on-tail distribution for runaway electrons
!>
!> This program uses fixed values for B, E and ne and Te, and also the timestep is huge. These are done
!> because otherwise reproducing the distribution would not be feasible with a Monte Carlo method.
!>
!> Parts of the code are commented out on purpose as uncommenting those will allow one to use this program
!> for postprocessing actual JOREK restart files with simple simulations.
module testccoll_helpers
  use particle_tracer
  use mod_particle_io
  use mod_particle_diagnostics
  use mod_gc_relativistic
  use hdf5_io_module
  use constants, only: EL_CHG, SPEED_OF_LIGHT, ATOMIC_MASS_UNIT
  use mod_coordinate_transforms, only: vector_cylindrical_to_cartesian
  
  implicit none

  real*8 :: Bnorm       = 4.0d0  !< The value of the (toroidal) magnetic field [T] in the test
  real*8 :: Enorm       = 3.d-2  !< The value of the (toroidal/parallel) electric field [V/m] in the test
  real*8 :: temperature = 16.2d3 !< The value of the electron and ion temperatures [eV] in the test
  real*8 :: density     = 1.d19  !< The value of the electron and ion densities [m^-3] in the test

contains

!< Initialize markers with fixed position and momentum
subroutine init_markers(sim, nprt, mass, chargenum, r, z, p, xi, jorek_data)
  implicit none

  type(particle_sim), intent(inout) :: sim
  integer*4, intent(in) :: nprt
  real*8, intent(in)    :: mass
  integer, intent(in)   :: chargenum
  real*8, intent(in)    :: r, z, p, xi !< p = p_SI / mc, xi = ppar / p
  logical, intent(in)   :: jorek_data  

  integer*4 :: iprt
  integer*4 :: ifail
  real*8 :: energy, pnorm, E(3), B(3), psi, U

  type(particle_gc_relativistic) :: particle_out, prtgc
  type(particle_kinetic_relativistic) :: prtprt

  sim%groups(1)%mass = mass

  do iprt = 1,nprt
     select type (prt=>sim%groups(1)%particles(iprt))
     type is (particle_gc_relativistic)

        prt%q = chargenum
        prt%x = [r, z, 0.0]


        if (jorek_data) then

          prt%i_elm = 0
          call find_RZ(sim%fields%node_list, sim%fields%element_list, &
               prt%x(1), prt%x(2), &
               prt%x(1), prt%x(2), prt%i_elm, prt%st(1), prt%st(2), ifail)

          energy = sqrt( 1.0 + p**2 ) * mass * (ATOMIC_MASS_UNIT*SPEED_OF_LIGHT**2) / EL_CHG
          particle_out = relativistic_gc_momenta_from_E_cospitch(&
               prt,energy, xi,sim%groups(1)%mass,&
               sim%fields,sim%time)
          
          prt%p = particle_out%p

        else

          prt%i_elm = 1

          B = [0.d0, 0.d0, Bnorm]
          pnorm = p * ( sim%groups(1)%mass * SPEED_OF_LIGHT )
          prt%p(1) = pnorm * xi
          prt%p(2) = ( pnorm**2 - prt%p(1)**2 ) / ( 2 * norm2(B) * sim%groups(1)%mass )

        end if

     type is (particle_kinetic_relativistic)

        prtgc%q = chargenum
        prtgc%x = [r, z, 0.0]

        if (jorek_data) then

          prtgc%i_elm = 0
          call find_RZ(sim%fields%node_list, sim%fields%element_list, &
               prtgc%x(1), prtgc%x(2), &
               prtgc%x(1), prtgc%x(2), prtgc%i_elm, prtgc%st(1), prtgc%st(2), ifail)

          energy = sqrt( 1.0 + p**2 ) * mass * (ATOMIC_MASS_UNIT*SPEED_OF_LIGHT**2) / EL_CHG
          particle_out = relativistic_gc_momenta_from_E_cospitch(&
               prtgc,energy, 0.d0,sim%groups(1)%mass,&
               sim%fields,sim%time)
          prtgc%p = particle_out%p

          call sim%fields%calc_EBpsiU(sim%time, prtgc%i_elm, prtgc%st, prtgc%x(3), E, B, psi, U)
          prtprt = relativistic_gc_to_relativistic_kinetic(sim%fields%node_list, sim%fields%element_list, &
               prtgc,sim%groups(1)%mass,B,0.0)

          prt%p = prtprt%p
          prt%x = prtprt%x
          prt%q = prtgc%q

          prt%i_elm = 0
          call find_RZ(sim%fields%node_list, sim%fields%element_list, &
               prt%x(1), prt%x(2), &
               prt%x(1), prt%x(2), prt%i_elm, prt%st(1), prt%st(2), ifail)

        else

          prtgc%i_elm = 1

          B = [0.d0, 0.d0, Bnorm]
          pnorm = p * ( sim%groups(1)%mass * SPEED_OF_LIGHT )

          prt%p = [sqrt(1.d0 - xi**2)*pnorm, xi*pnorm, 0.0]
          prt%x = prtgc%x
          prt%q = prtgc%q

          prt%i_elm = 1

        end if

     end select
  end do
   
end subroutine init_markers


subroutine write_state_hdf5(fnout, p, xi)
  implicit none
  character(len=40)  :: fnout
  real*8, intent(in) :: p(:), xi(:)
  integer*4 :: nprt
  integer(HID_T) :: file
  integer :: hdferr

  nprt = size(p,1)

  call h5open_f(hdferr)
  call h5fcreate_f(fnout, H5F_ACC_TRUNC_F, file, hdferr)
  if (hdferr .gt. 0) then
     write(*,*) "file open failed:", hdferr
     return
  end if

  call HDF5_array1D_saving(file, p,  nprt, "p")
  call HDF5_array1D_saving(file, xi, nprt, "xi")

  call h5fclose_f(file, hdferr)
  call h5close_f(hdferr)

end subroutine write_state_hdf5


end module testccoll_helpers



program test_bump

  use particle_tracer
  use mod_event
  use mod_particle_io
  use mod_particle_diagnostics
  use mod_fields_linear   
  use mod_fields_hermite_birkhoff 
  use mod_gc_relativistic
  use mod_ccoll_relativistic
  use mod_radreactforce
  use mod_kinetic_relativistic
  use mod_impurity, only: init_imp_adas
  use testccoll_helpers

  use constants, only: EL_CHG, SPEED_OF_LIGHT, ATOMIC_MASS_UNIT, PI, EPS_ZERO, MASS_ELECTRON, K_BOLTZ

  implicit none

! Set up the simulation variables
real*8      :: tstep, deltat, duration, mass
real*8      :: time_startsim, time_endsim, target_time, tracetime
integer*4   :: nstep, ifail
type(event) :: fieldreader

character(len=40) :: fnout !< File where the output is written

    
real*8, allocatable :: p(:), xi(:)
real*8 :: raxis, zaxis, taxis, saxis, psiaxis, ielmaxis
integer*4 :: iprt, nprt, istep, chargenum
logical :: full_orbit, jorek_data, part_screen
real*8 :: E(3), B(3), psi, U, p0, xi0, rc0, zc0
real*8 :: pnorm, pin, xiin, pout, xiout, rnd(2), rndprt(3), pinprt(3), poutprt(3)
real*8, dimension(:), allocatable    :: ni
real*8 :: the, thi(1), ne
real*8 :: bhat(3), bperp(3)
integer :: ierr

! For CPU time
real*8 :: t0, t1

type(ccoll_data) :: dat

!!! Simulation options begin !!!
tstep         = 1.e-7   !< Marker time step
duration      = 1e0     !< How long markers are traced
nprt          = 48*1    !< Number of markers
time_startsim = 0.0
full_orbit    = .false. !< Use full orbit markers instead of guiding-center
jorek_data    = .false. !< Use field from jorek restart files
part_screen   = .false. !< Use the partial screening collision operator

if (full_orbit) then
  fnout = "bump_fo.h5"
else
  fnout = "bump_gc.h5"
end if

! Define markers
rc0  = 6.2
zc0  = 0.0
p0   = 0.2
xi0  = -0.99
mass = 0.000548579909
chargenum = -1

call random_seed()
!!!

if (jorek_data) then
  call sim%initialize(num_groups=1)
  call init_imp_adas(0)
  call ccoll_init('ccolldata', dat)
else
  dat = ccoll_read_L0L1table('ccolldata')
  allocate(sim%groups(1))
  allocate( ni(1), dat%mi(1), dat%Z0(1), dat%Zi(1), dat%ai(1), dat%Ii(1))
  dat%mi(1) = atomic_mass_unit
  dat%Z0(1) = 1
  dat%Zi(1) = 1
  dat%ai(1) = 1
  dat%Ii(1) = 1.d0
end if

! Toggle here between GC and gyro-orbit

if (full_orbit) then
  allocate(particle_kinetic_relativistic::sim%groups(1)%particles(nprt))
else
  allocate(particle_gc_relativistic::sim%groups(1)%particles(nprt))
end if

! Allocate and initialize needed data arrays
allocate( p(nprt), xi(nprt) )

call cpu_time(t0)

! Begin simulation at new time slice by initializing the field at that point
sim%time = time_startsim

if (jorek_data) then
  fieldreader = event(read_jorek_fields_interp_linear(i=-1))!last_file_before_time(sim%time)))
  call with(sim,fieldreader)
  events = [fieldreader]
end if

! Initialize markers
call init_markers(sim, nprt, mass, chargenum, rc0, zc0, p0, xi0, jorek_data)

! Trace markers for the given duration
nstep  = nint(duration/tstep)
deltat = duration / nstep


do istep=1,nstep
   select type (prt => sim%groups(1)%particles)

   type is (particle_gc_relativistic)

      !$omp parallel do default(shared) &
      !$omp private(pnorm, pin, pout, xiin, xiout, rnd) &
      !$omp private(the, thi, ne, ni, E, B, psi, U) &
      !$omp private(iprt, ifail)
      do iprt=1,nprt

         the = temperature * EL_CHG / (MASS_ELECTRON * SPEED_OF_LIGHT**2)
         thi = [ temperature * EL_CHG / (dat%mi(1) * SPEED_OF_LIGHT**2) ]
         ne = density
         ni = [density]
         B  = [0.d0, 0.d0, Bnorm]
         E  = [0.d0, 0.d0, Enorm]

         if (prt(iprt)%i_elm .gt. 0) then

            if (jorek_data) then
              !** Use these when using actual JOREK data **!

              call runge_kutta_fixed_dt_gc_push_jorek_radreact(sim%fields,sim%time,deltat, sim%groups(1)%mass, prt(iprt))

              if (part_screen) then
                call ccoll_gc_relativistic_push_partialscreening(dat, prt(iprt), sim%fields, sim%groups(1)%mass, sim%time,deltat)
              else
                call ccoll_gc_relativistic_push(dat, prt(iprt), sim%fields, sim%groups(1)%mass, sim%time,deltat)
              end if
            
            else

              !** Test block **!
              ! Test field is uniform so we can push the marker explicitly (just accelerating it in E-field)
              prt(iprt)%p(1) = prt(iprt)%p(1) + (EL_CHG*prt(iprt)%q / ATOMIC_MASS_UNIT) * norm2(E) * deltat
              call radreactforce_gc(B, deltat, sim%groups(1)%mass, prt(iprt))
              
              pnorm = sqrt(prt(iprt)%p(2) * 2 * norm2(B) * sim%groups(1)%mass + prt(iprt)%p(1)**2)
              pin   = pnorm / ( sim%groups(1)%mass * SPEED_OF_LIGHT )
              xiin  = prt(iprt)%p(1) / pnorm

              call random_number(rnd)
              rnd = floor(2*rnd)
              rnd = -1.0 + 2.0 * rnd


              if (part_screen) then
                !** Use this instead to test the partial screening operator **!
                call ccoll_explicitpush_partialscreening(dat, ne, the, ni, pin, pout, xiin, xiout, deltat, rnd, 1.0e-4, ifail)
              else
                call ccoll_gc_relativistic_explicitpush(dat, sim%groups(1)%mass * ATOMIC_MASS_UNIT, prt(iprt)%q, &
                   ne, the, ni, thi, pin, pout, xiin, xiout, deltat, rnd, 1.0e-4, ifail)
              end if

              p(iprt)  = pout
              xi(iprt) = xiout

              ! Collisions don't move markers so we can just use the same B here to convert back to GC momentum
              pnorm = pout * ( sim%groups(1)%mass * SPEED_OF_LIGHT )
              prt(iprt)%p(1) = pnorm * xiout
              prt(iprt)%p(2) = ( pnorm**2 - prt(iprt)%p(1)**2 ) / ( 2 * norm2(B) * sim%groups(1)%mass )
              !** Test block ends **!

          end if
         end if
      end do
      !$omp end parallel do

   type is (particle_kinetic_relativistic)

      !$omp parallel do default(shared) &
      !$omp private(pnorm, rndprt, poutprt) &
      !$omp private(the, thi, ne, ni, E, B, psi, U) &
      !$omp private(bhat, bperp, xiout, xiin, pin, pout) &
      !$omp private(iprt, ifail)
      do iprt=1,nprt

         the = temperature * EL_CHG / (MASS_ELECTRON * SPEED_OF_LIGHT**2) 
         thi = [temperature * EL_CHG / (dat%mi(1) * SPEED_OF_LIGHT**2) ]
         ne = density
         ni = [density]
         B  = [0.d0, 0.d0, Bnorm]
         E  = [0.d0, 0.d0, Enorm]
          
         if (prt(iprt)%i_elm .gt. 0) then

            if (jorek_data) then
              !** Use these when using actual JOREK data **!

              call volume_preserving_radiation_push_jorek(prt(iprt),sim%fields,sim%groups(1)%mass,sim%time,deltat,ifail)

              if (part_screen) then
                call ccoll_kinetic_relativistic_push_partialscreening(dat, prt(iprt), sim%fields, sim%groups(1)%mass, sim%time,deltat)
              else
                call ccoll_kinetic_relativistic_push(dat, prt(iprt), sim%fields, sim%groups(1)%mass, sim%time,deltat)
              end if

            else

              !** Test block **!
              ! Test field is uniform so we can push the marker explicitly (just accelerating it in E-field)
              prt(iprt)%p =  prt(iprt)%p + EL_CHG * prt(iprt)%q * vector_cylindrical_to_cartesian(prt(iprt)%x(3), E) * deltat / ATOMIC_MASS_UNIT
              call radreactforce_kinetic(B, deltat, sim%groups(1)%mass, prt(iprt))
              
              ! For collisions we need to convert from ppar,mu to pnorm,pitch (and then revert)
              pnorm = norm2(prt(iprt)%p) / (sim%groups(1)%mass * SPEED_OF_LIGHT)

              call random_number(rndprt)
              rndprt = floor(2*rndprt)
              rndprt = -1.0 + 2.0 * rndprt

              pnorm = norm2(prt(iprt)%p)


              if (part_screen) then
              
                !** Use this instead to test the partial screening operator **!
                bhat = vector_cylindrical_to_cartesian(prt(iprt)%x(3), B) / norm2(B)
                pin  = norm2(prt(iprt)%p) / (mass * SPEED_OF_LIGHT)
                xiin = dot_product(prt(iprt)%p, bhat) / norm2(prt(iprt)%p)
                call ccoll_explicitpush_partialscreening(dat, ne, the, ni, pin, pout, xiin, xiout, deltat, rndprt, 1.0e-4, ifail)
                bperp = prt(iprt)%p - dot_product(prt(iprt)%p, bhat) * bhat
                bperp = bperp / norm2(bperp)
                poutprt = (xiout * bhat + sqrt( 1.d0 - xiout**2 ) * bperp ) * pout
              
              else

                call ccoll_kinetic_relativistic_explicitpush(dat, sim%groups(1)%mass * ATOMIC_MASS_UNIT, prt(iprt)%q, &
                    ne, the, ni, thi, deltat, rndprt, prt(iprt)%p / (sim%groups(1)%mass * SPEED_OF_LIGHT), poutprt)

              end if

              prt(iprt)%p = poutprt * (sim%groups(1)%mass * SPEED_OF_LIGHT)
              poutprt = prt(iprt)%p / (sim%groups(1)%mass * SPEED_OF_LIGHT)
              p(iprt)  = norm2(poutprt) 
              xi(iprt) = dot_product(prt(iprt)%p, vector_cylindrical_to_cartesian(prt(iprt)%x(3), B)) / ( norm2(prt(iprt)%p) * norm2(B) )

                !** Test block ends **!

          end if
         end if
      end do
      !$omp end parallel do

   end select

end do

call cpu_time(t1)
write(*,*) 'CPU time: ', t1-t0 

call write_state_hdf5(fnout, p, xi)

! Finalize the simulation
call ccoll_deallocate(dat)
deallocate ( p, xi, ni )
call sim%finalize

end program test_bump
