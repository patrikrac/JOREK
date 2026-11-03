!> Module for neutral-neutral collisions
!> Currently only self collisions are implemented, which can be switched on from the input
!> Sources cited in this file: 
!> [1]: Bird, G. A. (1994). Molecular Gas Dynamics and the Direct Simulation of Gas Flows.
!> [2]: Liebermann (2005). Principles of Plasma Discharges and Materials Processing.
!>
!> For more information on the neutral neutral collisions, see https://jorek.eu/wiki/doku.php?id=particles:neutral_neutral_collisions
module mod_neutral_collision
  use mod_particle_types
  use constants
  use mod_pcg32_rng,   only: pcg32_rng
  use mod_particle_sim
  use phys_module, only: n_period, use_manual_random_seed,n_plane !< we need the intrinsic fortran gamma function so have to use only
  use mod_interp, only: interp_RZ, interp_0
  use mod_event, only: mpi_minmeanmax
  use mod_particle_sorting
  use mpi
  use nodes_elements
  
  !$ use omp_lib

  implicit none
   
  private
  public :: type_neutral_collision, neutral_collisions_from_config, nonzero, gcd_neutral_collisions

  !> action object, storing P_max values per element
  type :: type_neutral_collision
    integer                                    :: group_num           !< which group number this collision is about
    real*8,          dimension(:), allocatable :: P_max_elm           !< [chance] rescaling factor for collision probability for this element, based on a large estimate for the actual chance a given physical particle will collide in the element (n_elm)
    type(pcg32_rng), dimension(:), allocatable :: rng                 !< rng object for this action specifically
    real*8,          dimension(3)              :: dTw                 !< the d_ref, T_ref and omega of the variable hard sphere model for this neutral species
    integer                                    :: each_nstep_part     !< run this particular action at every i_inner_loop = each_nstep_part
    logical                                    :: constructed=.false. !< whether initialization finished
  contains
    procedure :: initialize
    procedure :: do => neutral_self_collision
  end type 

  !> object to store an index array to randomly draw from the unused indices
  type :: random_draw
    integer :: size !< size of the index array
    integer, dimension(:), allocatable :: indices !< indices list (used indices are at the front) (size)
    integer :: used                               !< number of indices used
  contains
    procedure :: reset
    procedure :: next
  end type

  integer :: gcd_neutral_collisions = -9999991 !< greatest common divisor of the %each_nstep_part of all neutral collision objects

contains

!> Neutral-neutral collision within a species 
!> Uses a direct binary collisions approach where each all particles are sorted into collisional bins
!> Each element is divided equally in the toroidal direction (based on n_plane). The division of the 
!> element into bins in the poloidal direction is done by dividing the element up in an s,t grid
!> which is determined automatically by the amount of particles and the length of the element along s and t.
!> Within a collisional bin, superparticles are paired up randomly, and their real collision chance is determined from
!> P=n sigma v_r dt, with n the sampled background density of the species, sigma the cross section of the collision (which is
!> a function of v_r as the variable hard sphere model is used), v_r the relative velocity between the two particles and dt
!> the timestep. This real collisional chance (which approximates the chance that a physical particle would collide during this 
!> timestep) is then rescaled by the maximum real collisional chance of this element of the previous timestep, while the amount
!> of pairs to be tried for collision is lowered to compensate for this. This means fewer particles need to be tried for
!> collisions while retaining the correct collisional frequency, thus saving on computational time. This method is called the
!> no time counter (NTC) method (see: [1] sections 11.2-11.4).
!> Since we are using variable weights and we don't split the heavier particle into a colliding and non-colliding part
!> (which would create a new particle for every colliding pair), it is impossible to conserve both momentum and energy.
!> It is chosen to conserve energy and keep the large angle scattering behaviour also for the heavier particle to ensure
!> the correct diffusive behaviour.
!>
!> For more information on the neutral neutral collisions, see https://jorek.eu/wiki/doku.php?id=particles:neutral_neutral_collisions
subroutine neutral_self_collision(this, sim, nodes, elements)
  implicit none

  class(type_neutral_collision), intent(inout) :: this
  type(particle_sim),            intent(inout) :: sim
  type (type_node_list),         intent(in)    :: nodes    !< aux node list to sample neutral density from (neutral density is assumed to be at index 5)
  type (type_element_list),      intent(in)    :: elements !< element list (necessary to sample neutral density from)

  type(particle_kinetic_leapfrog)                                :: pa1            !< first particle of the collisional pair
  type(particle_kinetic_leapfrog)                                :: pa2            !< second particle of the collisional pair
  type(indices_in_elm),            dimension(:),     allocatable :: sorted_ind_arr !< object containing all particle indices as arrays per element number (n_elm)
  type(indices_in_elm),            dimension(:,:,:), allocatable :: i_pa_bin       !< object containing all this element's particle indices as arrays per bin (s bin,t bin, phi bin)
  type(random_draw)                                              :: i_random       !< particle index list (1 to n_pa_bin) to draw from to determine next random particle for pairing. Already used indices are at the front. (n_pa_bin)

  integer, parameter :: i_neutral_n=1 !< index in received nodes list of neutral density
  integer, parameter :: aim_pa_per_bin = 50 !< wanted amount of super particles in one collisional bin (element is split up to satisfy this). Bigger means better statistics of particle density, but also loss of locality. Much smaller than 50 (like 2-10 or so) is not advised because the momenta of the particles in the bin might get correlated over time
  
  integer, dimension(:),       allocatable :: i_pa_elm      !< global particle indices of the particles in this element (pa_in_elm)
  integer, dimension(:,:,:),   allocatable :: n_pa_bin_arr  !< number of particles in each collisional bin of this element (s bin, t bin, phi bin)
  integer, dimension(:,:,:),   allocatable :: i_loc_bin     !< local index of particles in each collisional bin of this element (s bin, t bin, phi bin)
  logical, dimension(:),       allocatable :: paired        !< array to keep track of which particle is not paired yet (pa_in_bin)
  
  ! arrays used for determining sorted_ind_arr
  integer, dimension(:),       allocatable :: pa_in_elm_arr     !< number of particles in the element (n_elm)

  real*8  :: dt        !< timestep over which the collisions must be calculated

  integer :: n_phi !< number of toroidal bins to do collisions in
  integer :: i, j, i_elm, i_loc, i_global, i_phi, is, it, ns, nt, i_pair, n_elm
  integer :: pa_in_elm !< number of particles in the element under consideration
  integer :: n_pa_bin  !< number of particles in this collisional bin
  integer :: i_thread

  real*8 :: P_real  !< [chance] chance that this particle will collide (with anything) this timestep, P_real = n \sigma_T v_r dt
  real*8 :: P_try   !< [chance] rescaled chance of this pair colliding P_try = P_real / P_max_elm(i_elm)
  real*8 :: w2, w1  !< [physical particles] weight of super particle 1
  real*8 :: w_s     !< [physical particles] weight of smaller super particle in binary collision w_s = min(w1,w2)
  real*8 :: w_g     !< [physical particles] weight of greater super particle in binary collision w_g = max(w1, w2)
  real*8 :: v_r     !< [m/s] scalar relative velocity of collisional pair
  real*8 :: sigma_T !< [m^2] total collisional cross section of the collision pair
  real*8 :: n_loc   !< [m^-3] local density for determining P_real
  real*8 :: E_i     !< [J*amu/kg] initial energy before collision (=w1 m1 v1i^2 + w2 m2 v2i^2)
  real*8 :: v_g     !< [m/s] scalar velocity of larger weight particle after the collision (used to conserve energy for variable weights)

  real*8 :: R, R_s, R_t, Z, Z_s, Z_t, P(1)
  real*8 :: nst
  real*8 :: lt,ls     !< [m] physical size of element / bin in s direction
  real*8 :: V_c       !< [m^3] grid cell (collisional bin) volume 
  real*8 :: RN(5)     !< random numbers for particle 1, particle 2, chance to collide, impact parameter and scattering plane angle
  real*8 :: P_max_now !< for updating this%P_max_elm
  integer :: N_try    !< number of pairs to be tried for collisions (NTC method, N_try = n_pa_bin/2 * P_max_real)
  integer :: i_pa1, i_pa2
  
  !elastic collision parameters
  real*8 :: alpha  !< scattering plane angle to Z-axis around *v1*
  real*8 :: Theta  !< scattering angle in normal coordinates
  real*8 :: m2, m1 !< mass of particle 1
  real*8 :: v1i(3) !< initial velocity of particle 1
  real*8 :: v2i(3) !< initial velocity of particle 2
  real*8 :: v1f(3) !< final   velocity of particle 1
  real*8 :: v2f(3) !< final   velocity of particle 2
  
  !global diagnostics
  integer, parameter :: n_diag=11     !< number of global diagnostics
  integer, parameter :: i_P_av=1      !< index of average collisions chance (=sum(P)/n_pairs)
  integer, parameter :: i_tau=2       !< index of average collision time (=dt/P_av)
  integer, parameter :: i_sigma_av=3  !< index of average collision cross section (=sum(sigma)/n_pairs)
  integer, parameter :: i_n_pairs=4   !< index of number of pairs (=sum(N_try))
  integer, parameter :: i_n_col=5     !< index of number of collided pairs
  integer, parameter :: i_w_col=6     !< index of sum of collided weight
  integer, parameter :: i_Pgt1=7      !< index of pairs with P > 1
  integer, parameter :: i_Pst1=8      !< index of pairs with P < 0
  integer, parameter :: i_V=9         !< index of total volume of all cells (=sum(V_c), should be equal to domain volume if all elements have colliding particles)
  integer, parameter :: i_d_av=10     !< index of average VHS diameter (=sum(d)/n_pairs)
  integer, parameter :: i_angle_av=11 !< index of average collisional angle fraction (=(sum(angle)/n_col)/(PI/2)). If this is 1, it is a random walk (so MFP = v*tau), if this is 0 the MFP is infinite, and if it is >!, then MFP < v*tau (with limit 0 for fraction 2?)  
  real*8 :: global_diag(n_diag) !< global diagnostics
  real*8 :: reduced_global_diag(n_diag) !< MPI reduced global_diag
  character(len=100) :: FMT !< writing format for diagnostics
  
  integer :: max_n_pa(2), max_n_pa_gl(2)
  real*8 :: P_max_mpi, P_max_global
  real*8 :: P_max_elm_max, P_max_elm_min !< min P_max_elm for logging purpose
  real*8 :: P_max_elm_max_red, P_max_elm_min_red !< MPI reduced P_max_elm_min
  integer :: ierr
  !$ real*8 :: w(2), mmm(3)

  ! --- start of code

  !check whether this action should be run right now
  if(.not. (mod(sim%istep_inner_loop,this%each_nstep_part)==0 .or. sim%istep_inner_loop==sim%nstep_inner_loop)) return

  !$ w(1) = omp_get_wtime()
  
  if(sim%my_id == 0) write(*,"(3A)") "--- self collisions of species ",sim%groups(this%group_num)%id," ---"

  if(.not. this%constructed) then
    if(sim%my_id==0) write(*,"(A)") "ERROR: something went wrong in the initialization of this neutral_collision object as it did not finish. Aborting"
    stop
  end if

  if(this%each_nstep_part == -9999991) then
    dt = sim%nstep_inner_loop*sim%tstep_part_adj
  else
    dt = this%each_nstep_part*sim%tstep_part_adj
  endif

  select type (pa => sim%groups(this%group_num)%particles)
  type is (particle_kinetic_leapfrog)
  
    n_elm = sim%fields%element_list%n_elements
    
    global_diag(:)=0.d0

    ! for now it is assumed that having n_plane toroidal bins is enough locality in the toroidal direction. 
    ! If it turns out that is not the case, it might make sense to have this as a lower limit and let the amount of toroidal bins scale with the amount of particles in the elements
    ! (just as the number of s and t bins scale as such)
    n_phi = n_plane
    
    ! sorting the particle indices into arrays by their element number
    call sort_particles_in_elm(sim%groups(this%group_num)%particles,n_elm,sorted_ind_arr,pa_in_elm_arr)

    max_n_pa = 0
    P_max_mpi = 0
    ! Start loop over each finite element number
    if(use_manual_random_seed) then
      !$ call omp_set_schedule(omp_sched_static,1)
    else
      !$ call omp_set_schedule(omp_sched_dynamic,1)
    end if
#ifdef __GFORTRAN__
    !$omp parallel do default(shared)                                                            &
#else
    !$omp parallel do default(none)                                                              &
    !$omp shared(this, sim, dt, nodes, elements,                                                  &
    !$omp n_elm, n_phi, sorted_ind_arr, pa_in_elm_arr)                               &
#endif
    !$omp schedule(runtime)                                                                      &
    !$omp private(i_pa_elm, pa_in_elm, i, i_global, i_phi, it, is, nst, ns, nt, R, R_s, R_t, Z, Z_s, Z_t,  &
    !$omp ls, lt, i_pa_bin, n_pa_bin_arr, n_pa_bin, pa1, pa2, V_c, P_max_now,  &
    !$omp i_random, i_pair, i_pa1, i_pa2, w1, w2, w_s, w_g, v_r, m1, m2, E_i, v1i, v2i, v_g, P, n_loc, &
    !$omp sigma_T, P_try, P_real, N_try, i_thread, RN, Theta, alpha, v1f, v2f, i_loc, i_loc_bin) &
    !$omp reduction(+:global_diag) reduction(max:max_n_pa, P_max_mpi)
    do i_elm=1,n_elm
      i_thread = 1 !default if not using OMP
      !$ i_thread = omp_get_thread_num()+1
      
      if(allocated(i_pa_elm)) deallocate(i_pa_elm)
      allocate(i_pa_elm(pa_in_elm_arr(i_elm)))
      i_pa_elm = sorted_ind_arr(i_elm)%pa_ind(:)
      
      pa_in_elm = size(i_pa_elm,1)        
      
      if(pa_in_elm > max_n_pa(1)) max_n_pa(1) = pa_in_elm

      !> skip collisions if there are nearly no particles
      if(pa_in_elm .lt. 2*n_phi) cycle
      
      ! --- determine how to split up this bin poloidally
      ! nst is real to give the best approximation to ns, nt
      nst = pa_in_elm/real(aim_pa_per_bin * n_phi) !< number of bins in poloidal plane

      if(nst .gt. 1.d0) then        
        !determine how to distribute the poloidal bins based on what size the element is along the s and t coordinates
        call interp_RZ(sim%fields%node_list,sim%fields%element_list,i_elm,0.5d0,0.5d0,R,R_s,R_t,Z,Z_s,Z_t)
        ls = sqrt(R_s**2 + Z_s**2)
        lt = sqrt(R_t**2 + Z_t**2)
        
        !aim for ns*nt = nst together with (ls/lt)nt = ns and (ls/lt)ns = nt
        ns = nint(sqrt(nst*ls/lt))
        nt = nint(sqrt(nst*lt/ls))

        !if ns = nt = 1 while nst != 1, set the biggest side to nst
        if(ns .le. 1 .and. nt .le. 1) then
          if (ls < lt) then
            ns = 1
            nt = floor(nst)
          else 
            nt = 1
            ns = floor(nst)
          end if
        end if
        !if the element is very elongated, make sure there's always at least one bin in both directions
        if(ns .le. 1) then
          ns = 1
          nt = floor(nst)
        end if
        if(nt .le. 1) then
          nt = 1
          ns = floor(nst)
        end if
      else 
        ns = 1
        nt = 1
      end if

      ! --- distribute the particles in the element over the toroidal and poloidal collisional bins based on their s, t and phi coordinates
      allocate(i_pa_bin(ns,nt,n_phi))
      allocate(n_pa_bin_arr(ns,nt,n_phi))
      
      n_pa_bin_arr = 0

      !loop to get amount of particles per bin
      do i=1,pa_in_elm
        i_global = i_pa_elm(i)
        is = ith_bin(pa(i_global)%st(1),ns)
        it = ith_bin(pa(i_global)%st(2),nt)
        
        i_phi = ceiling(pa(i_global)%x(3)/ (2.d0 * PI / float(n_period*n_phi)))
        if (i_phi .gt. n_phi) i_phi = mod(i_phi,n_phi)
        if (i_phi .lt. 1)     i_phi = mod(i_phi,n_phi) + n_phi
        
        n_pa_bin_arr(is,it,i_phi) = n_pa_bin_arr(is,it,i_phi) + 1
      end do !i_pa_elm

      !loop to allocate bin sizes in i_pa_bin
      do i_phi=1,n_phi
        do it=1,nt
          do is=1,ns
            allocate(i_pa_bin(is,it,i_phi)%pa_ind(n_pa_bin_arr(is,it,i_phi)))
          end do
        end do
      end do

      !fill i_pa_bin
      allocate(i_loc_bin(ns,nt,n_phi))
      i_loc_bin = 1
      do i=1,pa_in_elm
        i_global = i_pa_elm(i)
        is = ith_bin(pa(i_global)%st(1),ns)
        it = ith_bin(pa(i_global)%st(2),nt)
        
        i_phi = ceiling(pa(i_global)%x(3)/ (2.d0 * PI / float(n_period*n_phi)))
        if (i_phi .gt. n_phi) i_phi = mod(i_phi,n_phi)
        if (i_phi .lt. 1)     i_phi = mod(i_phi,n_phi) + n_phi
        
        i_loc = i_loc_bin(is,it,i_phi)
        i_loc_bin(is,it,i_phi) = i_loc_bin(is,it,i_phi) + 1
        i_pa_bin(is,it,i_phi)%pa_ind(i_loc) = i_global
      end do !i_pa_elm
      deallocate(i_loc_bin)
      
      ! --- loop through each collisional bin
      P_max_now = 0.d0
      do i_phi=1,n_phi 
        do it=1,nt
          do is=1,ns 
            n_pa_bin = n_pa_bin_arr(is,it,i_phi) ! n_pa_bin is the shorthand notation
            
            if(n_pa_bin > max_n_pa(2)) max_n_pa(2) = n_pa_bin
            if(n_pa_bin .le. 1) cycle !< can't collide 0 or 1 particles
            
            ! detemine bin volume (this would be necessary when the heavier particle would be split up, now it is only present as a sanity check)
            call interp_RZ(sim%fields%node_list,sim%fields%element_list,i_elm,(real(is)+0.5d0)/real(ns),(real(it)+0.5d0)/real(nt),R,R_s,R_t,Z,Z_s,Z_t)
            ls = sqrt(R_s**2 + Z_s**2)/real(ns)
            lt = sqrt(R_t**2 + Z_t**2)/real(nt)
            V_c = ls*lt*(2.d0*PI*R/real(n_phi))

            global_diag(i_V) = global_diag(i_V) + V_c !< sanity check on the total volume
            
            ! prepare for drawing random particles
            call i_random%reset(n_pa_bin)

            ! determine the number of pairs to be tried for collision
            N_try = nint(0.5d0*n_pa_bin*this%P_max_elm(i_elm))
            if(N_try < 1) N_try = 1 ! to make sure that for low collisional chance the number of collisions are not rounded off to 0
            ! if (N_try > n_pa_bin/2) some particles will be used twice. This is known (and warned for) beforehand because P_max_elm is known

            ! loop over to-be-tried collisional pairs for this bin
            do i_pair = 1,N_try !< dummy variable for loop
              !generate random number for the pair
              call this%rng(i_thread)%next(RN)

              !get random particle from the bin
              i_pa1 = i_random%next(RN(1))
              i_pa2 = i_random%next(RN(2))

              !copy from MPI pa array (with generic assignment =)
              pa1 = pa(i_pa_bin(is,it,i_phi)%pa_ind(i_pa1))
              pa2 = pa(i_pa_bin(is,it,i_phi)%pa_ind(i_pa2))

              ! calculate P for this collision pair
              w1 = pa1%weight
              w2 = pa2%weight
              w_g = max(w1, w2)
              w_s = min(w1, w2)
              v1i = pa1%v
              v2i = pa2%v
              v_r = norm2(v1i - v2i)
              m1 = sim%groups(this%group_num)%mass
              m2 = sim%groups(this%group_num)%mass
              sigma_T = calc_sigma_T(v_r, m1, m2, this%dtW)
              call interp_0(nodes, elements, pa1%i_elm, [i_neutral_n], 1, pa1%st(1), pa1%st(2), pa1%x(3), P)
              n_loc = P(1)

              ! the following is the required collisional chance when splitting up the heavier particle:
              !P_real = w_g * n_pa_bin * sim%n_mpi * sigma_T * v_r * dt / V_c ! in this scenario we should technically take all particles in the collisional bin across all MPI's, but we approximate that with n_pa_bin*sim%n_mpi

              ! because we are not splitting up the heavier particle:
              P_real = n_loc * sigma_T * v_r * dt

              if (P_real > P_max_now) P_max_now = P_real

              !diagnostics
              global_diag(i_sigma_av) = global_diag(i_sigma_av) + sigma_T
              global_diag(i_d_av)     = global_diag(i_d_av)     + sqrt(sigma_T/PI)
              global_diag(i_P_av)     = global_diag(i_P_av)     + max(P_real,0.d0) ! we want to count negative P_real as 0 for the average collision chance
              global_diag(i_n_pairs)  = global_diag(i_n_pairs)  + 1
              if (P_real > P_max_mpi) P_max_mpi = P_real
              
              ! rescale with N_try/(0.5*n_pa_bin) rather than with this%P_max_elm(i_elm) directly to correct for integer effects of N_try
              P_try = P_real * (0.5d0*n_pa_bin)/N_try

              if (P_try > 1.d0) then
                global_diag(i_Pgt1) = global_diag(i_Pgt1) + 1
              else if (P_try < 0.d0) then
                global_diag(i_Pst1) = global_diag(i_Pst1) + 1
              end if

              if (RN(3) .le. P_try) then ! do binary collision

                E_i = w1*m1*dot_product(v1i,v1i) + w2*m2*dot_product(v2i,v2i)

                Theta = PI - 2*asin(RN(4))
                alpha = TWOPI*RN(5)

                ! updates velocities of colliding part of super particles
                call binary_elastic_collision(m1,m2,v1i,v2i,Theta,alpha,v1f,v2f)

                global_diag(i_angle_av) = global_diag(i_angle_av) + angle(v1i,v1f) + angle(v2i,v2f)
                
                !Correct new velocities for non-colliding component in heaviest particle
                !Because the heavier particle is not explicitly split up into colliding and non-colliding parts, we cannot conserve momentum and energy simultaneously.
                !It is chosen to conserve energy because neutral momentum also changes at the wall.
                !The fully colliding particle keeps its resulting velocity (as that represents a single underlying species),
                !while the partially colliding particle gets new velocity direction as if it was the same weight as the lower weight particle, 
                !and the magnitude is determined from energy conservation according to 
                !E_i = ws ms vsf^2 + wg mg vg^2 --> vg = sqrt((E_i - ws ms vsf^2)/(wg mg)) (where s means smaller weight and g greater weight)
                if(w1 .gt. w2) then
                  pa2%v = v2f
                  v_g = sqrt((E_i - w_s*m2*dot_product(v2f,v2f))/(w_g*m1)) ! energy conserving new velocity
                  pa1%v = v_g * v1f / norm2(v1f)
                else
                  pa1%v = v1f
                  v_g = sqrt((E_i - w_s*m1*dot_product(v1f,v1f))/(w_g*m2))
                  pa2%v = v_g * v2f / norm2(v2f)
                end if 

                global_diag(i_w_col) = global_diag(i_w_col) + w1 + w2
                global_diag(i_n_col) = global_diag(i_n_col) + 2
                
                ! copy back into MPI pa array (with generic assignment =)
                pa(i_pa_bin(is,it,i_phi)%pa_ind(i_pa1)) = pa1
                pa(i_pa_bin(is,it,i_phi)%pa_ind(i_pa2)) = pa2

              end if !collision happens

            end do !loop over i_pair

          end do !is
        end do !it
      end do !i_phi toroidal bins
      
      ! update maximum collision chance if necessary (with a margin of 20% to keep P<1 in the future)
      if(1.2*P_max_now > this%P_max_elm(i_elm)) then ! update if the maximum chance now is bigger than previously
        this%P_max_elm(i_elm) = 1.2*P_max_now
      else if (2*P_max_now < this%P_max_elm(i_elm)) then ! if P_max_now is significantly smaller than before, we can carefully decrease P_max_elm to gain some speed in the future
        this%P_max_elm(i_elm) = 0.98*this%P_max_elm(i_elm)
      end if

      ! deallocation of private allocatables
      deallocate(n_pa_bin_arr)
      do i_phi=1,n_phi
        do it=1,nt
          do is=1,ns
            deallocate(i_pa_bin(is,it,i_phi)%pa_ind)
          end do
        end do
      end do
      deallocate(i_pa_bin)

    end do !i_elm
    !$omp end parallel do
    
    call MPI_REDUCE(P_max_mpi,     P_max_global,        1,      MPI_DOUBLE_PRECISION, MPI_MAX, 0, MPI_COMM_WORLD, ierr)
    call MPI_REDUCE(max_n_pa,      max_n_pa_gl,         2,      MPI_INTEGER,          MPI_MAX, 0, MPI_COMM_WORLD, ierr)
    P_max_elm_min = minval(this%P_max_elm)
    P_max_elm_max = maxval(this%P_max_elm)
    call MPI_REDUCE(P_max_elm_min, P_max_elm_min_red,   1,      MPI_DOUBLE_PRECISION, MPI_MIN, 0, MPI_COMM_WORLD, ierr)
    call MPI_REDUCE(P_max_elm_max, P_max_elm_max_red,   1,      MPI_DOUBLE_PRECISION, MPI_MAX, 0, MPI_COMM_WORLD, ierr)
    
    call MPI_REDUCE(global_diag,   reduced_global_diag, n_diag, MPI_DOUBLE_PRECISION, MPI_SUM, 0, MPI_COMM_WORLD, ierr)
    
    if(sim%my_id .eq. 0) then
      write(*,"(A,2I8,2es15.5)") "max (pa in elm/pa in bin/P_real/min tau) =",max_n_pa_gl,P_max_global,dt/nonzero(P_max_global)
      write(*,"(A,2es15.5)") "P_max_elm rescale values (min/max) ",P_max_elm_min_red,P_max_elm_max_red
      if(P_max_elm_max_red .ge. 1) write(*,"(A)") "WARNING: some particles will be tried more than once for collisions"
      if(reduced_global_diag(i_Pgt1) > 0.5) write(*,"(A,I10.0,A,I10.0,A)") "WARNING: P_try>1 for ",nint(reduced_global_diag(i_Pgt1))," out of ",nint(reduced_global_diag(i_n_pairs))," collision attempts during this timestep"
  
      !getting the averages by dividing the sum by number of pairs
      reduced_global_diag(i_P_av)     = reduced_global_diag(i_P_av)     / nonzero(reduced_global_diag(i_n_pairs))
      reduced_global_diag(i_sigma_av) = reduced_global_diag(i_sigma_av) / nonzero(reduced_global_diag(i_n_pairs))
      reduced_global_diag(i_d_av)     = reduced_global_diag(i_d_av)     / nonzero(reduced_global_diag(i_n_pairs))
      reduced_global_diag(i_angle_av) = reduced_global_diag(i_angle_av) / (nonzero(reduced_global_diag(i_n_col)) * (PI/2))
      reduced_global_diag(i_tau)      = dt/nonzero(reduced_global_diag(i_P_av))
      reduced_global_diag(i_V)        = reduced_global_diag(i_V)        / sim%n_mpi
      write(FMT,"(A,I2,A)") "(A,",n_diag,"es15.5)"  
      write(*,trim(FMT)) "diagnostics (P_real av/tau av/sigma av/pairs tried/pairs coll/weight coll/#P_try>1/#P_try<0/sum V_c/d av/av angle frac) =",reduced_global_diag
    end if
    !$ w(2) = omp_get_wtime()
    !$ mmm = mpi_minmeanmax(w(2)-w(1))
    !$ if (sim%my_id .eq. 0) write(*,"(A,3f9.4,A)") "Neutral self collision complete in (min/mean/max) ", mmm, " s"

  class default
    write(*,*) "neutral-neutral self collisions not implemented for this type, group=", i
    call exit(13)
  end select

end subroutine neutral_self_collision

!> initializes the neutral collision object
subroutine initialize(this, sim, group_num, dTw, each_nstep_part)
  use mod_random_seed
  use mod_math_operators, only: gcd

  implicit none  
  class(type_neutral_collision), intent(inout) :: this
  type(particle_sim),            intent(inout) :: sim
  integer,                       intent(in)    :: group_num
  real*8,                        intent(in)    :: dtW(3)
  integer,                       intent(in)    :: each_nstep_part

  integer :: i, seed, n_thread
  
  ! --- Setting up P_max_elm
  if(allocated(this%P_max_elm)) deallocate(this%P_max_elm)
  allocate(this%P_max_elm(sim%fields%element_list%n_elements))

  ! initial guess of P_max (the chance that the highest collisional particle will collide this timestep)
  ! is set to a reasonable value to benefit from the NTC algorithm but not to generate too many errors in the beginning
  this%P_max_elm(:) = 1.d-1

  ! --- Setting up random numbers
  seed = random_seed()
  n_thread = 1
  !$ n_thread = omp_get_max_threads()
  write(*,*) "id, n_mpi, n_thread",sim%my_id, sim%n_mpi, n_thread
  allocate(this%rng(n_thread))
  do i=1,n_thread
    call this%rng(i)%initialize(1, seed, n_thread, i)
  end do

  !setting up variables
  this%group_num = group_num
  this%dTw = dtW
  this%each_nstep_part = each_nstep_part
  if (each_nstep_part /= -9999991) then
    call sim%update_lcm_gcd(each_nstep_part)
    if(gcd_neutral_collisions == -9999991) then
      gcd_neutral_collisions = each_nstep_part
    else
      gcd_neutral_collisions = gcd(gcd_neutral_collisions,each_nstep_part)
    endif
  endif

  this%constructed = .true.
end subroutine initialize

!> determines the neutral collision actions from the config and returns the array of neutral collisions objects
function neutral_collisions_from_config(sim) result(neutral_collisions)
  use phys_module, only: part_group_configs, n_part_groups_max 
  use mod_particle_sim, only: group_num_from_id
  
  implicit none

  type(particle_sim), intent(inout)                        :: sim
  class(type_neutral_collision), allocatable, dimension(:) :: neutral_collisions !< array of neutral collisions objects 


  integer :: i, group_num, n_coll_objs, i_coll_obj
  character(len=3) :: id

  !determine number of collision objects
  n_coll_objs = 0
  do i=1, n_part_groups_max
    id = part_group_configs(i)%id
    if(id == "non") cycle
    if(.not. part_group_configs(i)%use_kin_neutral_coll) then
      if(any(abs(part_group_configs(i)%neutral_coll_dTw(:) + 1.d99)/1.d99 > 1.d-10)) then ! dTw was changed while use_kin_neutral_coll=.false.
        if(sim%my_id == 0) write(*,"(A,I2,A,I2,A,I2,A,I2,A)") "ERROR: part_group_configs(",i,")%use_kin_neutral_coll=.false. but part_group_configs(",i,")%neutral_coll_dTw has been set. If you want to use neutral collisions you need to set part_group_configs(",i,")%use_kin_neutral_coll=.true and define the three parameters of part_group_configs(",i,")%neutral_coll_dTw. Aborting."
        stop
      end if  
      cycle
    end if
    n_coll_objs = n_coll_objs+1
  end do

  allocate(neutral_collisions(n_coll_objs))

  !check input and generate collision objects
  i_coll_obj=0
  do i=1, n_part_groups_max
    id = part_group_configs(i)%id
    if(id == "non") cycle
    if(.not. part_group_configs(i)%use_kin_neutral_coll) cycle
    i_coll_obj = i_coll_obj + 1

    if(part_group_configs(i)%coupling_scheme /= "ncs") then
      if(sim%my_id == 0) write(*,"(A,I2,A,I2,3A)") "ERROR: neutral collisions can only be used for neutral particle groups, but part_group_configs(",i,")%use_kin_neutral_coll=.true. while part_group_configs(",i,")%coupling_scheme=",part_group_configs(i)%coupling_scheme," rather than ncs. Aborting."
      stop 
    end if
    
    !check sanity of VHS input dTw
    if(any(abs(part_group_configs(i)%neutral_coll_dTw(:) + 1.d99)/1.d99 < 1.d-10)) then ! it still has the default value
      if(sim%my_id == 0) write(*,"(A,I2,A,I2,A,I2,A)") "ERROR: part_group_configs(",i,")%use_kin_neutral_coll=.true. but not all parameters in part_group_configs(",i,")%neutral_coll_dTw have been defined. Please set the reference collisional diameter, temperature and viscosity index part_group_configs(",i,")%neutral_coll_dTw. Aborting."
      stop
    else if(any(part_group_configs(i)%neutral_coll_dTw(:) < 0)) then
      if(sim%my_id == 0) write(*,"(A,I2,A,3es14.4,A)") "ERROR: negative values in part_group_configs(",i,")%neutral_coll_dTw=",part_group_configs(i)%neutral_coll_dTw," Please check your input. Aborting."
      stop
    end if

    !sanity check and writing when to run the action
    if(part_group_configs(i)%ncoll_each_nstep_part /= -9999991) then
      if(part_group_configs(i)%ncoll_each_nstep_part <= 0) then
        if(sim%my_id == 0) write(*,"(A,I2,A,I2,A,I2,A)") "ERROR: part_group_configs(",i,")%ncoll_each_nstep_part <= 0, which is not possible. Aborting."
        stop
      endif
    else
      if(sim%my_id == 0) write(*,"(A,I2,A,I2,A,I2,A)") "part_group_configs(",i,")%ncoll_each_nstep_part was not set, so neutral self collisions action will be done once every fluid step."
    endif
    
    group_num = group_num_from_id(sim, id)
    call neutral_collisions(i_coll_obj)%initialize(sim,group_num,part_group_configs(i)%neutral_coll_dTw,part_group_configs(i)%ncoll_each_nstep_part)
  end do

  !sanity check
  if(i_coll_obj /= n_coll_objs) then
    if(sim%my_id == 0) write(*,"(A,2I5)") "ERROR in neutral collisions setup: number of objects to be made is inconsistent?",i_coll_obj,n_coll_objs
    stop 
  end if
  
end function neutral_collisions_from_config

!> calculates the elastic collisional cross section for D + D elastic collisions
!> variable hard sphere based on [1] eq 4.63
function calc_sigma_T(v_r, m1, m2, dTw) result(sigma_T)
  
  use constants, only: PI, K_BOLTZ
  
  implicit none
  
  real*8, intent(in)  :: v_r     !< relative velocity [m/s]
  real*8, intent(in)  :: m1      !< mass of first particle [AMU]
  real*8, intent(in)  :: m2      !< mass of second particle [AMU]
  real*8, intent(in)  :: dTw(3)  !< d_ref, T_ref and omega

  real*8              :: sigma_T !< total collisional cross section sigma_T

  ! local variables
  real*8 :: d     !< diameter [m]
  real*8 :: T_ref !< reference temperature for d_ref and omega [K]
  real*8 :: d_ref !< diameter at reference temperature [m]
  real*8 :: omega !< viscosity index
  real*8 :: m_r   !< reduced mass m=m1*m2/(m1+m2) [kg]

  if (abs(v_r) .lt. 1.d-12) then
    sigma_T = 0.d0
    return
  end if

  ! setting local variables
  d_ref = dTw(1)
  T_ref = dTw(2)
  omega = dTw(3)
  m_r   = m1*m2/(m1+m2)
  m_r   = m_r * ATOMIC_MASS_UNIT

  d = d_ref*sqrt(((2*K_BOLTZ*T_ref/max(1.d-40,m_r*v_r**2))**(omega-0.5d0) )/ gamma(2.5d0-omega)) !< gamma here is the gamma function, not phys_module gamma
  sigma_T = PI*d**2

end function calc_sigma_T

!> Calculates the resulting velocity vectors after a binary elastic collision 
!> Uses centre of mass scattering angle Theta and angle of the scattering plane alpha
subroutine binary_elastic_collision(m1,m2,v1i,v2i,Theta,alpha,v1f,v2f)
  use constants, only: PI
  use mod_math_operators, only: cross_product

  implicit none
  
  real*8, intent(in)  :: m1      !< mass of particle 1
  real*8, intent(in)  :: m2      !< mass of particle 2
  real*8, intent(in)  :: Theta   !< scattering angle of centre-of-mass system
  real*8, intent(in)  :: alpha   !< scattering plane angle to Z-axis around v1i
  real*8, intent(in)  :: v1i(3)  !< initial velocity of particle 1
  real*8, intent(in)  :: v2i(3)  !< initial velocity of particle 2
  real*8, intent(out) :: v1f(3)  !< final velocity of particle 1
  real*8, intent(out) :: v2f(3)  !< final velocity of particle 2

  !local parameters
  ! collision velocities in scattering plane [parallel to v1, perp to v1] where v1=v_r, v2=0, and p stands for prime, the final velocity
  real*8 :: v_r(3)     !< relative velocity in R,Z,phi
  real*8 :: v_r_hat(3) !< direction of v_r
  real*8 :: scalar_v1, scalar_v1p, scalar_v2p, v1p(2), v2p(2) 
  real*8 :: theta_1, theta_2, perp1(3), perp2(3)

  !sanity
  real*8 :: dp(3), dE, tol
  
  !this calculation follows the logic in [2], section 3.2, applied to 3D

  !go to 3D rest frame of particle 2
  v_r = v1i - v2i
  !go to scattering plane, 2D rest frame of particle 2 (i.e. v2i = 0 in this frame, while v1i = [v_r,0])
  scalar_v1 = norm2(v_r)

  !calculate scattering angles of particles in scattering plane from CM frame scattering angle Theta
  theta_2 = -(PI-Theta)/2
  theta_1 = atan(sin(Theta)/nonzero(m1/m2 + cos(Theta)))

  !calculate the velocities in scattering plane
  scalar_v1p = scalar_v1 * sqrt(1.d0/nonzero(1 + (m1/m2)*(sin(theta_1)/nonzero(sin(theta_2)))**2)) !< from energy balance, substituting scalar_v2p (see next line)
  scalar_v2p = -(m1/m2) * (sin(theta_1) / nonzero(sin(theta_2))) * scalar_v1p               !< from perpendicular momentum balance

  v2p = scalar_v2p*[cos(theta_2), sin(theta_2)]
  v1p = scalar_v1p*[cos(theta_1), sin(theta_1)]

  !transform back to 3D rest frame of particle 2
  !we need 2 orthonormal vectors to v_r to determine the scattering plane using alpha
  perp1 = cross_product(v_r,[1.d0,0.d0,0.d0]) !< cross product with R axis to get orthogonal vector
  perp2 = cross_product(v_r,perp1)            !< cross product with first perpendicular axis
  
  v_r_hat = v_r  /nonzero(norm2(v_r))   !< normalising to get orthonormal basis vectors
  perp1   = perp1/nonzero(norm2(perp1))
  perp2   = perp2/nonzero(norm2(perp2))

  !calculate the resulting velocity from parallel velocity component and perpendicular split up into the two perpendicular directions using alpha
  v1f = v1p(1)*v_r_hat + v1p(2)*cos(alpha)*perp1 + v1p(2)*sin(alpha)*perp2
  v2f = v2p(1)*v_r_hat + v2p(2)*cos(alpha)*perp1 + v2p(2)*sin(alpha)*perp2

  !transform back from rest frame of particle 2 to R,Z,phi frame
  v1f = v1f + v2i
  v2f = v2f + v2i

  ! check conservation quantities
  dp = (m1*(v1f-v1i) + m2*(v2f-v2i))/max(1.d0,norm2(m1*v1i + m2*v2i)) !< relative change in momentum due to collision, should be 0, m is in amu
  dE = (m1*(norm2(v1f)**2-norm2(v1i)**2) + m2*(norm2(v2f)**2-norm2(v2i)**2)) / max(1.d0,m1*norm2(v1i)**2 + m2*norm2(v2i)**2) !< relative change in energy [J/(kg/amu)]

  tol = 1.d-8
  if(abs(dp(1)) .gt. tol .or. abs(dp(1)) .gt. tol .or. abs(dp(1)) .gt. tol .or. abs(dE) .gt. tol) then
    write(*,*) "ERROR, momentum or energy not conserved in binary elastic collision (m1,m2,v1i,v2i,Theta,alpha,v1f,v2f,dp,dE)",m1,m2,v1i,v2i,Theta,alpha,v1f,v2f,dp,dE
  end if
  
end subroutine binary_elastic_collision

!> calculates in which bin (out of n bins) a value x=[0,1] should fall
pure function ith_bin(x_in,n) result(i)
  implicit none
  real*8,  intent(in) :: x_in !< value between 0 and 1, of which the corresponding bin should be determined
  integer, intent(in) :: n !< number of bins
  integer :: i !< ith element in the bin

  real*8 :: x !< copy of x_in
  real*8,parameter :: tol=1.d-10 !< numerical tolerance
  
  x = x_in

  ! avoid x exactly 0 or 1
  if (x < tol)        x = tol
  if (x > 1.d0 - tol) x = 1.d0 - tol

  i = ceiling(x*n)
end function ith_bin

!> resets the random draw object to a clean new array
subroutine reset(this,size)
  implicit none

  class(random_draw), intent(inout) :: this
  integer,            intent(in)    :: size !< size of the new random draw array

  integer :: i

  if(size < 0) then
    write(*,*) "ERROR in mod_particle collision, random_draw size < 0 : ", size
    stop
  end if

  if (allocated(this%indices)) then
    call dealloc_random_draw(this)
  end if
  allocate(this%indices(size))

  this%size=size
  this%used=0

  do i=1,size
    this%indices(i) = i
  end do

end subroutine reset

subroutine dealloc_random_draw(this)
  implicit none
  class(random_draw), intent(inout) :: this
  deallocate(this%indices)
end subroutine

!> draws a new unused random number from the indices list by moving unused indices to empty spots
!> Example for this%size 5: start with this%indices=[1,2,3,4,5] and this%used=0
!> this%indices | this%used | roll | roll result | draw_loc | outcome | 
!> 12345        | 0         | 1d5  | 3           | 3        | 3       | put 1 at position 3
!> -2145        | 1         | 1d4  | 2           | 3        | 1       | put 2 at position 3
!> --245        | 2         | 1d3  | 3           | 5        | 5       | put 2 at position 5
!> ---42        | 3         | 1d2  | 1           | 4        | 4       | nothing to be changed
!> ----2        | 4         | 1d1  | 1           | 5        | 2       |
function next(this, RN) result(i_drawn)
  implicit none

  class(random_draw), intent(inout) :: this
  real*8,             intent(in)    :: RN !< random number to get the next random index
  integer                           :: i_drawn

  integer :: draw_loc !< which index this draw draws from
  integer :: nth_draw !< the number of this draw (so number of previous draws (=this%used) + 1)

  nth_draw = this%used + 1

  ! check whether we have unused indices left
  if(nth_draw > this%size) then
    call this%reset(this%size)
    nth_draw = 1 !< because it was just reset, this is again the first draw
  end if

  ! determine the location in the index array to draw from
  draw_loc = this%used + ceiling(RN * (this%size - this%used))
  
  if(ceiling(RN) == 0) draw_loc = draw_loc + 1 ! treat the edge case that the random number = 0 exactly

  !sanity check on draw_loc
  if(draw_loc < 1 .or. draw_loc > this%size) then
    write(*,"(A,3I10,f10.5)") "ERROR in random draw. draw_loc/size/used/RN = ",draw_loc,this%size,this%used,RN
    write(*,*) "this%indices=",this%indices
    i_drawn=1
    return
  end if

  ! draw the number from the draw location
  i_drawn = this%indices(draw_loc)
  
  ! sanity check on drawn number
  if(i_drawn < 1 .or. i_drawn > this%size) then
    write(*,"(A,3I10,f10.5)") "ERROR in random draw. i_drawn/size/used/RN = ",i_drawn,this%size,this%used,RN
    write(*,*) "this%indices=",this%indices
    i_drawn=1
    return
  end if

  ! move the unused number (at index nth_draw) to the draw_loc
  this%indices(draw_loc)  = this%indices(nth_draw)

  ! set the used number to -1 to catch errors
  this%indices(nth_draw) = -1
    
  ! the draw is done
  this%used = this%used + 1
end function next

!> returns the input unless the input is 0, then returns 1.d-10 with the sign of the input (to avoid divide by 0 issues)
pure function nonzero(in) result(out)
  implicit none
  real*8, intent(in) :: in
  real*8 :: out

  real*8, parameter :: tol = 1.d-10

  out = in
  if (abs(in) < tol) out = sign(tol, in)
end function nonzero

!> find the angle between two vectors [0,pi]
pure function angle(v1,v2) result(alpha)
  implicit none

  real*8, intent(in) :: v1(3)
  real*8, intent(in) :: v2(3)
  real*8             :: alpha

  alpha = acos(dot_product(v1,v2)/nonzero(norm2(v1)*norm2(v2)))

end function angle

end module mod_neutral_collision
