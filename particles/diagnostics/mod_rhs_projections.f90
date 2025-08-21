!*****************************************************************
!* Example projection functions for generating a right-hand-side *
!*****************************************************************
module mod_rhs_projections

use mod_particle_types
use mod_particle_sim

implicit none
private

public proj_f, proj_f_interface, proj_one, proj_q, proj_vR, proj_vZ, proj_vPhi, proj_Ekin, proj_Ekin_keV, proj_jR, proj_jZ, proj_jPhi
public proj_R, proj_min_rad, proj_Z,proj_v,proj_vpar,proj_mu,proj_pow

interface
  function proj_f_interface(sim, group, particle)
    import particle_sim, particle_base
    type(particle_sim), intent(in) :: sim
    integer, intent(in) :: group
    class(particle_base), intent(in) :: particle !< Input particle
    real*8 :: proj_f_interface !< Value to be projected
  end function proj_f_interface
end interface

!> A wrapper type to contain a projection function and a group to apply it to
type proj_f
  procedure(proj_f_interface), nopass, pointer :: f
  integer :: group !< group number to apply this function to
end type proj_f

interface proj_f
  module procedure new_proj_f !< The constructor for this type
end interface proj_f

contains

  !> Project the particle density by using transformation function 1
  pure function proj_one(sim, group, particle)
    type(particle_sim), intent(in) :: sim
    integer, intent(in) :: group
    class(particle_base), intent(in) :: particle
    real*8 :: proj_one
    proj_one = 1.d0
  end function proj_one

  pure function proj_R(sim,group,particle)
    type(particle_sim), intent(in) :: sim
    integer, intent(in) :: group
    class(particle_base), intent(in) :: particle
    real*8 :: proj_R
    select type (p => particle)
      type is (particle_kinetic_leapfrog)
        proj_R = p%x(1)
      class default
        proj_R = 0.d0
    end select
  end function proj_R

  pure function proj_min_rad(sim,group,particle)
    use equil_info, only : ES
    type(particle_sim), intent(in) :: sim
    integer, intent(in) :: group
    class(particle_base), intent(in) :: particle
    real*8 :: proj_min_rad
    
    select type (p => particle)
      type is (particle_kinetic_leapfrog)
        
        proj_min_rad = sqrt((p%x(1)-ES%R_axis)**2+p%x(2)**2) !Small r 
      class default
        proj_min_rad = 0.d0
    end select
  end function proj_min_rad

  pure function proj_Z(sim,group,particle)
    type(particle_sim), intent(in) :: sim
    integer, intent(in) :: group
    class(particle_base), intent(in) :: particle
    real*8 :: proj_Z
    select type (p => particle)
      type is (particle_kinetic_leapfrog)
        proj_Z = p%x(2)

      class default
        proj_Z = 0.d0
    end select
  end function proj_Z

  pure function proj_phi(sim,group,particle)
    type(particle_sim), intent(in) :: sim
    integer, intent(in) :: group
    class(particle_base), intent(in) :: particle
    real*8 :: proj_phi
    select type (p => particle)
      type is (particle_kinetic_leapfrog)
        proj_phi = p%x(3)

      class default
        proj_phi = 0.d0
    end select
  end function proj_phi

  ! Debatable how accurate this is. kinetic_to_gc is not exact.
  function proj_mu(sim,group,particle)
    use mod_particle_types
    use mod_boris
    type(particle_sim), intent(in) :: sim
    integer, intent(in) :: group
    class(particle_base), intent(in) :: particle
    type(particle_gc)    :: particle_gc_tmp
    real*8 :: proj_mu,E(3),B(3),psi,U,mass
    mass=sim%groups(1)%mass
    select type (p => particle)
      type is (particle_kinetic_leapfrog)
        call sim%fields%calc_EBpsiU(sim%time,p%i_elm,p%st,p%x(3),E,B,psi,U)
        particle_gc_tmp=kinetic_to_gc(sim%fields%node_list, sim%fields%element_list, kinetic_leapfrog_to_kinetic(p, E, B, mass, 0.d0), B, mass)
        proj_mu = abs(particle_gc_tmp%mu)
      class default
        proj_mu = 0.d0
    end select
  end function proj_mu

  function proj_vpar(sim,group,particle)
    use mod_particle_types, only: particle_kinetic_leapfrog
    type(particle_sim), intent(in) :: sim
    integer, intent(in) :: group
    class(particle_base), intent(in) :: particle
    type(particle_gc)    :: particle_gc_tmp
    real*8 :: proj_vpar,E(3),B(3),psi,U

    select type (p => particle)
      type is (particle_kinetic_leapfrog)

        call sim%fields%calc_EBpsiU(sim%time,p%i_elm,p%st,p%x(3),E,B,psi,U)

        proj_vpar = dot_product(p%v,B)/sqrt(dot_product(B,B))
      class default
        proj_vpar = 0.d0
    end select
  end function proj_vpar

  function proj_pow(sim,group,particle)
    use constants, only: el_chg
    use mod_particle_types, only: particle_kinetic_leapfrog
    type(particle_sim), intent(in) :: sim
    integer, intent(in) :: group
    class(particle_base), intent(in) :: particle
    real*8 :: proj_pow,E(3),B(3),psi,U
    select type (p => particle)
      type is (particle_kinetic_leapfrog)

        call sim%fields%calc_EBpsiU(sim%time,p%i_elm,p%st,p%x(3),E,B,psi,U)
        proj_pow = p%q*el_chg*dot_product(p%v,E)

      class default
        proj_pow = 0.d0
    end select
  end function proj_pow

  pure function proj_v(sim,group,particle)
    use mod_particle_types, only: particle_kinetic_leapfrog
    type(particle_sim), intent(in) :: sim
    integer, intent(in) :: group
    class(particle_base), intent(in) :: particle
    real*8 :: proj_v
    select type (p => particle)
      type is (particle_kinetic_leapfrog)
        proj_v = sqrt(dot_product(p%v,p%v))
      class default
        proj_v = 0.d0
    end select
  end function proj_v

  pure function proj_vR(sim, group, particle)
    use mod_particle_types, only: particle_kinetic_leapfrog
    type(particle_sim), intent(in) :: sim
    integer, intent(in) :: group
    class(particle_base), intent(in) :: particle
    real*8 :: proj_vR
    select type (p => particle)
    type is (particle_kinetic_leapfrog)
      proj_vR = p%v(1)
    class default
      proj_vR = 0.d0
    end select
  end function proj_vR

  pure function proj_vZ(sim, group, particle)
    use mod_particle_types, only: particle_kinetic_leapfrog
    type(particle_sim), intent(in) :: sim
    integer, intent(in) :: group
    class(particle_base), intent(in) :: particle
    real*8 :: proj_vZ
    select type (p => particle)
    type is (particle_kinetic_leapfrog)
      proj_vZ = p%v(2)
    class default
      proj_vZ = 0.d0
    end select
  end function proj_vZ

  pure function proj_vPhi(sim, group, particle)
    use mod_particle_types, only: particle_kinetic_leapfrog
    type(particle_sim), intent(in) :: sim
    integer, intent(in) :: group
    class(particle_base), intent(in) :: particle
    real*8 :: proj_vPhi
    select type (p => particle)
    type is (particle_kinetic_leapfrog)
      proj_vPhi = p%v(3)
    class default
      proj_vPhi = 0.d0
    end select
  end function proj_vPhi

  !< Energy in joules
  pure function proj_Ekin(sim, group, particle) 
    use constants, only: atomic_mass_unit
    use mod_particle_types, only: particle_kinetic_leapfrog
    type(particle_sim), intent(in) :: sim
    integer, intent(in) :: group
    class(particle_base), intent(in) :: particle
    real*8 :: proj_Ekin
    select type (p => particle)
    type is (particle_kinetic_leapfrog)
      proj_Ekin = sim%groups(group)%mass * atomic_mass_unit * dot_product(p%v,p%v)/2.d0
    class default
      proj_Ekin = 0.d0
    end select
  end function proj_Ekin

  pure function proj_Ekin_keV(sim, group, particle) 
    use constants, only: atomic_mass_unit, el_chg
    use mod_particle_types, only: particle_kinetic_leapfrog
    type(particle_sim), intent(in) :: sim
    integer, intent(in) :: group
    class(particle_base), intent(in) :: particle
    real*8 :: proj_Ekin_keV
    select type (p => particle)
    type is (particle_kinetic_leapfrog)
      proj_Ekin_keV = sim%groups(group)%mass * atomic_mass_unit * dot_product(p%v,p%v)/2.d0 / el_chg /1.d3
    class default
      proj_Ekin_keV = 0.d0
    end select
  end function proj_Ekin_keV

  !> Project the particle charge by using transformation function q
  !> (only valid for particles of type kinetic(_leapfrog) or gc
  !>
  !> TODO: normalize projection with density. This function calculates
  !> the integral of q instead of the mean value.
  pure function proj_q(sim, group, particle)
    use constants, only: el_chg
    type(particle_sim), intent(in) :: sim
    integer, intent(in) :: group
    class(particle_base), intent(in) :: particle
    real*8 :: proj_q
    select type (p => particle)
    type is (particle_kinetic)
      proj_q = real(p%q, 8) * el_chg
    type is (particle_kinetic_leapfrog)
      proj_q = real(p%q, 8) * el_chg
    type is (particle_gc)
      proj_q = real(p%q, 8) * el_chg
    class default
      proj_q = 0.d0
    end select
  end function proj_q

  pure function proj_jR(sim, group, particle)
    use constants, only: el_chg
    type(particle_sim), intent(in) :: sim
    integer, intent(in) :: group
    class(particle_base), intent(in) :: particle
    real*8 :: proj_jR
    select type (p => particle)
    type is (particle_kinetic)
      proj_jR = real(p%q, 8) * el_chg * p%v(1)
    type is (particle_kinetic_leapfrog)
      proj_jR = real(p%q, 8) * el_chg * p%v(1)
    class default
      proj_jR = 0.d0
    end select
  end function proj_jR

  pure function proj_jZ(sim, group, particle)
    use constants, only: el_chg
    type(particle_sim), intent(in) :: sim
    integer, intent(in) :: group
    class(particle_base), intent(in) :: particle
    real*8 :: proj_jZ
    select type (p => particle)
    type is (particle_kinetic)
      proj_jZ = real(p%q, 8) * el_chg * p%v(2)
    type is (particle_kinetic_leapfrog)
      proj_jZ = real(p%q, 8) * el_chg * p%v(2)
    class default
      proj_jZ = 0.d0
    end select
  end function proj_jZ

  pure function proj_jPhi(sim, group, particle)
    use constants, only: el_chg
    type(particle_sim), intent(in) :: sim
    integer, intent(in) :: group
    class(particle_base), intent(in) :: particle
    real*8 :: proj_jPhi
    select type (p => particle)
    type is (particle_kinetic)
      proj_jPhi = real(p%q, 8) * el_chg * p%v(3)
    type is (particle_kinetic_leapfrog)
      proj_jPhi = real(p%q, 8) * el_chg * p%v(3)
    class default
      proj_jPhi = 0.d0
    end select
  end function proj_jPhi

  !> Constructor for projection function
  function new_proj_f(f, group)
    type(proj_f) :: new_proj_f
    procedure(proj_f_interface), pointer, intent(in) :: f
    integer, intent(in) :: group
    new_proj_f%f => f
    new_proj_f%group = group
  end function new_proj_f

end module mod_rhs_projections