!> Element matrix of the R^3 rho0-weighted mass M_w of the physics PC's flux
!! arm (physics_pc_sf_suu = flux, see mod_petsc_pc_sf_solver::sf_needs_mass_w).
!!
!!   M_w(du, v) = int R^3 rho0 v du dR dZ ,   rho0 = |rho_(n=0)|
!!
!! THE WEIGHT. rho0 is the AXISYMMETRIC part of the density only: the in = 1
!! harmonic of the nodal values (HZ(1,:) = 1 in basis_at_gaussian, so this is
!! exactly the toroidal average over the n_plane planes, without evaluating
!! them). With a phi-independent weight the operator is diagonal in the
!! toroidal index, so no FFT is needed and every slot m = 1..n_tor gets the
!! same poloidal mass up to the Jacobian blocks' slot factor (n_plane on
!! n = 0, n_plane / 2 on cos / sin); slots are not coupled. abs() as in the force routine
!! (r0 = abs(eq_g(mp,5,...)) in models/model199/mod_pc_elt_matrix_force_fft).
!! Models without a density variable (var_rho < 1, e.g. model001) use rho0 = 1.
!!
!! GEOMETRY. R and the Jacobian come from the n = 0 coordinate harmonic
!! (x(1,:,:)), as in the model element routines. On stellarator models
!! (n_coord_tor > 1) this is the toroidally averaged geometry, i.e. an
!! approximation of the true phi-dependent mass.
!!
!! LAYOUT. One variable, the same element-block layout as
!! pc_elt_matrix_force_fft: element index n_tor*n_degrees*(i-1) + n_tor*(j-1) + m
!! for vertex i, degree j, toroidal slot m -- ready for insertion with
!! MatSetValuesBlocked at block size n_tor.
!!
!! Model-independent: uses only mod_parameters / gauss / basis_at_gaussian
!! symbols that every model provides, and no FFTW.
module mod_pc_elt_mass_w

  implicit none
  private

  public :: pc_elt_mass_w

contains

!-----------------------------------------------------------------
!> Element matrix of the R^3 rho0 (n = 0) weighted mass, diagonal in the
!! toroidal slot index (see module header).
!-----------------------------------------------------------------
subroutine pc_elt_mass_w(element, nodes, ELM)

  use mod_parameters,    only: n_tor, n_plane, n_vertex_max, n_degrees, var_rho
  use data_structure,    only: type_element, type_node
  use gauss,             only: n_gauss, wgauss
  use basis_at_gaussian, only: H, H_s, H_t

  implicit none

  type(type_element), intent(in)  :: element             !< the element
  type(type_node),    intent(in)  :: nodes(n_vertex_max) !< its vertex nodes

#define DFV_MW (n_tor*n_vertex_max*n_degrees)

  real*8, dimension(DFV_MW, DFV_MW), intent(out) :: ELM  !< element matrix, n_tor-blocked

  real*8, dimension(n_gauss, n_gauss) :: x_g, x_s, x_t, y_s, y_t, r0_g
  real*8, dimension(n_vertex_max*n_degrees, n_vertex_max*n_degrees) :: ELM_pol

  integer :: i, j, k, l, ms, mt, m
  integer :: index_ij, index_kl, k_rho
  logical :: has_rho
  real*8  :: wst, xjac, BigR, wgt, v, u

  ELM     = 0.d0
  ELM_pol = 0.d0

  ! a run-time copy of the index, so models without density (var_rho = 0)
  ! never form a constant out-of-range subscript
  has_rho = (var_rho >= 1)
  k_rho   = max(var_rho, 1)

  x_g  = 0.d0; x_s = 0.d0; x_t = 0.d0
  y_s  = 0.d0; y_t = 0.d0
  r0_g = 0.d0

  do i = 1, n_vertex_max
    do j = 1, n_degrees
      do ms = 1, n_gauss
        do mt = 1, n_gauss
          x_g(ms,mt) = x_g(ms,mt) + nodes(i)%x(1,j,1) * element%size(i,j) * H  (i,j,ms,mt)
          x_s(ms,mt) = x_s(ms,mt) + nodes(i)%x(1,j,1) * element%size(i,j) * H_s(i,j,ms,mt)
          x_t(ms,mt) = x_t(ms,mt) + nodes(i)%x(1,j,1) * element%size(i,j) * H_t(i,j,ms,mt)
          y_s(ms,mt) = y_s(ms,mt) + nodes(i)%x(1,j,2) * element%size(i,j) * H_s(i,j,ms,mt)
          y_t(ms,mt) = y_t(ms,mt) + nodes(i)%x(1,j,2) * element%size(i,j) * H_t(i,j,ms,mt)
          ! the n = 0 harmonic only (in = 1, HZ(1,:) = 1)
          if (has_rho) then
            r0_g(ms,mt) = r0_g(ms,mt) + nodes(i)%values(1,j,k_rho) * element%size(i,j) * H(i,j,ms,mt)
          endif
        enddo
      enddo
    enddo
  enddo

  if (has_rho) then
    r0_g = abs(r0_g)
  else
    r0_g = 1.d0
  endif

  !--- poloidal mass, once
  do ms = 1, n_gauss
    do mt = 1, n_gauss

      wst  = wgauss(ms)*wgauss(mt)
      xjac = x_s(ms,mt)*y_t(ms,mt) - x_t(ms,mt)*y_s(ms,mt)
      BigR = x_g(ms,mt)
      wgt  = wst * BigR**3 * r0_g(ms,mt) * xjac

      do i = 1, n_vertex_max
        do j = 1, n_degrees
          index_ij = n_degrees*(i-1) + j
          v        = H(i,j,ms,mt) * element%size(i,j)
          do k = 1, n_vertex_max
            do l = 1, n_degrees
              index_kl = n_degrees*(k-1) + l
              u        = H(k,l,ms,mt) * element%size(k,l)
              ELM_pol(index_ij, index_kl) = ELM_pol(index_ij, index_kl) + wgt * v * u
            enddo
          enddo
        enddo
      enddo

    enddo
  enddo

  !--- the same poloidal mass on every toroidal slot, no slot coupling, with
  !--- the slot normalisation of the Jacobian blocks: element_matrix_fft's
  !--- unnormalised r2c FFT and its final 0.5 give an axisymmetric
  !--- coefficient the factor n_plane on the n = 0 slot and n_plane / 2 on
  !--- every cos / sin slot. The bending term B_12^T C_w B_12 is then
  !--- normalised like the Schur term B_12 B_22^-1 B_23 it stands for.
  do index_ij = 1, n_vertex_max*n_degrees
    do index_kl = 1, n_vertex_max*n_degrees
      do m = 1, n_tor
        ELM(n_tor*(index_ij-1) + m, n_tor*(index_kl-1) + m) = &
          merge(dble(n_plane), 0.5d0*dble(n_plane), m == 1) * ELM_pol(index_ij, index_kl)
      enddo
    enddo
  enddo

#undef DFV_MW

end subroutine pc_elt_mass_w

end module mod_pc_elt_mass_w
