!> Piecewise Field-Aligned Finite Element Method (PFAFEM) toroidal discretization
!! for JOREK models 180, 183, and 600, integrated with the Split-Field (SF)
!! physics-based MHD preconditioner.
!!
!! Key components:
!!  - Periodic C^2 cubic B-spline basis in toroidal angle phi (even & odd n_tor >= 4)
!!  - Orthogonal nodal Real-DFT (TorFFT) transform matrices Q, Q^T
!!  - Structured C^2 cubic B-spline field-line tangent representation (f_R, f_Z)
!!    with exact analytical 1st & 2nd derivatives (fieldLineJet: fl, A, H, dfl_dphi)
!!  - 12-component RK4 field-line tracer with Jacobian F and Hessian G
!!  - RTree + damped Newton Bezier inversion with OpenBC Hermite continuation
!!  - Shared-Point Bubnov-Galerkin trace cache, exact 2nd-order cylindrical pullback,
!!    axis Pmat transformation, and distributed PETSc CG L2 field projection.
module mod_pfafem
#ifdef USE_PETSC
#include <petsc/finclude/petsc.h>
  use petsc
#if PETSC_VERSION_LT(3,23,0)
#define VecGetArray VecGetArrayF90
#define VecRestoreArray VecRestoreArrayF90
#define VecGetArrayRead VecGetArrayReadF90
#define VecRestoreArrayRead VecRestoreArrayReadF90
#endif
#endif
  use constants, only: pi
  use mod_parameters, only: n_tor, n_plane, n_period, n_coord_tor, n_coord_period, &
                            n_vertex_max, n_degrees, n_order, n_var, var_Psi
  use gauss, only: n_gauss, xgauss, wgauss
  use data_structure, only: type_node_list, type_element_list
  implicit none
  private

  ! --- Per-element Shared-Point trace cache ---
  type, public :: type_pfafem_elem_cache
    integer :: n_unodes = 0
    integer, allocatable :: unodes(:)                 ! (1:n_unodes) global 1-based JOREK node indices
    integer :: n_vpairs = 0
    integer, allocatable :: vpairs(:,:)               ! (2, 1:n_vpairs) (u_row, u_col) in 1..n_unodes
    integer, allocatable :: pair_map(:,:)             ! (1:n_unodes, 1:n_unodes) -> ivp in 1..n_vpairs (0 if uncoupled)
    integer :: n_uj = 0
    integer, allocatable :: uj_map(:,:)               ! (1:n_unodes, 1:n_tor) -> iuj in 1..n_uj (0 if inactive)
    integer, allocatable :: uj_u(:), uj_jb(:)         ! (1:n_uj)
    integer, allocatable :: u_sp_lo(:), u_sp_hi(:)    ! (1:n_unodes) contiguous isp range for row vertex u1
    integer :: n_sp = 0
    integer, allocatable :: sp_map(:,:)               ! (1:n_uj, 1:n_uj) -> isp in 1..n_sp (0 if uncoupled)
    integer, allocatable :: sp_pairs(:,:)             ! (2, 1:n_sp) -> (iuj_row, iuj_col) in 1..n_uj
    ! Traced footprints: dims (lb=1..4, ms=1..n_gauss, mt=1..pfafem_ng_pol, iq=1..ng_tor, iep=1..ep_dim)
    ! e_tr > 0: normal or OpenBC Hermite-continued element; e_tr < 0: boundary-clamped element (-e_tr); 0: dropped
    integer, allocatable :: e_tr(:,:,:,:,:)
    integer, allocatable :: u_vert(:,:,:,:,:,:)       ! (kv=1..4, lb, ms, mt, iq, iep) -> u in 1..n_unodes
    real*8,  allocatable :: s_tr(:,:,:,:,:)
    real*8,  allocatable :: t_tr(:,:,:,:,:)
    real*8,  allocatable :: F_tr(:,:,:,:,:,:)         ! (1:4, lb, ms, mt, iq, iep): [F11, F12, F21, F22]
    real*8,  allocatable :: G_tr(:,:,:,:,:,:)         ! (1:6, lb, ms, mt, iq, iep): [GR_RR,RZ,ZZ, GZ_RR,RZ,ZZ]
  end type type_pfafem_elem_cache

  ! --- Public module state ---
  integer, parameter, public :: pfafem_ng_pol_max = 64
#ifdef STELLARATOR_MODEL
  integer, public :: pfafem_ng_pol = 24
#else
  integer, public :: pfafem_ng_pol = n_gauss
#endif
  logical, public :: pfafem_ready = .false.
  integer, public :: pfafem_cache_epoch = 0
  logical, public :: pfafem_is_axisym_bfield = .true.
  logical, public :: pfafem_cache_axisym = .true.
  integer, public :: pfafem_ep_dim = 1
  integer, public :: pfafem_ng_tor = 4
  real*8,  public :: pfafem_dphi = 0.d0
  real*8,  public :: pfafem_phi_per = 0.d0

  real*8, allocatable, public :: pfafem_tau_q(:)      ! (1:pfafem_ng_tor) Gauss points on [0,1]
  real*8, allocatable, public :: pfafem_w_q(:)        ! (1:pfafem_ng_tor) Gauss weights summing to 1
  real*8, allocatable, public :: pfafem_w_tor(:)      ! (1:pfafem_ng_tor) (n_plane / n_tor) * w_q
  real*8, allocatable, public :: pfafem_t_pol(:,:,:,:,:) ! (ms, mt=1..pfafem_ng_pol, iq, iep, iel_loc)
  real*8, allocatable, public :: pfafem_w_pol(:,:,:,:,:) ! (ms, mt=1..pfafem_ng_pol, iq, iep, iel_loc)
  real*8, allocatable, public :: pfafem_Q(:,:)        ! (1:n_tor, 1:n_tor) nodal-from-spectral TorFFT
  real*8, allocatable, public :: pfafem_QT(:,:)       ! (1:n_tor, 1:n_tor) spectral-from-nodal Q^T
  real*8 :: pfafem_gl_x(pfafem_ng_pol_max, pfafem_ng_pol_max) = 0.d0
  real*8 :: pfafem_gl_w(pfafem_ng_pol_max, pfafem_ng_pol_max) = 0.d0

  type(type_pfafem_elem_cache), allocatable, public :: pfafem_cache(:) ! (1:n_elements)

  ! Disjoint element range owned by this MPI rank during volume assembly [pfafem_el_lo .. pfafem_el_hi]
  integer, public :: pfafem_el_lo = 1
  integer, public :: pfafem_el_hi = 0

  ! Precomputed quadrature-point element geometry & raw 10-derivative basis jets on [pfafem_el_lo .. pfafem_el_hi]
  ! pfafem_qp_geom(1:21): [BigR, y_g, x_s, x_t, x_p, y_s, y_t, y_p, x_st, x_tt, y_st, y_tt,
  !                        xjac, xjac_x, xjac_y, x_p_x, x_p_y, y_p_x, y_p_y, wvol, psi_norm]
  ! pfafem_qp_db(1:10, 1:64): [rv, rR, rZ, rP, rRR, rZZ, rRZ, rPP, rRP, rZP] (matching d_ix(3,10))
  logical,    public :: pfafem_qp_cached = .false.
  real*8,     allocatable, public :: pfafem_qp_geom(:,:,:,:,:,:)
  integer(2), allocatable, public :: pfafem_qp_ngrp(:,:,:,:,:)
  integer(2), allocatable, public :: pfafem_qp_gu(:,:,:,:,:,:)
  integer(2), allocatable, public :: pfafem_qp_gjb(:,:,:,:,:,:)
  integer(2), allocatable, public :: pfafem_qp_giuj(:,:,:,:,:,:)
  real*8,     allocatable, public :: pfafem_qp_db(:,:,:,:,:,:,:)

#ifdef STELLARATOR_MODEL
  ! Precomputed Dommaschk vacuum potential derivatives at local rank's PFAFEM quadrature points:
  ! dims (0:2, 0:2, 0:2, ms=1..n_gauss, mt=1..pfafem_ng_pol, iq=1..pfafem_ng_tor, ep=1..n_tor, iel_loc=1..n_el_loc)
  real*8, allocatable, public :: pfafem_chi_q(:,:,:,:,:,:,:,:)
#endif

  ! --- Serializable structured C^2 cubic B-spline B-field tangent grid (for HDF5 restart I/O) ---
  logical, public :: pfafem_bgrid_valid = .false.
  integer, public :: pfafem_bgrid_dims(3) = [0, 0, 0]                ! [nr, nz, nphi]
  real*8,  public :: pfafem_bgrid_bounds(5) = [0.d0, 0.d0, 0.d0, 0.d0, 0.d0] ! [Rmin, Rmax, Zmin, Zmax, phi_per]
  real*8, allocatable, public :: pfafem_bgrid_coefs(:,:,:,:)         ! (2, nr+2, nz+2, max(1,nphi))

  ! --- Restart cross-discretization projection staging buffers ---
  logical, public :: pfafem_restart_needs_projection = .false.
  integer, public :: pfafem_restart_src_type = 0   ! 0 = none, 1 = Fourier, 2 = PFAFEM (different n_tor)
  integer, public :: pfafem_restart_src_ntor = 0
  real*8, allocatable, public :: pfafem_restart_values(:,:,:,:)      ! (n_tor_src, n_degrees, n_var, n_nodes)
  real*8, allocatable, public :: pfafem_restart_deltas(:,:,:,:)      ! (n_tor_src, n_degrees, n_var, n_nodes)
#ifdef STELLARATOR_MODEL
  real*8, allocatable, public :: pfafem_restart_jsrc(:,:,:)          ! (n_tor_src, n_degrees, n_nodes)
#endif

  ! --- Cached boundary-element indicator (1..n_elements), knot geometry, and compact lookup ---
  logical, allocatable :: elem_is_bnd(:)
  integer :: n_bnd_elems = 0
  integer, allocatable :: bnd_elem_list(:)
  real*8,  allocatable :: bnd_elem_knot_rz(:, :, :)
  real*8,  allocatable :: pfafem_knot_x(:, :, :, :) ! (2, n_degrees, n_nodes, n_tor)

  ! --- Public procedures ---
  public :: pfafem_bspline_eval
  public :: pfafem_slot
  public :: pfafem_torfft_matrices
  public :: pfafem_collocation_fit
  public :: pfafem_eval_coord_hz
  public :: pfafem_hermite2d_jet
  public :: pfafem_eval_elem_geom
  public :: pfafem_fieldline_jet
  public :: pfafem_trace_with_hessian
  public :: pfafem_axis_pmat
  public :: pfafem_init_eval_only
  public :: pfafem_build_cache
  public :: pfafem_eval_qp_basis
  public :: pfafem_eval_point_basis_val
  public :: pfafem_eval_point_jet
  public :: pfafem_trace_point_1st
  public :: pfafem_eval_cached_1st
  public :: pfafem_bcast_restart_state
  public :: pfafem_stage_fourier_from_nodes
  public :: pfafem_project_restart
  public :: pfafem_retrace_and_project
  public :: pfafem_cleanup

contains

  !=============================================================================
  !> Evaluate the 4 active uniform cubic B-spline basis functions and their
  !! 1st and 2nd derivatives w.r.t. local coordinate tau in [0, 1].
  !! Divide dnb by dphi and ddnb by dphi^2 for derivatives w.r.t. phi.
  !=============================================================================
  pure subroutine pfafem_bspline_eval(tau, nb, dnb, ddnb)
    real*8, intent(in)  :: tau
    real*8, intent(out) :: nb(4), dnb(4), ddnb(4)
    real*8 :: t, mt

    t  = tau
    mt = 1.d0 - t

    nb(1) = mt * mt * mt / 6.d0
    nb(2) = (3.d0*t*t*t - 6.d0*t*t + 4.d0) / 6.d0
    nb(3) = (-3.d0*t*t*t + 3.d0*t*t + 3.d0*t + 1.d0) / 6.d0
    nb(4) = t * t * t / 6.d0

    dnb(1) = -0.5d0 * mt * mt
    dnb(2) = (9.d0*t*t - 12.d0*t) / 6.d0
    dnb(3) = (-9.d0*t*t + 6.d0*t + 3.d0) / 6.d0
    dnb(4) = 0.5d0 * t * t

    ddnb(1) = mt
    ddnb(2) = 3.d0*t - 2.d0
    ddnb(3) = -3.d0*t + 1.d0
    ddnb(4) = t
  end subroutine pfafem_bspline_eval

  !=============================================================================
  !> Map toroidal cell eP in {0..n_tor_in-1} and local spline lb in {1..4}
  !! to 1-based periodic global B-spline slot jB in {1..n_tor_in}.
  !! Slot jB is centered (peaks with value 4/6) at plane phi = (jB - 1)*dphi,
  !! matching reference plane phi_ref = (eP - 2 + lb)*dphi modulo Phi_per.
  !=============================================================================
  pure integer function pfafem_slot(eP, lb, n_tor_in)
    integer, intent(in) :: eP, lb, n_tor_in
    pfafem_slot = 1 + modulo(eP - 2 + lb, n_tor_in)
  end function pfafem_slot

  !=============================================================================
  !> Build orthogonal Real-DFT (TorFFT) matrices Q and QT = Q^T of size n x n
  !! (n >= 1, both even and odd supported).
  !! Column layout (1-based col = m + 1 for 0-based harmonic slot m = 0..n-1):
  !!  - col 1 (m=0): 1/sqrt(n)
  !!  - cols 2k, 2k+1 (1 <= k <= (n-1)/2): sqrt(2/n)*cos(2*pi*k*(j-1)/n),
  !!                                      -sqrt(2/n)*sin(2*pi*k*(j-1)/n)
  !!  - col n (m=n-1, even n only): (-1)^(j-1) / sqrt(n)  (Nyquist mode)
  !=============================================================================
  pure subroutine pfafem_torfft_matrices(n, Q, QT)
    integer, intent(in)  :: n
    real*8,  intent(out) :: Q(n, n), QT(n, n)
    integer :: j, k, n_pairs
    real*8  :: s0, s2, ang

    Q = 0.d0
    s0 = 1.d0 / sqrt(dble(n))
    s2 = sqrt(2.d0 / dble(n))
    n_pairs = (n - 1) / 2

    do j = 1, n
      Q(j, 1) = s0
      do k = 1, n_pairs
        ang = 2.d0 * pi * dble(k * (j - 1)) / dble(n)
        Q(j, 2 * k)     =  s2 * cos(ang)
        Q(j, 2 * k + 1) = -s2 * sin(ang)
      end do
      if (mod(n, 2) == 0) then
        if (mod(j - 1, 2) == 0) then
          Q(j, n) =  s0
        else
          Q(j, n) = -s0
        end if
      end if
    end do
    QT = transpose(Q)
  end subroutine pfafem_torfft_matrices

  !=============================================================================
  !> Fit 1D periodic cubic B-spline coefficients c_spl(1:n) that interpolate
  !! nodal values f_planes(1:n) at planes phi_j = (j - 1)*dphi:
  !!   (1/6)*c(j-1) + (4/6)*c(j) + (1/6)*c(j+1) = f_planes(j)  (periodic).
  !! Solved analytically via DFT/circulant eigenvalues (exact for any n >= 3).
  !=============================================================================
  pure subroutine pfafem_collocation_fit(n, f_planes, c_spl)
    integer, intent(in)  :: n
    real*8,  intent(in)  :: f_planes(n)
    real*8,  intent(out) :: c_spl(n)
    real*8  :: Q(n, n), QT(n, n), spec(n), lam
    integer :: k, n_pairs

    call pfafem_torfft_matrices(n, Q, QT)
    spec = matmul(QT, f_planes)
    spec(1) = spec(1) / 1.d0   ! k = 0 eigenvalue: 4/6 + 2/6 = 1
    n_pairs = (n - 1) / 2
    do k = 1, n_pairs
      lam = (2.d0 + cos(2.d0 * pi * dble(k) / dble(n))) / 3.d0
      spec(2 * k)     = spec(2 * k)     / lam
      spec(2 * k + 1) = spec(2 * k + 1) / lam
    end do
    if (mod(n, 2) == 0) then
      ! Nyquist eigenvalue: 4/6 - 2/6 = 1/3
      spec(n) = spec(n) * 3.d0
    end if
    c_spl = matmul(Q, spec)
  end subroutine pfafem_collocation_fit

  !=============================================================================
  !> Solve 1D clamped-slope cubic B-spline interpolation on N >= 4 uniform nodes
  !! x_1 .. x_N, returning c_out(1:N+2) with O(h^4) fourth-order end slopes.
  !=============================================================================
  pure subroutine fit_1d_clamped_bspline(n, h, y, c_out)
    integer, intent(in)  :: n
    real*8,  intent(in)  :: h, y(n)
    real*8,  intent(out) :: c_out(n + 2)
    real*8  :: dl(n), dg(n), du(n), rhs(n), dy1, dyn, w
    integer :: i

    ! 4th-order one-sided first derivatives at endpoints
    dy1 = (-11.d0*y(1) + 18.d0*y(2) - 9.d0*y(3) + 2.d0*y(4)) / (6.d0 * h)
    dyn = ( 11.d0*y(n) - 18.d0*y(n-1) + 9.d0*y(n-2) - 2.d0*y(n-3)) / (6.d0 * h)

    dl  = 1.d0 / 6.d0
    dg  = 4.d0 / 6.d0
    du  = 1.d0 / 6.d0
    rhs = y

    ! Eliminate c_out(1) = c_out(3) - 2*h*dy1 and c_out(n+2) = c_out(n) + 2*h*dyn
    dl(1)   = 0.d0
    du(1)   = 2.d0 / 6.d0
    rhs(1)  = y(1) + (h * dy1) / 3.d0

    dl(n)   = 2.d0 / 6.d0
    du(n)   = 0.d0
    rhs(n)  = y(n) - (h * dyn) / 3.d0

    ! Thomas algorithm for strictly diagonally dominant tridiagonal system
    do i = 2, n
      w = dl(i) / dg(i - 1)
      dg(i)  = dg(i)  - w * du(i - 1)
      rhs(i) = rhs(i) - w * rhs(i - 1)
    end do

    c_out(n + 1) = rhs(n) / dg(n)
    do i = n - 1, 1, -1
      c_out(i + 1) = (rhs(i) - du(i) * c_out(i + 2)) / dg(i)
    end do
    c_out(1)     = c_out(3)     - 2.d0 * h * dy1
    c_out(n + 2) = c_out(n)     + 2.d0 * h * dyn
  end subroutine fit_1d_clamped_bspline

  !=============================================================================
  !> Gauss-Legendre rule on [0, 1] for toroidal interval quadrature + 1..pfafem_ng_pol
  !! lookup table for seam-split poloidal sub-interval quadrature.
  !=============================================================================
  subroutine init_tor_gauss(ng_req)
    integer, intent(in) :: ng_req
    integer :: i, k, it, n, m_ord
    real*8  :: z, p0, p1, p2, pn, pm, pp, dz

    n = max(2, min(64, ng_req))
    pfafem_ng_tor = n
    if (allocated(pfafem_tau_q)) deallocate(pfafem_tau_q, pfafem_w_q, pfafem_w_tor)
    allocate(pfafem_tau_q(n), pfafem_w_q(n), pfafem_w_tor(n))

    do i = 1, n
      z = cos(pi * (dble(i) - 0.25d0) / (dble(n) + 0.5d0))
      pp = 1.d0
      do it = 1, 100
        p0 = 1.d0
        p1 = z
        do k = 2, n
          p2 = ((2.d0 * dble(k) - 1.d0) * z * p1 - dble(k - 1) * p0) / dble(k)
          p0 = p1
          p1 = p2
        end do
        pn = p1
        pm = p0
        pp = dble(n) * (z * pn - pm) / (z * z - 1.d0)
        dz = pn / pp
        z  = z - dz
        if (abs(dz) < 1.d-16) exit
      end do
      pfafem_tau_q(i) = 0.5d0 * (1.d0 - z)
      pfafem_w_q(i)   = 1.d0 / ((1.d0 - z * z) * pp * pp)
      pfafem_w_tor(i) = (dble(n_plane) / dble(n_tor)) * pfafem_w_q(i)
    end do

    pfafem_gl_x = 0.d0
    pfafem_gl_w = 0.d0
    pfafem_gl_x(1, 1) = 0.5d0
    pfafem_gl_w(1, 1) = 1.d0
    do m_ord = 2, pfafem_ng_pol
      do i = 1, m_ord
        z = cos(pi * (dble(i) - 0.25d0) / (dble(m_ord) + 0.5d0))
        pp = 1.d0
        do it = 1, 100
          p0 = 1.d0
          p1 = z
          do k = 2, m_ord
            p2 = ((2.d0 * dble(k) - 1.d0) * z * p1 - dble(k - 1) * p0) / dble(k)
            p0 = p1
            p1 = p2
          end do
          pn = p1
          pm = p0
          pp = dble(m_ord) * (z * pn - pm) / (z * z - 1.d0)
          dz = pn / pp
          z  = z - dz
          if (abs(dz) < 1.d-16) exit
        end do
        pfafem_gl_x(i, m_ord) = 0.5d0 * (1.d0 - z)
        pfafem_gl_w(i, m_ord) = 1.d0 / ((1.d0 - z * z) * pp * pp)
      end do
    end do
  end subroutine init_tor_gauss

  !=============================================================================
  !> Evaluate 2D bicubic Hermite element basis functions at arbitrary (s, t)
  !! with canonical argument order (H0, Hs, Ht, [Hss, Hst, Htt]).
  !=============================================================================
  pure subroutine pfafem_hermite2d_jet(s, t, H0, Hs, Ht, Hss, Hst, Htt)
    use mod_basisfunctions, only: basisfunctions
    real*8, intent(in)            :: s, t
    real*8, intent(out)           :: H0(4, n_degrees), Hs(4, n_degrees), Ht(4, n_degrees)
    real*8, intent(out), optional :: Hss(4, n_degrees), Hst(4, n_degrees), Htt(4, n_degrees)

    if (present(Hss)) then
      call basisfunctions(s, t, H0, Hs, Ht, Hst, Hss, Htt)
    else
      call basisfunctions(s, t, H0, Hs, Ht)
    end if
  end subroutine pfafem_hermite2d_jet

  !=============================================================================
  !> Evaluate JOREK coordinate toroidal Fourier basis HZ_coord(1:n_coord_tor)
  !! and its 1st/2nd phi derivatives at arbitrary angle phi.
  !! Matches elements/basis_at_gaussian.f90:98-105 (-sin on odd slots!).
  !=============================================================================
  pure subroutine pfafem_eval_coord_hz(phi, hz_c, hzp_c, hzpp_c)
    use phys_module, only: mode_coord
    real*8, intent(in)  :: phi
    real*8, intent(out) :: hz_c(n_coord_tor)
    real*8, intent(out), optional :: hzp_c(n_coord_tor), hzpp_c(n_coord_tor)
    integer :: i
    real*8  :: mc, cs, sn

    hz_c(1) = 1.d0
    if (present(hzp_c))  hzp_c(1)  = 0.d0
    if (present(hzpp_c)) hzpp_c(1) = 0.d0

    do i = 1, (n_coord_tor - 1) / 2
      mc = dble(mode_coord(2 * i))
      cs = cos(mc * phi)
      sn = sin(mc * phi)
      hz_c(2 * i)     =  cs
      hz_c(2 * i + 1) = -sn
      if (present(hzp_c)) then
        hzp_c(2 * i)     = -mc * sn
        hzp_c(2 * i + 1) = -mc * cs
      end if
      if (present(hzpp_c)) then
        hzpp_c(2 * i)     = -mc * mc * cs
        hzpp_c(2 * i + 1) =  mc * mc * sn
      end if
    end do
  end subroutine pfafem_eval_coord_hz

  !=============================================================================
  !> Evaluate JOREK physics Fourier basis HZ(1:n_src) at arbitrary angle phi
  !! (+sin on odd slots, matching basis_at_gaussian.f90:84-96).
  !=============================================================================
  pure subroutine eval_phys_fourier_hz(n_src, phi, hz_f, hzp_f)
    integer, intent(in)  :: n_src
    real*8,  intent(in)  :: phi
    real*8,  intent(out) :: hz_f(n_src)
    real*8,  intent(out), optional :: hzp_f(n_src)
    integer :: i
    real*8  :: md, cs, sn

    hz_f(1) = 1.d0
    if (present(hzp_f)) hzp_f(1) = 0.d0
    do i = 1, (n_src - 1) / 2
      md = dble(i * n_period)
      cs = cos(md * phi)
      sn = sin(md * phi)
      hz_f(2 * i)     = cs
      hz_f(2 * i + 1) = sn
      if (present(hzp_f)) then
        hzp_f(2 * i)     = -md * sn
        hzp_f(2 * i + 1) =  md * cs
      end if
    end do
    if (mod(n_src, 2) == 0 .and. n_src >= 2) then
      md = dble((n_src / 2) * n_period)
      hz_f(n_src) = cos(md * phi)
      if (present(hzp_f)) hzp_f(n_src) = -md * sin(md * phi)
    end if
  end subroutine eval_phys_fourier_hz

  !=============================================================================
  !> Evaluate physical (R, Z) and parametric (s, t) 1st and optional 2nd
  !! derivatives of element ielm at arbitrary (s, t, phi).
  !=============================================================================
  pure subroutine pfafem_eval_elem_geom(node_list, element_list, ielm, s, t, phi, &
                                        R, Z, Rs, Rt, Zs, Zt, Rss, Rst, Rtt, Zss, Zst, Ztt)
    use mod_basisfunctions, only: basisfunctions
    type(type_node_list),    intent(in)  :: node_list
    type(type_element_list), intent(in)  :: element_list
    integer,                 intent(in)  :: ielm
    real*8,                  intent(in)  :: s, t, phi
    real*8,                  intent(out) :: R, Z, Rs, Rt, Zs, Zt
    real*8, optional,        intent(out) :: Rss, Rst, Rtt, Zss, Zst, Ztt

    real*8  :: H_l(4, n_degrees), Hs_l(4, n_degrees), Ht_l(4, n_degrees)
    real*8  :: Hst_l(4, n_degrees), Hss_l(4, n_degrees), Htt_l(4, n_degrees)
    real*8  :: hz_c(n_coord_tor), rx, zx, sz
    integer :: iv, io, in_c, inode

    if (present(Rss)) then
      call basisfunctions(s, t, H_l, Hs_l, Ht_l, Hst_l, Hss_l, Htt_l)
      Rss = 0.d0; Rst = 0.d0; Rtt = 0.d0
      Zss = 0.d0; Zst = 0.d0; Ztt = 0.d0
    else
      call basisfunctions(s, t, H_l, Hs_l, Ht_l)
    end if

    if (n_coord_tor == 1) then
      hz_c(1) = 1.d0
    else
      call pfafem_eval_coord_hz(phi, hz_c)
    end if

    R  = 0.d0; Z  = 0.d0
    Rs = 0.d0; Rt = 0.d0
    Zs = 0.d0; Zt = 0.d0

    do iv = 1, n_vertex_max
      inode = element_list%element(ielm)%vertex(iv)
      do io = 1, n_degrees
        sz = element_list%element(ielm)%size(iv, io)
        rx = 0.d0
        zx = 0.d0
        do in_c = 1, n_coord_tor
          rx = rx + node_list%node(inode)%x(in_c, io, 1) * hz_c(in_c)
          zx = zx + node_list%node(inode)%x(in_c, io, 2) * hz_c(in_c)
        end do
        rx = rx * sz
        zx = zx * sz
        R  = R  + rx * H_l(iv, io)
        Z  = Z  + zx * H_l(iv, io)
        Rs = Rs + rx * Hs_l(iv, io)
        Rt = Rt + rx * Ht_l(iv, io)
        Zs = Zs + zx * Hs_l(iv, io)
        Zt = Zt + zx * Ht_l(iv, io)
        if (present(Rss)) then
          Rss = Rss + rx * Hss_l(iv, io)
          Rst = Rst + rx * Hst_l(iv, io)
          Rtt = Rtt + rx * Htt_l(iv, io)
          Zss = Zss + zx * Hss_l(iv, io)
          Zst = Zst + zx * Hst_l(iv, io)
          Ztt = Ztt + zx * Htt_l(iv, io)
        end if
      end do
    end do
  end subroutine pfafem_eval_elem_geom

  !=============================================================================
  !> Fast evaluation of physical (R, Z) and parametric (s, t) 1st and optional
  !! 2nd derivatives of element ielm at toroidal knot plane jb in 1..n_tor using
  !! pre-contracted pfafem_knot_x(:, io, inode, jb) (avoids n_coord_tor loop).
  !=============================================================================
  pure subroutine pfafem_eval_knot_elem_geom(node_list, element_list, ielm, jb, s, t, phi_c, &
                                             R, Z, Rs, Rt, Zs, Zt, Rss, Rst, Rtt, Zss, Zst, Ztt)
    use mod_basisfunctions, only: basisfunctions
    type(type_node_list),    intent(in)  :: node_list
    type(type_element_list), intent(in)  :: element_list
    integer,                 intent(in)  :: ielm, jb
    real*8,                  intent(in)  :: s, t, phi_c
    real*8,                  intent(out) :: R, Z, Rs, Rt, Zs, Zt
    real*8, optional,        intent(out) :: Rss, Rst, Rtt, Zss, Zst, Ztt

    real*8  :: H_l(4, n_degrees), Hs_l(4, n_degrees), Ht_l(4, n_degrees)
    real*8  :: Hst_l(4, n_degrees), Hss_l(4, n_degrees), Htt_l(4, n_degrees)
    real*8  :: rx, zx, sz
    integer :: iv, io, inode

    if (.not. allocated(pfafem_knot_x)) then
      if (present(Rss)) then
        call pfafem_eval_elem_geom(node_list, element_list, ielm, s, t, phi_c, &
                                   R, Z, Rs, Rt, Zs, Zt, Rss, Rst, Rtt, Zss, Zst, Ztt)
      else
        call pfafem_eval_elem_geom(node_list, element_list, ielm, s, t, phi_c, &
                                   R, Z, Rs, Rt, Zs, Zt)
      end if
      return
    end if

    if (present(Rss)) then
      call basisfunctions(s, t, H_l, Hs_l, Ht_l, Hst_l, Hss_l, Htt_l)
      Rss = 0.d0; Rst = 0.d0; Rtt = 0.d0
      Zss = 0.d0; Zst = 0.d0; Ztt = 0.d0
    else
      call basisfunctions(s, t, H_l, Hs_l, Ht_l)
    end if

    R  = 0.d0; Z  = 0.d0
    Rs = 0.d0; Rt = 0.d0
    Zs = 0.d0; Zt = 0.d0

    do iv = 1, n_vertex_max
      inode = element_list%element(ielm)%vertex(iv)
      do io = 1, n_degrees
        sz = element_list%element(ielm)%size(iv, io)
        rx = pfafem_knot_x(1, io, inode, jb) * sz
        zx = pfafem_knot_x(2, io, inode, jb) * sz
        R  = R  + rx * H_l(iv, io)
        Z  = Z  + zx * H_l(iv, io)
        Rs = Rs + rx * Hs_l(iv, io)
        Rt = Rt + rx * Ht_l(iv, io)
        Zs = Zs + zx * Hs_l(iv, io)
        Zt = Zt + zx * Ht_l(iv, io)
        if (present(Rss)) then
          Rss = Rss + rx * Hss_l(iv, io)
          Rst = Rst + rx * Hst_l(iv, io)
          Rtt = Rtt + rx * Htt_l(iv, io)
          Zss = Zss + zx * Hss_l(iv, io)
          Zst = Zst + zx * Hst_l(iv, io)
          Ztt = Ztt + zx * Htt_l(iv, io)
        end if
      end do
    end do
  end subroutine pfafem_eval_knot_elem_geom

  !=============================================================================
  !> Damped Newton inversion of (R, Z) -> (s, t) on element ielm at angle phi,
  !! constrained to [lo, hi]^2. Handles degenerate axis-wedge corners cleanly.
  !=============================================================================
  logical function newton_invert_elem(node_list, element_list, ielm, R, Z, phi, &
                                      lo, hi, tol, s_out, t_out)
    type(type_node_list),    intent(in)  :: node_list
    type(type_element_list), intent(in)  :: element_list
    integer,                 intent(in)  :: ielm
    real*8,                  intent(in)  :: R, Z, phi, lo, hi, tol
    real*8,                  intent(out) :: s_out, t_out

    integer :: it, ls
    real*8  :: s, t, sn, tn, Rr, Zr, Rs, Rt, Zs, Zt
    real*8  :: fr, fz, err0, err1, det, ds, dt, alpha

    newton_invert_elem = .false.
    s = 0.5d0
    t = 0.5d0

    do it = 1, 45
      call pfafem_eval_elem_geom(node_list, element_list, ielm, s, t, phi, Rr, Zr, Rs, Rt, Zs, Zt)
      fr = R - Rr
      fz = Z - Zr
      err0 = hypot(fr, fz)
      if (err0 < tol) then
        s_out = s
        t_out = t
        newton_invert_elem = .true.
        return
      end if
      det = Rs * Zt - Rt * Zs
      if (abs(det) < 1.d-26) then
        ! Nudge slightly off collapsed axis edge s=0
        s = min(hi, max(lo, s + 1.d-3))
        cycle
      end if
      ds = ( Zt * fr - Rt * fz) / det
      dt = (-Zs * fr + Rs * fz) / det
      alpha = 1.d0
      do ls = 1, 5
        sn = min(hi, max(lo, s + alpha * ds))
        tn = min(hi, max(lo, t + alpha * dt))
        call pfafem_eval_elem_geom(node_list, element_list, ielm, sn, tn, phi, Rr, Zr, Rs, Rt, Zs, Zt)
        err1 = hypot(R - Rr, Z - Zr)
        if (err1 < err0 .or. ls == 5) exit
        alpha = 0.5d0 * alpha
      end do
      if (hypot(sn - s, tn - t) < 1.d-14 .and. err1 > 1.d-9) exit
      s = sn
      t = tn
    end do

    call pfafem_eval_elem_geom(node_list, element_list, ielm, s, t, phi, Rr, Zr, Rs, Rt, Zs, Zt)
    if (hypot(R - Rr, Z - Zr) < max(1.d-9, 10.d0 * tol)) then
      s_out = s
      t_out = t
      newton_invert_elem = .true.
    else
      s_out = s
      t_out = t
    end if
  end function newton_invert_elem

  !=============================================================================
  !> Locate physical point (R, Z, phi) in the JOREK mesh:
  !!  status_out = 1: standard interior element ([0, 1]^2 within 1e-8)
  !!  status_out = 2: OpenBC bicubic Hermite continuation on nearest boundary
  !!                  element ([-1.5, 2.5]^2, Jacobian non-degenerate)
  !!  status_out = 3: OpenBC clamped boundary point fallback (PoU-exact)
  !!  status_out = 0: outside domain / failed
  !=============================================================================
  subroutine pfafem_invert_point(node_list, element_list, R, Z, phi, allow_openbc, &
                                 e_out, s_out, t_out, status_out)
    use mod_element_rtree, only: elements_containing_point
    type(type_node_list),    intent(in)  :: node_list
    type(type_element_list), intent(in)  :: element_list
    real*8,                  intent(in)  :: R, Z, phi
    logical,                 intent(in)  :: allow_openbc
    integer,                 intent(out) :: e_out, status_out
    real*8,                  intent(out) :: s_out, t_out

    integer, allocatable :: cand(:)
    integer :: n_found, k, idx_b, jb_knot, ie, best_e, iv, inode
    real*8  :: phi_c, phi_cper, s_try, t_try, best_over, over
    real*8  :: Rc, Zc, dist2, hz_c(n_coord_tor), knot_pos
    real*8  :: R0, Z0, Rs0, Rt0, Zs0, Zt0, det0, R1, Z1, Rs1, Rt1, Zs1, Zt1, det1
    integer :: top_bnd_e(6)
    real*8  :: top_bnd_d2(6)

    e_out = 0
    s_out = 0.5d0
    t_out = 0.5d0
    status_out = 0
    if (.not. (R > 0.d0)) return

    phi_cper = 2.d0 * pi / dble(n_coord_period)
    phi_c = modulo(phi, phi_cper)

    ! 1. Fast RTree bounding-box lookup + interior [0, 1]^2 Newton inversion
    call elements_containing_point(R, Z, phi_c, cand)
    n_found = size(cand)
    best_e = 0
    best_over = 1.d30
    do k = 1, n_found
      ie = cand(k)
      if (ie < 1 .or. ie > element_list%n_elements) cycle
      if (newton_invert_elem(node_list, element_list, ie, R, Z, phi_c, &
                             -1.d-7, 1.d0 + 1.d-7, 1.d-12, s_try, t_try)) then
        over = max(0.d0, -s_try, s_try - 1.d0, -t_try, t_try - 1.d0)
        if (over <= 1.d-8) then
          e_out = ie
          s_out = min(1.d0, max(0.d0, s_try))
          t_out = min(1.d0, max(0.d0, t_try))
          status_out = 1
          deallocate(cand)
          return
        else if (over < best_over) then
          best_over = over
          best_e = ie
          s_out = s_try
          t_out = t_try
        end if
      end if
    end do
    deallocate(cand)

    if (best_e > 0 .and. best_over <= 1.d-6) then
      e_out = best_e
      s_out = min(1.d0, max(0.d0, s_out))
      t_out = min(1.d0, max(0.d0, t_out))
      status_out = 1
      return
    end if

    ! Fast scan over cached boundary elements (e.g. 64 instead of 5568)
    jb_knot = 0
    if (allocated(bnd_elem_knot_rz) .and. pfafem_dphi > 0.d0) then
      knot_pos = modulo(phi, pfafem_phi_per) / pfafem_dphi
      k = mod(nint(knot_pos), n_tor) + 1
      if (abs(knot_pos - dble(nint(knot_pos))) < 1.d-10) jb_knot = k
    end if
    if (jb_knot == 0) then
      if (n_coord_tor == 1) then
        hz_c(1) = 1.d0
      else
        call pfafem_eval_coord_hz(phi_c, hz_c)
      end if
    end if
    top_bnd_e  = 0
    top_bnd_d2 = 1.d30

    do idx_b = 1, n_bnd_elems
      ie = bnd_elem_list(idx_b)
      if (jb_knot > 0) then
        Rc = bnd_elem_knot_rz(1, idx_b, jb_knot)
        Zc = bnd_elem_knot_rz(2, idx_b, jb_knot)
      else
        Rc = 0.d0; Zc = 0.d0
        do iv = 1, n_vertex_max
          inode = element_list%element(ie)%vertex(iv)
          Rc = Rc + dot_product(node_list%node(inode)%x(1:n_coord_tor, 1, 1), hz_c)
          Zc = Zc + dot_product(node_list%node(inode)%x(1:n_coord_tor, 1, 2), hz_c)
        end do
        Rc = 0.25d0 * Rc
        Zc = 0.25d0 * Zc
      end if
      dist2 = (R - Rc)**2 + (Z - Zc)**2
      do k = 1, 6
        if (dist2 < top_bnd_d2(k)) then
          if (k < 6) then
            top_bnd_d2(k+1:6) = top_bnd_d2(k:5)
            top_bnd_e(k+1:6)  = top_bnd_e(k:5)
          end if
          top_bnd_d2(k) = dist2
          top_bnd_e(k)  = ie
          exit
        end if
      end do
    end do

    ! Check top boundary elements for exact interior or near-boundary hit missed by vertex bbox
    do k = 1, 6
      ie = top_bnd_e(k)
      if (ie < 1) cycle
      if (newton_invert_elem(node_list, element_list, ie, R, Z, phi_c, &
                             -1.d-4, 1.d0 + 1.d-3, 1.d-12, s_try, t_try)) then
        e_out = ie
        s_out = min(1.d0, max(0.d0, s_try))
        t_out = min(1.d0, max(0.d0, t_try))
        status_out = 1
        return
      end if
    end do

    if (.not. allow_openbc) return

    ! 2. OpenBC Hermite continuation on nearest boundary elements (s, t in [-1.5, 2.5])
    best_e = 0
    best_over = 1.d30
    do k = 1, 6
      ie = top_bnd_e(k)
      if (ie < 1) cycle
      if (newton_invert_elem(node_list, element_list, ie, R, Z, phi_c, &
                             -1.5d0, 2.5d0, 1.d-11, s_try, t_try)) then
        call pfafem_eval_elem_geom(node_list, element_list, ie, 0.5d0, 0.5d0, phi_c, &
                                   R0, Z0, Rs0, Rt0, Zs0, Zt0)
        call pfafem_eval_elem_geom(node_list, element_list, ie, s_try, t_try, phi_c, &
                                   R1, Z1, Rs1, Rt1, Zs1, Zt1)
        det0 = Rs0 * Zt0 - Rt0 * Zs0
        det1 = Rs1 * Zt1 - Rt1 * Zs1
        ! Require well-conditioned continued Jacobian with same orientation
        if (det0 * det1 > 0.d0 .and. abs(det1) > 0.15d0 * abs(det0)) then
          over = max(0.d0, -s_try, s_try - 1.d0) + max(0.d0, -t_try, t_try - 1.d0)
          if (over < best_over) then
            best_over = over
            best_e = ie
            s_out = s_try
            t_out = t_try
          end if
        end if
      end if
    end do

    if (best_e > 0) then
      e_out = best_e
      status_out = 2
      return
    end if

    ! 3. Safe boundary-clamped fallback on closest boundary element (PoU-exact)
    if (top_bnd_e(1) > 0) then
      ie = top_bnd_e(1)
      s_try = 0.5d0; t_try = 0.5d0
      if (newton_invert_elem(node_list, element_list, ie, R, Z, phi_c, &
                             -4.d0, 5.d0, 1.d-8, s_try, t_try)) then
        e_out = ie
        s_out = min(1.d0, max(0.d0, s_try))
        t_out = min(1.d0, max(0.d0, t_try))
      else
        ! Project onto [0,1]^2 boundary of closest boundary element by 5x5 coarse + clamped Newton
        call project_onto_elem_clamped(node_list, element_list, ie, R, Z, phi_c, s_out, t_out)
        e_out = ie
      end if
      status_out = 3
    end if
  end subroutine pfafem_invert_point

  !=============================================================================
  !> Find closest (s, t) in [0, 1]^2 of element ie to (R, Z) at angle phi.
  !=============================================================================
  pure subroutine project_onto_elem_clamped(node_list, element_list, ie, R, Z, phi, s_cl, t_cl)
    type(type_node_list),    intent(in)  :: node_list
    type(type_element_list), intent(in)  :: element_list
    integer,                 intent(in)  :: ie
    real*8,                  intent(in)  :: R, Z, phi
    real*8,                  intent(out) :: s_cl, t_cl
    integer :: is, it
    real*8  :: st, tt, Rr, Zr, Rs, Rt, Zs, Zt, d2, best_d2

    best_d2 = 1.d30
    s_cl = 0.5d0
    t_cl = 0.5d0
    do is = 0, 8
      st = dble(is) / 8.d0
      do it = 0, 8
        tt = dble(it) / 8.d0
        call pfafem_eval_elem_geom(node_list, element_list, ie, st, tt, phi, Rr, Zr, Rs, Rt, Zs, Zt)
        d2 = (R - Rr)**2 + (Z - Zr)**2
        if (d2 < best_d2) then
          best_d2 = d2
          s_cl = st
          t_cl = tt
        end if
      end do
    end do
  end subroutine project_onto_elem_clamped

  !=============================================================================
  !> Evaluate JOREK's field-line ODE right-hand side (f_R, f_Z) = R*(B_R, B_Z)/B_phi
  !! directly from JOREK finite element fields at (R, Z, phi) on (ie, s, t).
  !=============================================================================
  subroutine eval_jorek_fl_at_elem(node_list, element_list, ie, s, t, R, Z, phi, &
                                   use_current_pfafem, fl_out, ok)
    use phys_module, only: F0
    use mod_basisfunctions, only: basisfunctions
#ifdef STELLARATOR_MODEL
    use mod_chi, only: get_chi
#endif
    type(type_node_list),    intent(in)  :: node_list
    type(type_element_list), intent(in)  :: element_list
    integer,                 intent(in)  :: ie
    real*8,                  intent(in)  :: s, t, R, Z, phi
    logical,                 intent(in)  :: use_current_pfafem
    real*8,                  intent(out) :: fl_out(2)
    logical,                 intent(out) :: ok

    real*8  :: H_l(4, n_degrees), Hs_l(4, n_degrees), Ht_l(4, n_degrees)
    real*8  :: Rr, Zr, Rs, Rt, Zs, Zt, xjac, sz
    real*8  :: ps_s, ps_t, ps_p, ps_R, ps_Z, ps_phi, BR, BZ, BP
    integer :: iv, io, in_t, inode
#ifdef STELLARATOR_MODEL
    real*8  :: chi(0:n_order-1, 0:n_order-1, 0:n_order-1)
    real*8  :: hz_c(n_coord_tor), hzp_c(n_coord_tor), xp, yp, bmag
    logical :: use_gvec_bfield
#endif
    real*8, allocatable :: hz_f(:), hzp_f(:)

    ok = .false.
    fl_out = 0.d0
    if (R <= 1.d-8) return

    ! If re-tracing from an active PFAFEM state, gather Psi0 and grad(Psi0) via PFAFEM gather!
    if (use_current_pfafem .and. pfafem_ready) then
      call gather_psi_grad_pfafem(node_list, element_list, R, Z, phi, ps_R, ps_Z, ps_phi, ok)
      if (.not. ok) return
#ifdef STELLARATOR_MODEL
      ps_R   = ps_R   / F0
      ps_Z   = ps_Z   / F0
      ps_phi = ps_phi / F0
      chi = get_chi(R, Z, modulo(phi, 2.d0 * pi / dble(n_coord_period)), node_list, element_list, ie, s, t, 1)
      BR = chi(1,0,0) + (-ps_phi * chi(0,1,0) + ps_Z * chi(0,0,1)) / R
      BZ = chi(0,1,0) + ( ps_phi * chi(1,0,0) - ps_R * chi(0,0,1)) / R
      BP = chi(0,0,1) / R + (-ps_Z * chi(1,0,0) + ps_R * chi(0,1,0))
#else
      BR =  ps_Z / R
      BZ = -ps_R / R
      BP =  F0 / R
#endif
      if (abs(BP) < 1.d-20) then
        ok = .false.
        return
      end if
      fl_out(1) = R * BR / BP
      fl_out(2) = R * BZ / BP
      call apply_bnd_tangent_collar(node_list, element_list, ie, s, t, phi, fl_out)
      ok = .true.
      return
    end if

    call basisfunctions(s, t, H_l, Hs_l, Ht_l)
    call pfafem_eval_elem_geom(node_list, element_list, ie, s, t, phi, Rr, Zr, Rs, Rt, Zs, Zt)
    xjac = Rs * Zt - Rt * Zs

#ifdef STELLARATOR_MODEL
    ! In model 180 before solving for Psi, node%b_field holds the imported GVEC 3D B-field
    use_gvec_bfield = .false.
#if JOREK_MODEL == 180
    inode = element_list%element(ie)%vertex(1)
    if (maxval(abs(node_list%node(inode)%b_field)) > 1.d-20) use_gvec_bfield = .true.
    if (use_gvec_bfield) then
      call pfafem_eval_coord_hz(phi, hz_c)
      BR = 0.d0; BZ = 0.d0; BP = 0.d0
      do iv = 1, n_vertex_max
        inode = element_list%element(ie)%vertex(iv)
        do io = 1, n_degrees
          sz = element_list%element(ie)%size(iv, io) * H_l(iv, io)
          do in_t = 1, n_coord_tor
            BR = BR + node_list%node(inode)%b_field(in_t, io, 1) * sz * hz_c(in_t)
            BZ = BZ + node_list%node(inode)%b_field(in_t, io, 2) * sz * hz_c(in_t)
            BP = BP + node_list%node(inode)%b_field(in_t, io, 3) * sz * hz_c(in_t)
          end do
        end do
      end do
      bmag = abs(BP)
      if (bmag > 1.d-20) then
        fl_out(1) = R * BR / BP
        fl_out(2) = R * BZ / BP
        call apply_bnd_tangent_collar(node_list, element_list, ie, s, t, phi, fl_out)
        ok = .true.
        return
      end if
    end if
#endif

    chi = get_chi(R, Z, modulo(phi, 2.d0 * pi / dble(n_coord_period)), node_list, element_list, ie, s, t, 1)
    if (abs(xjac) < 1.d-22) then
      ! On collapsed magnetic axis vertex: Psi_R = Psi_Z = Psi_phi = 0 -> pure vacuum B_v = grad(chi)
      BR = chi(1,0,0)
      BZ = chi(0,1,0)
      BP = chi(0,0,1) / R
    else
      call pfafem_eval_coord_hz(phi, hz_c, hzp_c)
      xp = 0.d0; yp = 0.d0
      do iv = 1, n_vertex_max
        inode = element_list%element(ie)%vertex(iv)
        do io = 1, n_degrees
          sz = element_list%element(ie)%size(iv, io) * H_l(iv, io)
          do in_t = 1, n_coord_tor
            xp = xp + node_list%node(inode)%x(in_t, io, 1) * sz * hzp_c(in_t)
            yp = yp + node_list%node(inode)%x(in_t, io, 2) * sz * hzp_c(in_t)
          end do
        end do
      end do
      ps_s = 0.d0; ps_t = 0.d0; ps_p = 0.d0
      if (pfafem_restart_needs_projection .and. pfafem_restart_src_type == 1) then
        allocate(hz_f(pfafem_restart_src_ntor), hzp_f(pfafem_restart_src_ntor))
        call eval_phys_fourier_hz(pfafem_restart_src_ntor, phi, hz_f, hzp_f)
        do iv = 1, n_vertex_max
          inode = element_list%element(ie)%vertex(iv)
          do io = 1, n_degrees
            sz = element_list%element(ie)%size(iv, io)
            do in_t = 1, pfafem_restart_src_ntor
              ps_s = ps_s + pfafem_restart_values(in_t, io, var_Psi, inode) * sz * Hs_l(iv, io) * hz_f(in_t)
              ps_t = ps_t + pfafem_restart_values(in_t, io, var_Psi, inode) * sz * Ht_l(iv, io) * hz_f(in_t)
              ps_p = ps_p + pfafem_restart_values(in_t, io, var_Psi, inode) * sz * H_l(iv, io)  * hzp_f(in_t)
            end do
          end do
        end do
        deallocate(hz_f, hzp_f)
      else
        allocate(hz_f(n_tor), hzp_f(n_tor))
        call eval_phys_fourier_hz(n_tor, phi, hz_f, hzp_f)
        do iv = 1, n_vertex_max
          inode = element_list%element(ie)%vertex(iv)
          do io = 1, n_degrees
            sz = element_list%element(ie)%size(iv, io)
            do in_t = 1, n_tor
              ps_s = ps_s + node_list%node(inode)%values(in_t, io, var_Psi) * sz * Hs_l(iv, io) * hz_f(in_t)
              ps_t = ps_t + node_list%node(inode)%values(in_t, io, var_Psi) * sz * Ht_l(iv, io) * hz_f(in_t)
              ps_p = ps_p + node_list%node(inode)%values(in_t, io, var_Psi) * sz * H_l(iv, io)  * hzp_f(in_t)
            end do
          end do
        end do
        deallocate(hz_f, hzp_f)
      end if
      ps_R   = ( Zt * ps_s - Zs * ps_t) / (xjac * F0)
      ps_Z   = (-Rt * ps_s + Rs * ps_t) / (xjac * F0)
      ps_phi = ps_p / F0 - ps_R * xp - ps_Z * yp
      BR = chi(1,0,0) + (-ps_phi * chi(0,1,0) + ps_Z * chi(0,0,1)) / R
      BZ = chi(0,1,0) + ( ps_phi * chi(1,0,0) - ps_R * chi(0,0,1)) / R
      BP = chi(0,0,1) / R + (-ps_Z * chi(1,0,0) + ps_R * chi(0,1,0))
    end if
#else
    ! Tokamak reduced MHD (model 600): B_R = psi_Z / R, B_Z = -psi_R / R, B_phi = F0 / R
    if (abs(xjac) < 1.d-22) then
      ! At collapsed magnetic axis vertex s=0, B_pol = 0 -> fl = (0, 0)
      fl_out = 0.d0
      ok = .true.
      return
    end if
    ps_s = 0.d0; ps_t = 0.d0
    if (pfafem_restart_needs_projection .and. pfafem_restart_src_type == 1) then
      allocate(hz_f(pfafem_restart_src_ntor))
      call eval_phys_fourier_hz(pfafem_restart_src_ntor, phi, hz_f)
      do iv = 1, n_vertex_max
        inode = element_list%element(ie)%vertex(iv)
        do io = 1, n_degrees
          sz = element_list%element(ie)%size(iv, io)
          do in_t = 1, pfafem_restart_src_ntor
            ps_s = ps_s + pfafem_restart_values(in_t, io, var_Psi, inode) * sz * Hs_l(iv, io) * hz_f(in_t)
            ps_t = ps_t + pfafem_restart_values(in_t, io, var_Psi, inode) * sz * Ht_l(iv, io) * hz_f(in_t)
          end do
        end do
      end do
      deallocate(hz_f)
    else
      ! At initial equilibrium setup (not restart), slot 1 holds the axisymmetric equilibrium psi
      do iv = 1, n_vertex_max
        inode = element_list%element(ie)%vertex(iv)
        do io = 1, n_degrees
          sz = element_list%element(ie)%size(iv, io)
          ps_s = ps_s + node_list%node(inode)%values(1, io, var_Psi) * sz * Hs_l(iv, io)
          ps_t = ps_t + node_list%node(inode)%values(1, io, var_Psi) * sz * Ht_l(iv, io)
        end do
      end do
    end if
    ps_R = ( Zt * ps_s - Zs * ps_t) / xjac
    ps_Z = (-Rt * ps_s + Rs * ps_t) / xjac
    BR =  ps_Z / R
    BZ = -ps_R / R
    BP =  F0 / R
#endif

    if (abs(BP) < 1.d-20) return
    fl_out(1) = R * BR / BP
    fl_out(2) = R * BZ / BP
    call apply_bnd_tangent_collar(node_list, element_list, ie, s, t, phi, fl_out)
    ok = .true.
  end subroutine eval_jorek_fl_at_elem

  !=============================================================================
  !> Smoothly project the field-line tangent (f_R, f_Z) onto the constant-s
  !! coordinate foliation across an outer vacuum collar s_norm in [s0, s1]
  !! when bloating_factor > 1.001 (extended-boundary stellarator domains where
  !! B.n /= 0 on the outer Dirichlet wall s_norm = 1).
  !! Preserves 100% exact field alignment inside the plasma & SOL (s_norm <= s0),
  !! while making s_norm = 1 an invariant torus of the PFAFEM basis map so all
  !! footprints remain interior and Dirichlet BCs hold for all phi in [0, 2*pi).
  !=============================================================================
  pure subroutine apply_bnd_tangent_collar(node_list, element_list, ie, s, t, phi, fl_io)
    use phys_module, only: bloating_factor
    use mod_basisfunctions, only: basisfunctions
    type(type_node_list),    intent(in)    :: node_list
    type(type_element_list), intent(in)    :: element_list
    integer,                 intent(in)    :: ie
    real*8,                  intent(in)    :: s, t, phi
    real*8,                  intent(inout) :: fl_io(2)

#ifdef STELLARATOR_MODEL
    real*8  :: H_l(4, n_degrees), s_norm_pt, s_lcfs, s0_col, s1_col, u_col, w_col
    real*8  :: Rr, Zr, Rs, Rt, Zs, Zt, xjac, xp_m, yp_m, sz, df_R, df_Z, s_dot, phi_c
    real*8  :: hz_c(n_coord_tor), hzp_c(n_coord_tor)
    integer :: iv, io, in_t, inode

    if (.not. (bloating_factor > 1.001d0) .and. n_coord_tor <= 1) return

    call basisfunctions(s, t, H_l)
    s_norm_pt = 0.d0
    do iv = 1, n_vertex_max
      inode = element_list%element(ie)%vertex(iv)
      do io = 1, n_degrees
        s_norm_pt = s_norm_pt + node_list%node(inode)%r_tor_eq(io) * element_list%element(ie)%size(iv, io) * H_l(iv, io)
      end do
    end do

    if (bloating_factor > 1.001d0) then
      s_lcfs = 1.d0 / bloating_factor
      s0_col = max(0.72d0, 1.d0 - 0.45d0 * (1.d0 - s_lcfs))
      s1_col = 0.920d0
    else
      s0_col = 0.900d0
      s1_col = 0.965d0
    end if
    if (s_norm_pt <= s0_col) return

    phi_c = modulo(phi, 2.d0 * pi / dble(n_coord_period))
    call pfafem_eval_elem_geom(node_list, element_list, ie, s, t, phi_c, Rr, Zr, Rs, Rt, Zs, Zt)
    xjac = Rs * Zt - Rt * Zs
    if (abs(xjac) < 1.d-22) return

    if (n_coord_tor == 1) then
      xp_m = 0.d0; yp_m = 0.d0
    else
      call pfafem_eval_coord_hz(phi_c, hz_c, hzp_c)
      xp_m = 0.d0; yp_m = 0.d0
      do iv = 1, n_vertex_max
        inode = element_list%element(ie)%vertex(iv)
        do io = 1, n_degrees
          sz = element_list%element(ie)%size(iv, io) * H_l(iv, io)
          do in_t = 1, n_coord_tor
            xp_m = xp_m + node_list%node(inode)%x(in_t, io, 1) * sz * hzp_c(in_t)
            yp_m = yp_m + node_list%node(inode)%x(in_t, io, 2) * sz * hzp_c(in_t)
          end do
        end do
      end do
    end if

    if (s_norm_pt >= s1_col) then
      w_col = 0.d0
    else
      u_col = (s_norm_pt - s0_col) / (s1_col - s0_col)
      w_col = 1.d0 - (u_col**4) * (35.d0 - 84.d0 * u_col + 70.d0 * u_col * u_col - 20.d0 * (u_col**3))
    end if

    df_R  = fl_io(1) - xp_m
    df_Z  = fl_io(2) - yp_m
    s_dot = (df_R * Zt - df_Z * Rt) / xjac
    fl_io(1) = fl_io(1) - (1.d0 - w_col) * s_dot * Rs
    fl_io(2) = fl_io(2) - (1.d0 - w_col) * s_dot * Zs
#endif
  end subroutine apply_bnd_tangent_collar

  !=============================================================================
  !> C^1-exact exterior continuation of the constant-s collared tangent field
  !! across the outward normal ray chart X_lin(s_m, t) = X(1, t) + (s_m - 1)*X_s(1, t)
  !! for s_m in (1, 12], eliminating any spline derivative jump across dOmega so
  !! zero footprints ever cross the outer wall (openbc_hermite = 0).
  !=============================================================================
  subroutine eval_collar_exterior_ray(node_list, element_list, R, Z, phi, &
                                      use_current_pfafem, fl_out, ok)
    use mod_basisfunctions, only: basisfunctions
    type(type_node_list),    intent(in)  :: node_list
    type(type_element_list), intent(in)  :: element_list
    real*8,                  intent(in)  :: R, Z, phi
    logical,                 intent(in)  :: use_current_pfafem
    real*8,                  intent(out) :: fl_out(2)
    logical,                 intent(out) :: ok

    integer :: idx_b, k, it_n, ie, best_e, iv, io, in_t, inode, top_e(8)
    real*8  :: phi_c, hz_c(n_coord_tor), hzp_c(n_coord_tor), Rc, Zc, dist2, top_d2(8)
    real*8  :: sm_try, t_try, best_sm, best_t, best_score, score
    real*8  :: Rb, Zb, Rs, Rt, Zs, Zt, Rss, Rst, Rtt, Zss, Zst, Ztt
    real*8  :: res_R, res_Z, J11, J12, J21, J22, detJ, dsm, dt_n, step_fac
    real*8  :: t_cl, d_s, Rb1, Zb1, Rs1, Rt1, Zs1, Zt1, Rss1, Rst1, Rtt1, Zss1, Zst1, Ztt1
    real*8  :: Rb0, Zb0, Rs0, Rt0, Zs0, Zt0, j1, j0, fl1(2), fl0(2)
    real*8  :: H1(4, n_degrees), Hs1(4, n_degrees), Ht1(4, n_degrees), H0(4, n_degrees)
    real*8  :: xp1, yp1, xsp1, ysp1, xp0, yp0, sz1, szs1, sz0, tdot1, tdot0, tdot_ext
    logical :: ok1, ok0

    ok = .false.
    fl_out = 0.d0
    if (n_bnd_elems <= 0) return

    phi_c = modulo(phi, 2.d0 * pi / dble(n_coord_period))
    if (n_coord_tor == 1) then
      hz_c(1)  = 1.d0
      hzp_c(1) = 0.d0
    else
      call pfafem_eval_coord_hz(phi_c, hz_c, hzp_c)
    end if

    top_e  = 0
    top_d2 = 1.d30
    do idx_b = 1, n_bnd_elems
      ie = bnd_elem_list(idx_b)
      Rc = 0.d0; Zc = 0.d0
      do iv = 1, n_vertex_max
        inode = element_list%element(ie)%vertex(iv)
        Rc = Rc + dot_product(node_list%node(inode)%x(1:n_coord_tor, 1, 1), hz_c)
        Zc = Zc + dot_product(node_list%node(inode)%x(1:n_coord_tor, 1, 2), hz_c)
      end do
      dist2 = (R - 0.25d0 * Rc)**2 + (Z - 0.25d0 * Zc)**2
      do k = 1, 8
        if (dist2 < top_d2(k)) then
          if (k < 8) then
            top_d2(k+1:8) = top_d2(k:7)
            top_e(k+1:8)  = top_e(k:7)
          end if
          top_d2(k) = dist2
          top_e(k)  = ie
          exit
        end if
      end do
    end do

    best_e     = 0
    best_sm    = 1.d0
    best_t     = 0.5d0
    best_score = 1.d30

    do k = 1, 8
      ie = top_e(k)
      if (ie < 1) cycle
      call pfafem_eval_elem_geom(node_list, element_list, ie, 1.0d0, 0.5d0, phi_c, &
                                 Rb, Zb, Rs, Rt, Zs, Zt)
      detJ = Rs * Zt - Rt * Zs
      if (abs(detJ) < 1.d-22) cycle
      sm_try = 1.0d0 + min(11.d0, max(0.01d0, ((R - Rb) * Zt - (Z - Zb) * Rt) / detJ))
      t_try  = 0.5d0 + min(0.6d0, max(-0.6d0, (Rs * (Z - Zb) - Zs * (R - Rb)) / detJ))

      do it_n = 1, 16
        call pfafem_eval_elem_geom(node_list, element_list, ie, 1.0d0, t_try, phi_c, &
                                   Rb, Zb, Rs, Rt, Zs, Zt, Rss, Rst, Rtt, Zss, Zst, Ztt)
        res_R = Rb + (sm_try - 1.0d0) * Rs - R
        res_Z = Zb + (sm_try - 1.0d0) * Zs - Z
        if (res_R * res_R + res_Z * res_Z <= 1.d-22) exit
        J11  = Rs
        J12  = Rt + (sm_try - 1.0d0) * Rst
        J21  = Zs
        J22  = Zt + (sm_try - 1.0d0) * Zst
        detJ = J11 * J22 - J12 * J21
        if (abs(detJ) < 1.d-22) exit
        dsm = -( J22 * res_R - J12 * res_Z) / detJ
        dt_n = -(-J21 * res_R + J11 * res_Z) / detJ
        step_fac = 1.d0
        if (max(abs(dsm), abs(dt_n)) > 1.5d0) step_fac = 1.5d0 / max(abs(dsm), abs(dt_n))
        sm_try = min(14.0d0, max(0.95d0, sm_try + step_fac * dsm))
        t_try  = min(1.35d0, max(-0.35d0, t_try  + step_fac * dt_n))
      end do

      call pfafem_eval_elem_geom(node_list, element_list, ie, 1.0d0, t_try, phi_c, &
                                 Rb, Zb, Rs, Rt, Zs, Zt)
      res_R = Rb + (sm_try - 1.0d0) * Rs - R
      res_Z = Zb + (sm_try - 1.0d0) * Zs - Z
      if (res_R * res_R + res_Z * res_Z <= 1.d-18 .and. sm_try >= 1.0d0 - 1.d-4 .and. sm_try <= 13.5d0) then
        score = 100.d0 * max(0.d0, -t_try, t_try - 1.d0) + 1.d-3 * max(0.d0, sm_try - 1.0d0)
        if (score < best_score) then
          best_score = score
          best_e     = ie
          best_sm    = max(1.0d0, sm_try)
          best_t     = t_try
        end if
      end if
    end do

    if (best_e < 1 .or. best_score > 15.d0) return

    ie   = best_e
    t_cl = min(1.d0, max(0.d0, best_t))
    d_s  = min(2.5d0, max(0.d0, best_sm - 1.0d0))

    call pfafem_eval_elem_geom(node_list, element_list, ie, 1.0d0,  t_cl, phi_c, &
                               Rb1, Zb1, Rs1, Rt1, Zs1, Zt1, Rss1, Rst1, Rtt1, Zss1, Zst1, Ztt1)
    call pfafem_eval_elem_geom(node_list, element_list, ie, 0.95d0, t_cl, phi_c, &
                               Rb0, Zb0, Rs0, Rt0, Zs0, Zt0)
    j1 = Rs1 * Zt1 - Rt1 * Zs1
    j0 = Rs0 * Zt0 - Rt0 * Zs0
    if (abs(j1) < 1.d-22 .or. abs(j0) < 1.d-22) return

    call eval_jorek_fl_at_elem(node_list, element_list, ie, 1.0d0,  t_cl, Rb1, Zb1, phi, use_current_pfafem, fl1, ok1)
    if (.not. ok1) return
    call eval_jorek_fl_at_elem(node_list, element_list, ie, 0.95d0, t_cl, Rb0, Zb0, phi, use_current_pfafem, fl0, ok0)
    if (.not. ok0) return

    if (n_coord_tor == 1) then
      xp1  = 0.d0; yp1  = 0.d0
      xsp1 = 0.d0; ysp1 = 0.d0
      xp0  = 0.d0; yp0  = 0.d0
    else
      call basisfunctions(1.0d0,  t_cl, H1, Hs1, Ht1)
      call basisfunctions(0.95d0, t_cl, H0)
      xp1  = 0.d0; yp1  = 0.d0
      xsp1 = 0.d0; ysp1 = 0.d0
      xp0  = 0.d0; yp0  = 0.d0
      do iv = 1, n_vertex_max
        inode = element_list%element(ie)%vertex(iv)
        do io = 1, n_degrees
          sz1  = element_list%element(ie)%size(iv, io) * H1(iv, io)
          szs1 = element_list%element(ie)%size(iv, io) * Hs1(iv, io)
          sz0  = element_list%element(ie)%size(iv, io) * H0(iv, io)
          do in_t = 1, n_coord_tor
            xp1  = xp1  + node_list%node(inode)%x(in_t, io, 1) * sz1  * hzp_c(in_t)
            yp1  = yp1  + node_list%node(inode)%x(in_t, io, 2) * sz1  * hzp_c(in_t)
            xsp1 = xsp1 + node_list%node(inode)%x(in_t, io, 1) * szs1 * hzp_c(in_t)
            ysp1 = ysp1 + node_list%node(inode)%x(in_t, io, 2) * szs1 * hzp_c(in_t)
            xp0  = xp0  + node_list%node(inode)%x(in_t, io, 1) * sz0  * hzp_c(in_t)
            yp0  = yp0  + node_list%node(inode)%x(in_t, io, 2) * sz0  * hzp_c(in_t)
          end do
        end do
      end do
    end if

    tdot1    = (Rs1 * (fl1(2) - yp1) - Zs1 * (fl1(1) - xp1)) / j1
    tdot0    = (Rs0 * (fl0(2) - yp0) - Zs0 * (fl0(1) - xp0)) / j0
    tdot_ext = tdot1 + ((tdot1 - tdot0) / 0.05d0) * d_s

    fl_out(1) = (xp1 + d_s * xsp1) + tdot_ext * (Rt1 + d_s * Rst1)
    fl_out(2) = (yp1 + d_s * ysp1) + tdot_ext * (Zt1 + d_s * Zst1)
    ok = .true.
  end subroutine eval_collar_exterior_ray

  !=============================================================================
  !> Gather Psi0 and its cylindrical gradient (ps_R, ps_Z, ps_phi) at (R, Z, phi)
  !! from the CURRENT PFAFEM basis (used when re-tracing dynamically at step > 1).
  !=============================================================================
  subroutine gather_psi_grad_pfafem(node_list, element_list, R, Z, phi, ps_R, ps_Z, ps_phi, ok)
    use phys_module, only: pfafem_n_substeps, pfafem_openbc
    use mod_basisfunctions, only: basisfunctions
    type(type_node_list),    intent(in)  :: node_list
    type(type_element_list), intent(in)  :: element_list
    real*8,                  intent(in)  :: R, Z, phi
    real*8,                  intent(out) :: ps_R, ps_Z, ps_phi
    logical,                 intent(out) :: ok

    integer :: eP, lb, jb, em, status_inv, iv, io, inode
    real*8  :: phin, tau, nb(4), dnb(4), ddnb(4), phi_ref, phi_c
    real*8  :: tr12(12), sm, tm, Rr, Zr, Rs, Rt, Zs, Zt, xjac, sz
    real*8  :: H_l(4, n_degrees), Hs_l(4, n_degrees), Ht_l(4, n_degrees)
    real*8  :: fl(2), A(4), Hjet(6), dfl_dp(2), Nv, d_RFA, d_ZFA, gRm, gZm, uk
    logical :: tr_ok, jet_ok, any_lb

    ps_R = 0.d0; ps_Z = 0.d0; ps_phi = 0.d0
    ok = .false.

    call pfafem_fieldline_jet(R, Z, phi, fl, A, Hjet, dfl_dp, jet_ok)
    if (.not. jet_ok) return

    phin = modulo(phi, pfafem_phi_per)
    eP   = min(n_tor - 1, max(0, int(phin / pfafem_dphi)))
    tau  = min(1.d0, max(0.d0, phin / pfafem_dphi - dble(eP)))
    call pfafem_bspline_eval(tau, nb, dnb, ddnb)

    any_lb = .false.
    do lb = 1, 4
      if (abs(nb(lb)) <= 1.d-14 .and. abs(dnb(lb)) <= 1.d-14) cycle
      jb = pfafem_slot(eP, lb, n_tor)
      phi_ref = dble(eP - 2 + lb) * pfafem_dphi
      call pfafem_trace_with_hessian(R, Z, phin, phi_ref, pfafem_n_substeps, tr12, tr_ok)
      if (.not. tr_ok) return
      phi_c = dble(jb - 1) * pfafem_dphi
      call pfafem_invert_point(node_list, element_list, tr12(1), tr12(2), phi_c, &
                               pfafem_openbc, em, sm, tm, status_inv)
      if (status_inv == 0) return
      call basisfunctions(sm, tm, H_l, Hs_l, Ht_l)
      call pfafem_eval_elem_geom(node_list, element_list, em, sm, tm, phi_c, Rr, Zr, Rs, Rt, Zs, Zt)
      xjac = Rs * Zt - Rt * Zs
      if (abs(xjac) < 1.d-22) then
        sm = min(1.d0, sm + 1.d-4)
        call basisfunctions(sm, tm, H_l, Hs_l, Ht_l)
        call pfafem_eval_elem_geom(node_list, element_list, em, sm, tm, phi_c, Rr, Zr, Rs, Rt, Zs, Zt)
        xjac = Rs * Zt - Rt * Zs
        if (abs(xjac) < 1.d-22) return
      end if

      do iv = 1, n_vertex_max
        inode = element_list%element(em)%vertex(iv)
        do io = 1, n_degrees
          sz = element_list%element(em)%size(iv, io)
          uk = node_list%node(inode)%values(jb, io, var_Psi)
          Nv = H_l(iv, io) * sz
          if (status_inv == 3) then
            gRm = 0.d0; gZm = 0.d0
          else
            d_RFA = ( Zt * Hs_l(iv, io) - Zs * Ht_l(iv, io)) * sz / xjac
            d_ZFA = (-Rt * Hs_l(iv, io) + Rs * Ht_l(iv, io)) * sz / xjac
            gRm = tr12(3) * d_RFA + tr12(5) * d_ZFA
            gZm = tr12(4) * d_RFA + tr12(6) * d_ZFA
          end if
          ps_R   = ps_R   + uk * gRm * nb(lb)
          ps_Z   = ps_Z   + uk * gZm * nb(lb)
          ps_phi = ps_phi + uk * (Nv * dnb(lb) / pfafem_dphi - (fl(1) * gRm + fl(2) * gZm) * nb(lb))
        end do
      end do
      any_lb = .true.
    end do
    ok = any_lb
  end subroutine gather_psi_grad_pfafem

  !=============================================================================
  !> Fit structured C^2 cubic B-spline representation of (f_R, f_Z) = R*(B_R, B_Z)/B_phi
  !! over [R_min, R_max] x [Z_min, Z_max] (x [0, Phi_per) when 3D).
  !=============================================================================
  subroutine pfafem_fit_bfield_spline(my_id, node_list, element_list, use_current_pfafem)
    use phys_module, only: pfafem_spline_nr, pfafem_spline_nz, bloating_factor
#if defined(STELLARATOR_MODEL) && defined(USE_DOMM)
    use mod_chi, only: get_chi_domm
#endif
    integer,                 intent(in) :: my_id
    type(type_node_list),    intent(in) :: node_list
    type(type_element_list), intent(in) :: element_list
    logical,                 intent(in) :: use_current_pfafem

    integer :: nr, nz, nphi, i, j, k, ie, status_inv, it_lap, mpi_ierr
    real*8  :: Rmin, Rmax, Zmin, Zmax, padR, padZ, hR, hZ, dphi_sp
    real*8  :: R, Z, phi, sm, tm, fl2(2), non_axisym_norm, base_norm
    real*8  :: hz_c(n_coord_tor), Rnode, Znode
    logical :: ok_pt, mpi_is_init
    real*8,  allocatable :: raw_fl(:,:,:,:), tmp_r(:), tmp_c(:), tmp_phi(:), tmp_cphi(:)
    logical, allocatable :: known(:,:,:)
#if defined(STELLARATOR_MODEL) && defined(USE_DOMM)
    real*8  :: chi(0:n_order-1, 0:n_order-1, 0:n_order-1)
#endif

    nr = max(16, pfafem_spline_nr)
    nz = max(16, pfafem_spline_nz)
    if (bloating_factor > 1.001d0 .or. n_coord_tor > 1) then
      nr = max(160, nr)
      nz = max(160, nz)
    end if

    ! Determine whether B(R, Z, phi) is axisymmetric or 3D
#ifdef STELLARATOR_MODEL
    pfafem_is_axisym_bfield = .false.
#else
    pfafem_is_axisym_bfield = .true.
    if (pfafem_restart_needs_projection .and. pfafem_restart_src_type == 1) then
      if (pfafem_restart_src_ntor > 1) then
        if (maxval(abs(pfafem_restart_values(2:pfafem_restart_src_ntor, :, var_Psi, :))) > 1.d-13) then
          pfafem_is_axisym_bfield = .false.
        end if
      end if
    else if (use_current_pfafem .and. pfafem_ready) then
      ! Check whether nodal B-spline coefficients of Psi vary across toroidal slots 1..n_tor
      non_axisym_norm = 0.d0
      base_norm = 0.d0
      do i = 1, node_list%n_nodes
        base_norm = max(base_norm, maxval(abs(node_list%node(i)%values(:, 1, var_Psi))))
        do k = 2, n_tor
          non_axisym_norm = max(non_axisym_norm, &
            maxval(abs(node_list%node(i)%values(k, :, var_Psi) - node_list%node(i)%values(1, :, var_Psi))))
        end do
      end do
      if (non_axisym_norm > 1.d-11 * max(1.d0, base_norm)) then
        pfafem_is_axisym_bfield = .false.
      end if
    end if
#endif

    call MPI_Initialized(mpi_is_init, mpi_ierr)
    if (mpi_is_init) then
      call MPI_Bcast(pfafem_is_axisym_bfield, 1, MPI_LOGICAL, 0, MPI_COMM_WORLD, mpi_ierr)
    end if
    if (pfafem_is_axisym_bfield) then
      nphi = 1
    else if (bloating_factor > 1.001d0 .or. n_coord_tor > 1) then
      nphi = max(12, 2 * n_tor, 2 * n_coord_tor * max(1, n_coord_period / max(1, n_period)))
    else
      nphi = max(12, 2 * n_tor, n_coord_tor + 1)
    end if

    ! Compute mesh bounding box across all toroidal planes and pad by 12%
    Rmin =  1.d30; Rmax = -1.d30
    Zmin =  1.d30; Zmax = -1.d30
    do k = 1, max(1, nphi)
      phi = dble(k - 1) * pfafem_phi_per / dble(max(1, nphi))
      if (n_coord_tor == 1) then
        hz_c(1) = 1.d0
      else
        call pfafem_eval_coord_hz(phi, hz_c)
      end if
      do i = 1, node_list%n_nodes
        Rnode = dot_product(node_list%node(i)%x(1:n_coord_tor, 1, 1), hz_c)
        Znode = dot_product(node_list%node(i)%x(1:n_coord_tor, 1, 2), hz_c)
        Rmin = min(Rmin, Rnode); Rmax = max(Rmax, Rnode)
        Zmin = min(Zmin, Znode); Zmax = max(Zmax, Znode)
      end do
    end do
    padR = 0.12d0 * (Rmax - Rmin)
    padZ = 0.12d0 * (Zmax - Zmin)
    Rmin = max(0.05d0 * Rmin, Rmin - padR)
    Rmax = Rmax + padR
    Zmin = Zmin - padZ
    Zmax = Zmax + padZ

    ! Rank 0 samples (f_R, f_Z) BEFORE replacing pfafem_bgrid_* so dynamic retrace
    ! evaluates grad(Psi) on the existing field-aligned basis without clobbering it!
    if (my_id == 0) then
      allocate(raw_fl(2, nr, nz, nphi), known(nr, nz, nphi))
      raw_fl = 0.d0
      known  = .false.
      hR = (Rmax - Rmin) / dble(nr - 1)
      hZ = (Zmax - Zmin) / dble(nz - 1)
      dphi_sp = pfafem_phi_per / dble(nphi)

      !$omp parallel do collapse(2) schedule(dynamic, 4) &
      !$omp private(k, j, i, phi, Z, R, ie, sm, tm, status_inv, fl2, ok_pt) &
#if defined(STELLARATOR_MODEL) && defined(USE_DOMM)
      !$omp private(chi) &
#endif
      !$omp shared(raw_fl, known, nr, nz, nphi, Rmin, Zmin, hR, hZ, dphi_sp, &
      !$omp        node_list, element_list, use_current_pfafem, bloating_factor)
      do k = 1, nphi
        do j = 1, nz
          phi = dble(k - 1) * dphi_sp
          Z   = Zmin + dble(j - 1) * hZ
          do i = 1, nr
            R = Rmin + dble(i - 1) * hR
            call pfafem_invert_point(node_list, element_list, R, Z, phi, .false., ie, sm, tm, status_inv)
            if (status_inv == 1) then
              call eval_jorek_fl_at_elem(node_list, element_list, ie, sm, tm, R, Z, phi, &
                                         (use_current_pfafem .and. pfafem_ready), fl2, ok_pt)
              if (ok_pt) then
                raw_fl(:, i, j, k) = fl2
                known(i, j, k)     = .true.
                cycle
              end if
            else if (bloating_factor > 1.001d0 .or. n_coord_tor > 1) then
              call eval_collar_exterior_ray(node_list, element_list, R, Z, phi, &
                                            (use_current_pfafem .and. pfafem_ready), fl2, ok_pt)
              if (ok_pt) then
                raw_fl(:, i, j, k) = fl2
                known(i, j, k)     = .true.
                cycle
              end if
            end if

#if defined(STELLARATOR_MODEL) && defined(USE_DOMM)
            if (.not. (bloating_factor > 1.001d0)) then
              ! Outside the plasma mesh in non-collared Dommaschk stellarator models, vacuum field grad(chi) is exact!
              chi = get_chi_domm(R, Z, modulo(phi, 2.d0 * pi / dble(n_coord_period)))
              if (abs(chi(0,0,1)) > 1.d-20) then
                raw_fl(1, i, j, k) = R * R * chi(1,0,0) / chi(0,0,1)
                raw_fl(2, i, j, k) = R * R * chi(0,1,0) / chi(0,0,1)
                known(i, j, k)     = .true.
              end if
            end if
#endif
          end do
        end do
      end do
      !$omp end parallel do

      ! Smoothly extend any remaining exterior box-corner grid nodes (tokamak vacuum)
      ! by nearest-known seed + 40 harmonic relaxation sweeps so there is zero Gibbs jump across dOmega.
      call fill_and_smooth_exterior(nr, nz, nphi, raw_fl, known)

      pfafem_bgrid_dims   = [nr, nz, nphi]
      pfafem_bgrid_bounds = [Rmin, Rmax, Zmin, Zmax, pfafem_phi_per]
      if (allocated(pfafem_bgrid_coefs)) deallocate(pfafem_bgrid_coefs)
      allocate(pfafem_bgrid_coefs(2, nr + 2, nz + 2, nphi))
      pfafem_bgrid_coefs = 0.d0

      ! Tensor-product 1D cubic B-spline collocation:
      ! 1) Fit along R (length nr -> nr+2)
      allocate(tmp_r(nr), tmp_c(nr + 2))
      do k = 1, nphi
        do j = 1, nz
          do i = 1, 2
            tmp_r = raw_fl(i, 1:nr, j, k)
            call fit_1d_clamped_bspline(nr, hR, tmp_r, tmp_c)
            pfafem_bgrid_coefs(i, 1:nr + 2, j + 1, k) = tmp_c
          end do
        end do
      end do
      deallocate(tmp_r, tmp_c)

      ! 2) Fit along Z (length nz -> nz+2) for every R coefficient index 1..nr+2
      allocate(tmp_r(nz), tmp_c(nz + 2))
      do k = 1, nphi
        do i = 1, nr + 2
          tmp_r = pfafem_bgrid_coefs(1, i, 2:nz + 1, k)
          call fit_1d_clamped_bspline(nz, hZ, tmp_r, tmp_c)
          pfafem_bgrid_coefs(1, i, 1:nz + 2, k) = tmp_c

          tmp_r = pfafem_bgrid_coefs(2, i, 2:nz + 1, k)
          call fit_1d_clamped_bspline(nz, hZ, tmp_r, tmp_c)
          pfafem_bgrid_coefs(2, i, 1:nz + 2, k) = tmp_c
        end do
      end do
      deallocate(tmp_r, tmp_c)

      ! 3) Fit periodic cubic B-spline along phi when nphi > 1
      if (nphi > 1) then
        allocate(tmp_phi(nphi), tmp_cphi(nphi))
        do j = 1, nz + 2
          do i = 1, nr + 2
            tmp_phi = pfafem_bgrid_coefs(1, i, j, 1:nphi)
            call pfafem_collocation_fit(nphi, tmp_phi, tmp_cphi)
            pfafem_bgrid_coefs(1, i, j, 1:nphi) = tmp_cphi

            tmp_phi = pfafem_bgrid_coefs(2, i, j, 1:nphi)
            call pfafem_collocation_fit(nphi, tmp_phi, tmp_cphi)
            pfafem_bgrid_coefs(2, i, j, 1:nphi) = tmp_cphi
          end do
        end do
        deallocate(tmp_phi, tmp_cphi)
      end if

      deallocate(raw_fl, known)
    else
      pfafem_bgrid_dims   = [nr, nz, nphi]
      pfafem_bgrid_bounds = [Rmin, Rmax, Zmin, Zmax, pfafem_phi_per]
      if (allocated(pfafem_bgrid_coefs)) deallocate(pfafem_bgrid_coefs)
      allocate(pfafem_bgrid_coefs(2, nr + 2, nz + 2, nphi))
      pfafem_bgrid_coefs = 0.d0
    end if

    if (mpi_is_init) then
      call MPI_Bcast(pfafem_bgrid_dims, 3, MPI_INTEGER, 0, MPI_COMM_WORLD, mpi_ierr)
      call MPI_Bcast(pfafem_bgrid_bounds, 5, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, mpi_ierr)
      call MPI_Bcast(pfafem_bgrid_coefs, size(pfafem_bgrid_coefs), MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, mpi_ierr)
    end if
    pfafem_bgrid_valid = .true.
  end subroutine pfafem_fit_bfield_spline

  !=============================================================================
  !> Seed unknown exterior structured-grid nodes from nearest known neighbor
  !! and apply harmonic relaxation so exterior values join C^1/C^2-smoothly.
  !=============================================================================
  subroutine fill_and_smooth_exterior(nr, nz, nphi, raw_fl, known)
    integer, intent(in)    :: nr, nz, nphi
    real*8,  intent(inout) :: raw_fl(2, nr, nz, nphi)
    logical, intent(in)    :: known(nr, nz, nphi)
    logical :: filled(nr, nz), next_filled(nr, nz)
    integer :: k, i, j, im, ip, jm, jp, it, cnt
    real*8  :: s1, s2

    do k = 1, nphi
      filled = known(:, :, k)
      if (all(filled)) cycle
      if (.not. any(filled)) cycle

      ! Flood-fill nearest neighbor seed
      do it = 1, nr + nz
        if (all(filled)) exit
        next_filled = filled
        do j = 1, nz
          jm = max(1, j - 1)
          jp = min(nz, j + 1)
          do i = 1, nr
            if (filled(i, j)) cycle
            im = max(1, i - 1)
            ip = min(nr, i + 1)
            cnt = 0; s1 = 0.d0; s2 = 0.d0
            if (i > 1  .and. filled(im, j)) then; cnt = cnt + 1; s1 = s1 + raw_fl(1, im, j, k); s2 = s2 + raw_fl(2, im, j, k); end if
            if (i < nr .and. filled(ip, j)) then; cnt = cnt + 1; s1 = s1 + raw_fl(1, ip, j, k); s2 = s2 + raw_fl(2, ip, j, k); end if
            if (j > 1  .and. filled(i, jm)) then; cnt = cnt + 1; s1 = s1 + raw_fl(1, i, jm, k); s2 = s2 + raw_fl(2, i, jm, k); end if
            if (j < nz .and. filled(i, jp)) then; cnt = cnt + 1; s1 = s1 + raw_fl(1, i, jp, k); s2 = s2 + raw_fl(2, i, jp, k); end if
            if (cnt > 0) then
              raw_fl(1, i, j, k) = s1 / dble(cnt)
              raw_fl(2, i, j, k) = s2 / dble(cnt)
              next_filled(i, j)  = .true.
            end if
          end do
        end do
        filled = next_filled
      end do

      ! 40 Gauss-Seidel harmonic smoothing sweeps on unknown exterior nodes only
      do it = 1, 40
        do j = 1, nz
          jm = max(1, j - 1)
          jp = min(nz, j + 1)
          do i = 1, nr
            if (known(i, j, k)) cycle
            im = max(1, i - 1)
            ip = min(nr, i + 1)
            cnt = 0; s1 = 0.d0; s2 = 0.d0
            if (i > 1)  then; cnt = cnt + 1; s1 = s1 + raw_fl(1, im, j, k); s2 = s2 + raw_fl(2, im, j, k); end if
            if (i < nr) then; cnt = cnt + 1; s1 = s1 + raw_fl(1, ip, j, k); s2 = s2 + raw_fl(2, ip, j, k); end if
            if (j > 1)  then; cnt = cnt + 1; s1 = s1 + raw_fl(1, i, jm, k); s2 = s2 + raw_fl(2, i, jm, k); end if
            if (j < nz) then; cnt = cnt + 1; s1 = s1 + raw_fl(1, i, jp, k); s2 = s2 + raw_fl(2, i, jp, k); end if
            if (cnt > 0) then
              raw_fl(1, i, j, k) = s1 / dble(cnt)
              raw_fl(2, i, j, k) = s2 / dble(cnt)
            end if
          end do
        end do
      end do
    end do
  end subroutine fill_and_smooth_exterior

  !=============================================================================
  !> Evaluate analytic 2nd-order field-line jet from the C^2 cubic B-spline grid:
  !!  fl(1:2) = (f_R, f_Z)
  !!  A(1:4)  = [df_R/dR, df_R/dZ, df_Z/dR, df_Z/dZ]
  !!  H(1:6)  = [d^2f_R/dR^2, d^2f_R/dRdZ, d^2f_R/dZ^2,
  !!             d^2f_Z/dR^2, d^2f_Z/dRdZ, d^2f_Z/dZ^2]
  !!  dfl_dphi(1:2) = [df_R/dphi, df_Z/dphi]
  !=============================================================================
  pure subroutine pfafem_fieldline_jet(R, Z, phi, fl, A, H, dfl_dphi, ok)
    real*8,  intent(in)  :: R, Z, phi
    real*8,  intent(out) :: fl(2), A(4), H(6), dfl_dphi(2)
    logical, intent(out) :: ok

    integer :: nr, nz, nphi, iR, iZ, eP, a_idx, b_idx, c_idx, kp
    real*8  :: Rmin, Rmax, Zmin, Zmax, phi_per, inv_hR, inv_hZ, dphi_sp
    real*8  :: uR, uZ, tR, tZ, phin, tP
    real*8  :: bR(4), dbR(4), ddbR(4), bZ(4), dbZ(4), ddbZ(4), bP(4), dbP(4), ddbP(4)
    real*8  :: c1, c2, d1P, d2P, wr, wz, wr_r, wz_z, wr_rr, wz_zz

    fl = 0.d0; A = 0.d0; H = 0.d0; dfl_dphi = 0.d0
    ok = .false.
    if (.not. pfafem_bgrid_valid) return

    nr = pfafem_bgrid_dims(1)
    nz = pfafem_bgrid_dims(2)
    nphi = pfafem_bgrid_dims(3)
    Rmin = pfafem_bgrid_bounds(1); Rmax = pfafem_bgrid_bounds(2)
    Zmin = pfafem_bgrid_bounds(3); Zmax = pfafem_bgrid_bounds(4)
    phi_per = pfafem_bgrid_bounds(5)

    if (R < Rmin .or. R > Rmax .or. Z < Zmin .or. Z > Zmax) return

    inv_hR = dble(nr - 1) / (Rmax - Rmin)
    inv_hZ = dble(nz - 1) / (Zmax - Zmin)

    uR = (R - Rmin) * inv_hR
    uZ = (Z - Zmin) * inv_hZ
    iR = min(nr - 1, max(1, int(uR) + 1))
    iZ = min(nz - 1, max(1, int(uZ) + 1))
    tR = uR - dble(iR - 1)
    tZ = uZ - dble(iZ - 1)

    call pfafem_bspline_eval(tR, bR, dbR, ddbR)
    call pfafem_bspline_eval(tZ, bZ, dbZ, ddbZ)
    dbR  = dbR  * inv_hR
    ddbR = ddbR * inv_hR * inv_hR
    dbZ  = dbZ  * inv_hZ
    ddbZ = ddbZ * inv_hZ * inv_hZ

    if (nphi == 1) then
      do b_idx = 1, 4
        wz    = bZ(b_idx)
        wz_z  = dbZ(b_idx)
        wz_zz = ddbZ(b_idx)
        do a_idx = 1, 4
          wr    = bR(a_idx)
          wr_r  = dbR(a_idx)
          wr_rr = ddbR(a_idx)
          c1 = pfafem_bgrid_coefs(1, iR - 1 + a_idx, iZ - 1 + b_idx, 1)
          c2 = pfafem_bgrid_coefs(2, iR - 1 + a_idx, iZ - 1 + b_idx, 1)

          fl(1) = fl(1) + c1 * wr * wz
          fl(2) = fl(2) + c2 * wr * wz

          A(1) = A(1) + c1 * wr_r * wz
          A(2) = A(2) + c1 * wr   * wz_z
          A(3) = A(3) + c2 * wr_r * wz
          A(4) = A(4) + c2 * wr   * wz_z

          H(1) = H(1) + c1 * wr_rr * wz
          H(2) = H(2) + c1 * wr_r  * wz_z
          H(3) = H(3) + c1 * wr    * wz_zz
          H(4) = H(4) + c2 * wr_rr * wz
          H(5) = H(5) + c2 * wr_r  * wz_z
          H(6) = H(6) + c2 * wr    * wz_zz
        end do
      end do
    else
      dphi_sp = phi_per / dble(nphi)
      phin = modulo(phi, phi_per)
      eP = min(nphi - 1, max(0, int(phin / dphi_sp)))
      tP = min(1.d0, max(0.d0, phin / dphi_sp - dble(eP)))
      call pfafem_bspline_eval(tP, bP, dbP, ddbP)
      dbP = dbP / dphi_sp

      do b_idx = 1, 4
        wz    = bZ(b_idx)
        wz_z  = dbZ(b_idx)
        wz_zz = ddbZ(b_idx)
        do a_idx = 1, 4
          wr    = bR(a_idx)
          wr_r  = dbR(a_idx)
          wr_rr = ddbR(a_idx)
          c1 = 0.d0; c2 = 0.d0
          d1P = 0.d0; d2P = 0.d0
          do c_idx = 1, 4
            kp = pfafem_slot(eP, c_idx, nphi)
            c1  = c1  + pfafem_bgrid_coefs(1, iR - 1 + a_idx, iZ - 1 + b_idx, kp) * bP(c_idx)
            c2  = c2  + pfafem_bgrid_coefs(2, iR - 1 + a_idx, iZ - 1 + b_idx, kp) * bP(c_idx)
            d1P = d1P + pfafem_bgrid_coefs(1, iR - 1 + a_idx, iZ - 1 + b_idx, kp) * dbP(c_idx)
            d2P = d2P + pfafem_bgrid_coefs(2, iR - 1 + a_idx, iZ - 1 + b_idx, kp) * dbP(c_idx)
          end do

          fl(1) = fl(1) + c1 * wr * wz
          fl(2) = fl(2) + c2 * wr * wz

          dfl_dphi(1) = dfl_dphi(1) + d1P * wr * wz
          dfl_dphi(2) = dfl_dphi(2) + d2P * wr * wz

          A(1) = A(1) + c1 * wr_r * wz
          A(2) = A(2) + c1 * wr   * wz_z
          A(3) = A(3) + c2 * wr_r * wz
          A(4) = A(4) + c2 * wr   * wz_z

          H(1) = H(1) + c1 * wr_rr * wz
          H(2) = H(2) + c1 * wr_r  * wz_z
          H(3) = H(3) + c1 * wr    * wz_zz
          H(4) = H(4) + c2 * wr_rr * wz
          H(5) = H(5) + c2 * wr_r  * wz_z
          H(6) = H(6) + c2 * wr    * wz_zz
        end do
      end do
    end if
    ok = .true.
  end subroutine pfafem_fieldline_jet

  !=============================================================================
  !> Right-hand side of the 12-component 2nd-order variational field-line ODE:
  !!  s(1:2)  = [R, Z]
  !!  s(3:6)  = [F11, F12, F21, F22]
  !!  s(7:12) = [GR_RR, GR_RZ, GR_ZZ, GZ_RR, GZ_RZ, GZ_ZZ]
  !=============================================================================
  pure subroutine trace_rhs12(y, phi, dy, ok)
    real*8,  intent(in)  :: y(12), phi
    real*8,  intent(out) :: dy(12)
    logical, intent(out) :: ok

    real*8  :: fl(2), A(4), H(6), dfl_dp(2), F(2, 2), v, h_i(3)
    integer :: i, j, q, jj, kk
    integer, parameter :: jk(2, 3) = reshape([1, 1,  1, 2,  2, 2], [2, 3])

    dy = 0.d0
    call pfafem_fieldline_jet(y(1), y(2), phi, fl, A, H, dfl_dp, ok)
    if (.not. ok) return

    dy(1) = fl(1)
    dy(2) = fl(2)

    F(1, 1) = y(3); F(1, 2) = y(4)
    F(2, 1) = y(5); F(2, 2) = y(6)

    do i = 1, 2
      do j = 1, 2
        dy(2 + 2*(i - 1) + j) = A(2*(i - 1) + 1) * F(1, j) + A(2*(i - 1) + 2) * F(2, j)
      end do
    end do

    do i = 1, 2
      h_i = H(3*(i - 1) + 1 : 3*i)
      do q = 1, 3
        jj = jk(1, q)
        kk = jk(2, q)
        v = A(2*(i - 1) + 1) * y(6 + q) + A(2*(i - 1) + 2) * y(9 + q)
        v = v + h_i(1) * F(1, jj) * F(1, kk) &
              + h_i(2) * (F(1, jj) * F(2, kk) + F(2, jj) * F(1, kk)) &
              + h_i(3) * F(2, jj) * F(2, kk)
        dy(6 + 3*(i - 1) + q) = v
      end do
    end do
  end subroutine trace_rhs12

  !=============================================================================
  !> Integrate 12-component field-line map [R, Z, F(2x2), G(2x3)] from phi_start
  !! to phi_end using classical RK4 (matching FieldLineTracer::traceWithHessian).
  !=============================================================================
  pure subroutine pfafem_trace_with_hessian(R0, Z0, phi_start, phi_end, n_substeps, tr12, ok)
    real*8,  intent(in)  :: R0, Z0, phi_start, phi_end
    integer, intent(in)  :: n_substeps
    real*8,  intent(out) :: tr12(12)
    logical, intent(out) :: ok

    integer :: nst, st
    real*8  :: dphi, phi_cur, k1(12), k2(12), k3(12), k4(12), yt(12)

    tr12 = [R0, Z0, 1.d0, 0.d0, 0.d0, 1.d0, 0.d0, 0.d0, 0.d0, 0.d0, 0.d0, 0.d0]
    ok   = .true.
    if (n_substeps < 1 .or. abs(phi_end - phi_start) < 1.d-13) return

    nst  = max(n_substeps, ceiling(abs(phi_end - phi_start) / 0.05d0))
    dphi = (phi_end - phi_start) / dble(nst)
    phi_cur = phi_start

    do st = 1, nst
      call trace_rhs12(tr12, phi_cur, k1, ok)
      if (.not. ok) then; tr12(1) = -1.d0; return; end if

      yt = tr12 + 0.5d0 * dphi * k1
      call trace_rhs12(yt, phi_cur + 0.5d0 * dphi, k2, ok)
      if (.not. ok) then; tr12(1) = -1.d0; return; end if

      yt = tr12 + 0.5d0 * dphi * k2
      call trace_rhs12(yt, phi_cur + 0.5d0 * dphi, k3, ok)
      if (.not. ok) then; tr12(1) = -1.d0; return; end if

      yt = tr12 + dphi * k3
      call trace_rhs12(yt, phi_cur + dphi, k4, ok)
      if (.not. ok) then; tr12(1) = -1.d0; return; end if

      tr12 = tr12 + (dphi / 6.d0) * (k1 + 2.d0 * k2 + 2.d0 * k3 + k4)
      phi_cur = phi_cur + dphi
      if (tr12(1) <= 0.d0) then
        ok = .false.
        tr12(1) = -1.d0
        return
      end if
    end do
  end subroutine pfafem_trace_with_hessian

  !=============================================================================
  !> Fast evaluation of position-only field-line tangent fl(1:2) = (f_R, f_Z)
  !! from the C^2 cubic B-spline grid (identical to pfafem_fieldline_jet fl(1:2)).
  !=============================================================================
  pure subroutine pfafem_fieldline_fl(R, Z, phi, fl, ok)
    real*8,  intent(in)  :: R, Z, phi
    real*8,  intent(out) :: fl(2)
    logical, intent(out) :: ok

    integer :: nr, nz, nphi, iR, iZ, eP, a_idx, b_idx, c_idx, kp
    real*8  :: Rmin, Rmax, Zmin, Zmax, phi_per, inv_hR, inv_hZ, dphi_sp
    real*8  :: uR, uZ, tR, tZ, phin, tP
    real*8  :: bR(4), dbR(4), ddbR(4), bZ(4), dbZ(4), ddbZ(4), bP(4), dbP(4), ddbP(4)
    real*8  :: c1, c2, wr, wz

    fl = 0.d0
    ok = .false.
    if (.not. pfafem_bgrid_valid) return

    nr = pfafem_bgrid_dims(1)
    nz = pfafem_bgrid_dims(2)
    nphi = pfafem_bgrid_dims(3)
    Rmin = pfafem_bgrid_bounds(1); Rmax = pfafem_bgrid_bounds(2)
    Zmin = pfafem_bgrid_bounds(3); Zmax = pfafem_bgrid_bounds(4)
    phi_per = pfafem_bgrid_bounds(5)

    if (R < Rmin .or. R > Rmax .or. Z < Zmin .or. Z > Zmax) return

    inv_hR = dble(nr - 1) / (Rmax - Rmin)
    inv_hZ = dble(nz - 1) / (Zmax - Zmin)

    uR = (R - Rmin) * inv_hR
    uZ = (Z - Zmin) * inv_hZ
    iR = min(nr - 1, max(1, int(uR) + 1))
    iZ = min(nz - 1, max(1, int(uZ) + 1))
    tR = uR - dble(iR - 1)
    tZ = uZ - dble(iZ - 1)

    call pfafem_bspline_eval(tR, bR, dbR, ddbR)
    call pfafem_bspline_eval(tZ, bZ, dbZ, ddbZ)

    if (nphi == 1) then
      do b_idx = 1, 4
        wz = bZ(b_idx)
        do a_idx = 1, 4
          wr = bR(a_idx)
          fl(1) = fl(1) + pfafem_bgrid_coefs(1, iR - 1 + a_idx, iZ - 1 + b_idx, 1) * wr * wz
          fl(2) = fl(2) + pfafem_bgrid_coefs(2, iR - 1 + a_idx, iZ - 1 + b_idx, 1) * wr * wz
        end do
      end do
    else
      dphi_sp = phi_per / dble(nphi)
      phin = modulo(phi, phi_per)
      eP = min(nphi - 1, max(0, int(phin / dphi_sp)))
      tP = min(1.d0, max(0.d0, phin / dphi_sp - dble(eP)))
      call pfafem_bspline_eval(tP, bP, dbP, ddbP)

      do b_idx = 1, 4
        wz = bZ(b_idx)
        do a_idx = 1, 4
          wr = bR(a_idx)
          c1 = 0.d0; c2 = 0.d0
          do c_idx = 1, 4
            kp = pfafem_slot(eP, c_idx, nphi)
            c1 = c1 + pfafem_bgrid_coefs(1, iR - 1 + a_idx, iZ - 1 + b_idx, kp) * bP(c_idx)
            c2 = c2 + pfafem_bgrid_coefs(2, iR - 1 + a_idx, iZ - 1 + b_idx, kp) * bP(c_idx)
          end do
          fl(1) = fl(1) + c1 * wr * wz
          fl(2) = fl(2) + c2 * wr * wz
        end do
      end do
    end if
    ok = .true.
  end subroutine pfafem_fieldline_fl

  !=============================================================================
  !> Position-only RK4 field-line tracer matching pfafem_trace_with_hessian(1:2)
  !! bit-for-bit (identical substep count and step size).
  !=============================================================================
  pure subroutine pfafem_trace_rz(R0, Z0, phi_start, phi_end, n_substeps, R1, Z1, ok)
    real*8,  intent(in)  :: R0, Z0, phi_start, phi_end
    integer, intent(in)  :: n_substeps
    real*8,  intent(out) :: R1, Z1
    logical, intent(out) :: ok

    integer :: nst, st
    real*8  :: dphi, phi_cur, y(2), k1(2), k2(2), k3(2), k4(2), yt(2)

    R1 = R0
    Z1 = Z0
    ok = .true.
    if (n_substeps < 1 .or. abs(phi_end - phi_start) < 1.d-13) return

    nst     = max(n_substeps, ceiling(abs(phi_end - phi_start) / 0.05d0))
    dphi    = (phi_end - phi_start) / dble(nst)
    phi_cur = phi_start
    y(1)    = R0
    y(2)    = Z0

    do st = 1, nst
      call pfafem_fieldline_fl(y(1), y(2), phi_cur, k1, ok)
      if (.not. ok) then; R1 = -1.d0; return; end if

      yt = y + 0.5d0 * dphi * k1
      call pfafem_fieldline_fl(yt(1), yt(2), phi_cur + 0.5d0 * dphi, k2, ok)
      if (.not. ok) then; R1 = -1.d0; return; end if

      yt = y + 0.5d0 * dphi * k2
      call pfafem_fieldline_fl(yt(1), yt(2), phi_cur + 0.5d0 * dphi, k3, ok)
      if (.not. ok) then; R1 = -1.d0; return; end if

      yt = y + dphi * k3
      call pfafem_fieldline_fl(yt(1), yt(2), phi_cur + dphi, k4, ok)
      if (.not. ok) then; R1 = -1.d0; return; end if

      y = y + (dphi / 6.d0) * (k1 + 2.d0 * k2 + 2.d0 * k3 + k4)
      phi_cur = phi_cur + dphi
      if (y(1) <= 0.d0) then
        ok = .false.
        R1 = -1.d0
        return
      end if
    end do
    R1 = y(1)
    Z1 = y(2)
  end subroutine pfafem_trace_rz

  !=============================================================================
  !> Classify traced point (R, Z) at knot plane phi_c into signed element ID
  !! (+em interior/Hermite, -em clamped, 0 dropped), checking candidate element
  !! hints first when strictly inside (1e-7, 1 - 1e-7)^2 to avoid RTree lookups.
  !=============================================================================
  subroutine pfafem_classify_traced_pt(node_list, element_list, R, Z, phi_c, &
                                       allow_openbc, hint_a, hint_b, sig_out)
    type(type_node_list),    intent(in)  :: node_list
    type(type_element_list), intent(in)  :: element_list
    real*8,                  intent(in)  :: R, Z, phi_c
    logical,                 intent(in)  :: allow_openbc
    integer,                 intent(in)  :: hint_a, hint_b
    integer,                 intent(out) :: sig_out

    integer :: em, status_inv
    real*8  :: phi_cm, s_try, t_try

    sig_out = 0
    if (.not. (R > 0.d0)) return
    phi_cm = modulo(phi_c, 2.d0 * pi / dble(n_coord_period))

    if (hint_a >= 1 .and. hint_a <= element_list%n_elements) then
      if (newton_invert_elem(node_list, element_list, hint_a, R, Z, phi_cm, &
                             -1.d-7, 1.d0 + 1.d-7, 1.d-12, s_try, t_try)) then
        if (s_try > 1.d-7 .and. s_try < 1.d0 - 1.d-7 .and. &
            t_try > 1.d-7 .and. t_try < 1.d0 - 1.d-7) then
          sig_out = hint_a
          return
        end if
      end if
    end if

    if (hint_b >= 1 .and. hint_b <= element_list%n_elements .and. hint_b /= hint_a) then
      if (newton_invert_elem(node_list, element_list, hint_b, R, Z, phi_cm, &
                             -1.d-7, 1.d0 + 1.d-7, 1.d-12, s_try, t_try)) then
        if (s_try > 1.d-7 .and. s_try < 1.d0 - 1.d-7 .and. &
            t_try > 1.d-7 .and. t_try < 1.d0 - 1.d-7) then
          sig_out = hint_b
          return
        end if
      end if
    end if

    call pfafem_invert_point(node_list, element_list, R, Z, phi_c, allow_openbc, &
                             em, s_try, t_try, status_inv)
    if (status_inv == 0) then
      sig_out = 0
    else if (status_inv == 3) then
      sig_out = -em
    else
      sig_out = em
    end if
  end subroutine pfafem_classify_traced_pt

  !=============================================================================
  !> Build composite seam-split Gauss-Legendre quadrature rule (t_out, w_out)
  !! of fixed total length pfafem_ng_pol on [0, 1]_t for poloidal line
  !! (ielm, eP, iq, ms), locating all element-seam crossings of the 4 active
  !! traced field lines lb = 1..4 to 1.5e-8 accuracy so every sub-interval
  !! integrates C^infty polynomial segments of the symmetric weak form!
  !=============================================================================
  subroutine pfafem_build_pol_seam_rule(node_list, element_list, ielm, eP, iq, ms, &
                                        phi_q, t_out, w_out, n_seams_out)
    use phys_module, only: pfafem_n_substeps, pfafem_openbc
    type(type_node_list),    intent(in)  :: node_list
    type(type_element_list), intent(in)  :: element_list
    integer,                 intent(in)  :: ielm, eP, iq, ms
    real*8,                  intent(in)  :: phi_q
    real*8,                  intent(out) :: t_out(pfafem_ng_pol), w_out(pfafem_ng_pol)
    integer,                 intent(out) :: n_seams_out

    integer, parameter :: N_SAMP = 17
    integer :: k, lb, jb, it, j, k_max, n_raw_s, n_seams, n_sub, n_base, rem, idx, np
    integer :: hint_prev, ha, hb, sa, sb, sm_sig, sig_k(N_SAMP), i_min, m_best, m
    integer :: n_pt(pfafem_ng_pol)
    real*8  :: s_q, ts(N_SAMP), Rsamp(N_SAMP), Zsamp(N_SAMP), Rs, Rt, Zs, Zt
    real*8  :: phi_ref, phi_c, Rtr, Ztr, ta, tb, tm, Rm, Zm, key_s
    real*8  :: raw_s(64), seams(64), knots(0:pfafem_ng_pol)
    real*8  :: min_gap, max_w, gap, t0, dt_sub
    logical :: tr_ok

    if (pfafem_ng_pol == n_gauss) then
      t_out(1:n_gauss) = xgauss(1:n_gauss)
      w_out(1:n_gauss) = wgauss(1:n_gauss)
      n_seams_out = 0
      return
    end if

    s_q = xgauss(ms)
    do k = 1, N_SAMP
      ts(k) = dble(k - 1) / dble(N_SAMP - 1)
      call pfafem_eval_elem_geom(node_list, element_list, ielm, s_q, ts(k), phi_q, &
                                 Rsamp(k), Zsamp(k), Rs, Rt, Zs, Zt)
    end do

    n_raw_s = 0
    do lb = 1, 4
      phi_ref = dble(eP - 2 + lb) * pfafem_dphi
      jb      = pfafem_slot(eP, lb, n_tor)
      phi_c   = dble(jb - 1) * pfafem_dphi

      hint_prev = 0
      do k = 1, N_SAMP
        call pfafem_trace_rz(Rsamp(k), Zsamp(k), phi_q, phi_ref, pfafem_n_substeps, Rtr, Ztr, tr_ok)
        if (.not. tr_ok) then
          sig_k(k) = 0
        else
          call pfafem_classify_traced_pt(node_list, element_list, Rtr, Ztr, phi_c, &
                                         pfafem_openbc, hint_prev, 0, sig_k(k))
          if (sig_k(k) > 0) hint_prev = sig_k(k)
        end if
      end do

      do k = 1, N_SAMP - 1
        if (sig_k(k) /= sig_k(k + 1)) then
          ta = ts(k)
          tb = ts(k + 1)
          sa = sig_k(k)
          sb = sig_k(k + 1)
          ha = max(0, sa)
          hb = max(0, sb)
          do it = 1, 22
            tm = 0.5d0 * (ta + tb)
            call pfafem_eval_elem_geom(node_list, element_list, ielm, s_q, tm, phi_q, &
                                       Rm, Zm, Rs, Rt, Zs, Zt)
            call pfafem_trace_rz(Rm, Zm, phi_q, phi_ref, pfafem_n_substeps, Rtr, Ztr, tr_ok)
            if (.not. tr_ok) then
              sm_sig = 0
            else
              call pfafem_classify_traced_pt(node_list, element_list, Rtr, Ztr, phi_c, &
                                             pfafem_openbc, ha, hb, sm_sig)
            end if
            if (sm_sig == sa) then
              ta = tm
            else
              tb = tm
              sb = sm_sig
              hb = max(0, sb)
            end if
          end do
          if (n_raw_s < 64) then
            n_raw_s = n_raw_s + 1
            raw_s(n_raw_s) = 0.5d0 * (ta + tb)
          end if
        end if
      end do
    end do

    ! Sort raw interior seams in ascending order
    do k = 2, n_raw_s
      key_s = raw_s(k)
      j = k - 1
      do while (j >= 1)
        if (raw_s(j) <= key_s) exit
        raw_s(j + 1) = raw_s(j)
        j = j - 1
      end do
      raw_s(j + 1) = key_s
    end do

    ! Deduplicate interior seams in (1e-6, 1 - 1e-6) within 1e-5 tolerance
    n_seams = 0
    do k = 1, n_raw_s
      if (raw_s(k) <= 1.d-6 .or. raw_s(k) >= 1.d0 - 1.d-6) cycle
      if (n_seams == 0) then
        n_seams = 1
        seams(1) = raw_s(k)
      else if (raw_s(k) - seams(n_seams) < 1.d-5) then
        seams(n_seams) = 0.5d0 * (seams(n_seams) + raw_s(k))
      else
        n_seams = n_seams + 1
        seams(n_seams) = raw_s(k)
      end if
    end do
    n_seams_out = n_seams

    ! Allow up to pfafem_ng_pol / 2 - 1 = 11 interior seams (12 sub-intervals)
    ! so even max_seams = 9 in 3D W7-X keeps 100% of seams unmerged!
    k_max = max(0, pfafem_ng_pol / 2 - 1)
    do while (n_seams > k_max)
      i_min = 1
      min_gap = seams(2) - seams(1)
      do k = 2, n_seams - 1
        gap = seams(k + 1) - seams(k)
        if (gap < min_gap) then
          min_gap = gap
          i_min   = k
        end if
      end do
      seams(i_min) = 0.5d0 * (seams(i_min) + seams(i_min + 1))
      do k = i_min + 1, n_seams - 1
        seams(k) = seams(k + 1)
      end do
      n_seams = n_seams - 1
    end do

    n_sub = n_seams + 1
    knots(0) = 0.d0
    do k = 1, n_seams
      knots(k) = seams(k)
    end do
    knots(n_sub) = 1.d0

    ! Refine widest sub-interval when n_sub < 3 so h-refinement stays balanced
    do while (n_sub < 3 .and. 3 * (n_sub + 1) <= pfafem_ng_pol)
      m_best = 1
      max_w  = knots(1) - knots(0)
      do m = 2, n_sub
        if (knots(m) - knots(m - 1) > max_w) then
          max_w  = knots(m) - knots(m - 1)
          m_best = m
        end if
      end do
      do m = n_sub, m_best, -1
        knots(m + 1) = knots(m)
      end do
      knots(m_best) = 0.5d0 * (knots(m_best - 1) + knots(m_best + 1))
      n_sub = n_sub + 1
    end do

    n_base = merge(3, 2, 3 * n_sub <= pfafem_ng_pol)
    n_pt(1:n_sub) = n_base
    rem = pfafem_ng_pol - n_base * n_sub
    do it = 1, rem
      m_best = 1
      max_w  = (knots(1) - knots(0)) / dble(n_pt(1))
      do m = 2, n_sub
        gap = (knots(m) - knots(m - 1)) / dble(n_pt(m))
        if (gap > max_w) then
          max_w  = gap
          m_best = m
        end if
      end do
      n_pt(m_best) = n_pt(m_best) + 1
    end do

    idx = 0
    do m = 1, n_sub
      t0     = knots(m - 1)
      dt_sub = knots(m) - knots(m - 1)
      np     = n_pt(m)
      do j = 1, np
        idx = idx + 1
        t_out(idx) = t0 + dt_sub * pfafem_gl_x(j, np)
        w_out(idx) = dt_sub * pfafem_gl_w(j, np)
      end do
    end do
  end subroutine pfafem_build_pol_seam_rule

  !=============================================================================
  !> Build the 4x4 magnetic-axis basis transformation matrix Pmat(io_old, jo_new)
  !! for vertex kv (1..4) of element ielm, matching grids/mod_axis_treatment.f90:
  !!   old_dofs = Pmat * new_dofs,   H_global(jo_new) = sum_io H_local(io)*Pmat(io, jo_new).
  !=============================================================================
  pure subroutine pfafem_axis_pmat(node_list, element_list, ielm, kv, pmat)
    use phys_module, only: treat_axis, n_tht, n_flux, bloating_factor
    type(type_node_list),    intent(in)  :: node_list
    type(type_element_list), intent(in)  :: element_list
    integer,                 intent(in)  :: ielm, kv
    real*8,                  intent(out) :: pmat(4, 4)

    integer :: jo, inode, ax_dof
#ifdef STELLARATOR_MODEL
    real*8  :: theta
#endif

    pmat = 0.d0
    do jo = 1, 4
      pmat(jo, jo) = 1.d0
    end do
    if (.not. treat_axis) return
    if (kv /= 1 .and. kv /= 4) return

    inode  = element_list%element(ielm)%vertex(kv)
    ax_dof = node_list%node(inode)%axis_dof
    if (ax_dof /= 2 .and. ax_dof /= 3) return

#ifdef STELLARATOR_MODEL
    if (kv <= 2) then
      theta = 2.d0 * pi * dble(mod(ielm - 1, n_tht)) / dble(n_tht)
    else
      theta = 2.d0 * pi * dble(mod(ielm, n_tht)) / dble(n_tht)
    end if
#endif

    if (ax_dof == 2) then
#ifdef STELLARATOR_MODEL
      pmat(3, 2) = cos(theta) / (3.d0 * dble(n_flux - 1) / bloating_factor)
      pmat(3, 3) = sin(theta) / (3.d0 * dble(n_flux - 1) / bloating_factor)
      pmat(4, 2) = -2.d0 * pi * sin(theta) / (9.d0 * dble(n_tht * (n_flux - 1)) / bloating_factor)
      pmat(4, 3) =  2.d0 * pi * cos(theta) / (9.d0 * dble(n_tht * (n_flux - 1)) / bloating_factor)
#else
      pmat(3, 2) = node_list%node(inode)%x(1, 3, 1)
      pmat(3, 3) = node_list%node(inode)%x(1, 3, 2)
      pmat(4, 2) = node_list%node(inode)%x(1, 4, 1)
      pmat(4, 3) = node_list%node(inode)%x(1, 4, 2)
#endif
      pmat(2, 2) = 0.d0
      pmat(4, 4) = 0.d0
    else if (ax_dof == 3) then
#ifdef STELLARATOR_MODEL
      pmat(2, 2) = cos(theta) / (3.d0 * dble(n_flux - 1) / bloating_factor)
      pmat(2, 3) = sin(theta) / (3.d0 * dble(n_flux - 1) / bloating_factor)
      pmat(4, 2) = -2.d0 * pi * sin(theta) / (9.d0 * dble(n_tht * (n_flux - 1)) / bloating_factor)
      pmat(4, 3) =  2.d0 * pi * cos(theta) / (9.d0 * dble(n_tht * (n_flux - 1)) / bloating_factor)
#else
      pmat(2, 2) = node_list%node(inode)%x(1, 2, 1)
      pmat(2, 3) = node_list%node(inode)%x(1, 2, 2)
      pmat(4, 2) = node_list%node(inode)%x(1, 4, 1)
      pmat(4, 3) = node_list%node(inode)%x(1, 4, 2)
#endif
      pmat(3, 3) = 0.d0
      pmat(4, 4) = 0.d0
    end if
  end subroutine pfafem_axis_pmat

  !=============================================================================
  !> Lightweight initialization for standalone post-processing tools (e.g. jorek2vtk,
  !! jorek2vtk_3d) that evaluate PFAFEM fields via pfafem_eval_point_jet without
  !! allocating full 3D element quadrature caches or requiring MPI_Init.
  !=============================================================================
  subroutine pfafem_init_eval_only(my_id, node_list, element_list)
    integer,                 intent(in) :: my_id
    type(type_node_list),    intent(in) :: node_list
    type(type_element_list), intent(in) :: element_list

    integer :: nel, ielm, iv, inode, io, jb, r
    real*8  :: phi_c, R_q, Z_q, Rs, Rt, Zs, Zt, hz_knot(n_coord_tor)

    nel = element_list%n_elements
    pfafem_phi_per = 2.d0 * pi / dble(n_period)
    pfafem_dphi    = pfafem_phi_per / dble(n_tor)

    if (allocated(elem_is_bnd)) deallocate(elem_is_bnd)
    if (allocated(bnd_elem_list)) deallocate(bnd_elem_list)
    if (allocated(bnd_elem_knot_rz)) deallocate(bnd_elem_knot_rz)
    if (allocated(pfafem_knot_x)) deallocate(pfafem_knot_x)

    allocate(elem_is_bnd(nel))
    elem_is_bnd = .false.
    do ielm = 1, nel
      do iv = 1, n_vertex_max
        inode = element_list%element(ielm)%vertex(iv)
        if (node_list%node(inode)%boundary /= 0) elem_is_bnd(ielm) = .true.
      end do
    end do
    n_bnd_elems = count(elem_is_bnd)
    allocate(bnd_elem_list(max(1, n_bnd_elems)))
    allocate(bnd_elem_knot_rz(2, max(1, n_bnd_elems), n_tor))
    bnd_elem_list    = 0
    bnd_elem_knot_rz = 0.d0
    r = 0
    do ielm = 1, nel
      if (.not. elem_is_bnd(ielm)) cycle
      r = r + 1
      bnd_elem_list(r) = ielm
      do jb = 1, n_tor
        phi_c = modulo(dble(jb - 1) * pfafem_dphi, 2.d0 * pi / dble(n_coord_period))
        call pfafem_eval_elem_geom(node_list, element_list, ielm, 0.5d0, 0.5d0, phi_c, &
                                   R_q, Z_q, Rs, Rt, Zs, Zt)
        bnd_elem_knot_rz(1, r, jb) = R_q
        bnd_elem_knot_rz(2, r, jb) = Z_q
      end do
    end do

    allocate(pfafem_knot_x(2, n_degrees, node_list%n_nodes, n_tor))
    do jb = 1, n_tor
      phi_c = modulo(dble(jb - 1) * pfafem_dphi, 2.d0 * pi / dble(n_coord_period))
      if (n_coord_tor == 1) then
        hz_knot(1) = 1.d0
      else
        call pfafem_eval_coord_hz(phi_c, hz_knot)
      end if
      do inode = 1, node_list%n_nodes
        do io = 1, n_degrees
          pfafem_knot_x(1, io, inode, jb) = dot_product(node_list%node(inode)%x(1:n_coord_tor, io, 1), hz_knot)
          pfafem_knot_x(2, io, inode, jb) = dot_product(node_list%node(inode)%x(1:n_coord_tor, io, 2), hz_knot)
        end do
      end do
    end do

    if (.not. pfafem_bgrid_valid) then
      call pfafem_fit_bfield_spline(my_id, node_list, element_list, .true.)
    else
      pfafem_is_axisym_bfield = (pfafem_bgrid_dims(3) == 1)
    end if
    pfafem_ready = .true.
  end subroutine pfafem_init_eval_only

  !=============================================================================
  !> Build or rebuild the global PFAFEM Shared-Point trace cache across all elements.
  !=============================================================================
  subroutine pfafem_build_cache(my_id, node_list, element_list, build_bgrid, use_current_pfafem)
    use phys_module, only: pfafem_n_gauss_tor, pfafem_n_gauss_pol, pfafem_n_substeps, pfafem_openbc
#ifdef STELLARATOR_MODEL
    use mod_chi, only: get_chi
#endif
    integer,                 intent(in) :: my_id
    type(type_node_list),    intent(in) :: node_list
    type(type_element_list), intent(in) :: element_list
    logical,                 intent(in) :: build_bgrid, use_current_pfafem

    integer :: n_cpu, mpi_ierr, nel, ielm, iel_loc, iv, io, inode, ep_dim, ng_t
    integer :: el_lo, el_hi, n_loc, iep, eP, iq, ms, mt, lb, jb, em, status_inv, n_seams_line
    real*8  :: s_q, t_q, phi_q, phi_ref, phi_c, R_q, Z_q, Rs, Rt, Zs, Zt, sm, tm, tr12(12)
    real*8  :: t_line(pfafem_ng_pol_max), w_line(pfafem_ng_pol_max), dummy_r(1)
    real*8  :: hz_knot(n_coord_tor)
    logical :: tr_ok
    integer :: n_int_cnt, n_ext_cnt, n_clmp_cnt, n_drop_cnt, g_counts(4)
    integer :: n_seam_max, n_seam_sum, n_lines_loc, g_seam_max, g_seam_sum(2)
    integer :: pts_per_el, i_off, r_off, r, r_lo, r_hi, r_n
    integer, allocatable :: recv_cnt_i(:), disp_i(:), send_i(:), all_i(:)
    real*8,  allocatable :: send_r(:)
#ifndef USE_PETSC
    integer, allocatable :: recv_cnt_r(:), disp_r(:)
    real*8,  allocatable :: all_r(:)
#endif

    call MPI_Comm_size(MPI_COMM_WORLD, n_cpu, mpi_ierr)
    nel = element_list%n_elements

    pfafem_phi_per = 2.d0 * pi / dble(n_period)
    pfafem_dphi    = pfafem_phi_per / dble(n_tor)

    if (pfafem_n_gauss_pol <= n_gauss) then
      pfafem_ng_pol = n_gauss
    else
      pfafem_ng_pol = min(pfafem_ng_pol_max, pfafem_n_gauss_pol)
    end if
    call init_tor_gauss(pfafem_n_gauss_tor)
    ng_t = pfafem_ng_tor

    if (allocated(pfafem_Q)) deallocate(pfafem_Q, pfafem_QT)
    allocate(pfafem_Q(n_tor, n_tor), pfafem_QT(n_tor, n_tor))
    call pfafem_torfft_matrices(n_tor, pfafem_Q, pfafem_QT)

    ! Mark boundary elements (any vertex with node%boundary /= 0) and cache centroids at knot planes
    if (allocated(elem_is_bnd)) deallocate(elem_is_bnd)
    if (allocated(bnd_elem_list)) deallocate(bnd_elem_list)
    if (allocated(bnd_elem_knot_rz)) deallocate(bnd_elem_knot_rz)
    if (allocated(pfafem_knot_x)) deallocate(pfafem_knot_x)
    n_bnd_elems = 0
    allocate(elem_is_bnd(nel))
    elem_is_bnd = .false.
    do ielm = 1, nel
      do iv = 1, n_vertex_max
        inode = element_list%element(ielm)%vertex(iv)
        if (node_list%node(inode)%boundary /= 0) elem_is_bnd(ielm) = .true.
      end do
    end do
    n_bnd_elems = count(elem_is_bnd)
    if (allocated(bnd_elem_list)) deallocate(bnd_elem_list)
    if (allocated(bnd_elem_knot_rz)) deallocate(bnd_elem_knot_rz)
    allocate(bnd_elem_list(max(1, n_bnd_elems)))
    allocate(bnd_elem_knot_rz(2, max(1, n_bnd_elems), n_tor))
    bnd_elem_list    = 0
    bnd_elem_knot_rz = 0.d0
    r = 0
    do ielm = 1, nel
      if (.not. elem_is_bnd(ielm)) cycle
      r = r + 1
      bnd_elem_list(r) = ielm
      do jb = 1, n_tor
        phi_c = modulo(dble(jb - 1) * pfafem_dphi, 2.d0 * pi / dble(n_coord_period))
        call pfafem_eval_elem_geom(node_list, element_list, ielm, 0.5d0, 0.5d0, phi_c, &
                                   R_q, Z_q, Rs, Rt, Zs, Zt)
        bnd_elem_knot_rz(1, r, jb) = R_q
        bnd_elem_knot_rz(2, r, jb) = Z_q
      end do
    end do

    ! Pre-contract nodal geometry onto the n_tor toroidal knot planes for O(1) lookup in pfafem_eval_qp_basis
    if (allocated(pfafem_knot_x)) deallocate(pfafem_knot_x)
    allocate(pfafem_knot_x(2, n_degrees, node_list%n_nodes, n_tor))
    do jb = 1, n_tor
      phi_c = modulo(dble(jb - 1) * pfafem_dphi, 2.d0 * pi / dble(n_coord_period))
      if (n_coord_tor == 1) then
        hz_knot(1) = 1.d0
      else
        call pfafem_eval_coord_hz(phi_c, hz_knot)
      end if
      do inode = 1, node_list%n_nodes
        do io = 1, n_degrees
          pfafem_knot_x(1, io, inode, jb) = dot_product(node_list%node(inode)%x(1:n_coord_tor, io, 1), hz_knot)
          pfafem_knot_x(2, io, inode, jb) = dot_product(node_list%node(inode)%x(1:n_coord_tor, io, 2), hz_knot)
        end do
      end do
    end do

    ! Fit structured C^2 cubic B-spline B-field if requested (or if not loaded from restart)
    if (build_bgrid .or. .not. pfafem_bgrid_valid) then
      call pfafem_fit_bfield_spline(my_id, node_list, element_list, use_current_pfafem)
    else
      pfafem_is_axisym_bfield = (pfafem_bgrid_dims(3) == 1)
    end if

    pfafem_cache_axisym = (pfafem_is_axisym_bfield .and. n_coord_tor == 1)
    pfafem_ep_dim = merge(1, n_tor, pfafem_cache_axisym)
    ep_dim = pfafem_ep_dim

    ! Disjoint element ownership for parallel tracing AND volume assembly
    pfafem_el_lo = (my_id * nel) / n_cpu + 1
    pfafem_el_hi = ((my_id + 1) * nel) / n_cpu
    el_lo = pfafem_el_lo
    el_hi = pfafem_el_hi
    n_loc = max(0, el_hi - el_lo + 1)

    if (allocated(pfafem_t_pol)) deallocate(pfafem_t_pol, pfafem_w_pol)
    allocate(pfafem_t_pol(n_gauss, pfafem_ng_pol, ng_t, ep_dim, max(1, n_loc)))
    allocate(pfafem_w_pol(n_gauss, pfafem_ng_pol, ng_t, ep_dim, max(1, n_loc)))
    pfafem_t_pol = 0.d0
    pfafem_w_pol = 0.d0

    pts_per_el = 4 * n_gauss * pfafem_ng_pol * ng_t * ep_dim
    allocate(send_i(max(1, n_loc * pts_per_el)))
    allocate(send_r(max(1, n_loc * pts_per_el * 12)))
    send_i = 0
    send_r = 0.d0
    dummy_r(1) = 0.d0

    n_int_cnt   = 0
    n_ext_cnt   = 0
    n_clmp_cnt  = 0
    n_drop_cnt  = 0
    n_seam_max  = 0
    n_seam_sum  = 0
    n_lines_loc = n_loc * ep_dim * ng_t * n_gauss

    !$omp parallel do schedule(dynamic, 4) &
    !$omp private(ielm, iel_loc, iep, eP, iq, mt, ms, s_q, t_q, phi_q, R_q, Z_q, Rs, Rt, Zs, Zt, &
    !$omp         t_line, w_line, n_seams_line, lb, jb, phi_ref, phi_c, tr12, tr_ok, &
    !$omp         em, sm, tm, status_inv, i_off, r_off) &
    !$omp reduction(+:n_int_cnt, n_ext_cnt, n_clmp_cnt, n_drop_cnt, n_seam_sum) &
    !$omp reduction(max:n_seam_max)
    do ielm = el_lo, el_hi
      iel_loc = ielm - el_lo + 1
      do iep = 1, ep_dim
        eP = iep - 1
        do iq = 1, ng_t
          phi_q = (dble(eP) + pfafem_tau_q(iq)) * pfafem_dphi
          do ms = 1, n_gauss
            s_q = xgauss(ms)
            call pfafem_build_pol_seam_rule(node_list, element_list, ielm, eP, iq, ms, &
                                            phi_q, t_line(1:pfafem_ng_pol), w_line(1:pfafem_ng_pol), n_seams_line)
            pfafem_t_pol(ms, 1:pfafem_ng_pol, iq, iep, iel_loc) = t_line(1:pfafem_ng_pol)
            pfafem_w_pol(ms, 1:pfafem_ng_pol, iq, iep, iel_loc) = w_line(1:pfafem_ng_pol)
            n_seam_sum = n_seam_sum + n_seams_line
            n_seam_max = max(n_seam_max, n_seams_line)

            do mt = 1, pfafem_ng_pol
              t_q = t_line(mt)
              call pfafem_eval_elem_geom(node_list, element_list, ielm, s_q, t_q, phi_q, &
                                         R_q, Z_q, Rs, Rt, Zs, Zt)
              do lb = 1, 4
                i_off = ((((ielm - el_lo) * ep_dim + (iep - 1)) * ng_t + (iq - 1)) * pfafem_ng_pol + (mt - 1)) &
                        * n_gauss * 4 + (ms - 1) * 4 + lb
                r_off = (i_off - 1) * 12

                phi_ref = dble(eP - 2 + lb) * pfafem_dphi
                jb = pfafem_slot(eP, lb, n_tor)
                phi_c = dble(jb - 1) * pfafem_dphi

                call pfafem_trace_with_hessian(R_q, Z_q, phi_q, phi_ref, pfafem_n_substeps, tr12, tr_ok)
                if (.not. tr_ok) then
                  send_i(i_off) = 0
                  n_drop_cnt = n_drop_cnt + 1
                  cycle
                end if

                call pfafem_invert_point(node_list, element_list, tr12(1), tr12(2), phi_c, &
                                         pfafem_openbc, em, sm, tm, status_inv)
                if (status_inv == 0) then
                  send_i(i_off) = 0
                  n_drop_cnt = n_drop_cnt + 1
                else if (status_inv == 3) then
                  send_i(i_off) = -em
                  send_r(r_off + 1) = sm
                  send_r(r_off + 2) = tm
                  send_r(r_off + 3 : r_off + 12) = 0.d0
                  n_clmp_cnt = n_clmp_cnt + 1
                else
                  send_i(i_off) = em
                  send_r(r_off + 1)  = sm
                  send_r(r_off + 2)  = tm
                  send_r(r_off + 3 : r_off + 12) = tr12(3:12)
                  if (status_inv == 1) then
                    n_int_cnt = n_int_cnt + 1
                  else
                    n_ext_cnt = n_ext_cnt + 1
                  end if
                end if
              end do
            end do
          end do
        end do
      end do
    end do
    !$omp end parallel do

    ! Exchange compact integer footprints across MPI ranks so global_matrix_structure has full sparsity
    allocate(all_i(nel * pts_per_el))
#ifndef USE_PETSC
    allocate(all_r(nel * pts_per_el * 12))
#endif
    if (n_cpu == 1) then
      all_i = send_i(1 : nel * pts_per_el)
#ifndef USE_PETSC
      all_r = send_r(1 : nel * pts_per_el * 12)
#endif
      g_counts   = [n_int_cnt, n_ext_cnt, n_clmp_cnt, n_drop_cnt]
      g_seam_max = n_seam_max
      g_seam_sum = [n_seam_sum, n_lines_loc]
    else
      allocate(recv_cnt_i(n_cpu), disp_i(n_cpu))
#ifndef USE_PETSC
      allocate(recv_cnt_r(n_cpu), disp_r(n_cpu))
#endif
      do r = 0, n_cpu - 1
        r_lo = (r * nel) / n_cpu + 1
        r_hi = ((r + 1) * nel) / n_cpu
        r_n  = max(0, r_hi - r_lo + 1)
        recv_cnt_i(r + 1) = r_n * pts_per_el
        disp_i(r + 1)     = (r_lo - 1) * pts_per_el
#ifndef USE_PETSC
        recv_cnt_r(r + 1) = r_n * pts_per_el * 12
        disp_r(r + 1)     = (r_lo - 1) * pts_per_el * 12
#endif
      end do
      call MPI_Allgatherv(send_i, n_loc * pts_per_el, MPI_INTEGER, &
                          all_i, recv_cnt_i, disp_i, MPI_INTEGER, MPI_COMM_WORLD, mpi_ierr)
#ifndef USE_PETSC
      call MPI_Allgatherv(send_r, n_loc * pts_per_el * 12, MPI_DOUBLE_PRECISION, &
                          all_r, recv_cnt_r, disp_r, MPI_DOUBLE_PRECISION, MPI_COMM_WORLD, mpi_ierr)
#endif
      call MPI_Allreduce([n_int_cnt, n_ext_cnt, n_clmp_cnt, n_drop_cnt], g_counts, 4, &
                         MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, mpi_ierr)
      call MPI_Allreduce(n_seam_max, g_seam_max, 1, MPI_INTEGER, MPI_MAX, MPI_COMM_WORLD, mpi_ierr)
      call MPI_Allreduce([n_seam_sum, n_lines_loc], g_seam_sum, 2, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, mpi_ierr)
      deallocate(recv_cnt_i, disp_i)
#ifndef USE_PETSC
      deallocate(recv_cnt_r, disp_r)
#endif
    end if
    deallocate(send_i)

    if (my_id == 0) then
      if (pfafem_ng_pol == n_gauss) then
        write(*,'(A,I3,A,I3)') ' [PFAFEM] Inexact poloidal Gauss rule: ng_pol=', pfafem_ng_pol, &
          ' ng_tor=', pfafem_ng_tor
      else
        write(*,'(A,I3,A,F5.2,A,I2)') ' [PFAFEM] Seam-split poloidal rule: ng_pol=', pfafem_ng_pol, &
          ' avg_seams=', dble(g_seam_sum(1)) / dble(max(1, g_seam_sum(2))), ' max_seams=', g_seam_max
      end if
    end if

    ! Unpack into pfafem_cache(1:nel) and build unique coupled vertex & pair tables
    if (allocated(pfafem_cache)) deallocate(pfafem_cache)
    allocate(pfafem_cache(nel))

#ifdef USE_PETSC
    !$omp parallel do schedule(dynamic, 8) private(ielm)
    do ielm = 1, nel
      if (ielm >= el_lo .and. ielm <= el_hi) then
        call populate_elem_cache_entry(element_list, ielm, ep_dim, ng_t, pts_per_el, &
                                       all_i((ielm - 1)*pts_per_el + 1 : ielm*pts_per_el), &
                                       send_r((ielm - el_lo)*pts_per_el*12 + 1), &
                                       .true., pfafem_cache(ielm))
      else
        call populate_elem_cache_entry(element_list, ielm, ep_dim, ng_t, pts_per_el, &
                                       all_i((ielm - 1)*pts_per_el + 1 : ielm*pts_per_el), &
                                       dummy_r, .false., pfafem_cache(ielm))
      end if
    end do
    !$omp end parallel do
    deallocate(all_i, send_r)
#else
    deallocate(send_r)
    !$omp parallel do schedule(dynamic, 8) private(ielm)
    do ielm = 1, nel
      call populate_elem_cache_entry(element_list, ielm, ep_dim, ng_t, pts_per_el, &
                                     all_i((ielm - 1)*pts_per_el + 1 : ielm*pts_per_el), &
                                     all_r((ielm - 1)*pts_per_el*12 + 1), &
                                     .true., pfafem_cache(ielm))
    end do
    !$omp end parallel do
    deallocate(all_i, all_r)
#endif

#ifdef STELLARATOR_MODEL
    ! Precompute Dommaschk chi(0:2, 0:2, 0:2) at this rank's physical PFAFEM quadrature points
    if (allocated(pfafem_chi_q)) deallocate(pfafem_chi_q)
    allocate(pfafem_chi_q(0:2, 0:2, 0:2, n_gauss, pfafem_ng_pol, ng_t, n_tor, max(1, n_loc)))
    pfafem_chi_q = 0.d0
    !$omp parallel do schedule(dynamic, 4) &
    !$omp private(ielm, iel_loc, eP, iep, iq, mt, ms, phi_q, s_q, t_q, R_q, Z_q, Rs, Rt, Zs, Zt)
    do ielm = el_lo, el_hi
      iel_loc = ielm - el_lo + 1
      do eP = 0, n_tor - 1
        iep = merge(1, eP + 1, ep_dim == 1)
        do iq = 1, ng_t
          phi_q = (dble(eP) + pfafem_tau_q(iq)) * pfafem_dphi
          do ms = 1, n_gauss
            s_q = xgauss(ms)
            do mt = 1, pfafem_ng_pol
              t_q = pfafem_t_pol(ms, mt, iq, iep, iel_loc)
              call pfafem_eval_elem_geom(node_list, element_list, ielm, s_q, t_q, phi_q, &
                                         R_q, Z_q, Rs, Rt, Zs, Zt)
              pfafem_chi_q(:, :, :, ms, mt, iq, eP + 1, iel_loc) = &
                get_chi(R_q, Z_q, phi_q, node_list, element_list, ielm, s_q, t_q)
            end do
          end do
        end do
      end do
    end do
    !$omp end parallel do
#endif

    pfafem_ready = .true.
    pfafem_cache_epoch = pfafem_cache_epoch + 1
#if JOREK_MODEL != 180
    call pfafem_build_qp_cache(my_id, node_list, element_list)
#endif
    call pfafem_verify_pou(my_id, node_list, element_list, g_counts)
  end subroutine pfafem_build_cache

  !=============================================================================
  !> Free precomputed local-element quadrature geometry & basis jet arrays.
  !=============================================================================
  subroutine pfafem_dealloc_qp_cache()
    pfafem_qp_cached = .false.
    if (allocated(pfafem_qp_geom)) deallocate(pfafem_qp_geom)
    if (allocated(pfafem_qp_ngrp)) deallocate(pfafem_qp_ngrp)
    if (allocated(pfafem_qp_gu))   deallocate(pfafem_qp_gu)
    if (allocated(pfafem_qp_gjb))  deallocate(pfafem_qp_gjb)
    if (allocated(pfafem_qp_giuj)) deallocate(pfafem_qp_giuj)
    if (allocated(pfafem_qp_db))   deallocate(pfafem_qp_db)
  end subroutine pfafem_dealloc_qp_cache

  !=============================================================================
  !> Precompute owned-element [pfafem_el_lo .. pfafem_el_hi] quadrature-point
  !! physical element geometry and raw 10-derivative PFAFEM basis jets in OpenMP
  !! parallel whenever per-rank footprint <= 3.5 GB.
  !=============================================================================
  subroutine pfafem_build_qp_cache(my_id, node_list, element_list)
    integer,                 intent(in) :: my_id
    type(type_node_list),    intent(in) :: node_list
    type(type_element_list), intent(in) :: element_list

    integer :: n_loc, ng_t, ielm, iel_loc, eP, iep, iq, ms, mt, i, j, inode
    integer :: n_raw, raw_u(64), raw_io(64), raw_jb(64), n_grp, g, a0, a, u_g, jb_g
    integer :: max_iuj_loc, max_iuj_glob, mpi_ierr
    real*8  :: qp_bytes_loc, qp_bytes_max
    real*8  :: phi_q, t_q, w_t, sz, s_norm_q, hz_c(n_coord_tor), hzp_c(n_coord_tor), hzpp_c(n_coord_tor)
    real*8  :: H0_q(4, n_degrees), Hs_q(4, n_degrees), Ht_q(4, n_degrees)
    real*8  :: Hss_q(4, n_degrees), Hst_q(4, n_degrees), Htt_q(4, n_degrees)
    real*8  :: x_iq0(4, 4), x_iq1(4, 4), x_iq2(4, 4), y_iq0(4, 4), y_iq1(4, 4), y_iq2(4, 4), rtor_iq(4, 4)
    real*8  :: x_g, x_s, x_t, x_p, x_ss, x_st, x_tt, x_sp, x_tp
    real*8  :: y_g, y_s, y_t, y_p, y_ss, y_st, y_tt, y_sp, y_tp
    real*8  :: xjac, xjac_x, xjac_y, x_p_x, x_p_y, y_p_x, y_p_y, BigR, wvol
    real*8  :: rv(64), rR(64), rZ(64), rP(64), rRR(64), rRZ(64), rZZ(64), rRP(64), rZP(64), rPP(64)

    call pfafem_dealloc_qp_cache()

    n_loc = max(0, pfafem_el_hi - pfafem_el_lo + 1)
    ng_t  = pfafem_ng_tor
    max_iuj_loc = 0
    do ielm = pfafem_el_lo, pfafem_el_hi
      max_iuj_loc = max(max_iuj_loc, pfafem_cache(ielm)%n_unodes, pfafem_cache(ielm)%n_uj)
    end do
    max_iuj_loc = max(max_iuj_loc, n_tor)

    qp_bytes_loc = dble(n_gauss * pfafem_ng_pol * ng_t * n_tor) * dble(max(1, n_loc)) * 5386.d0
    call MPI_Allreduce(qp_bytes_loc, qp_bytes_max, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, mpi_ierr)
    call MPI_Allreduce(max_iuj_loc, max_iuj_glob, 1, MPI_INTEGER, MPI_MAX, MPI_COMM_WORLD, mpi_ierr)

    if (qp_bytes_max > 3.5d9 .or. max_iuj_glob > 32000) then
      if (my_id == 0) then
        write(*,'(A,F8.1,A)') ' [PFAFEM] QP basis cache skipped (peak ', qp_bytes_max * 1.d-6, ' MB/rank > cap)'
      end if
      return
    end if

    allocate(pfafem_qp_geom(21, n_gauss, pfafem_ng_pol, ng_t, n_tor, max(1, n_loc)))
    allocate(pfafem_qp_ngrp(n_gauss, pfafem_ng_pol, ng_t, n_tor, max(1, n_loc)))
    allocate(pfafem_qp_gu(16, n_gauss, pfafem_ng_pol, ng_t, n_tor, max(1, n_loc)))
    allocate(pfafem_qp_gjb(16, n_gauss, pfafem_ng_pol, ng_t, n_tor, max(1, n_loc)))
    allocate(pfafem_qp_giuj(16, n_gauss, pfafem_ng_pol, ng_t, n_tor, max(1, n_loc)))
    allocate(pfafem_qp_db(10, 64, n_gauss, pfafem_ng_pol, ng_t, n_tor, max(1, n_loc)))

    pfafem_qp_geom = 0.d0
    pfafem_qp_ngrp = 0_2
    pfafem_qp_gu   = 0_2
    pfafem_qp_gjb  = 0_2
    pfafem_qp_giuj = 0_2
    pfafem_qp_db   = 0.d0

    !$omp parallel do schedule(dynamic, 2) &
    !$omp private(ielm, iel_loc, eP, iep, iq, ms, mt, i, j, inode, phi_q, t_q, w_t, sz, s_norm_q, &
    !$omp         hz_c, hzp_c, hzpp_c, H0_q, Hs_q, Ht_q, Hss_q, Hst_q, Htt_q, &
    !$omp         x_iq0, x_iq1, x_iq2, y_iq0, y_iq1, y_iq2, rtor_iq, &
    !$omp         x_g, x_s, x_t, x_p, x_ss, x_st, x_tt, x_sp, x_tp, &
    !$omp         y_g, y_s, y_t, y_p, y_ss, y_st, y_tt, y_sp, y_tp, &
    !$omp         xjac, xjac_x, xjac_y, x_p_x, x_p_y, y_p_x, y_p_y, BigR, wvol, &
    !$omp         n_raw, raw_u, raw_io, raw_jb, n_grp, g, a0, a, u_g, jb_g, &
    !$omp         rv, rR, rZ, rP, rRR, rRZ, rZZ, rRP, rZP, rPP)
    do ielm = pfafem_el_lo, pfafem_el_hi
      iel_loc = ielm - pfafem_el_lo + 1
#ifdef STELLARATOR_MODEL
      do i = 1, n_vertex_max
        inode = element_list%element(ielm)%vertex(i)
        do j = 1, n_order + 1
          rtor_iq(i, j) = element_list%element(ielm)%size(i, j) * node_list%node(inode)%r_tor_eq(j)
        end do
      end do
#else
      rtor_iq = 0.d0
#endif
      do eP = 0, n_tor - 1
        iep = merge(1, eP + 1, pfafem_cache_axisym)
        do iq = 1, ng_t
          phi_q = (dble(eP) + pfafem_tau_q(iq)) * pfafem_dphi
          if (n_coord_tor == 1) then
            hz_c(1)   = 1.d0
            hzp_c(1)  = 0.d0
            hzpp_c(1) = 0.d0
          else
            call pfafem_eval_coord_hz(phi_q, hz_c, hzp_c, hzpp_c)
          end if

          do i = 1, n_vertex_max
            inode = element_list%element(ielm)%vertex(i)
            do j = 1, n_order + 1
              sz = element_list%element(ielm)%size(i, j)
              x_iq0(i, j) = sz * dot_product(node_list%node(inode)%x(1:n_coord_tor, j, 1), hz_c)
              x_iq1(i, j) = sz * dot_product(node_list%node(inode)%x(1:n_coord_tor, j, 1), hzp_c)
              x_iq2(i, j) = sz * dot_product(node_list%node(inode)%x(1:n_coord_tor, j, 1), hzpp_c)
              y_iq0(i, j) = sz * dot_product(node_list%node(inode)%x(1:n_coord_tor, j, 2), hz_c)
              y_iq1(i, j) = sz * dot_product(node_list%node(inode)%x(1:n_coord_tor, j, 2), hzp_c)
              y_iq2(i, j) = sz * dot_product(node_list%node(inode)%x(1:n_coord_tor, j, 2), hzpp_c)
            end do
          end do

          do ms = 1, n_gauss
            do mt = 1, pfafem_ng_pol
              t_q = pfafem_t_pol(ms, mt, iq, iep, iel_loc)
              w_t = pfafem_w_pol(ms, mt, iq, iep, iel_loc)
              call pfafem_hermite2d_jet(xgauss(ms), t_q, H0_q, Hs_q, Ht_q, Hss_q, Hst_q, Htt_q)

              x_g = 0.d0; x_s = 0.d0; x_t = 0.d0; x_p = 0.d0
              x_ss = 0.d0; x_st = 0.d0; x_tt = 0.d0; x_sp = 0.d0; x_tp = 0.d0
              y_g = 0.d0; y_s = 0.d0; y_t = 0.d0; y_p = 0.d0
              y_ss = 0.d0; y_st = 0.d0; y_tt = 0.d0; y_sp = 0.d0; y_tp = 0.d0
              s_norm_q = 0.d0

              do i = 1, n_vertex_max
                do j = 1, n_order + 1
                  x_g  = x_g  + x_iq0(i, j) * H0_q(i, j)
                  x_s  = x_s  + x_iq0(i, j) * Hs_q(i, j)
                  x_t  = x_t  + x_iq0(i, j) * Ht_q(i, j)
                  x_p  = x_p  + x_iq1(i, j) * H0_q(i, j)
                  x_ss = x_ss + x_iq0(i, j) * Hss_q(i, j)
                  x_st = x_st + x_iq0(i, j) * Hst_q(i, j)
                  x_tt = x_tt + x_iq0(i, j) * Htt_q(i, j)
                  x_sp = x_sp + x_iq1(i, j) * Hs_q(i, j)
                  x_tp = x_tp + x_iq1(i, j) * Ht_q(i, j)

                  y_g  = y_g  + y_iq0(i, j) * H0_q(i, j)
                  y_s  = y_s  + y_iq0(i, j) * Hs_q(i, j)
                  y_t  = y_t  + y_iq0(i, j) * Ht_q(i, j)
                  y_p  = y_p  + y_iq1(i, j) * H0_q(i, j)
                  y_ss = y_ss + y_iq0(i, j) * Hss_q(i, j)
                  y_st = y_st + y_iq0(i, j) * Hst_q(i, j)
                  y_tt = y_tt + y_iq0(i, j) * Htt_q(i, j)
                  y_sp = y_sp + y_iq1(i, j) * Hs_q(i, j)
                  y_tp = y_tp + y_iq1(i, j) * Ht_q(i, j)

                  s_norm_q = s_norm_q + rtor_iq(i, j) * H0_q(i, j)
                end do
              end do

              xjac   = x_s * y_t - x_t * y_s
              xjac_x = (x_ss*y_t**2 - y_ss*x_t*y_t - 2.d0*x_st*y_s*y_t &
                      + y_st*(x_s*y_t + x_t*y_s) + x_tt*y_s**2 - y_tt*x_s*y_s) / xjac
              xjac_y = (y_tt*x_s**2 - x_tt*y_s*x_s - 2.d0*y_st*x_t*x_s &
                      + x_st*(y_t*x_s + y_s*x_t) + y_ss*x_t**2 - x_ss*y_t*x_t) / xjac
              x_p_x  = (x_sp * y_t - x_tp * y_s) / xjac
              x_p_y  = (x_tp * x_s - x_sp * x_t) / xjac
              y_p_x  = (y_sp * y_t - y_tp * y_s) / xjac
              y_p_y  = (y_tp * x_s - y_sp * x_t) / xjac

              BigR = x_g
              wvol = wgauss(ms) * w_t * pfafem_w_tor(iq) * BigR * xjac

              pfafem_qp_geom(1,  ms, mt, iq, eP + 1, iel_loc) = BigR
              pfafem_qp_geom(2,  ms, mt, iq, eP + 1, iel_loc) = y_g
              pfafem_qp_geom(3,  ms, mt, iq, eP + 1, iel_loc) = x_s
              pfafem_qp_geom(4,  ms, mt, iq, eP + 1, iel_loc) = x_t
              pfafem_qp_geom(5,  ms, mt, iq, eP + 1, iel_loc) = x_p
              pfafem_qp_geom(6,  ms, mt, iq, eP + 1, iel_loc) = y_s
              pfafem_qp_geom(7,  ms, mt, iq, eP + 1, iel_loc) = y_t
              pfafem_qp_geom(8,  ms, mt, iq, eP + 1, iel_loc) = y_p
              pfafem_qp_geom(9,  ms, mt, iq, eP + 1, iel_loc) = x_st
              pfafem_qp_geom(10, ms, mt, iq, eP + 1, iel_loc) = x_tt
              pfafem_qp_geom(11, ms, mt, iq, eP + 1, iel_loc) = y_st
              pfafem_qp_geom(12, ms, mt, iq, eP + 1, iel_loc) = y_tt
              pfafem_qp_geom(13, ms, mt, iq, eP + 1, iel_loc) = xjac
              pfafem_qp_geom(14, ms, mt, iq, eP + 1, iel_loc) = xjac_x
              pfafem_qp_geom(15, ms, mt, iq, eP + 1, iel_loc) = xjac_y
              pfafem_qp_geom(16, ms, mt, iq, eP + 1, iel_loc) = x_p_x
              pfafem_qp_geom(17, ms, mt, iq, eP + 1, iel_loc) = x_p_y
              pfafem_qp_geom(18, ms, mt, iq, eP + 1, iel_loc) = y_p_x
              pfafem_qp_geom(19, ms, mt, iq, eP + 1, iel_loc) = y_p_y
              pfafem_qp_geom(20, ms, mt, iq, eP + 1, iel_loc) = wvol
              pfafem_qp_geom(21, ms, mt, iq, eP + 1, iel_loc) = s_norm_q

              call pfafem_eval_qp_basis(node_list, element_list, ielm, eP, iq, ms, mt, &
                                        BigR, y_g, phi_q, .false., n_raw, raw_u, raw_io, raw_jb, &
                                        rv, rR, rZ, rP, rRR, rRZ, rZZ, rRP, rZP, rPP)
              n_grp = n_raw / 4
              pfafem_qp_ngrp(ms, mt, iq, eP + 1, iel_loc) = int(n_grp, 2)
              do g = 1, n_grp
                a0   = 4 * (g - 1)
                u_g  = raw_u(a0 + 1)
                jb_g = raw_jb(a0 + 1)
                pfafem_qp_gu(g, ms, mt, iq, eP + 1, iel_loc)   = int(u_g, 2)
                pfafem_qp_gjb(g, ms, mt, iq, eP + 1, iel_loc)  = int(jb_g, 2)
                pfafem_qp_giuj(g, ms, mt, iq, eP + 1, iel_loc) = int(pfafem_cache(ielm)%uj_map(u_g, jb_g), 2)
              end do
              do a = 1, n_raw
                pfafem_qp_db(1,  a, ms, mt, iq, eP + 1, iel_loc) = rv(a)
                pfafem_qp_db(2,  a, ms, mt, iq, eP + 1, iel_loc) = rR(a)
                pfafem_qp_db(3,  a, ms, mt, iq, eP + 1, iel_loc) = rZ(a)
                pfafem_qp_db(4,  a, ms, mt, iq, eP + 1, iel_loc) = rP(a)
                pfafem_qp_db(5,  a, ms, mt, iq, eP + 1, iel_loc) = rRR(a)
                pfafem_qp_db(6,  a, ms, mt, iq, eP + 1, iel_loc) = rZZ(a)
                pfafem_qp_db(7,  a, ms, mt, iq, eP + 1, iel_loc) = rRZ(a)
                pfafem_qp_db(8,  a, ms, mt, iq, eP + 1, iel_loc) = rPP(a)
                pfafem_qp_db(9,  a, ms, mt, iq, eP + 1, iel_loc) = rRP(a)
                pfafem_qp_db(10, a, ms, mt, iq, eP + 1, iel_loc) = rZP(a)
              end do
            end do
          end do
        end do
      end do
    end do
    !$omp end parallel do

    pfafem_qp_cached = .true.
    if (my_id == 0) then
      write(*,'(A,F7.1,A)') ' [PFAFEM] QP basis & geometry cache built: ', qp_bytes_max * 1.d-6, ' MB/rank'
    end if
  end subroutine pfafem_build_qp_cache

  !=============================================================================
  !> Populate one element's cache entry, unique vertex list, and co-occurring pairs.
  !=============================================================================
  pure subroutine populate_elem_cache_entry(element_list, ielm, ep_dim, ng_t, pts_per_el, &
                                            el_i, el_r, keep_jets, ce)
    type(type_element_list),      intent(in)  :: element_list
    integer,                      intent(in)  :: ielm, ep_dim, ng_t, pts_per_el
    integer,                      intent(in)  :: el_i(pts_per_el)
    real*8,                       intent(in)  :: el_r(*)
    logical,                      intent(in)  :: keep_jets
    type(type_pfafem_elem_cache), intent(out) :: ce

    integer :: iep, iq, mt, ms, lb, kv, idx, roff, em, inode, u, u1, u2, n_u, n_vp
    integer :: eP, jb, iuj1, iuj2, n_uj_cnt, n_sp_cnt, qp_iuj(16), n_qp_iuj
    integer :: tmp_nodes(4 + 4 * pts_per_el)
    integer :: qp_u(16), n_qp_u, a, b

    allocate(ce%e_tr(4, n_gauss, pfafem_ng_pol, ng_t, ep_dim))
    allocate(ce%u_vert(4, 4, n_gauss, pfafem_ng_pol, ng_t, ep_dim))
    if (keep_jets) then
      allocate(ce%s_tr(4, n_gauss, pfafem_ng_pol, ng_t, ep_dim))
      allocate(ce%t_tr(4, n_gauss, pfafem_ng_pol, ng_t, ep_dim))
      allocate(ce%F_tr(4, 4, n_gauss, pfafem_ng_pol, ng_t, ep_dim))
      allocate(ce%G_tr(6, 4, n_gauss, pfafem_ng_pol, ng_t, ep_dim))
    end if

    ce%u_vert = 0
    n_u = 0

    ! Always include the physical element's own 4 vertices first
    do kv = 1, n_vertex_max
      inode = element_list%element(ielm)%vertex(kv)
      call add_unique_int(tmp_nodes, n_u, inode, u)
    end do

    idx = 0
    do iep = 1, ep_dim
      do iq = 1, ng_t
        do mt = 1, pfafem_ng_pol
          do ms = 1, n_gauss
            do lb = 1, 4
              idx  = idx + 1
              roff = (idx - 1) * 12
              em   = el_i(idx)
              ce%e_tr(lb, ms, mt, iq, iep) = em
              if (keep_jets) then
                ce%s_tr(lb, ms, mt, iq, iep)      = el_r(roff + 1)
                ce%t_tr(lb, ms, mt, iq, iep)      = el_r(roff + 2)
                ce%F_tr(1:4, lb, ms, mt, iq, iep) = el_r(roff + 3 : roff + 6)
                ce%G_tr(1:6, lb, ms, mt, iq, iep) = el_r(roff + 7 : roff + 12)
              end if
              if (em /= 0) then
                do kv = 1, n_vertex_max
                  inode = element_list%element(abs(em))%vertex(kv)
                  call add_unique_int(tmp_nodes, n_u, inode, u)
                  ce%u_vert(kv, lb, ms, mt, iq, iep) = u
                end do
              end if
            end do
          end do
        end do
      end do
    end do

    ce%n_unodes = n_u
    allocate(ce%unodes(n_u))
    ce%unodes = tmp_nodes(1:n_u)

    allocate(ce%pair_map(n_u, n_u))
    ce%pair_map = 0
    do u = 1, n_u
      ce%pair_map(u, u) = 1
    end do
    ! Physical element's own vertex pairs
    do a = 1, min(4, n_u)
      do b = 1, min(4, n_u)
        ce%pair_map(a, b) = 1
      end do
    end do

    ! Mark all co-occurring (u1, u2) pairs at every quadrature point
    do iep = 1, ep_dim
      do iq = 1, ng_t
        do mt = 1, pfafem_ng_pol
          do ms = 1, n_gauss
            n_qp_u = 0
            do lb = 1, 4
              if (ce%e_tr(lb, ms, mt, iq, iep) /= 0) then
                do kv = 1, n_vertex_max
                  n_qp_u = n_qp_u + 1
                  qp_u(n_qp_u) = ce%u_vert(kv, lb, ms, mt, iq, iep)
                end do
              end if
            end do
            do a = 1, n_qp_u
              u1 = qp_u(a)
              do b = 1, n_qp_u
                u2 = qp_u(b)
                ce%pair_map(u1, u2) = 1
              end do
            end do
          end do
        end do
      end do
    end do

    n_vp = count(ce%pair_map > 0)
    ce%n_vpairs = n_vp
    allocate(ce%vpairs(2, n_vp))
    idx = 0
    do u1 = 1, n_u
      do u2 = 1, n_u
        if (ce%pair_map(u1, u2) > 0) then
          idx = idx + 1
          ce%pair_map(u1, u2) = idx
          ce%vpairs(1, idx)   = u1
          ce%vpairs(2, idx)   = u2
        end if
      end do
    end do

    ! Build sparse active (u, jb) slot list and co-occurring (iuj1, iuj2) pair table
    allocate(ce%uj_map(n_u, n_tor))
    ce%uj_map = 0
    do eP = 0, n_tor - 1
      iep = merge(1, eP + 1, ep_dim == 1)
      do iq = 1, ng_t
        do mt = 1, pfafem_ng_pol
          do ms = 1, n_gauss
            do lb = 1, 4
              if (ce%e_tr(lb, ms, mt, iq, iep) /= 0) then
                jb = pfafem_slot(eP, lb, n_tor)
                do kv = 1, n_vertex_max
                  u = ce%u_vert(kv, lb, ms, mt, iq, iep)
                  ce%uj_map(u, jb) = 1
                end do
              end if
            end do
          end do
        end do
      end do
    end do

    n_uj_cnt = count(ce%uj_map > 0)
    ce%n_uj  = n_uj_cnt
    allocate(ce%uj_u(max(1, n_uj_cnt)), ce%uj_jb(max(1, n_uj_cnt)))
    ce%uj_u  = 0
    ce%uj_jb = 0
    idx = 0
    do u = 1, n_u
      do jb = 1, n_tor
        if (ce%uj_map(u, jb) > 0) then
          idx = idx + 1
          ce%uj_map(u, jb) = idx
          ce%uj_u(idx)     = u
          ce%uj_jb(idx)    = jb
        end if
      end do
    end do

    allocate(ce%sp_map(max(1, n_uj_cnt), max(1, n_uj_cnt)))
    ce%sp_map = 0
    do iuj1 = 1, n_uj_cnt
      ce%sp_map(iuj1, iuj1) = 1
    end do
    do eP = 0, n_tor - 1
      iep = merge(1, eP + 1, ep_dim == 1)
      do iq = 1, ng_t
        do mt = 1, pfafem_ng_pol
          do ms = 1, n_gauss
            n_qp_iuj = 0
            do lb = 1, 4
              if (ce%e_tr(lb, ms, mt, iq, iep) /= 0) then
                jb = pfafem_slot(eP, lb, n_tor)
                do kv = 1, n_vertex_max
                  u = ce%u_vert(kv, lb, ms, mt, iq, iep)
                  n_qp_iuj = n_qp_iuj + 1
                  qp_iuj(n_qp_iuj) = ce%uj_map(u, jb)
                end do
              end if
            end do
            do a = 1, n_qp_iuj
              iuj1 = qp_iuj(a)
              do b = 1, n_qp_iuj
                iuj2 = qp_iuj(b)
                ce%sp_map(iuj1, iuj2) = 1
              end do
            end do
          end do
        end do
      end do
    end do

    n_sp_cnt = count(ce%sp_map > 0)
    ce%n_sp  = n_sp_cnt
    allocate(ce%sp_pairs(2, max(1, n_sp_cnt)))
    allocate(ce%u_sp_lo(n_u), ce%u_sp_hi(n_u))
    ce%sp_pairs = 0
    ce%u_sp_lo  = 1
    ce%u_sp_hi  = 0
    idx = 0
    do iuj1 = 1, n_uj_cnt
      u1 = ce%uj_u(iuj1)
      do iuj2 = 1, n_uj_cnt
        if (ce%sp_map(iuj1, iuj2) > 0) then
          idx = idx + 1
          ce%sp_map(iuj1, iuj2) = idx
          ce%sp_pairs(1, idx)   = iuj1
          ce%sp_pairs(2, idx)   = iuj2
          if (ce%u_sp_hi(u1) == 0) ce%u_sp_lo(u1) = idx
          ce%u_sp_hi(u1) = idx
        end if
      end do
    end do

    if (.not. keep_jets) then
      deallocate(ce%e_tr, ce%u_vert)
    end if
  end subroutine populate_elem_cache_entry

  pure subroutine add_unique_int(arr, n, val, pos)
    integer, intent(inout) :: arr(:), n
    integer, intent(in)    :: val
    integer, intent(out)   :: pos
    integer :: k
    do k = 1, n
      if (arr(k) == val) then
        pos = k
        return
      end if
    end do
    n = n + 1
    arr(n) = val
    pos = n
  end subroutine add_unique_int

  !=============================================================================
  !> Evaluate all active PFAFEM scalar basis functions (at most 64) and their
  !! exact 1st and 2nd cylindrical derivatives (R, Z, phi) at quadrature point
  !! (eP, iq, ms, mt) of element ielm.
  !!  - apply_axis_pmat = .false.: raw local Hermite basis for gathering node%values / %deltas
  !!  - apply_axis_pmat = .true. : global-DOF basis (with Pmat on axis vertices) for A, RHS, W, M
  !=============================================================================
  subroutine pfafem_eval_qp_basis(node_list, element_list, ielm, eP, iq, ms, mt, &
                                  R_q, Z_q, phi_q, apply_axis_pmat, &
                                  n_act, act_u, act_io, act_jb, &
                                  b_val, b_dR, b_dZ, b_dP, &
                                  b_dRR, b_dRZ, b_dZZ, b_dRP, b_dZP, b_dPP)
    use phys_module, only: treat_axis
    use mod_basisfunctions, only: basisfunctions
    type(type_node_list),    intent(in)  :: node_list
    type(type_element_list), intent(in)  :: element_list
    integer,                 intent(in)  :: ielm, eP, iq, ms, mt
    real*8,                  intent(in)  :: R_q, Z_q, phi_q
    logical,                 intent(in)  :: apply_axis_pmat
    integer,                 intent(out) :: n_act
    integer,                 intent(out) :: act_u(64), act_io(64), act_jb(64)
    real*8,                  intent(out) :: b_val(64), b_dR(64), b_dZ(64), b_dP(64)
    real*8, optional,        intent(out) :: b_dRR(64), b_dRZ(64), b_dZZ(64)
    real*8, optional,        intent(out) :: b_dRP(64), b_dZP(64), b_dPP(64)

    logical :: want_hess, jet_ok
    integer :: iep, lb, jb, em_raw, em, kv, io, jo, inode
    real*8  :: tau, nb(4), dnb(4), ddnb(4), b0, bp, bpp, phi_c, sm, tm
    real*8  :: fl(2), A(4), Hjet(6), dfl_dp(2)
    real*8  :: H_l(4, n_degrees), Hs_l(4, n_degrees), Ht_l(4, n_degrees)
    real*8  :: Hst_l(4, n_degrees), Hss_l(4, n_degrees), Htt_l(4, n_degrees)
    real*8  :: Rr, Zr, Rs, Rt, Zs, Zt, Rss, Rst, Rtt, Zss, Zst, Ztt
    real*8  :: xjac, xjac_x, xjac_y, sz, F11, F12, F21, F22, G(6)
    real*8  :: hs, ht, hss, hst, htt, lam, lam_R, lam_Z, lam_RR, lam_RZ, lam_ZZ
    real*8  :: L, L_R, L_Z, L_P, L_RR, L_RZ, L_ZZ, L_RP, L_ZP, L_PP
    real*8  :: loc_v(4), loc_R(4), loc_Z(4), loc_P(4)
    real*8  :: loc_RR(4), loc_RZ(4), loc_ZZ(4), loc_RP(4), loc_ZP(4), loc_PP(4)
    real*8  :: pmat(4, 4)

    want_hess = present(b_dRR)
    n_act = 0

    call pfafem_fieldline_jet(R_q, Z_q, phi_q, fl, A, Hjet, dfl_dp, jet_ok)
    if (.not. jet_ok) then
      fl = 0.d0; A = 0.d0; Hjet = 0.d0; dfl_dp = 0.d0
    end if

    tau = pfafem_tau_q(iq)
    call pfafem_bspline_eval(tau, nb, dnb, ddnb)
    iep = merge(1, eP + 1, pfafem_cache_axisym)

    do lb = 1, 4
      em_raw = pfafem_cache(ielm)%e_tr(lb, ms, mt, iq, iep)
      if (em_raw == 0) cycle
      em  = abs(em_raw)
      jb  = pfafem_slot(eP, lb, n_tor)
      phi_c = dble(jb - 1) * pfafem_dphi

      b0  = nb(lb)
      bp  = dnb(lb) / pfafem_dphi
      bpp = ddnb(lb) / (pfafem_dphi * pfafem_dphi)

      sm = pfafem_cache(ielm)%s_tr(lb, ms, mt, iq, iep)
      tm = pfafem_cache(ielm)%t_tr(lb, ms, mt, iq, iep)

      if (want_hess) then
        call basisfunctions(sm, tm, H_l, Hs_l, Ht_l, Hst_l, Hss_l, Htt_l)
        call pfafem_eval_knot_elem_geom(node_list, element_list, em, jb, sm, tm, phi_c, &
                                        Rr, Zr, Rs, Rt, Zs, Zt, Rss, Rst, Rtt, Zss, Zst, Ztt)
      else
        call basisfunctions(sm, tm, H_l, Hs_l, Ht_l)
        call pfafem_eval_knot_elem_geom(node_list, element_list, em, jb, sm, tm, phi_c, &
                                        Rr, Zr, Rs, Rt, Zs, Zt)
      end if

      xjac = Rs * Zt - Rt * Zs
      if (em_raw > 0 .and. abs(xjac) < 1.d-22) then
        ! Nudge away from collapsed s=0 axis edge so 1st/2nd poloidal derivatives are smooth
        sm = min(1.d0, sm + 1.d-4)
        if (want_hess) then
          call basisfunctions(sm, tm, H_l, Hs_l, Ht_l, Hst_l, Hss_l, Htt_l)
          call pfafem_eval_knot_elem_geom(node_list, element_list, em, jb, sm, tm, phi_c, &
                                          Rr, Zr, Rs, Rt, Zs, Zt, Rss, Rst, Rtt, Zss, Zst, Ztt)
        else
          call basisfunctions(sm, tm, H_l, Hs_l, Ht_l)
          call pfafem_eval_knot_elem_geom(node_list, element_list, em, jb, sm, tm, phi_c, &
                                          Rr, Zr, Rs, Rt, Zs, Zt)
        end if
        xjac = Rs * Zt - Rt * Zs
      end if

      if (em_raw > 0 .and. abs(xjac) >= 1.d-22) then
        F11 = pfafem_cache(ielm)%F_tr(1, lb, ms, mt, iq, iep)
        F12 = pfafem_cache(ielm)%F_tr(2, lb, ms, mt, iq, iep)
        F21 = pfafem_cache(ielm)%F_tr(3, lb, ms, mt, iq, iep)
        F22 = pfafem_cache(ielm)%F_tr(4, lb, ms, mt, iq, iep)
        if (want_hess) then
          G = pfafem_cache(ielm)%G_tr(1:6, lb, ms, mt, iq, iep)
          xjac_x = (Rss*Zt**2 - Zss*Rt*Zt - 2.d0*Rst*Zs*Zt &
                  + Zst*(Rs*Zt + Rt*Zs) + Rtt*Zs**2 - Ztt*Rs*Zs) / xjac
          xjac_y = (Ztt*Rs**2 - Rtt*Zs*Rs - 2.d0*Zst*Rt*Rs &
                  + Rst*(Zt*Rs + Zs*Rt) + Zss*Rt**2 - Rss*Zt*Rt) / xjac
        end if
      else
        F11 = 0.d0; F12 = 0.d0; F21 = 0.d0; F22 = 0.d0
        G   = 0.d0; xjac_x = 0.d0; xjac_y = 0.d0
      end if

      do kv = 1, n_vertex_max
        do io = 1, n_degrees
          sz  = element_list%element(em)%size(kv, io)
          lam = H_l(kv, io) * sz

          if (em_raw < 0 .or. abs(xjac) < 1.d-22) then
            L    = lam
            L_R  = 0.d0; L_Z  = 0.d0; L_P  = 0.d0
            L_RR = 0.d0; L_RZ = 0.d0; L_ZZ = 0.d0
            L_RP = 0.d0; L_ZP = 0.d0; L_PP = 0.d0
          else
            hs = Hs_l(kv, io) * sz
            ht = Ht_l(kv, io) * sz
            lam_R = ( Zt * hs - Zs * ht) / xjac
            lam_Z = (-Rt * hs + Rs * ht) / xjac

            L   = lam
            L_R = F11 * lam_R + F21 * lam_Z
            L_Z = F12 * lam_R + F22 * lam_Z
            L_P = -(fl(1) * L_R + fl(2) * L_Z)

            if (want_hess) then
              hss = Hss_l(kv, io) * sz
              hst = Hst_l(kv, io) * sz
              htt = Htt_l(kv, io) * sz

              lam_RR = (hss*Zt**2 - 2.d0*hst*Zs*Zt + htt*Zs**2 &
                      + hs*(Zst*Zt - Ztt*Zs) + ht*(Zst*Zs - Zss*Zt)) / (xjac*xjac) &
                      - xjac_x * (hs*Zt - ht*Zs) / (xjac*xjac)
              lam_ZZ = (hss*Rt**2 - 2.d0*hst*Rs*Rt + htt*Rs**2 &
                      + hs*(Rst*Rt - Rtt*Rs) + ht*(Rst*Rs - Rss*Rt)) / (xjac*xjac) &
                      - xjac_y * (-hs*Rt + ht*Rs) / (xjac*xjac)
              lam_RZ = (-hss*Zt*Rt - htt*Rs*Zs + hst*(Zs*Rt + Zt*Rs) &
                      - hs*(Rst*Zt - Rtt*Zs) - ht*(Rst*Zs - Rss*Zt)) / (xjac*xjac) &
                      - xjac_x * (-hs*Rt + ht*Rs) / (xjac*xjac)

              L_RR = lam_RR * F11 * F11 + 2.d0 * lam_RZ * F11 * F21 + lam_ZZ * F21 * F21 &
                   + lam_R * G(1) + lam_Z * G(4)
              L_RZ = lam_RR * F11 * F12 + lam_RZ * (F11 * F22 + F21 * F12) + lam_ZZ * F21 * F22 &
                   + lam_R * G(2) + lam_Z * G(5)
              L_ZZ = lam_RR * F12 * F12 + 2.d0 * lam_RZ * F12 * F22 + lam_ZZ * F22 * F22 &
                   + lam_R * G(3) + lam_Z * G(6)

              L_RP = -(A(1) * L_R + A(3) * L_Z + fl(1) * L_RR + fl(2) * L_RZ)
              L_ZP = -(A(2) * L_R + A(4) * L_Z + fl(1) * L_RZ + fl(2) * L_ZZ)
              L_PP = -(dfl_dp(1) * L_R + dfl_dp(2) * L_Z + fl(1) * L_RP + fl(2) * L_ZP)
            end if
          end if

          loc_v(io) = L * b0
          loc_R(io) = L_R * b0
          loc_Z(io) = L_Z * b0
          loc_P(io) = L_P * b0 + L * bp
          if (want_hess) then
            loc_RR(io) = L_RR * b0
            loc_RZ(io) = L_RZ * b0
            loc_ZZ(io) = L_ZZ * b0
            loc_RP(io) = L_RP * b0 + L_R * bp
            loc_ZP(io) = L_ZP * b0 + L_Z * bp
            loc_PP(io) = L_PP * b0 + 2.d0 * L_P * bp + L * bpp
          end if
        end do

        ! Apply axis Pmat transformation when building global-DOF test/trial basis functions
        inode = element_list%element(em)%vertex(kv)
        if (apply_axis_pmat .and. treat_axis .and. (kv == 1 .or. kv == 4) &
            .and. node_list%node(inode)%axis_node) then
          call pfafem_axis_pmat(node_list, element_list, em, kv, pmat)
          do jo = 1, n_degrees
            n_act = n_act + 1
            act_u(n_act)  = pfafem_cache(ielm)%u_vert(kv, lb, ms, mt, iq, iep)
            act_io(n_act) = jo
            act_jb(n_act) = jb
            b_val(n_act)  = dot_product(loc_v, pmat(:, jo))
            b_dR(n_act)   = dot_product(loc_R, pmat(:, jo))
            b_dZ(n_act)   = dot_product(loc_Z, pmat(:, jo))
            b_dP(n_act)   = dot_product(loc_P, pmat(:, jo))
            if (want_hess) then
              b_dRR(n_act) = dot_product(loc_RR, pmat(:, jo))
              b_dRZ(n_act) = dot_product(loc_RZ, pmat(:, jo))
              b_dZZ(n_act) = dot_product(loc_ZZ, pmat(:, jo))
              b_dRP(n_act) = dot_product(loc_RP, pmat(:, jo))
              b_dZP(n_act) = dot_product(loc_ZP, pmat(:, jo))
              b_dPP(n_act) = dot_product(loc_PP, pmat(:, jo))
            end if
          end do
        else
          do jo = 1, n_degrees
            n_act = n_act + 1
            act_u(n_act)  = pfafem_cache(ielm)%u_vert(kv, lb, ms, mt, iq, iep)
            act_io(n_act) = jo
            act_jb(n_act) = jb
            b_val(n_act)  = loc_v(jo)
            b_dR(n_act)   = loc_R(jo)
            b_dZ(n_act)   = loc_Z(jo)
            b_dP(n_act)   = loc_P(jo)
            if (want_hess) then
              b_dRR(n_act) = loc_RR(jo)
              b_dRZ(n_act) = loc_RZ(jo)
              b_dZZ(n_act) = loc_ZZ(jo)
              b_dRP(n_act) = loc_RP(jo)
              b_dZP(n_act) = loc_ZP(jo)
              b_dPP(n_act) = loc_PP(jo)
            end if
          end do
        end if
      end do
    end do
  end subroutine pfafem_eval_qp_basis

  !=============================================================================
  !> Verify Partition of Unity (sum Phi = 1), zero gradient (sum grad Phi = 0),
  !! and zero Hessian (sum D^2 Phi = 0) across all quadrature points and print
  !! summary diagnostics on rank 0.
  !=============================================================================
  subroutine pfafem_verify_pou(my_id, node_list, element_list, g_counts)
    integer,                 intent(in) :: my_id, g_counts(4)
    type(type_node_list),    intent(in) :: node_list
    type(type_element_list), intent(in) :: element_list

    integer :: ielm, iel_loc, iq, ms, mt, n_act, a, act_u(64), act_io(64), act_jb(64), mpi_ierr
    real*8  :: phi_q, t_q, R_q, Z_q, Rs, Rt, Zs, Zt
    real*8  :: b_v(64), b_R(64), b_Z(64), b_P(64)
    real*8  :: b_RR(64), b_RZ(64), b_ZZ(64), b_RP(64), b_ZP(64), b_PP(64)
    real*8  :: sv, sgR, sgZ, sgP, shRR, shRZ, shZZ, shPP
    real*8  :: loc_max(3), glob_max(3)

    loc_max = 0.d0
    !$omp parallel do schedule(dynamic, 8) &
    !$omp private(ielm, iel_loc, iq, mt, ms, phi_q, t_q, R_q, Z_q, Rs, Rt, Zs, Zt, n_act, a, &
    !$omp         act_u, act_io, act_jb, b_v, b_R, b_Z, b_P, b_RR, b_RZ, b_ZZ, b_RP, b_ZP, b_PP, &
    !$omp         sv, sgR, sgZ, sgP, shRR, shRZ, shZZ, shPP) &
    !$omp reduction(max:loc_max)
    do ielm = pfafem_el_lo, pfafem_el_hi
      iel_loc = ielm - pfafem_el_lo + 1
      do iq = 1, pfafem_ng_tor
        phi_q = pfafem_tau_q(iq) * pfafem_dphi
        do mt = 1, pfafem_ng_pol
          do ms = 1, n_gauss
            t_q = pfafem_t_pol(ms, mt, iq, 1, iel_loc)
            call pfafem_eval_elem_geom(node_list, element_list, ielm, xgauss(ms), t_q, phi_q, &
                                       R_q, Z_q, Rs, Rt, Zs, Zt)
            call pfafem_eval_qp_basis(node_list, element_list, ielm, 0, iq, ms, mt, &
                                      R_q, Z_q, phi_q, .true., n_act, act_u, act_io, act_jb, &
                                      b_v, b_R, b_Z, b_P, b_RR, b_RZ, b_ZZ, b_RP, b_ZP, b_PP)
            if (n_act == 0) cycle
            sv = 0.d0; sgR = 0.d0; sgZ = 0.d0; sgP = 0.d0
            shRR = 0.d0; shRZ = 0.d0; shZZ = 0.d0; shPP = 0.d0
            do a = 1, n_act
              if (act_io(a) == 1) then
                sv   = sv   + b_v(a)
                sgR  = sgR  + b_R(a)
                sgZ  = sgZ  + b_Z(a)
                sgP  = sgP  + b_P(a)
                shRR = shRR + b_RR(a)
                shRZ = shRZ + b_RZ(a)
                shZZ = shZZ + b_ZZ(a)
                shPP = shPP + b_PP(a)
              end if
            end do
            loc_max(1) = max(loc_max(1), abs(1.d0 - sv))
            loc_max(2) = max(loc_max(2), max(abs(sgR), abs(sgZ), abs(sgP)))
            loc_max(3) = max(loc_max(3), max(abs(shRR), abs(shRZ), abs(shZZ), abs(shPP)))
          end do
        end do
      end do
    end do
    !$omp end parallel do

    call MPI_Allreduce(loc_max, glob_max, 3, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, mpi_ierr)
    if (my_id == 0) then
      write(*,'(A,I4,A,I3,A,I3,A,L1)') ' [PFAFEM] Cache ready: n_tor=', n_tor, &
        ' ng_tor=', pfafem_ng_tor, ' ng_pol=', pfafem_ng_pol, ' axisym_B=', pfafem_is_axisym_bfield
      write(*,'(A,I9,A,I8,A,I8,A,I7)') ' [PFAFEM] Footprints: interior=', g_counts(1), &
        ' openbc_hermite=', g_counts(2), ' openbc_clamped=', g_counts(3), ' dropped=', g_counts(4)
      write(*,'(A,ES10.3,A,ES10.3,A,ES10.3)') ' [PFAFEM] PoU check: max|1-sum|=', glob_max(1), &
        ' max|grad(1)|=', glob_max(2), ' max|hess(1)|=', glob_max(3)
    end if
  end subroutine pfafem_verify_pou

  !=============================================================================
  !> Evaluate active PFAFEM scalar basis values (up to 64) at an arbitrary
  !! reference point (s_pt, t_pt) on element ielm at toroidal Gauss point (eP, iq)
  !! by tracing on the fly (used for boundary surface integrals and BCs).
  !=============================================================================
  subroutine pfafem_eval_point_basis_val(node_list, element_list, ielm, s_pt, t_pt, eP, iq, &
                                         apply_axis_pmat, n_act, act_inode, act_io, act_jb, b_val)
    use phys_module, only: treat_axis, pfafem_n_substeps, pfafem_openbc
    use mod_basisfunctions, only: basisfunctions
    type(type_node_list),    intent(in)  :: node_list
    type(type_element_list), intent(in)  :: element_list
    integer,                 intent(in)  :: ielm, eP, iq
    real*8,                  intent(in)  :: s_pt, t_pt
    logical,                 intent(in)  :: apply_axis_pmat
    integer,                 intent(out) :: n_act
    integer,                 intent(out) :: act_inode(64), act_io(64), act_jb(64)
    real*8,                  intent(out) :: b_val(64)

    integer :: lb, jb, em, status_inv, kv, io, jo, inode
    real*8  :: phi_q, phi_ref, phi_c, R_q, Z_q, Rs, Rt, Zs, Zt, sm, tm, tr12(12)
    real*8  :: nb(4), dnb(4), ddnb(4), H_l(4, n_degrees), Hs_l(4, n_degrees), Ht_l(4, n_degrees)
    real*8  :: loc_v(4), pmat(4, 4)
    logical :: tr_ok

    n_act = 0
    phi_q = (dble(eP) + pfafem_tau_q(iq)) * pfafem_dphi
    call pfafem_eval_elem_geom(node_list, element_list, ielm, s_pt, t_pt, phi_q, &
                               R_q, Z_q, Rs, Rt, Zs, Zt)
    call pfafem_bspline_eval(pfafem_tau_q(iq), nb, dnb, ddnb)

    do lb = 1, 4
      phi_ref = dble(eP - 2 + lb) * pfafem_dphi
      jb      = pfafem_slot(eP, lb, n_tor)
      phi_c   = dble(jb - 1) * pfafem_dphi

      call pfafem_trace_with_hessian(R_q, Z_q, phi_q, phi_ref, pfafem_n_substeps, tr12, tr_ok)
      if (.not. tr_ok) cycle

      call pfafem_invert_point(node_list, element_list, tr12(1), tr12(2), phi_c, &
                               pfafem_openbc, em, sm, tm, status_inv)
      if (status_inv == 0) cycle

      call basisfunctions(sm, tm, H_l, Hs_l, Ht_l)
      do kv = 1, n_vertex_max
        inode = element_list%element(em)%vertex(kv)
        do io = 1, n_degrees
          loc_v(io) = H_l(kv, io) * element_list%element(em)%size(kv, io) * nb(lb)
        end do

        if (apply_axis_pmat .and. treat_axis .and. (kv == 1 .or. kv == 4) &
            .and. node_list%node(inode)%axis_node) then
          call pfafem_axis_pmat(node_list, element_list, em, kv, pmat)
          do jo = 1, n_degrees
            n_act = n_act + 1
            act_inode(n_act) = inode
            act_io(n_act)    = jo
            act_jb(n_act)    = jb
            b_val(n_act)     = dot_product(loc_v, pmat(:, jo))
          end do
        else
          do jo = 1, n_degrees
            n_act = n_act + 1
            act_inode(n_act) = inode
            act_io(n_act)    = jo
            act_jb(n_act)    = jb
            b_val(n_act)     = loc_v(jo)
          end do
        end if
      end do
    end do
  end subroutine pfafem_eval_point_basis_val

  !=============================================================================
  !> Evaluate active PFAFEM scalar basis values and 1st (and optional 2nd)
  !! derivatives w.r.t. (R, Z, phi) at an arbitrary physical point (R_q, Z_q, phi_q)
  !! by tracing on the fly (used by 3D volume/boundary diagnostics).
  !=============================================================================
  subroutine pfafem_eval_point_jet(node_list, element_list, R_q, Z_q, phi_q, &
                                   apply_axis_pmat, n_act, act_inode, act_io, act_jb, &
                                   b_val, b_dR, b_dZ, b_dP, &
                                   b_dRR, b_dRZ, b_dZZ, b_dRP, b_dZP, b_dPP)
    use phys_module, only: treat_axis, pfafem_n_substeps, pfafem_openbc
    use mod_basisfunctions, only: basisfunctions
    type(type_node_list),    intent(in)  :: node_list
    type(type_element_list), intent(in)  :: element_list
    real*8,                  intent(in)  :: R_q, Z_q, phi_q
    logical,                 intent(in)  :: apply_axis_pmat
    integer,                 intent(out) :: n_act
    integer,                 intent(out) :: act_inode(64), act_io(64), act_jb(64)
    real*8,                  intent(out) :: b_val(64), b_dR(64), b_dZ(64), b_dP(64)
    real*8, optional,        intent(out) :: b_dRR(64), b_dRZ(64), b_dZZ(64)
    real*8, optional,        intent(out) :: b_dRP(64), b_dZP(64), b_dPP(64)

    logical :: want_hess, jet_ok, tr_ok
    integer :: eP, lb, jb, em, em_raw, status_inv, kv, io, jo, inode
    real*8  :: phi_mod, tau, phi_ref, phi_c, sm, tm, tr12(12)
    real*8  :: nb(4), dnb(4), ddnb(4), b0, bp, bpp
    real*8  :: fl(2), A(4), Hjet(6), dfl_dp(2)
    real*8  :: H_l(4, n_degrees), Hs_l(4, n_degrees), Ht_l(4, n_degrees)
    real*8  :: Hst_l(4, n_degrees), Hss_l(4, n_degrees), Htt_l(4, n_degrees)
    real*8  :: Rr, Zr, Rs, Rt, Zs, Zt, Rss, Rst, Rtt, Zss, Zst, Ztt
    real*8  :: xjac, xjac_x, xjac_y, sz, F11, F12, F21, F22, G(6)
    real*8  :: hs, ht, hss, hst, htt, lam, lam_R, lam_Z, lam_RR, lam_RZ, lam_ZZ
    real*8  :: L, L_R, L_Z, L_P, L_RR, L_RZ, L_ZZ, L_RP, L_ZP, L_PP
    real*8  :: loc_v(4), loc_R(4), loc_Z(4), loc_P(4)
    real*8  :: loc_RR(4), loc_RZ(4), loc_ZZ(4), loc_RP(4), loc_ZP(4), loc_PP(4)
    real*8  :: pmat(4, 4)

    want_hess = present(b_dRR)
    n_act = 0

    phi_mod = modulo(phi_q, pfafem_phi_per)
    eP      = min(n_tor - 1, max(0, int(phi_mod / pfafem_dphi)))
    tau     = max(0.d0, min(1.d0, phi_mod / pfafem_dphi - dble(eP)))

    call pfafem_fieldline_jet(R_q, Z_q, phi_mod, fl, A, Hjet, dfl_dp, jet_ok)
    if (.not. jet_ok) then
      fl = 0.d0; A = 0.d0; Hjet = 0.d0; dfl_dp = 0.d0
    end if
    call pfafem_bspline_eval(tau, nb, dnb, ddnb)

    do lb = 1, 4
      phi_ref = dble(eP - 2 + lb) * pfafem_dphi
      jb      = pfafem_slot(eP, lb, n_tor)
      phi_c   = dble(jb - 1) * pfafem_dphi

      call pfafem_trace_with_hessian(R_q, Z_q, phi_mod, phi_ref, pfafem_n_substeps, tr12, tr_ok)
      if (.not. tr_ok) cycle

      call pfafem_invert_point(node_list, element_list, tr12(1), tr12(2), phi_c, &
                               pfafem_openbc, em, sm, tm, status_inv)
      if (status_inv == 0) cycle
      em_raw = merge(-em, em, status_inv == 3)

      b0  = nb(lb)
      bp  = dnb(lb) / pfafem_dphi
      bpp = ddnb(lb) / (pfafem_dphi * pfafem_dphi)

      if (want_hess) then
        call basisfunctions(sm, tm, H_l, Hs_l, Ht_l, Hst_l, Hss_l, Htt_l)
        call pfafem_eval_elem_geom(node_list, element_list, em, sm, tm, phi_c, &
                                   Rr, Zr, Rs, Rt, Zs, Zt, Rss, Rst, Rtt, Zss, Zst, Ztt)
      else
        call basisfunctions(sm, tm, H_l, Hs_l, Ht_l)
        call pfafem_eval_elem_geom(node_list, element_list, em, sm, tm, phi_c, &
                                   Rr, Zr, Rs, Rt, Zs, Zt)
      end if

      xjac = Rs * Zt - Rt * Zs
      if (em_raw > 0 .and. abs(xjac) < 1.d-22) then
        sm = min(1.d0, sm + 1.d-4)
        if (want_hess) then
          call basisfunctions(sm, tm, H_l, Hs_l, Ht_l, Hst_l, Hss_l, Htt_l)
          call pfafem_eval_elem_geom(node_list, element_list, em, sm, tm, phi_c, &
                                     Rr, Zr, Rs, Rt, Zs, Zt, Rss, Rst, Rtt, Zss, Zst, Ztt)
        else
          call basisfunctions(sm, tm, H_l, Hs_l, Ht_l)
          call pfafem_eval_elem_geom(node_list, element_list, em, sm, tm, phi_c, &
                                     Rr, Zr, Rs, Rt, Zs, Zt)
        end if
        xjac = Rs * Zt - Rt * Zs
      end if

      if (em_raw > 0 .and. abs(xjac) >= 1.d-22) then
        F11 = tr12(3); F12 = tr12(4); F21 = tr12(5); F22 = tr12(6)
        if (want_hess) then
          G = tr12(7:12)
          xjac_x = (Rss*Zt**2 - Zss*Rt*Zt - 2.d0*Rst*Zs*Zt &
                  + Zst*(Rs*Zt + Rt*Zs) + Rtt*Zs**2 - Ztt*Rs*Zs) / xjac
          xjac_y = (Ztt*Rs**2 - Rtt*Zs*Rs - 2.d0*Zst*Rt*Rs &
                  + Rst*(Zt*Rs + Zs*Rt) + Zss*Rt**2 - Rss*Zt*Rt) / xjac
        end if
      else
        F11 = 0.d0; F12 = 0.d0; F21 = 0.d0; F22 = 0.d0
        G   = 0.d0; xjac_x = 0.d0; xjac_y = 0.d0
      end if

      do kv = 1, n_vertex_max
        do io = 1, n_degrees
          sz  = element_list%element(em)%size(kv, io)
          lam = H_l(kv, io) * sz

          if (em_raw < 0 .or. abs(xjac) < 1.d-22) then
            L    = lam
            L_R  = 0.d0; L_Z  = 0.d0; L_P  = 0.d0
            L_RR = 0.d0; L_RZ = 0.d0; L_ZZ = 0.d0
            L_RP = 0.d0; L_ZP = 0.d0; L_PP = 0.d0
          else
            hs = Hs_l(kv, io) * sz
            ht = Ht_l(kv, io) * sz
            lam_R = ( Zt * hs - Zs * ht) / xjac
            lam_Z = (-Rt * hs + Rs * ht) / xjac

            L   = lam
            L_R = F11 * lam_R + F21 * lam_Z
            L_Z = F12 * lam_R + F22 * lam_Z
            L_P = -(fl(1) * L_R + fl(2) * L_Z)

            if (want_hess) then
              hss = Hss_l(kv, io) * sz
              hst = Hst_l(kv, io) * sz
              htt = Htt_l(kv, io) * sz

              lam_RR = (hss*Zt**2 - 2.d0*hst*Zs*Zt + htt*Zs**2 &
                      + hs*(Zst*Zt - Ztt*Zs) + ht*(Zst*Zs - Zss*Zt)) / (xjac*xjac) &
                      - xjac_x * (hs*Zt - ht*Zs) / (xjac*xjac)
              lam_ZZ = (hss*Rt**2 - 2.d0*hst*Rs*Rt + htt*Rs**2 &
                      + hs*(Rst*Rt - Rtt*Rs) + ht*(Rst*Rs - Rss*Rt)) / (xjac*xjac) &
                      - xjac_y * (-hs*Rt + ht*Rs) / (xjac*xjac)
              lam_RZ = (-hss*Zt*Rt - htt*Rs*Zs + hst*(Zs*Rt + Zt*Rs) &
                      - hs*(Rst*Zt - Rtt*Zs) - ht*(Rst*Zs - Rss*Zt)) / (xjac*xjac) &
                      - xjac_x * (-hs*Rt + ht*Rs) / (xjac*xjac)

              L_RR = lam_RR * F11 * F11 + 2.d0 * lam_RZ * F11 * F21 + lam_ZZ * F21 * F21 &
                   + lam_R * G(1) + lam_Z * G(4)
              L_RZ = lam_RR * F11 * F12 + lam_RZ * (F11 * F22 + F21 * F12) + lam_ZZ * F21 * F22 &
                   + lam_R * G(2) + lam_Z * G(5)
              L_ZZ = lam_RR * F12 * F12 + 2.d0 * lam_RZ * F12 * F22 + lam_ZZ * F22 * F22 &
                   + lam_R * G(3) + lam_Z * G(6)

              L_RP = -(A(1) * L_R + A(3) * L_Z + fl(1) * L_RR + fl(2) * L_RZ)
              L_ZP = -(A(2) * L_R + A(4) * L_Z + fl(1) * L_RZ + fl(2) * L_ZZ)
              L_PP = -(dfl_dp(1) * L_R + dfl_dp(2) * L_Z + fl(1) * L_RP + fl(2) * L_ZP)
            end if
          end if

          loc_v(io) = L * b0
          loc_R(io) = L_R * b0
          loc_Z(io) = L_Z * b0
          loc_P(io) = L_P * b0 + L * bp
          if (want_hess) then
            loc_RR(io) = L_RR * b0
            loc_RZ(io) = L_RZ * b0
            loc_ZZ(io) = L_ZZ * b0
            loc_RP(io) = L_RP * b0 + L_R * bp
            loc_ZP(io) = L_ZP * b0 + L_Z * bp
            loc_PP(io) = L_PP * b0 + 2.d0 * L_P * bp + L * bpp
          end if
        end do

        inode = element_list%element(em)%vertex(kv)
        if (apply_axis_pmat .and. treat_axis .and. (kv == 1 .or. kv == 4) &
            .and. node_list%node(inode)%axis_node) then
          call pfafem_axis_pmat(node_list, element_list, em, kv, pmat)
          do jo = 1, n_degrees
            n_act = n_act + 1
            act_inode(n_act) = inode
            act_io(n_act)    = jo
            act_jb(n_act)    = jb
            b_val(n_act)     = dot_product(loc_v, pmat(:, jo))
            b_dR(n_act)      = dot_product(loc_R, pmat(:, jo))
            b_dZ(n_act)      = dot_product(loc_Z, pmat(:, jo))
            b_dP(n_act)      = dot_product(loc_P, pmat(:, jo))
            if (want_hess) then
              b_dRR(n_act) = dot_product(loc_RR, pmat(:, jo))
              b_dRZ(n_act) = dot_product(loc_RZ, pmat(:, jo))
              b_dZZ(n_act) = dot_product(loc_ZZ, pmat(:, jo))
              b_dRP(n_act) = dot_product(loc_RP, pmat(:, jo))
              b_dZP(n_act) = dot_product(loc_ZP, pmat(:, jo))
              b_dPP(n_act) = dot_product(loc_PP, pmat(:, jo))
            end if
          end do
        else
          do jo = 1, n_degrees
            n_act = n_act + 1
            act_inode(n_act) = inode
            act_io(n_act)    = jo
            act_jb(n_act)    = jb
            b_val(n_act)     = loc_v(jo)
            b_dR(n_act)      = loc_R(jo)
            b_dZ(n_act)      = loc_Z(jo)
            b_dP(n_act)      = loc_P(jo)
            if (want_hess) then
              b_dRR(n_act) = loc_RR(jo)
              b_dRZ(n_act) = loc_RZ(jo)
              b_dZZ(n_act) = loc_ZZ(jo)
              b_dRP(n_act) = loc_RP(jo)
              b_dZP(n_act) = loc_ZP(jo)
              b_dPP(n_act) = loc_PP(jo)
            end if
          end do
        end if
      end do
    end do
  end subroutine pfafem_eval_point_jet

  !=============================================================================
  !> Pre-trace 1st-derivative footprint and inverse-Jacobian pullback matrix M
  !! at an arbitrary physical point (R_q, Z_q, phi_q) so 3D diagnostics can
  !! reuse the exact footprints across steps whenever pfafem_cache_epoch is unchanged.
  !=============================================================================
  subroutine pfafem_trace_point_1st(node_list, element_list, R_q, Z_q, phi_q, &
                                    em_out, sm_out, tm_out, M_out, fl_out)
    use phys_module, only: pfafem_n_substeps, pfafem_openbc
    use mod_basisfunctions, only: basisfunctions
    type(type_node_list),    intent(in)  :: node_list
    type(type_element_list), intent(in)  :: element_list
    real*8,                  intent(in)  :: R_q, Z_q, phi_q
    integer,                 intent(out) :: em_out(4)
    real*8,                  intent(out) :: sm_out(4), tm_out(4), M_out(4, 4), fl_out(2)

    logical :: jet_ok, tr_ok
    integer :: eP, lb, jb, em, em_raw, status_inv
    real*8  :: phi_mod, phi_ref, phi_c, sm, tm, tr12(12)
    real*8  :: A(4), Hjet(6), dfl_dp(2)
    real*8  :: H_l(4, n_degrees), Hs_l(4, n_degrees), Ht_l(4, n_degrees)
    real*8  :: Rr, Zr, Rs, Rt, Zs, Zt, xjac, F11, F12, F21, F22

    em_out = 0
    sm_out = 0.d0
    tm_out = 0.d0
    M_out  = 0.d0

    phi_mod = modulo(phi_q, pfafem_phi_per)
    eP      = min(n_tor - 1, max(0, int(phi_mod / pfafem_dphi)))

    call pfafem_fieldline_jet(R_q, Z_q, phi_mod, fl_out, A, Hjet, dfl_dp, jet_ok)
    if (.not. jet_ok) fl_out = 0.d0

    do lb = 1, 4
      phi_ref = dble(eP - 2 + lb) * pfafem_dphi
      jb      = pfafem_slot(eP, lb, n_tor)
      phi_c   = dble(jb - 1) * pfafem_dphi

      call pfafem_trace_with_hessian(R_q, Z_q, phi_mod, phi_ref, pfafem_n_substeps, tr12, tr_ok)
      if (.not. tr_ok) cycle

      call pfafem_invert_point(node_list, element_list, tr12(1), tr12(2), phi_c, &
                               pfafem_openbc, em, sm, tm, status_inv)
      if (status_inv == 0) cycle
      em_raw = merge(-em, em, status_inv == 3)

      call basisfunctions(sm, tm, H_l, Hs_l, Ht_l)
      call pfafem_eval_elem_geom(node_list, element_list, em, sm, tm, phi_c, &
                                 Rr, Zr, Rs, Rt, Zs, Zt)
      xjac = Rs * Zt - Rt * Zs
      if (em_raw > 0 .and. abs(xjac) < 1.d-22) then
        sm = min(1.d0, sm + 1.d-4)
        call pfafem_eval_elem_geom(node_list, element_list, em, sm, tm, phi_c, &
                                   Rr, Zr, Rs, Rt, Zs, Zt)
        xjac = Rs * Zt - Rt * Zs
      end if

      em_out(lb) = em_raw
      sm_out(lb) = sm
      tm_out(lb) = tm
      if (em_raw > 0 .and. abs(xjac) >= 1.d-22) then
        F11 = tr12(3); F12 = tr12(4); F21 = tr12(5); F22 = tr12(6)
        M_out(1, lb) = ( F11 * Zt - F21 * Rt) / xjac
        M_out(2, lb) = (-F11 * Zs + F21 * Rs) / xjac
        M_out(3, lb) = ( F12 * Zt - F22 * Rt) / xjac
        M_out(4, lb) = (-F12 * Zs + F22 * Rs) / xjac
      end if
    end do
  end subroutine pfafem_trace_point_1st

  !=============================================================================
  !> Fast evaluation of raw local PFAFEM basis values and 1st derivatives
  !! (R, Z, phi) from a footprint pre-traced by pfafem_trace_point_1st.
  !=============================================================================
  subroutine pfafem_eval_cached_1st(element_list, phi_q, em_in, sm_in, tm_in, M_in, fl_in, &
                                    n_act, act_inode, act_io, act_jb, b_val, b_dR, b_dZ, b_dP)
    use mod_basisfunctions, only: basisfunctions
    type(type_element_list), intent(in)  :: element_list
    real*8,                  intent(in)  :: phi_q
    integer,                 intent(in)  :: em_in(4)
    real*8,                  intent(in)  :: sm_in(4), tm_in(4), M_in(4, 4), fl_in(2)
    integer,                 intent(out) :: n_act
    integer,                 intent(out) :: act_inode(64), act_io(64), act_jb(64)
    real*8,                  intent(out) :: b_val(64), b_dR(64), b_dZ(64), b_dP(64)

    integer :: eP, lb, jb, em_raw, em, kv, io, inode
    real*8  :: phi_mod, tau, nb(4), dnb(4), ddnb(4), b0, bp
    real*8  :: H_l(4, n_degrees), Hs_l(4, n_degrees), Ht_l(4, n_degrees)
    real*8  :: sz, hs, ht, L, L_R, L_Z, L_P

    n_act = 0
    phi_mod = modulo(phi_q, pfafem_phi_per)
    eP      = min(n_tor - 1, max(0, int(phi_mod / pfafem_dphi)))
    tau     = max(0.d0, min(1.d0, phi_mod / pfafem_dphi - dble(eP)))
    call pfafem_bspline_eval(tau, nb, dnb, ddnb)

    do lb = 1, 4
      em_raw = em_in(lb)
      if (em_raw == 0) cycle
      em = abs(em_raw)
      jb = pfafem_slot(eP, lb, n_tor)
      b0 = nb(lb)
      bp = dnb(lb) / pfafem_dphi

      call basisfunctions(sm_in(lb), tm_in(lb), H_l, Hs_l, Ht_l)

      do kv = 1, n_vertex_max
        inode = element_list%element(em)%vertex(kv)
        do io = 1, n_degrees
          sz = element_list%element(em)%size(kv, io)
          L  = H_l(kv, io) * sz
          if (em_raw < 0) then
            L_R = 0.d0; L_Z = 0.d0; L_P = 0.d0
          else
            hs  = Hs_l(kv, io) * sz
            ht  = Ht_l(kv, io) * sz
            L_R = M_in(1, lb) * hs + M_in(2, lb) * ht
            L_Z = M_in(3, lb) * hs + M_in(4, lb) * ht
            L_P = -(fl_in(1) * L_R + fl_in(2) * L_Z)
          end if
          n_act = n_act + 1
          act_inode(n_act) = inode
          act_io(n_act)    = io
          act_jb(n_act)    = jb
          b_val(n_act)     = L * b0
          b_dR(n_act)      = L_R * b0
          b_dZ(n_act)      = L_Z * b0
          b_dP(n_act)      = L_P * b0 + L * bp
        end do
      end do
    end do
  end subroutine pfafem_eval_cached_1st

  !=============================================================================
  !> Broadcast PFAFEM restart state (B-field spline grid + cross-discretization
  !! staging arrays read on rank 0 by import_restart) to all MPI ranks.
  !=============================================================================
  subroutine pfafem_bcast_restart_state(my_id, node_list)
    integer,              intent(in) :: my_id
    type(type_node_list), intent(in) :: node_list

    integer :: mpi_ierr, nr, nz, nphi, n_nodes
    logical :: has_del
#ifdef STELLARATOR_MODEL
    logical :: has_jsrc
#endif

    n_nodes = node_list%n_nodes

    call MPI_Bcast(pfafem_bgrid_valid, 1, MPI_LOGICAL, 0, MPI_COMM_WORLD, mpi_ierr)
    if (pfafem_bgrid_valid) then
      call MPI_Bcast(pfafem_bgrid_dims, 3, MPI_INTEGER, 0, MPI_COMM_WORLD, mpi_ierr)
      call MPI_Bcast(pfafem_bgrid_bounds, 5, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, mpi_ierr)
      nr   = pfafem_bgrid_dims(1)
      nz   = pfafem_bgrid_dims(2)
      nphi = max(1, pfafem_bgrid_dims(3))
      if (my_id /= 0) then
        if (allocated(pfafem_bgrid_coefs)) deallocate(pfafem_bgrid_coefs)
        allocate(pfafem_bgrid_coefs(2, nr + 2, nz + 2, nphi))
      end if
      call MPI_Bcast(pfafem_bgrid_coefs, size(pfafem_bgrid_coefs), MPI_DOUBLE_PRECISION, &
                     0, MPI_COMM_WORLD, mpi_ierr)
    end if

    call MPI_Bcast(pfafem_restart_needs_projection, 1, MPI_LOGICAL, 0, MPI_COMM_WORLD, mpi_ierr)
    if (pfafem_restart_needs_projection) then
      call MPI_Bcast(pfafem_restart_src_type, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, mpi_ierr)
      call MPI_Bcast(pfafem_restart_src_ntor, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, mpi_ierr)
      if (my_id == 0) has_del = allocated(pfafem_restart_deltas)
      call MPI_Bcast(has_del, 1, MPI_LOGICAL, 0, MPI_COMM_WORLD, mpi_ierr)
      if (my_id /= 0) then
        if (allocated(pfafem_restart_values)) deallocate(pfafem_restart_values)
        allocate(pfafem_restart_values(pfafem_restart_src_ntor, n_degrees, n_var, n_nodes))
        if (allocated(pfafem_restart_deltas)) deallocate(pfafem_restart_deltas)
        if (has_del) allocate(pfafem_restart_deltas(pfafem_restart_src_ntor, n_degrees, n_var, n_nodes))
      end if
      call MPI_Bcast(pfafem_restart_values, size(pfafem_restart_values), MPI_DOUBLE_PRECISION, &
                     0, MPI_COMM_WORLD, mpi_ierr)
      if (has_del) then
        call MPI_Bcast(pfafem_restart_deltas, size(pfafem_restart_deltas), MPI_DOUBLE_PRECISION, &
                       0, MPI_COMM_WORLD, mpi_ierr)
      end if
#ifdef STELLARATOR_MODEL
      if (my_id == 0) has_jsrc = allocated(pfafem_restart_jsrc)
      call MPI_Bcast(has_jsrc, 1, MPI_LOGICAL, 0, MPI_COMM_WORLD, mpi_ierr)
      if (my_id /= 0) then
        if (allocated(pfafem_restart_jsrc)) deallocate(pfafem_restart_jsrc)
        if (has_jsrc) allocate(pfafem_restart_jsrc(pfafem_restart_src_ntor, n_degrees, n_nodes))
      end if
      if (has_jsrc) then
        call MPI_Bcast(pfafem_restart_jsrc, size(pfafem_restart_jsrc), MPI_DOUBLE_PRECISION, &
                       0, MPI_COMM_WORLD, mpi_ierr)
      end if
#endif
    end if
  end subroutine pfafem_bcast_restart_state

  !=============================================================================
  !> Stage the current node_list%node(:)%values (in standard JOREK Fourier basis,
  !! e.g. right after equilibrium + initial_conditions when .not. restart) into
  !! pfafem_restart_values so pfafem_build_cache + pfafem_project_restart project
  !! the initial equilibrium and perturbations onto the PFAFEM basis.
  !=============================================================================
  subroutine pfafem_stage_fourier_from_nodes(node_list)
#ifdef STELLARATOR_MODEL
    use phys_module, only: current_prof_initialized
#endif
    type(type_node_list), intent(in) :: node_list
    integer :: i

    if (allocated(pfafem_restart_values)) deallocate(pfafem_restart_values)
    if (allocated(pfafem_restart_deltas)) deallocate(pfafem_restart_deltas)
    allocate(pfafem_restart_values(n_tor, n_degrees, n_var, node_list%n_nodes))
    do i = 1, node_list%n_nodes
      pfafem_restart_values(:, :, :, i) = node_list%node(i)%values(1:n_tor, 1:n_degrees, 1:n_var)
    end do
#ifdef STELLARATOR_MODEL
    if (allocated(pfafem_restart_jsrc)) deallocate(pfafem_restart_jsrc)
    if (current_prof_initialized) then
      allocate(pfafem_restart_jsrc(n_tor, n_degrees, node_list%n_nodes))
      do i = 1, node_list%n_nodes
        pfafem_restart_jsrc(:, :, i) = node_list%node(i)%j_source(1:n_tor, 1:n_degrees)
      end do
    end if
#endif
    pfafem_restart_src_type = 1
    pfafem_restart_src_ntor = n_tor
    pfafem_restart_needs_projection = .true.
  end subroutine pfafem_stage_fourier_from_nodes

  !=============================================================================
  !> Helper: check whether side iside (1..4) of element ielm is an exterior boundary
  !! edge and map 1D Gauss coordinate u_s in [0,1] to element (s_b, t_b).
  !=============================================================================
  logical function get_elem_bnd_side_st(node_list, element_list, ielm, iside, u_s, s_b, t_b)
    type(type_node_list),    intent(in)  :: node_list
    type(type_element_list), intent(in)  :: element_list
    integer,                 intent(in)  :: ielm, iside
    real*8,                  intent(in)  :: u_s
    real*8,                  intent(out) :: s_b, t_b
    integer :: iv1, iv2, n1, n2

    s_b = 0.d0; t_b = 0.d0
    get_elem_bnd_side_st = .false.
    if (element_list%element(ielm)%neighbours(iside) /= 0) return
    iv1 = iside
    iv2 = mod(iside, 4) + 1
    n1  = element_list%element(ielm)%vertex(iv1)
    n2  = element_list%element(ielm)%vertex(iv2)
    if (node_list%node(n1)%boundary == 0 .or. node_list%node(n2)%boundary == 0) return

    select case (iside)
    case (1); s_b = u_s;  t_b = 0.d0
    case (2); s_b = 1.d0; t_b = u_s
    case (3); s_b = u_s;  t_b = 1.d0
    case (4); s_b = 0.d0; t_b = u_s
    end select
    get_elem_bnd_side_st = .true.
  end function get_elem_bnd_side_st

  !=============================================================================
  !> Project restart fields (Fourier -> PFAFEM or PFAFEM(n_tor_old) -> PFAFEM(n_tor))
  !! onto the active PFAFEM basis via distributed PETSc CG + Block-Jacobi mass solve.
  !=============================================================================
  subroutine pfafem_project_restart(my_id, node_list, element_list)
    use phys_module, only: pfafem_values_active
    use mod_basisfunctions, only: basisfunctions
    integer,                 intent(in)    :: my_id
    type(type_node_list),    intent(inout) :: node_list
    type(type_element_list), intent(in)    :: element_list

    integer :: n_loc, ielm, iel_loc, eP, iep, iq, ms, mt, iside, iv, io, in_t, ivar, inode
    real*8  :: phi_q, s_q, t_q, s_b, t_b, R_q, Z_q, Rs, Rt, Zs, Zt, sz
    real*8  :: H_l(4, n_degrees), Hs_l(4, n_degrees), Ht_l(4, n_degrees)
    real*8, allocatable :: val_qp(:,:,:,:,:,:), del_qp(:,:,:,:,:,:)
    real*8, allocatable :: val_bnd(:,:,:,:,:,:), del_bnd(:,:,:,:,:,:)
    real*8, allocatable :: jsrc_qp(:,:,:,:,:), jsrc_bnd(:,:,:,:,:)
    real*8, allocatable :: hz_f(:)
    logical :: has_deltas, has_jsrc

    if (.not. pfafem_restart_needs_projection) return
    if (.not. pfafem_ready) return

    n_loc = max(0, pfafem_el_hi - pfafem_el_lo + 1)
    has_deltas = allocated(pfafem_restart_deltas)
    has_jsrc   = .false.
#ifdef STELLARATOR_MODEL
    has_jsrc   = allocated(pfafem_restart_jsrc)
#endif

    allocate(val_qp(n_var, n_gauss, pfafem_ng_pol, pfafem_ng_tor, n_tor, max(1, n_loc)))
    allocate(val_bnd(n_var, n_gauss, 4, pfafem_ng_tor, n_tor, max(1, n_loc)))
    val_qp  = 0.d0
    val_bnd = 0.d0
    if (has_deltas) then
      allocate(del_qp(n_var, n_gauss, pfafem_ng_pol, pfafem_ng_tor, n_tor, max(1, n_loc)))
      allocate(del_bnd(n_var, n_gauss, 4, pfafem_ng_tor, n_tor, max(1, n_loc)))
      del_qp  = 0.d0
      del_bnd = 0.d0
    end if
    if (has_jsrc) then
      allocate(jsrc_qp(n_gauss, pfafem_ng_pol, pfafem_ng_tor, n_tor, max(1, n_loc)))
      allocate(jsrc_bnd(n_gauss, 4, pfafem_ng_tor, n_tor, max(1, n_loc)))
      jsrc_qp  = 0.d0
      jsrc_bnd = 0.d0
    end if

    if (pfafem_restart_src_type == 1) then
      ! Source is standard JOREK Fourier harmonics (1..pfafem_restart_src_ntor)
      !$omp parallel do schedule(dynamic, 4) &
      !$omp private(ielm, iel_loc, eP, iep, iq, mt, ms, iside, phi_q, s_q, t_q, s_b, t_b, &
      !$omp         H_l, Hs_l, Ht_l, hz_f, iv, inode, io, sz, ivar, in_t)
      do ielm = pfafem_el_lo, pfafem_el_hi
        iel_loc = ielm - pfafem_el_lo + 1
        allocate(hz_f(pfafem_restart_src_ntor))
        do eP = 0, n_tor - 1
          iep = merge(1, eP + 1, pfafem_cache_axisym)
          do iq = 1, pfafem_ng_tor
            phi_q = (dble(eP) + pfafem_tau_q(iq)) * pfafem_dphi
            call eval_phys_fourier_hz(pfafem_restart_src_ntor, phi_q, hz_f)
            do ms = 1, n_gauss
              s_q = xgauss(ms)
              do mt = 1, pfafem_ng_pol
                t_q = pfafem_t_pol(ms, mt, iq, iep, iel_loc)
                call basisfunctions(s_q, t_q, H_l, Hs_l, Ht_l)
                do iv = 1, n_vertex_max
                  inode = element_list%element(ielm)%vertex(iv)
                  do io = 1, n_degrees
                    sz = element_list%element(ielm)%size(iv, io) * H_l(iv, io)
                    do ivar = 1, n_var
                      do in_t = 1, pfafem_restart_src_ntor
                        val_qp(ivar, ms, mt, iq, eP + 1, iel_loc) = &
                          val_qp(ivar, ms, mt, iq, eP + 1, iel_loc) + &
                          pfafem_restart_values(in_t, io, ivar, inode) * sz * hz_f(in_t)
                        if (has_deltas) then
                          del_qp(ivar, ms, mt, iq, eP + 1, iel_loc) = &
                            del_qp(ivar, ms, mt, iq, eP + 1, iel_loc) + &
                            pfafem_restart_deltas(in_t, io, ivar, inode) * sz * hz_f(in_t)
                        end if
                      end do
                    end do
#ifdef STELLARATOR_MODEL
                    if (has_jsrc) then
                      do in_t = 1, pfafem_restart_src_ntor
                        jsrc_qp(ms, mt, iq, eP + 1, iel_loc) = &
                          jsrc_qp(ms, mt, iq, eP + 1, iel_loc) + &
                          pfafem_restart_jsrc(in_t, io, inode) * sz * hz_f(in_t)
                      end do
                    end if
#endif
                  end do
                end do
              end do
            end do

            if (elem_is_bnd(ielm)) then
              do iside = 1, 4
                do ms = 1, n_gauss
                  if (.not. get_elem_bnd_side_st(node_list, element_list, ielm, iside, xgauss(ms), s_b, t_b)) cycle
                  call basisfunctions(s_b, t_b, H_l, Hs_l, Ht_l)
                  do iv = 1, n_vertex_max
                    inode = element_list%element(ielm)%vertex(iv)
                    do io = 1, n_degrees
                      sz = element_list%element(ielm)%size(iv, io) * H_l(iv, io)
                      do ivar = 1, n_var
                        do in_t = 1, pfafem_restart_src_ntor
                          val_bnd(ivar, ms, iside, iq, eP + 1, iel_loc) = &
                            val_bnd(ivar, ms, iside, iq, eP + 1, iel_loc) + &
                            pfafem_restart_values(in_t, io, ivar, inode) * sz * hz_f(in_t)
                          if (has_deltas) then
                            del_bnd(ivar, ms, iside, iq, eP + 1, iel_loc) = &
                              del_bnd(ivar, ms, iside, iq, eP + 1, iel_loc) + &
                              pfafem_restart_deltas(in_t, io, ivar, inode) * sz * hz_f(in_t)
                          end if
                        end do
                      end do
#ifdef STELLARATOR_MODEL
                      if (has_jsrc) then
                        do in_t = 1, pfafem_restart_src_ntor
                          jsrc_bnd(ms, iside, iq, eP + 1, iel_loc) = &
                            jsrc_bnd(ms, iside, iq, eP + 1, iel_loc) + &
                            pfafem_restart_jsrc(in_t, io, inode) * sz * hz_f(in_t)
                        end do
                      end if
#endif
                    end do
                  end do
                end do
              end do
            end if
          end do
        end do
        deallocate(hz_f)
      end do
      !$omp end parallel do
    else if (pfafem_restart_src_type == 2) then
      ! Source is PFAFEM with different n_tor = pfafem_restart_src_ntor
      call eval_pfafem_src_ntor_qp(node_list, element_list, has_deltas, has_jsrc, &
                                   val_qp, val_bnd, del_qp, del_bnd, jsrc_qp, jsrc_bnd)
    end if

    call solve_pfafem_l2_projection(my_id, node_list, element_list, has_deltas, has_jsrc, &
                                    val_qp, val_bnd, del_qp, del_bnd, jsrc_qp, jsrc_bnd)

    deallocate(val_qp, val_bnd)
    if (allocated(del_qp)) deallocate(del_qp, del_bnd)
    if (allocated(jsrc_qp)) deallocate(jsrc_qp, jsrc_bnd)
    if (allocated(pfafem_restart_values)) deallocate(pfafem_restart_values)
    if (allocated(pfafem_restart_deltas)) deallocate(pfafem_restart_deltas)
#ifdef STELLARATOR_MODEL
    if (allocated(pfafem_restart_jsrc)) deallocate(pfafem_restart_jsrc)
#endif
    pfafem_restart_needs_projection = .false.
    pfafem_restart_src_type = 0
    pfafem_restart_src_ntor = 0
    pfafem_values_active = .true.

    if (my_id == 0) then
      write(*,'(A,I4)') ' [PFAFEM] Completed restart field projection onto PFAFEM n_tor=', n_tor
    end if
  end subroutine pfafem_project_restart

  !=============================================================================
  !> Evaluate PFAFEM restart source with different n_tor_src at current QP set.
  !=============================================================================
  subroutine eval_pfafem_src_ntor_qp(node_list, element_list, has_deltas, has_jsrc, &
                                     val_qp, val_bnd, del_qp, del_bnd, jsrc_qp, jsrc_bnd)
    use phys_module, only: pfafem_n_substeps, pfafem_openbc
    use mod_basisfunctions, only: basisfunctions
    type(type_node_list),    intent(in)    :: node_list
    type(type_element_list), intent(in)    :: element_list
    logical,                 intent(in)    :: has_deltas, has_jsrc
    real*8,                  intent(inout) :: val_qp(n_var, n_gauss, pfafem_ng_pol, pfafem_ng_tor, n_tor, *)
    real*8,                  intent(inout) :: val_bnd(n_var, n_gauss, 4, pfafem_ng_tor, n_tor, *)
    real*8,                  intent(inout), optional :: del_qp(n_var, n_gauss, pfafem_ng_pol, pfafem_ng_tor, n_tor, *)
    real*8,                  intent(inout), optional :: del_bnd(n_var, n_gauss, 4, pfafem_ng_tor, n_tor, *)
    real*8,                  intent(inout), optional :: jsrc_qp(n_gauss, pfafem_ng_pol, pfafem_ng_tor, n_tor, *)
    real*8,                  intent(inout), optional :: jsrc_bnd(n_gauss, 4, pfafem_ng_tor, n_tor, *)

    integer :: ielm, iel_loc, eP, iep, iq, ms, mt, iside, n_src, eP_s, lb, jb_s, em, status_inv, iv, io, inode, ivar
    real*8  :: dphi_s, phi_q, t_q, phin, tau_s, nb(4), dnb(4), ddnb(4), phi_ref, phi_c, s_b, t_b
    real*8  :: R_q, Z_q, Rs, Rt, Zs, Zt, tr12(12), sm, tm, H_l(4, n_degrees), Hs_l(4, n_degrees), Ht_l(4, n_degrees), w
    logical :: tr_ok

    n_src  = pfafem_restart_src_ntor
    dphi_s = pfafem_phi_per / dble(n_src)

    do ielm = pfafem_el_lo, pfafem_el_hi
      iel_loc = ielm - pfafem_el_lo + 1
      do eP = 0, n_tor - 1
        iep = merge(1, eP + 1, pfafem_cache_axisym)
        do iq = 1, pfafem_ng_tor
          phi_q = (dble(eP) + pfafem_tau_q(iq)) * pfafem_dphi
          phin  = modulo(phi_q, pfafem_phi_per)
          eP_s  = min(n_src - 1, max(0, int(phin / dphi_s)))
          tau_s = min(1.d0, max(0.d0, phin / dphi_s - dble(eP_s)))
          call pfafem_bspline_eval(tau_s, nb, dnb, ddnb)
          do ms = 1, n_gauss
            do mt = 1, pfafem_ng_pol
              t_q = pfafem_t_pol(ms, mt, iq, iep, iel_loc)
              call pfafem_eval_elem_geom(node_list, element_list, ielm, xgauss(ms), t_q, phi_q, &
                                         R_q, Z_q, Rs, Rt, Zs, Zt)
              do lb = 1, 4
                jb_s = pfafem_slot(eP_s, lb, n_src)
                phi_ref = dble(eP_s - 2 + lb) * dphi_s
                phi_c   = dble(jb_s - 1) * dphi_s
                call pfafem_trace_with_hessian(R_q, Z_q, phin, phi_ref, pfafem_n_substeps, tr12, tr_ok)
                if (.not. tr_ok) cycle
                call pfafem_invert_point(node_list, element_list, tr12(1), tr12(2), phi_c, &
                                         pfafem_openbc, em, sm, tm, status_inv)
                if (status_inv == 0) cycle
                call basisfunctions(sm, tm, H_l, Hs_l, Ht_l)
                do iv = 1, n_vertex_max
                  inode = element_list%element(em)%vertex(iv)
                  do io = 1, n_degrees
                    w = element_list%element(em)%size(iv, io) * H_l(iv, io) * nb(lb)
                    do ivar = 1, n_var
                      val_qp(ivar, ms, mt, iq, eP + 1, iel_loc) = &
                        val_qp(ivar, ms, mt, iq, eP + 1, iel_loc) + pfafem_restart_values(jb_s, io, ivar, inode) * w
                      if (has_deltas .and. present(del_qp)) then
                        del_qp(ivar, ms, mt, iq, eP + 1, iel_loc) = &
                          del_qp(ivar, ms, mt, iq, eP + 1, iel_loc) + pfafem_restart_deltas(jb_s, io, ivar, inode) * w
                      end if
                    end do
#ifdef STELLARATOR_MODEL
                    if (has_jsrc .and. present(jsrc_qp)) then
                      jsrc_qp(ms, mt, iq, eP + 1, iel_loc) = &
                        jsrc_qp(ms, mt, iq, eP + 1, iel_loc) + pfafem_restart_jsrc(jb_s, io, inode) * w
                    end if
#endif
                  end do
                end do
              end do
            end do
          end do

          if (elem_is_bnd(ielm)) then
            do iside = 1, 4
              do ms = 1, n_gauss
                if (.not. get_elem_bnd_side_st(node_list, element_list, ielm, iside, xgauss(ms), s_b, t_b)) cycle
                call pfafem_eval_elem_geom(node_list, element_list, ielm, s_b, t_b, phi_q, &
                                           R_q, Z_q, Rs, Rt, Zs, Zt)
                do lb = 1, 4
                  jb_s = pfafem_slot(eP_s, lb, n_src)
                  phi_ref = dble(eP_s - 2 + lb) * dphi_s
                  phi_c   = dble(jb_s - 1) * dphi_s
                  call pfafem_trace_with_hessian(R_q, Z_q, phin, phi_ref, pfafem_n_substeps, tr12, tr_ok)
                  if (.not. tr_ok) cycle
                  call pfafem_invert_point(node_list, element_list, tr12(1), tr12(2), phi_c, &
                                           pfafem_openbc, em, sm, tm, status_inv)
                  if (status_inv == 0) cycle
                  call basisfunctions(sm, tm, H_l, Hs_l, Ht_l)
                  do iv = 1, n_vertex_max
                    inode = element_list%element(em)%vertex(iv)
                    do io = 1, n_degrees
                      w = element_list%element(em)%size(iv, io) * H_l(iv, io) * nb(lb)
                      do ivar = 1, n_var
                        val_bnd(ivar, ms, iside, iq, eP + 1, iel_loc) = &
                          val_bnd(ivar, ms, iside, iq, eP + 1, iel_loc) + pfafem_restart_values(jb_s, io, ivar, inode) * w
                        if (has_deltas .and. present(del_bnd)) then
                          del_bnd(ivar, ms, iside, iq, eP + 1, iel_loc) = &
                            del_bnd(ivar, ms, iside, iq, eP + 1, iel_loc) + pfafem_restart_deltas(jb_s, io, ivar, inode) * w
                        end if
                      end do
#ifdef STELLARATOR_MODEL
                      if (has_jsrc .and. present(jsrc_bnd)) then
                        jsrc_bnd(ms, iside, iq, eP + 1, iel_loc) = &
                          jsrc_bnd(ms, iside, iq, eP + 1, iel_loc) + pfafem_restart_jsrc(jb_s, io, inode) * w
                      end if
#endif
                    end do
                  end do
                end do
              end do
            end do
          end if
        end do
      end do
    end do
  end subroutine eval_pfafem_src_ntor_qp

  !=============================================================================
  !> Dynamically re-trace from the evolving magnetic field B(t) and L2-project
  !! both node%values and node%deltas from the old traced basis onto the new basis.
  !=============================================================================
  subroutine pfafem_retrace_and_project(my_id, node_list, element_list)
#ifdef STELLARATOR_MODEL
    use phys_module, only: keep_current_prof, current_prof_initialized
#endif
    integer,                 intent(in)    :: my_id
    type(type_node_list),    intent(inout) :: node_list
    type(type_element_list), intent(in)    :: element_list

    integer :: n_loc, i, old_dims(3), new_dims(3)
    real*8  :: old_bounds(5), new_bounds(5)
    real*8, allocatable :: old_coefs(:,:,:,:), new_coefs(:,:,:,:)
    real*8, allocatable :: val_qp(:,:,:,:,:,:), del_qp(:,:,:,:,:,:)
    real*8, allocatable :: val_bnd(:,:,:,:,:,:), del_bnd(:,:,:,:,:,:)
    real*8, allocatable :: jsrc_qp(:,:,:,:,:), jsrc_bnd(:,:,:,:,:)
    logical :: do_jsrc

    if (.not. pfafem_ready) then
      call pfafem_build_cache(my_id, node_list, element_list, .true., .false.)
      return
    end if

    do_jsrc = .false.
#ifdef STELLARATOR_MODEL
    do_jsrc = (keep_current_prof .or. current_prof_initialized)
#endif

    ! 1. Snapshot old B-field spline and current nodal coefficients before rebuilding cache
    old_dims   = pfafem_bgrid_dims
    old_bounds = pfafem_bgrid_bounds
    allocate(old_coefs(size(pfafem_bgrid_coefs, 1), size(pfafem_bgrid_coefs, 2), &
                       size(pfafem_bgrid_coefs, 3), size(pfafem_bgrid_coefs, 4)))
    old_coefs = pfafem_bgrid_coefs

    if (allocated(pfafem_restart_values)) deallocate(pfafem_restart_values)
    if (allocated(pfafem_restart_deltas)) deallocate(pfafem_restart_deltas)
    allocate(pfafem_restart_values(n_tor, n_degrees, n_var, node_list%n_nodes))
    allocate(pfafem_restart_deltas(n_tor, n_degrees, n_var, node_list%n_nodes))
    do i = 1, node_list%n_nodes
      pfafem_restart_values(:, :, :, i) = node_list%node(i)%values(1:n_tor, 1:n_degrees, 1:n_var)
      pfafem_restart_deltas(:, :, :, i) = node_list%node(i)%deltas(1:n_tor, 1:n_degrees, 1:n_var)
    end do
#ifdef STELLARATOR_MODEL
    if (allocated(pfafem_restart_jsrc)) deallocate(pfafem_restart_jsrc)
    if (do_jsrc) then
      allocate(pfafem_restart_jsrc(n_tor, n_degrees, node_list%n_nodes))
      do i = 1, node_list%n_nodes
        pfafem_restart_jsrc(:, :, i) = node_list%node(i)%j_source(1:n_tor, 1:n_degrees)
      end do
    end if
#endif
    pfafem_restart_src_ntor = n_tor

    ! 2. Fit new B(t) spline from old cache and rebuild trace cache + seam-split rule along new B(t)
    call pfafem_build_cache(my_id, node_list, element_list, .true., .true.)

    ! 3. Evaluate old PFAFEM state at the NEW seam-split quadrature points, then restore new B-spline
    n_loc = max(0, pfafem_el_hi - pfafem_el_lo + 1)
    allocate(val_qp(n_var, n_gauss, pfafem_ng_pol, pfafem_ng_tor, n_tor, max(1, n_loc)))
    allocate(del_qp(n_var, n_gauss, pfafem_ng_pol, pfafem_ng_tor, n_tor, max(1, n_loc)))
    allocate(val_bnd(n_var, n_gauss, 4, pfafem_ng_tor, n_tor, max(1, n_loc)))
    allocate(del_bnd(n_var, n_gauss, 4, pfafem_ng_tor, n_tor, max(1, n_loc)))
    val_qp  = 0.d0; del_qp  = 0.d0
    val_bnd = 0.d0; del_bnd = 0.d0
    if (do_jsrc) then
      allocate(jsrc_qp(n_gauss, pfafem_ng_pol, pfafem_ng_tor, n_tor, max(1, n_loc)))
      allocate(jsrc_bnd(n_gauss, 4, pfafem_ng_tor, n_tor, max(1, n_loc)))
      jsrc_qp  = 0.d0
      jsrc_bnd = 0.d0
    end if

    new_dims   = pfafem_bgrid_dims
    new_bounds = pfafem_bgrid_bounds
    call move_alloc(pfafem_bgrid_coefs, new_coefs)
    pfafem_bgrid_dims   = old_dims
    pfafem_bgrid_bounds = old_bounds
    call move_alloc(old_coefs, pfafem_bgrid_coefs)

    call eval_pfafem_src_ntor_qp(node_list, element_list, .true., do_jsrc, &
                                 val_qp, val_bnd, del_qp, del_bnd, jsrc_qp, jsrc_bnd)

    deallocate(pfafem_bgrid_coefs)
    pfafem_bgrid_dims   = new_dims
    pfafem_bgrid_bounds = new_bounds
    call move_alloc(new_coefs, pfafem_bgrid_coefs)

    deallocate(pfafem_restart_values, pfafem_restart_deltas)
#ifdef STELLARATOR_MODEL
    if (allocated(pfafem_restart_jsrc)) deallocate(pfafem_restart_jsrc)
#endif
    pfafem_restart_src_ntor = 0

    ! 4. Project sampled val_qp/val_bnd and del_qp/del_bnd onto the NEW PFAFEM basis
    call solve_pfafem_l2_projection(my_id, node_list, element_list, .true., do_jsrc, &
                                    val_qp, val_bnd, del_qp, del_bnd, jsrc_qp, jsrc_bnd)

    deallocate(val_qp, del_qp, val_bnd, del_bnd)
    if (allocated(jsrc_qp)) deallocate(jsrc_qp, jsrc_bnd)
  end subroutine pfafem_retrace_and_project

  !=============================================================================
  !> Assemble scalar PFAFEM mass matrix M_pf (including consistent boundary
  !! surface trace weight) and solve M_pf * u = b for all n_var fields (plus
  !! optional deltas and stellarator j_source) via PETSc KSPCG + PCBJACOBI.
  !=============================================================================
  subroutine solve_pfafem_l2_projection(my_id, node_list, element_list, do_deltas, do_jsrc, &
                                        val_qp, val_bnd, del_qp, del_bnd, jsrc_qp, jsrc_bnd)
    use phys_module, only: treat_axis
    use mod_axis_treatment, only: new_to_old_dofs_on_the_axis
    integer,                 intent(in)    :: my_id
    type(type_node_list),    intent(inout) :: node_list
    type(type_element_list), intent(in)    :: element_list
    logical,                 intent(in)    :: do_deltas, do_jsrc
    real*8,                  intent(in)    :: val_qp(n_var, n_gauss, pfafem_ng_pol, pfafem_ng_tor, n_tor, *)
    real*8,                  intent(in)    :: val_bnd(n_var, n_gauss, 4, pfafem_ng_tor, n_tor, *)
    real*8,                  intent(in), optional :: del_qp(n_var, n_gauss, pfafem_ng_pol, pfafem_ng_tor, n_tor, *)
    real*8,                  intent(in), optional :: del_bnd(n_var, n_gauss, 4, pfafem_ng_tor, n_tor, *)
    real*8,                  intent(in), optional :: jsrc_qp(n_gauss, pfafem_ng_pol, pfafem_ng_tor, n_tor, *)
    real*8,                  intent(in), optional :: jsrc_bnd(n_gauss, 4, pfafem_ng_tor, n_tor, *)

#ifdef USE_PETSC
    Mat :: M_pf
    Vec :: vec_b, vec_x, vec_diag
    KSP :: ksp_m
    PC  :: pc_m
    PetscErrorCode :: ierr
    PetscInt :: n_glob, n_loc_rows, ind_lo, ind_hi, ind_tot, bs, r_idx, c_idx
    PetscInt :: r_rows(n_tor), c_cols(n_tor), its_m
    PetscScalar, pointer :: p_arr(:)
    PetscReal :: b_nrm, r_nrm
    integer :: n_cpu, mpi_ierr, i, j, in_t, ivar, ielm, iel_loc, eP, iep, iq, ms, mt, iside
    integer :: n_act, a, b, u1, u2, io1, io2, jb1, jb2, ind1, ind2, inode, ivp, n_vp, n_u, n_own
    integer :: act_u(64), act_in(64), act_io(64), act_jb(64), max_its_proj
    real*8  :: phi_q, t_q, w_t, s_b, t_b, R_q, Z_q, Rs, Rt, Zs, Zt, wvol, wsurf, max_rel_res_proj
    real*8  :: b_v(64), b_R(64), b_Z(64), b_P(64), dmax
    real*8  :: new_dofs(4), old_dofs(4), blk_t(n_tor, n_tor)
    real*8, parameter :: alpha_bnd = 50.d0
    integer, allocatable :: d_nnz(:), o_nnz(:)
    PetscInt, allocatable :: d_nnz_aij(:), o_nnz_aij(:)
    integer, allocatable :: seen_list(:)
    integer :: n_seen, k
    real*8, allocatable :: rhs_v(:,:), rhs_d(:,:), rhs_j(:), sol_all(:)
    real*8, allocatable :: el_m(:,:,:,:,:), el_rv(:,:,:,:), el_rd(:,:,:,:), el_rj(:,:,:)

    call MPI_Comm_size(MPI_COMM_WORLD, n_cpu, mpi_ierr)
    ind_tot = 0
    do i = 1, node_list%n_nodes
      ind_tot = max(ind_tot, maxval(node_list%node(i)%index))
    end do
    if (ind_tot <= 0) return

    ind_lo = (my_id * ind_tot) / n_cpu + 1
    ind_hi = ((my_id + 1) * ind_tot) / n_cpu
    bs         = n_tor
    n_own      = max(0, int(ind_hi - ind_lo + 1))
    n_loc_rows = n_own * bs
    n_glob     = ind_tot * bs

    ! Exact scalar AIJ preallocation for M_pf (enables direct MUMPS LU solve)
    allocate(d_nnz(max(1, n_own)), o_nnz(max(1, n_own)))
    d_nnz = 1; o_nnz = 0
    if (n_own > 0) then
      call count_mass_block_nnz(node_list, element_list, int(ind_lo), int(ind_hi), int(ind_tot), d_nnz, o_nnz)
    end if
    allocate(d_nnz_aij(max(1, int(n_loc_rows))), o_nnz_aij(max(1, int(n_loc_rows))))
    d_nnz_aij = 1; o_nnz_aij = 0
    do i = 1, n_own
      do j = 1, int(bs)
        d_nnz_aij((i - 1) * int(bs) + j) = min(n_loc_rows, int(d_nnz(i) * int(bs), kind=kind(n_loc_rows)))
        o_nnz_aij((i - 1) * int(bs) + j) = min(n_glob - n_loc_rows, int(o_nnz(i) * int(bs), kind=kind(n_loc_rows)))
      end do
    end do
    deallocate(d_nnz, o_nnz)

    PetscCallA(MatCreate(MPI_COMM_WORLD, M_pf, ierr))
    PetscCallA(MatSetSizes(M_pf, n_loc_rows, n_loc_rows, n_glob, n_glob, ierr))
    PetscCallA(MatSetType(M_pf, MATAIJ, ierr))
    PetscCallA(MatSeqAIJSetPreallocation(M_pf, 0, d_nnz_aij, ierr))
    PetscCallA(MatMPIAIJSetPreallocation(M_pf, 0, d_nnz_aij, 0, o_nnz_aij, ierr))
    PetscCallA(MatSetOption(M_pf, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE, ierr))
    PetscCallA(MatZeroEntries(M_pf, ierr))
    deallocate(d_nnz_aij, o_nnz_aij)

    allocate(rhs_v(n_glob, n_var))
    rhs_v = 0.d0
    if (do_deltas) then
      allocate(rhs_d(n_glob, n_var))
      rhs_d = 0.d0
    end if
    if (do_jsrc) then
      allocate(rhs_j(n_glob))
      rhs_j = 0.d0
    end if

    ! Assemble element mass blocks and RHS load vectors in parallel over disjoint owned elements
    !$omp parallel do schedule(dynamic, 4) default(shared) &
    !$omp   private(ielm, iel_loc, n_vp, n_u, el_m, el_rv, el_rd, el_rj, eP, iep, iq, phi_q, t_q, w_t, mt, ms, &
    !$omp           R_q, Z_q, Rs, Rt, Zs, Zt, wvol, n_act, act_u, act_io, act_jb, &
    !$omp           b_v, b_R, b_Z, b_P, a, b, u1, u2, io1, io2, jb1, jb2, inode, ind1, ind2, &
    !$omp           r_idx, c_idx, ivar, ivp, r_rows, c_cols, blk_t, iside, s_b, t_b, wsurf, act_in, ierr, j)
    do ielm = pfafem_el_lo, pfafem_el_hi
      iel_loc = ielm - pfafem_el_lo + 1
      n_vp = pfafem_cache(ielm)%n_vpairs
      n_u  = pfafem_cache(ielm)%n_unodes
      allocate(el_m(n_tor, n_tor, n_degrees, n_degrees, n_vp))
      allocate(el_rv(n_tor, n_degrees, n_u, n_var))
      el_m  = 0.d0
      el_rv = 0.d0
      if (do_deltas) then
        allocate(el_rd(n_tor, n_degrees, n_u, n_var))
        el_rd = 0.d0
      end if
      if (do_jsrc) then
        allocate(el_rj(n_tor, n_degrees, n_u))
        el_rj = 0.d0
      end if

      do eP = 0, n_tor - 1
        iep = merge(1, eP + 1, pfafem_cache_axisym)
        do iq = 1, pfafem_ng_tor
          phi_q = (dble(eP) + pfafem_tau_q(iq)) * pfafem_dphi
          do ms = 1, n_gauss
            do mt = 1, pfafem_ng_pol
              t_q = pfafem_t_pol(ms, mt, iq, iep, iel_loc)
              w_t = pfafem_w_pol(ms, mt, iq, iep, iel_loc)
              call pfafem_eval_elem_geom(node_list, element_list, ielm, xgauss(ms), t_q, phi_q, &
                                         R_q, Z_q, Rs, Rt, Zs, Zt)
              wvol = wgauss(ms) * w_t * pfafem_w_tor(iq) * R_q * abs(Rs * Zt - Rt * Zs)
              if (wvol <= 0.d0) cycle

              call pfafem_eval_qp_basis(node_list, element_list, ielm, eP, iq, ms, mt, &
                                        R_q, Z_q, phi_q, .true., n_act, act_u, act_io, act_jb, &
                                        b_v, b_R, b_Z, b_P)

              do a = 1, n_act
                u1  = act_u(a); io1 = act_io(a); jb1 = act_jb(a)
                do ivar = 1, n_var
                  el_rv(jb1, io1, u1, ivar) = el_rv(jb1, io1, u1, ivar) + &
                    wvol * b_v(a) * val_qp(ivar, ms, mt, iq, eP + 1, iel_loc)
                  if (do_deltas .and. present(del_qp)) then
                    el_rd(jb1, io1, u1, ivar) = el_rd(jb1, io1, u1, ivar) + &
                      wvol * b_v(a) * del_qp(ivar, ms, mt, iq, eP + 1, iel_loc)
                  end if
                end do
                if (do_jsrc .and. present(jsrc_qp)) then
                  el_rj(jb1, io1, u1) = el_rj(jb1, io1, u1) + &
                    wvol * b_v(a) * jsrc_qp(ms, mt, iq, eP + 1, iel_loc)
                end if

                do b = 1, n_act
                  u2  = act_u(b); io2 = act_io(b); jb2 = act_jb(b)
                  ivp = pfafem_cache(ielm)%pair_map(u1, u2)
                  if (ivp > 0) then
                    el_m(jb1, jb2, io1, io2, ivp) = el_m(jb1, jb2, io1, io2, ivp) + wvol * b_v(a) * b_v(b)
                  end if
                end do
              end do
            end do
          end do
        end do
      end do

      !$omp critical(pfafem_l2_mass_scatter)
      do u1 = 1, n_u
        inode = pfafem_cache(ielm)%unodes(u1)
        do io1 = 1, n_degrees
          ind1 = node_list%node(inode)%index(io1)
          if (ind1 <= 0) cycle
          do jb1 = 1, n_tor
            r_idx = (ind1 - 1) * n_tor + jb1
            rhs_v(r_idx, 1:n_var) = rhs_v(r_idx, 1:n_var) + el_rv(jb1, io1, u1, 1:n_var)
            if (do_deltas) rhs_d(r_idx, 1:n_var) = rhs_d(r_idx, 1:n_var) + el_rd(jb1, io1, u1, 1:n_var)
            if (do_jsrc)   rhs_j(r_idx)          = rhs_j(r_idx)          + el_rj(jb1, io1, u1)
          end do
        end do
      end do

      ! Consistent boundary surface trace integral on exterior edges (only 64 boundary elements total)
      if (elem_is_bnd(ielm)) then
        do eP = 0, n_tor - 1
          do iq = 1, pfafem_ng_tor
            phi_q = (dble(eP) + pfafem_tau_q(iq)) * pfafem_dphi
            do iside = 1, 4
              do ms = 1, n_gauss
                if (.not. get_elem_bnd_side_st(node_list, element_list, ielm, iside, xgauss(ms), s_b, t_b)) cycle
                call pfafem_eval_elem_geom(node_list, element_list, ielm, s_b, t_b, phi_q, &
                                           R_q, Z_q, Rs, Rt, Zs, Zt)
                wsurf = alpha_bnd * wgauss(ms) * pfafem_w_tor(iq) * R_q * abs(Rs * Zt - Rt * Zs)
                if (wsurf <= 0.d0) cycle

                call pfafem_eval_point_basis_val(node_list, element_list, ielm, s_b, t_b, eP, iq, &
                                                 .true., n_act, act_in, act_io, act_jb, b_v)
                do a = 1, n_act
                  ind1 = node_list%node(act_in(a))%index(act_io(a))
                  if (ind1 <= 0) cycle
                  r_idx = (ind1 - 1) * n_tor + act_jb(a)
                  do ivar = 1, n_var
                    rhs_v(r_idx, ivar) = rhs_v(r_idx, ivar) + wsurf * b_v(a) * val_bnd(ivar, ms, iside, iq, eP + 1, iel_loc)
                    if (do_deltas .and. present(del_bnd)) then
                      rhs_d(r_idx, ivar) = rhs_d(r_idx, ivar) + wsurf * b_v(a) * del_bnd(ivar, ms, iside, iq, eP + 1, iel_loc)
                    end if
                  end do
                  if (do_jsrc .and. present(jsrc_bnd)) then
                    rhs_j(r_idx) = rhs_j(r_idx) + wsurf * b_v(a) * jsrc_bnd(ms, iside, iq, eP + 1, iel_loc)
                  end if

                  do b = 1, n_act
                    ind2 = node_list%node(act_in(b))%index(act_io(b))
                    if (ind2 <= 0) cycle
                    c_idx = (ind2 - 1) * n_tor + act_jb(b)
                    call MatSetValue(M_pf, r_idx - 1, c_idx - 1, wsurf * b_v(a) * b_v(b), ADD_VALUES, ierr)
                  end do
                end do
              end do
            end do
          end do
        end do
      end if

      ! Scatter element mass blocks into M_pf (row-major transpose for MatSetValues)
      do ivp = 1, n_vp
        u1 = pfafem_cache(ielm)%vpairs(1, ivp)
        u2 = pfafem_cache(ielm)%vpairs(2, ivp)
        do io1 = 1, n_degrees
          ind1 = node_list%node(pfafem_cache(ielm)%unodes(u1))%index(io1)
          if (ind1 <= 0) cycle
          do j = 1, int(bs)
            r_rows(j) = (ind1 - 1) * bs + j - 1
          end do
          do io2 = 1, n_degrees
            ind2 = node_list%node(pfafem_cache(ielm)%unodes(u2))%index(io2)
            if (ind2 <= 0) cycle
            if (maxval(abs(el_m(:, :, io1, io2, ivp))) > 0.d0) then
              do j = 1, int(bs)
                c_cols(j) = (ind2 - 1) * bs + j - 1
              end do
              blk_t = transpose(el_m(:, :, io1, io2, ivp))
              call MatSetValues(M_pf, bs, r_rows, bs, c_cols, blk_t, ADD_VALUES, ierr)
            end if
          end do
        end do
      end do
      !$omp end critical(pfafem_l2_mass_scatter)

      deallocate(el_m, el_rv)
      if (allocated(el_rd)) deallocate(el_rd)
      if (allocated(el_rj)) deallocate(el_rj)
    end do
    !$omp end parallel do

    PetscCallA(MatAssemblyBegin(M_pf, MAT_FINAL_ASSEMBLY, ierr))
    PetscCallA(MatAssemblyEnd(M_pf, MAT_FINAL_ASSEMBLY, ierr))

    call MPI_Allreduce(MPI_IN_PLACE, rhs_v, size(rhs_v), MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, mpi_ierr)
    if (do_deltas) then
      call MPI_Allreduce(MPI_IN_PLACE, rhs_d, size(rhs_d), MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, mpi_ierr)
    end if
    if (do_jsrc) then
      call MPI_Allreduce(MPI_IN_PLACE, rhs_j, size(rhs_j), MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, mpi_ierr)
    end if

    ! Regularize dead / penalized axis dof=4 rows on M_pf diagonal
    PetscCallA(MatCreateVecs(M_pf, vec_x, vec_b, ierr))
    PetscCallA(VecDuplicate(vec_b, vec_diag, ierr))
    PetscCallA(MatGetDiagonal(M_pf, vec_diag, ierr))
    PetscCallA(VecNorm(vec_diag, NORM_INFINITY, dmax, ierr))
    PetscCallA(VecGetArrayRead(vec_diag, p_arr, ierr))
    n_seen = 0
    allocate(seen_list(max(1, n_loc_rows)))
    do k = 1, n_loc_rows
      if (abs(p_arr(k)) <= 1.d-20 * max(1.d0, dmax)) then
        r_idx = (ind_lo - 1) * bs + k - 1
        n_seen = n_seen + 1
        seen_list(n_seen) = r_idx + 1
        rhs_v(r_idx + 1, :) = 0.d0
        if (do_deltas) rhs_d(r_idx + 1, :) = 0.d0
        if (do_jsrc)   rhs_j(r_idx + 1)    = 0.d0
      end if
    end do
    PetscCallA(VecRestoreArrayRead(vec_diag, p_arr, ierr))
    do k = 1, n_seen
      r_idx = seen_list(k) - 1
      PetscCallA(MatSetValue(M_pf, r_idx, r_idx, 1.d0, INSERT_VALUES, ierr))
    end do
    deallocate(seen_list)
    PetscCallA(MatAssemblyBegin(M_pf, MAT_FINAL_ASSEMBLY, ierr))
    PetscCallA(MatAssemblyEnd(M_pf, MAT_FINAL_ASSEMBLY, ierr))

    ! Set up exact MUMPS LU + FGMRES iterative refinement solver for M_pf (89k scalar rows)
    PetscCallA(PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-pfafem_proj_mat_mumps_icntl_14", "80", ierr))
    PetscCallA(PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-pfafem_proj_mat_mumps_icntl_20", "0", ierr))
    PetscCallA(PetscOptionsSetValue(PETSC_NULL_OPTIONS, "-pfafem_proj_mat_mumps_icntl_24", "1", ierr))
    PetscCallA(KSPCreate(MPI_COMM_WORLD, ksp_m, ierr))
    PetscCallA(KSPSetOptionsPrefix(ksp_m, "pfafem_proj_", ierr))
    PetscCallA(KSPSetOperators(ksp_m, M_pf, M_pf, ierr))
    PetscCallA(KSPSetType(ksp_m, KSPFGMRES, ierr))
    PetscCallA(KSPGMRESSetRestart(ksp_m, 20, ierr))
    PetscCallA(KSPGetPC(ksp_m, pc_m, ierr))
#if defined(PETSC_HAVE_MUMPS)
    PetscCallA(PCSetType(pc_m, PCLU, ierr))
    PetscCallA(PCFactorSetMatSolverType(pc_m, MATSOLVERMUMPS, ierr))
    PetscCallA(KSPSetTolerances(ksp_m, 1.d-13, 1.d-30, 1.d10, 20, ierr))
#else
    PetscCallA(PCSetType(pc_m, PCJACOBI, ierr))
    PetscCallA(KSPSetTolerances(ksp_m, 1.d-12, 1.d-30, 1.d10, 4000, ierr))
#endif
    PetscCallA(KSPSetFromOptions(ksp_m, ierr))
    PetscCallA(PCSetFromOptions(pc_m, ierr))
    PetscCallA(KSPSetUp(ksp_m, ierr))

    allocate(sol_all(n_glob))
    max_its_proj     = 0
    max_rel_res_proj = 0.d0

    ! Solve for each variable's values
    do ivar = 1, n_var
      PetscCallA(VecGetArray(vec_b, p_arr, ierr))
      if (n_loc_rows > 0) then
        p_arr(1:n_loc_rows) = rhs_v((ind_lo - 1)*bs + 1 : ind_hi*bs, ivar)
      end if
      PetscCallA(VecRestoreArray(vec_b, p_arr, ierr))
      PetscCallA(VecNorm(vec_b, NORM_2, b_nrm, ierr))
      PetscCallA(VecZeroEntries(vec_x, ierr))
      if (b_nrm > 0.d0) then
        PetscCallA(KSPSolve(ksp_m, vec_b, vec_x, ierr))
        PetscCallA(KSPGetIterationNumber(ksp_m, its_m, ierr))
        max_its_proj = max(max_its_proj, int(its_m))
        PetscCallA(MatMult(M_pf, vec_x, vec_diag, ierr))
        PetscCallA(VecAXPY(vec_diag, -1.d0, vec_b, ierr))
        PetscCallA(VecNorm(vec_diag, NORM_2, r_nrm, ierr))
        max_rel_res_proj = max(max_rel_res_proj, dble(r_nrm) / dble(b_nrm))
      end if

      sol_all = 0.d0
      PetscCallA(VecGetArrayRead(vec_x, p_arr, ierr))
      if (n_loc_rows > 0) then
        sol_all((ind_lo - 1)*bs + 1 : ind_hi*bs) = p_arr(1:n_loc_rows)
      end if
      PetscCallA(VecRestoreArrayRead(vec_x, p_arr, ierr))
      call MPI_Allreduce(MPI_IN_PLACE, sol_all, n_glob, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, mpi_ierr)

      do i = 1, node_list%n_nodes
        if (node_list%node(i)%constrained) cycle
        do in_t = 1, n_tor
          new_dofs = 0.d0
          do j = 1, n_degrees
            ind1 = node_list%node(i)%index(j)
            if (ind1 > 0) then
              new_dofs(j) = sol_all((ind1 - 1)*bs + in_t)
            else
              new_dofs(j) = node_list%node(i)%values(in_t, j, ivar)
            end if
          end do
          if (treat_axis .and. node_list%node(i)%axis_node) then
            call new_to_old_dofs_on_the_axis(node_list, i, new_dofs, old_dofs)
            node_list%node(i)%values(in_t, :, ivar) = old_dofs
          else
            node_list%node(i)%values(in_t, :, ivar) = new_dofs
          end if
        end do
      end do
    end do

    ! Solve for each variable's deltas if requested
    if (do_deltas) then
      do ivar = 1, n_var
        PetscCallA(VecGetArray(vec_b, p_arr, ierr))
        if (n_loc_rows > 0) then
          p_arr(1:n_loc_rows) = rhs_d((ind_lo - 1)*bs + 1 : ind_hi*bs, ivar)
        end if
        PetscCallA(VecRestoreArray(vec_b, p_arr, ierr))
        PetscCallA(VecZeroEntries(vec_x, ierr))
        PetscCallA(KSPSolve(ksp_m, vec_b, vec_x, ierr))

        sol_all = 0.d0
        PetscCallA(VecGetArrayRead(vec_x, p_arr, ierr))
        if (n_loc_rows > 0) then
          sol_all((ind_lo - 1)*bs + 1 : ind_hi*bs) = p_arr(1:n_loc_rows)
        end if
        PetscCallA(VecRestoreArrayRead(vec_x, p_arr, ierr))
        call MPI_Allreduce(MPI_IN_PLACE, sol_all, n_glob, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, mpi_ierr)

        do i = 1, node_list%n_nodes
          if (node_list%node(i)%constrained) cycle
          do in_t = 1, n_tor
            new_dofs = 0.d0
            do j = 1, n_degrees
              ind1 = node_list%node(i)%index(j)
              if (ind1 > 0) new_dofs(j) = sol_all((ind1 - 1)*bs + in_t)
            end do
            if (treat_axis .and. node_list%node(i)%axis_node) then
              call new_to_old_dofs_on_the_axis(node_list, i, new_dofs, old_dofs)
              node_list%node(i)%deltas(in_t, :, ivar) = old_dofs
            else
              node_list%node(i)%deltas(in_t, :, ivar) = new_dofs
            end if
          end do
        end do
      end do
    end if

#ifdef STELLARATOR_MODEL
    ! Solve for stellarator j_source if requested
    if (do_jsrc) then
      PetscCallA(VecGetArray(vec_b, p_arr, ierr))
      if (n_loc_rows > 0) then
        p_arr(1:n_loc_rows) = rhs_j((ind_lo - 1)*bs + 1 : ind_hi*bs)
      end if
      PetscCallA(VecRestoreArray(vec_b, p_arr, ierr))
      PetscCallA(VecZeroEntries(vec_x, ierr))
      PetscCallA(KSPSolve(ksp_m, vec_b, vec_x, ierr))

      sol_all = 0.d0
      PetscCallA(VecGetArrayRead(vec_x, p_arr, ierr))
      if (n_loc_rows > 0) then
        sol_all((ind_lo - 1)*bs + 1 : ind_hi*bs) = p_arr(1:n_loc_rows)
      end if
      PetscCallA(VecRestoreArrayRead(vec_x, p_arr, ierr))
      call MPI_Allreduce(MPI_IN_PLACE, sol_all, n_glob, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, mpi_ierr)

      do i = 1, node_list%n_nodes
        if (node_list%node(i)%constrained) cycle
        do in_t = 1, n_tor
          new_dofs = 0.d0
          do j = 1, n_degrees
            ind1 = node_list%node(i)%index(j)
            if (ind1 > 0) new_dofs(j) = sol_all((ind1 - 1)*bs + in_t)
          end do
          if (treat_axis .and. node_list%node(i)%axis_node) then
            call new_to_old_dofs_on_the_axis(node_list, i, new_dofs, old_dofs)
            node_list%node(i)%j_source(in_t, :) = old_dofs
          else
            node_list%node(i)%j_source(in_t, :) = new_dofs
          end if
        end do
      end do
    end if
#endif

    if (my_id == 0) then
      write(*,'(A,I4,A,ES10.3)') ' [PFAFEM] L2 projection: max_its=', max_its_proj, &
        ' max_rel_res=', max_rel_res_proj
    end if
    deallocate(sol_all, rhs_v)
    if (allocated(rhs_d)) deallocate(rhs_d)
    if (allocated(rhs_j)) deallocate(rhs_j)
    PetscCallA(KSPDestroy(ksp_m, ierr))
    PetscCallA(VecDestroy(vec_diag, ierr))
    PetscCallA(VecDestroy(vec_x, ierr))
    PetscCallA(VecDestroy(vec_b, ierr))
    PetscCallA(MatDestroy(M_pf, ierr))
#endif
  end subroutine solve_pfafem_l2_projection

  !=============================================================================
  !> Count exact diagonal/off-diagonal block nonzeros per owned index row [ind_lo..ind_hi].
  !=============================================================================
  subroutine count_mass_block_nnz(node_list, element_list, ind_lo, ind_hi, ind_tot, d_nnz, o_nnz)
    type(type_node_list),    intent(in)  :: node_list
    type(type_element_list), intent(in)  :: element_list
    integer,                 intent(in)  :: ind_lo, ind_hi, ind_tot
    integer,                 intent(out) :: d_nnz(:), o_nnz(:)

    type :: row_adj_t
      integer :: n = 0
      integer, allocatable :: cols(:)
    end type row_adj_t
    type(row_adj_t), allocatable :: adj(:)
    integer :: n_own, ielm, ivp, u1, u2, n1, n2, io1, io2, ind1, ind2, r, k, cap
    integer, allocatable :: tmp(:)
    logical :: found

    n_own = ind_hi - ind_lo + 1
    allocate(adj(n_own))
    do r = 1, n_own
      allocate(adj(r)%cols(32))
      adj(r)%n = 1
      adj(r)%cols(1) = ind_lo + r - 1
    end do

    do ielm = 1, element_list%n_elements
      do ivp = 1, pfafem_cache(ielm)%n_vpairs
        n1 = pfafem_cache(ielm)%unodes(pfafem_cache(ielm)%vpairs(1, ivp))
        n2 = pfafem_cache(ielm)%unodes(pfafem_cache(ielm)%vpairs(2, ivp))
        do io1 = 1, n_degrees
          ind1 = node_list%node(n1)%index(io1)
          if (ind1 < ind_lo .or. ind1 > ind_hi) cycle
          r = ind1 - ind_lo + 1
          do io2 = 1, n_degrees
            ind2 = node_list%node(n2)%index(io2)
            if (ind2 <= 0 .or. ind2 > ind_tot) cycle
            found = .false.
            do k = 1, adj(r)%n
              if (adj(r)%cols(k) == ind2) then
                found = .true.
                exit
              end if
            end do
            if (.not. found) then
              if (adj(r)%n == size(adj(r)%cols)) then
                cap = 2 * size(adj(r)%cols)
                allocate(tmp(cap))
                tmp(1:adj(r)%n) = adj(r)%cols
                call move_alloc(tmp, adj(r)%cols)
              end if
              adj(r)%n = adj(r)%n + 1
              adj(r)%cols(adj(r)%n) = ind2
            end if
          end do
        end do
      end do
    end do

    do r = 1, n_own
      d_nnz(r) = 0
      o_nnz(r) = 0
      do k = 1, adj(r)%n
        ind2 = adj(r)%cols(k)
        if (ind2 >= ind_lo .and. ind2 <= ind_hi) then
          d_nnz(r) = d_nnz(r) + 1
        else
          o_nnz(r) = o_nnz(r) + 1
        end if
      end do
      deallocate(adj(r)%cols)
    end do
    deallocate(adj)
  end subroutine count_mass_block_nnz

  !=============================================================================
  !> Free all PFAFEM module allocations.
  !=============================================================================
  subroutine pfafem_cleanup()
    use phys_module, only: pfafem_values_active
    call pfafem_dealloc_qp_cache()
    if (allocated(pfafem_tau_q)) deallocate(pfafem_tau_q, pfafem_w_q, pfafem_w_tor)
    if (allocated(pfafem_t_pol)) deallocate(pfafem_t_pol, pfafem_w_pol)
    if (allocated(pfafem_Q))     deallocate(pfafem_Q, pfafem_QT)
    if (allocated(pfafem_cache)) deallocate(pfafem_cache)
    if (allocated(pfafem_bgrid_coefs)) deallocate(pfafem_bgrid_coefs)
    if (allocated(pfafem_restart_values)) deallocate(pfafem_restart_values)
    if (allocated(pfafem_restart_deltas)) deallocate(pfafem_restart_deltas)
    if (allocated(elem_is_bnd)) deallocate(elem_is_bnd)
    if (allocated(bnd_elem_list)) deallocate(bnd_elem_list)
    if (allocated(bnd_elem_knot_rz)) deallocate(bnd_elem_knot_rz)
    if (allocated(pfafem_knot_x)) deallocate(pfafem_knot_x)
    n_bnd_elems = 0
#ifdef STELLARATOR_MODEL
    if (allocated(pfafem_chi_q)) deallocate(pfafem_chi_q)
    if (allocated(pfafem_restart_jsrc)) deallocate(pfafem_restart_jsrc)
#endif
    pfafem_ready = .false.
    pfafem_cache_epoch = 0
    pfafem_bgrid_valid = .false.
    pfafem_restart_needs_projection = .false.
    pfafem_values_active = .false.
  end subroutine pfafem_cleanup

end module mod_pfafem
