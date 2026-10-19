!> model600's force operator W (see models/model199/mod_pc_elt_matrix_force_fft.f90):
!! with its extensions off model600's blocks entering W are model199's (the
!! u row's pressure coupling one integration by parts apart, the T equation
!! in pressure form giving the same dp), so W is model199's curvature term,
!! with GAMMA from phys_module.
!!
!! V_PAR (with_vpar, stages b0 / b1; optional ELM_v): the same pass assembles
!! v_par's sound-wave Schur terms, the p channel of S = B - L M^-1 U with
!! (rho, T) eliminated by their mass. With D_B f = (F0/R) d_phi f + [f, psi0]
!! (R B.grad f), P_B = D_B p0 and c = (theta dt)^2/opz:
!!   W_vv(v,V) = + c (D_B v) (V P_B + gamma p0 D_B V) / R       (-B_7p M^-1 B_p7)
!!   W_vu(v,u) = - 2 c gamma p0 (D_B v) dZ u                    (-B_7p M^-1 B_pu)
!!   W_uv(v,V) = + 2 c dZ v (V P_B + gamma p0 D_B V)            (-B_up M^-1 B_p7)
!! W_vu drops the R[p0,u] half of dp_u, which for an axisymmetric p0(psi0)
!! cancels against the psi channel -B_71 M_psi^-1 B_12 u; vu_full keeps it,
!!   W_vu(v,u) = - c (D_B v) (R [p0,u] + 2 gamma p0 dZ u),
!! for a sweep that puts psi_w into v_par's row. gamma_w replaces GAMMA in
!! these three terms only.
module mod_pc_elt_matrix_force_fft

  use mod_pc_fft_scatter, only: scatter_fft_to_elm, scatter_fft_to_elm_n, &
                                scatter_fft_to_elm_k, scatter_fft_to_elm_kn, pc_my_fft

  implicit none
  private

  public :: pc_elt_matrix_force_fft

contains

subroutine pc_elt_matrix_force_fft(element, nodes, ELM, ELM_v, gamma_w, vu_full)

  use mod_parameters
  use data_structure, only: type_element, type_node
  use gauss
  use basis_at_gaussian
  use phys_module

  implicit none

  type(type_element), intent(in) :: element
  type(type_node),    intent(in) :: nodes(n_vertex_max)

#define NFV (n_vertex_max*n_degrees)
#define DFV (n_tor*n_vertex_max*n_degrees)

  real*8, dimension(DFV, DFV), intent(out) :: ELM
  !> v_par's W terms, assembled when present: 1 = W_vv, 2 = W_vu, 3 = W_uv
  real*8, dimension(DFV, DFV, 3), intent(out), optional :: ELM_v
  real*8, intent(in), optional :: gamma_w    !< their adiabatic index (default GAMMA)
  logical, intent(in), optional :: vu_full   !< W_vu with the R [p0,u] half

  real*8, dimension(n_plane, NFV, NFV) :: ELM_p
  !> v_par's blocks per toroidal channel: p, d_phi on trial (n), test (k), both (kn)
  real*8, allocatable, dimension(:,:,:,:) :: EV_p, EV_n, EV_k, EV_kn
  logical :: do_v, vfull

  real*8, dimension(n_gauss, n_gauss) :: x_g, x_s, x_t, y_s, y_t
  !> psi0 (1), rho0 (2), T0 (3) and their s, t derivatives; d_phi of rho0, T0
  real*8, dimension(n_plane, 3, n_gauss, n_gauss) :: eq_g, eq_s, eq_t
  real*8, dimension(n_plane, 2, n_gauss, n_gauss) :: eqp_rt
  real*8 :: f_v(NFV), f_x(NFV), f_y(NFV)    !< basis values and gradients at the Gauss point
  integer, parameter :: vpt(3) = [var_psi, var_rho, var_T]

  integer :: i, j, k, ms, mt, mp, in, q, b
  integer :: index_ij, index_kl
  real*8  :: wst, xjac, BigR, theta, zeta_t, pref, GAMMA_l, sz, f_s, f_t
  real*8  :: ps0_x, ps0_y, r0, r0_x, r0_y, T0, T0_x, T0_y, p0, p0_x, p0_y
  real*8  :: v, v_y, u, u_x, u_y, a_curv, a_uu
  real*8  :: g_v, p0_p, P_B, cv, b_tst, b_tri, qv, qv_n, qu
  real*8     :: in_fft(1:n_plane)
  complex*16 :: out_fft(1:n_plane)

  ELM   = 0.d0
  ELM_p = 0.d0
  do_v  = present(ELM_v)
  if (do_v) then
    ELM_v = 0.d0
    allocate(EV_p(n_plane, NFV, NFV, 3), EV_n(n_plane, NFV, NFV, 3), &
             EV_k(n_plane, NFV, NFV, 3), EV_kn(n_plane, NFV, NFV, 3))
    EV_p = 0.d0; EV_n = 0.d0; EV_k = 0.d0; EV_kn = 0.d0
  endif

  GAMMA_l = gamma
  g_v     = gamma
  if (present(gamma_w)) g_v = gamma_w
  vfull   = .false.
  if (present(vu_full)) vfull = vu_full

  theta  = time_evol_theta
  zeta_t = time_evol_zeta * 2.0d0 * tstep / (tstep + tstep_prev)
  pref   = (theta * tstep)**2 / (1.d0 + zeta_t)

  x_g = 0.d0; x_s = 0.d0; x_t = 0.d0; y_s = 0.d0; y_t = 0.d0
  eq_g = 0.d0; eq_s = 0.d0; eq_t = 0.d0; eqp_rt = 0.d0
  do i = 1, n_vertex_max
    do j = 1, n_degrees
      sz = element%size(i,j)
      do ms = 1, n_gauss
        do mt = 1, n_gauss
          x_g(ms,mt) = x_g(ms,mt) + nodes(i)%x(1,j,1) * sz * H(i,j,ms,mt)
          x_s(ms,mt) = x_s(ms,mt) + nodes(i)%x(1,j,1) * sz * H_s(i,j,ms,mt)
          x_t(ms,mt) = x_t(ms,mt) + nodes(i)%x(1,j,1) * sz * H_t(i,j,ms,mt)
          y_s(ms,mt) = y_s(ms,mt) + nodes(i)%x(1,j,2) * sz * H_s(i,j,ms,mt)
          y_t(ms,mt) = y_t(ms,mt) + nodes(i)%x(1,j,2) * sz * H_t(i,j,ms,mt)
          do mp = 1, n_plane
            do k = 1, 3
              do in = 1, n_tor
                eq_g(mp,k,ms,mt) = eq_g(mp,k,ms,mt) &
                     + nodes(i)%values(in,j,vpt(k)) * sz * H(i,j,ms,mt)   * HZ(in,mp)
                eq_s(mp,k,ms,mt) = eq_s(mp,k,ms,mt) &
                     + nodes(i)%values(in,j,vpt(k)) * sz * H_s(i,j,ms,mt) * HZ(in,mp)
                eq_t(mp,k,ms,mt) = eq_t(mp,k,ms,mt) &
                     + nodes(i)%values(in,j,vpt(k)) * sz * H_t(i,j,ms,mt) * HZ(in,mp)
              enddo
            enddo
            if (do_v) then
              do in = 1, n_tor
                eqp_rt(mp,1,ms,mt) = eqp_rt(mp,1,ms,mt) &
                     + nodes(i)%values(in,j,var_rho) * sz * H(i,j,ms,mt) * HZ_p(in,mp)
                eqp_rt(mp,2,ms,mt) = eqp_rt(mp,2,ms,mt) &
                     + nodes(i)%values(in,j,var_T) * sz * H(i,j,ms,mt) * HZ_p(in,mp)
              enddo
            endif
          enddo
        enddo
      enddo
    enddo
  enddo

  do ms = 1, n_gauss
   do mt = 1, n_gauss
    wst  = wgauss(ms)*wgauss(mt)
    xjac = x_s(ms,mt)*y_t(ms,mt) - x_t(ms,mt)*y_s(ms,mt)
    BigR = x_g(ms,mt)
    do q = 1, NFV
      i = (q - 1) / n_degrees + 1; j = mod(q - 1, n_degrees) + 1
      f_v(q) = h(i,j,ms,mt) * element%size(i,j)
      f_s    = h_s(i,j,ms,mt) * element%size(i,j)
      f_t    = h_t(i,j,ms,mt) * element%size(i,j)
      f_x(q) = (  y_t(ms,mt)*f_s - y_s(ms,mt)*f_t ) / xjac
      f_y(q) = ( -x_t(ms,mt)*f_s + x_s(ms,mt)*f_t ) / xjac
    enddo

    do mp = 1, n_plane
      ps0_x = (  y_t(ms,mt)*eq_s(mp,1,ms,mt) - y_s(ms,mt)*eq_t(mp,1,ms,mt) ) / xjac
      ps0_y = ( -x_t(ms,mt)*eq_s(mp,1,ms,mt) + x_s(ms,mt)*eq_t(mp,1,ms,mt) ) / xjac
      r0   = abs(eq_g(mp,2,ms,mt))
      r0_x = (  y_t(ms,mt)*eq_s(mp,2,ms,mt) - y_s(ms,mt)*eq_t(mp,2,ms,mt) ) / xjac
      r0_y = ( -x_t(ms,mt)*eq_s(mp,2,ms,mt) + x_s(ms,mt)*eq_t(mp,2,ms,mt) ) / xjac
      T0   = abs(eq_g(mp,3,ms,mt))
      T0_x = (  y_t(ms,mt)*eq_s(mp,3,ms,mt) - y_s(ms,mt)*eq_t(mp,3,ms,mt) ) / xjac
      T0_y = ( -x_t(ms,mt)*eq_s(mp,3,ms,mt) + x_s(ms,mt)*eq_t(mp,3,ms,mt) ) / xjac
      p0_x = r0_x*T0 + r0*T0_x
      p0_y = r0_y*T0 + r0*T0_y
      p0   = r0 * T0
      if (do_v) then
        ! P_B = (F0/R) d_phi p0 + [p0, psi0] (zero for p0 = p0(psi0))
        p0_p = eqp_rt(mp,1,ms,mt) * T0 + r0 * eqp_rt(mp,2,ms,mt)
        P_B  = F0 / BigR * p0_p + ( p0_x*ps0_y - p0_y*ps0_x )
        cv   = pref * xjac
      endif

      do index_ij = 1, NFV
        v = f_v(index_ij); v_y = f_y(index_ij)
        b_tst = f_x(index_ij)*ps0_y - v_y*ps0_x                ! [v, psi0]
        do index_kl = 1, NFV
          u = f_v(index_kl); u_x = f_x(index_kl); u_y = f_y(index_kl)
          a_curv = - 2.d0 * BigR**2 * v_y * ( p0_x*u_y - p0_y*u_x ) &
                   - 4.d0 * GAMMA_l * BigR * p0 * v_y * u_y
          a_uu = pref * ( a_curv ) * xjac
          ELM_p(mp,index_ij,index_kl) = ELM_p(mp,index_ij,index_kl) + wst * a_uu

          if (.not. do_v) cycle
          ! D_B v = (F0/R) v_p + b_tst; the trial u is V in W_vv / W_uv
          b_tri = u_x*ps0_y - u_y*ps0_x
          qv   = u * P_B + g_v * p0 * b_tri               ! V P_B + g p0 [V, psi0]
          qv_n = g_v * p0 * F0 / BigR * u                 ! g p0 (F0/R) d_phi V
          EV_p (mp,index_ij,index_kl,1) = EV_p (mp,index_ij,index_kl,1) + wst * cv * b_tst * qv / BigR
          EV_n (mp,index_ij,index_kl,1) = EV_n (mp,index_ij,index_kl,1) + wst * cv * b_tst * qv_n / BigR
          EV_k (mp,index_ij,index_kl,1) = EV_k (mp,index_ij,index_kl,1) + wst * cv * F0 / BigR**2 * v * qv
          EV_kn(mp,index_ij,index_kl,1) = EV_kn(mp,index_ij,index_kl,1) + wst * cv * F0 / BigR**2 * v * qv_n
          qu = 2.d0 * g_v * p0 * u_y
          if (vfull) qu = qu + BigR * ( p0_x*u_y - p0_y*u_x )
          EV_p (mp,index_ij,index_kl,2) = EV_p (mp,index_ij,index_kl,2) - wst * cv * b_tst * qu
          EV_k (mp,index_ij,index_kl,2) = EV_k (mp,index_ij,index_kl,2) - wst * cv * F0 / BigR * v * qu
          EV_p (mp,index_ij,index_kl,3) = EV_p (mp,index_ij,index_kl,3) + wst * cv * 2.d0 * v_y * qv
          EV_n (mp,index_ij,index_kl,3) = EV_n (mp,index_ij,index_kl,3) + wst * cv * 2.d0 * v_y * qv_n
        enddo
      enddo
    enddo
   enddo
  enddo

  do i = 1, NFV
    do j = 1, NFV
      call one_channel(ELM_p(:,i,j), 0, ELM)
    enddo
  enddo
  ! the scatter accumulates each (row, col) pair twice (as mod_elt_matrix_fft)
  ELM = 0.5d0 * ELM

  if (do_v) then
    do b = 1, 3
      do i = 1, NFV
        do j = 1, NFV
          call one_channel(EV_p (:,i,j,b), 0, ELM_v(:,:,b))
          call one_channel(EV_n (:,i,j,b), 1, ELM_v(:,:,b))
          call one_channel(EV_k (:,i,j,b), 2, ELM_v(:,:,b))
          call one_channel(EV_kn(:,i,j,b), 3, ELM_v(:,:,b))
        enddo
      enddo
    enddo
    ELM_v = 0.5d0 * ELM_v
    deallocate(EV_p, EV_n, EV_k, EV_kn)
  endif

contains

  !> One toroidal channel of entry (i, j) into E: 0 = p, 1 = n, 2 = k, 3 = kn.
  subroutine one_channel(c, ch, E)
    real*8, intent(in)    :: c(n_plane)
    integer, intent(in)   :: ch
    real*8, intent(inout) :: E(DFV, DFV)
    if (maxval(abs(c)) == 0.d0) return
    in_fft = c
#ifdef USE_FFTW
    call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#else
    call pc_my_fft(in_fft, out_fft, n_plane)
#endif
    select case (ch)
    case (0); call scatter_fft_to_elm   (out_fft, i, j, E, DFV)
    case (1); call scatter_fft_to_elm_n (out_fft, i, j, E, DFV)
    case (2); call scatter_fft_to_elm_k (out_fft, i, j, E, DFV)
    case (3); call scatter_fft_to_elm_kn(out_fft, i, j, E, DFV)
    end select
  end subroutine one_channel

end subroutine pc_elt_matrix_force_fft

end module mod_pc_elt_matrix_force_fft
