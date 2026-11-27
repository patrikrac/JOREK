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
!! cancels against the psi channel -B_71 M_psi^-1 B_12 u.
module mod_pc_elt_matrix_force_fft

  use mod_pc_fft_scatter, only: scatter_fft_to_elm, scatter_fft_to_elm_n, &
                                scatter_fft_to_elm_k, scatter_fft_to_elm_kn, pc_my_fft

  implicit none
  private

  public :: pc_elt_matrix_force_fft, pc_elt_matrix_force_pfafem

contains

subroutine pc_elt_matrix_force_fft(element, nodes, ELM, ELM_v)

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

  real*8, dimension(n_plane, NFV, NFV) :: ELM_p
  !> v_par's blocks per toroidal channel: p, d_phi on trial (n), test (k), both (kn)
  real*8, allocatable, dimension(:,:,:,:) :: EV_p, EV_n, EV_k, EV_kn
  logical :: do_v

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
  real*8  :: p0_p, P_B, cv, b_tst, b_tri, qv, qv_n, qu
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
          qv   = u * P_B + GAMMA_l * p0 * b_tri           ! V P_B + gamma p0 [V, psi0]
          qv_n = GAMMA_l * p0 * F0 / BigR * u             ! gamma p0 (F0/R) d_phi V
          EV_p (mp,index_ij,index_kl,1) = EV_p (mp,index_ij,index_kl,1) + wst * cv * b_tst * qv / BigR
          EV_n (mp,index_ij,index_kl,1) = EV_n (mp,index_ij,index_kl,1) + wst * cv * b_tst * qv_n / BigR
          EV_k (mp,index_ij,index_kl,1) = EV_k (mp,index_ij,index_kl,1) + wst * cv * F0 / BigR**2 * v * qv
          EV_kn(mp,index_ij,index_kl,1) = EV_kn(mp,index_ij,index_kl,1) + wst * cv * F0 / BigR**2 * v * qv_n
          qu = 2.d0 * GAMMA_l * p0 * u_y
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

subroutine pc_elt_matrix_force_pfafem(node_list, element_list, ielm, W_vp, WV_vp)
  use mod_parameters
  use data_structure, only: type_node_list, type_element_list
  use gauss, only: n_gauss, xgauss, wgauss
  use phys_module
  use mod_pfafem, only: pfafem_cache, pfafem_el_lo, pfafem_el_hi, pfafem_ng_tor, pfafem_ng_pol, &
                        pfafem_ep_dim, pfafem_t_pol, pfafem_w_pol, pfafem_tau_q, &
                        pfafem_w_tor, pfafem_dphi, pfafem_qp_cached, pfafem_qp_geom, &
                        pfafem_qp_ngrp, pfafem_qp_gu, pfafem_qp_gjb, pfafem_qp_db, &
                        pfafem_eval_elem_geom, pfafem_eval_qp_basis
  implicit none

  type(type_node_list),    intent(in)            :: node_list
  type(type_element_list), intent(in)            :: element_list
  integer,                 intent(in)            :: ielm
  real*8,                  intent(out)           :: W_vp(:, :, :, :, :)
  real*8,                  intent(out), optional :: WV_vp(:, :, :, :, :, :)

  logical :: do_v, has_axis_u, use_qp_c
  integer :: eP, ep1, iep_q, iq, ms, mt, iel_loc, a, b, unode, ivp
  integer :: n_grp, g_a, a0, io_a, u_a, jb_a
  integer :: iv_T, iv_Ti, iv_Te
  integer :: n_raw, raw_u(64), raw_io(64), raw_jb(64)
  integer :: n_act, act_u(64), act_io(64), act_jb(64)
  real*8  :: GAMMA_l, theta, zeta_t, pref, phi_q, t_q, w_t
  real*8  :: BigR, y_g, Rs, Rt, Zs, Zt, xjac, wst_tor, cv
  real*8  :: rv(64), rR(64), rZ(64), rP(64), r_jet4(4)
  real*8  :: bv(64), bR(64), bZ(64), bP(64)
  real*8  :: vpsi_a, vr_a, vt_a
  real*8  :: ps0_x, ps0_y, r0_raw, r0, r0_x, r0_y, r0_p
  real*8  :: T0_raw, T0, T0_x, T0_y, T0_p, p0, p0_x, p0_y, p0_p, P_B
  real*8  :: v_y, u_x, u_y, a_curv, a_uu, Db_tst, Db_tri, q_tri

  W_vp = 0.d0
  do_v = present(WV_vp)
  if (do_v) WV_vp = 0.d0

  GAMMA_l = gamma
  theta   = time_evol_theta
  zeta_t  = time_evol_zeta * 2.0d0 * tstep / (tstep + tstep_prev)
  pref    = (theta * tstep)**2 / (1.d0 + zeta_t)

  iv_T  = max(1, var_T)
  iv_Ti = max(1, var_Ti)
  iv_Te = max(1, var_Te)

  iel_loc    = ielm - pfafem_el_lo + 1
  use_qp_c   = pfafem_qp_cached .and. (ielm >= pfafem_el_lo) .and. (ielm <= pfafem_el_hi)
  has_axis_u = .false.
  if (treat_axis) then
    do a = 1, pfafem_cache(ielm)%n_unodes
      if (node_list%node(pfafem_cache(ielm)%unodes(a))%axis_node) then
        has_axis_u = .true.
        exit
      end if
    end do
  end if

  do eP = 0, n_tor - 1
    ep1   = eP + 1
    iep_q = merge(1, ep1, pfafem_ep_dim == 1)
    do iq = 1, pfafem_ng_tor
      phi_q = (dble(eP) + pfafem_tau_q(iq)) * pfafem_dphi
      do ms = 1, n_gauss
        do mt = 1, pfafem_ng_pol
          if (ielm >= pfafem_el_lo .and. ielm <= pfafem_el_hi) then
            t_q = pfafem_t_pol(ms, mt, iq, iep_q, iel_loc)
            w_t = pfafem_w_pol(ms, mt, iq, iep_q, iel_loc)
          else
            t_q = xgauss(min(n_gauss, mt))
            w_t = wgauss(min(n_gauss, mt))
          end if
          if (use_qp_c) then
            BigR = pfafem_qp_geom(1,  ms, mt, iq, ep1, iel_loc)
            y_g  = pfafem_qp_geom(2,  ms, mt, iq, ep1, iel_loc)
            xjac = pfafem_qp_geom(13, ms, mt, iq, ep1, iel_loc)
          else
            call pfafem_eval_elem_geom(node_list, element_list, ielm, xgauss(ms), t_q, phi_q, &
                                       BigR, y_g, Rs, Rt, Zs, Zt)
            xjac = Rs * Zt - Rt * Zs
          end if
          wst_tor = wgauss(ms) * w_t * pfafem_w_tor(iq)
          cv      = pref * xjac

          ps0_x  = 0.d0; ps0_y = 0.d0
          r0_raw = 0.d0; r0_x  = 0.d0; r0_y = 0.d0; r0_p = 0.d0
          T0_raw = 0.d0; T0_x  = 0.d0; T0_y = 0.d0; T0_p = 0.d0

          if (use_qp_c) then
            n_grp = int(pfafem_qp_ngrp(ms, mt, iq, ep1, iel_loc))
            n_raw = 4 * n_grp
            do g_a = 1, n_grp
              u_a   = int(pfafem_qp_gu(g_a, ms, mt, iq, ep1, iel_loc))
              jb_a  = int(pfafem_qp_gjb(g_a, ms, mt, iq, ep1, iel_loc))
              unode = pfafem_cache(ielm)%unodes(u_a)
              a0    = 4 * (g_a - 1)
              do io_a = 1, 4
                a         = a0 + io_a
                raw_u(a)  = u_a
                raw_io(a) = io_a
                raw_jb(a) = jb_a
                r_jet4 = pfafem_qp_db(1:4, a, ms, mt, iq, ep1, iel_loc)
                rv(a) = r_jet4(1); rR(a) = r_jet4(2); rZ(a) = r_jet4(3); rP(a) = r_jet4(4)
                vpsi_a = node_list%node(unode)%values(jb_a, io_a, var_psi)
                vr_a   = node_list%node(unode)%values(jb_a, io_a, var_rho)
                if (with_TiTe) then
                  vt_a = node_list%node(unode)%values(jb_a, io_a, iv_Ti) &
                       + node_list%node(unode)%values(jb_a, io_a, iv_Te)
                else
                  vt_a = node_list%node(unode)%values(jb_a, io_a, iv_T)
                end if
                ps0_x  = ps0_x  + vpsi_a * r_jet4(2)
                ps0_y  = ps0_y  + vpsi_a * r_jet4(3)
                r0_raw = r0_raw + vr_a   * r_jet4(1)
                r0_x   = r0_x   + vr_a   * r_jet4(2)
                r0_y   = r0_y   + vr_a   * r_jet4(3)
                r0_p   = r0_p   + vr_a   * r_jet4(4)
                T0_raw = T0_raw + vt_a   * r_jet4(1)
                T0_x   = T0_x   + vt_a   * r_jet4(2)
                T0_y   = T0_y   + vt_a   * r_jet4(3)
                T0_p   = T0_p   + vt_a   * r_jet4(4)
              end do
            end do
          else
            call pfafem_eval_qp_basis(node_list, element_list, ielm, eP, iq, ms, mt, &
                                      BigR, y_g, phi_q, .false., n_raw, raw_u, raw_io, raw_jb, &
                                      rv, rR, rZ, rP)
            do a = 1, n_raw
              unode  = pfafem_cache(ielm)%unodes(raw_u(a))
              vpsi_a = node_list%node(unode)%values(raw_jb(a), raw_io(a), var_psi)
              vr_a   = node_list%node(unode)%values(raw_jb(a), raw_io(a), var_rho)
              if (with_TiTe) then
                vt_a = node_list%node(unode)%values(raw_jb(a), raw_io(a), iv_Ti) &
                     + node_list%node(unode)%values(raw_jb(a), raw_io(a), iv_Te)
              else
                vt_a = node_list%node(unode)%values(raw_jb(a), raw_io(a), iv_T)
              end if

              ps0_x  = ps0_x  + vpsi_a * rR(a)
              ps0_y  = ps0_y  + vpsi_a * rZ(a)
              r0_raw = r0_raw + vr_a   * rv(a)
              r0_x   = r0_x   + vr_a   * rR(a)
              r0_y   = r0_y   + vr_a   * rZ(a)
              r0_p   = r0_p   + vr_a   * rP(a)
              T0_raw = T0_raw + vt_a   * rv(a)
              T0_x   = T0_x   + vt_a   * rR(a)
              T0_y   = T0_y   + vt_a   * rZ(a)
              T0_p   = T0_p   + vt_a   * rP(a)
            end do
          end if

          r0   = abs(r0_raw)
          T0   = abs(T0_raw)
          p0   = r0 * T0
          p0_x = r0_x * T0 + r0 * T0_x
          p0_y = r0_y * T0 + r0 * T0_y
          if (do_v) then
            p0_p = r0_p * T0 + r0 * T0_p
            P_B  = (F0 / BigR) * p0_p + (p0_x * ps0_y - p0_y * ps0_x)
          end if

          if (has_axis_u) then
            call pfafem_eval_qp_basis(node_list, element_list, ielm, eP, iq, ms, mt, &
                                      BigR, y_g, phi_q, .true., n_act, act_u, act_io, act_jb, &
                                      bv, bR, bZ, bP)
          else
            n_act = n_raw
            act_u(1:n_act)  = raw_u(1:n_act)
            act_io(1:n_act) = raw_io(1:n_act)
            act_jb(1:n_act) = raw_jb(1:n_act)
            bv(1:n_act) = rv(1:n_act); bR(1:n_act) = rR(1:n_act)
            bZ(1:n_act) = rZ(1:n_act); bP(1:n_act) = rP(1:n_act)
          end if

          do a = 1, n_act
            v_y = bZ(a)
            if (do_v) Db_tst = (F0 / BigR) * bP(a) + (bR(a) * ps0_y - v_y * ps0_x)

            do b = 1, n_act
              ivp    = pfafem_cache(ielm)%pair_map(act_u(a), act_u(b))
              u_x    = bR(b)
              u_y    = bZ(b)
              a_curv = -2.d0 * BigR**2 * v_y * (p0_x * u_y - p0_y * u_x) &
                       -4.d0 * GAMMA_l * BigR * p0 * v_y * u_y
              a_uu   = cv * a_curv

              W_vp(act_jb(a), act_jb(b), act_io(a), act_io(b), ivp) = &
                W_vp(act_jb(a), act_jb(b), act_io(a), act_io(b), ivp) + wst_tor * a_uu

              if (do_v) then
                Db_tri = (F0 / BigR) * bP(b) + (u_x * ps0_y - u_y * ps0_x)
                q_tri  = bv(b) * P_B + GAMMA_l * p0 * Db_tri

                WV_vp(act_jb(a), act_jb(b), act_io(a), act_io(b), ivp, 1) = &
                  WV_vp(act_jb(a), act_jb(b), act_io(a), act_io(b), ivp, 1) &
                  + wst_tor * cv * (Db_tst * q_tri / BigR)

                WV_vp(act_jb(a), act_jb(b), act_io(a), act_io(b), ivp, 2) = &
                  WV_vp(act_jb(a), act_jb(b), act_io(a), act_io(b), ivp, 2) &
                  - wst_tor * cv * Db_tst * (2.d0 * GAMMA_l * p0 * u_y)

                WV_vp(act_jb(a), act_jb(b), act_io(a), act_io(b), ivp, 3) = &
                  WV_vp(act_jb(a), act_jb(b), act_io(a), act_io(b), ivp, 3) &
                  + wst_tor * cv * (2.d0 * v_y * q_tri)
              end if
            end do
          end do

        end do
      end do
    end do
  end do
end subroutine pc_elt_matrix_force_pfafem

end module mod_pc_elt_matrix_force_fft
