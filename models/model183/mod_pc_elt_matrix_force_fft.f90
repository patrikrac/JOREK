!> model183's force operator W (stellarator reduced MHD, Nikulsin et al.:
!! B = grad chi + grad Psi x grad chi, geometry R(s,t,phi), Z(s,t,phi)), the
!! matrix ADDED to B_22; see models/model199/mod_pc_elt_matrix_force_fft.f90.
!! With Bv2 = |grad chi|^2, [a,b] = grad chi . (grad a x grad b) and
!! pref = (theta dt)^2/(1+zeta), the curvature part of the composed Schur
!! complement (bending and kink go through pair_w's explicit (psi, j)):
!!
!!   W(du,v) = - pref int D(du) [1/Bv2, v] dV,  D(du) = ( [p0,du] - gamma p0 [Bv2,du]/Bv2 ) / Bv2
!!
!! (the tokamak limit, Bv2 ~ 1/R^2, gives model199's curvature term). Each
!! side is split by the phi-derivative order a = 0, 1 on its toroidal basis
!! function (HZ, HZ_p), scattered through the p / n / k / kn channels.
!! Physical derivatives on the phi-dependent grid: f_phi = f_p - x_p f_R - y_p f_Z.
module mod_pc_elt_matrix_force_fft

  use mod_pc_fft_scatter, only: scatter_fft_to_elm, scatter_fft_to_elm_n, &
                                scatter_fft_to_elm_k, scatter_fft_to_elm_kn, pc_my_fft

  implicit none
  private

  public :: pc_elt_matrix_force_fft, pc_elt_matrix_force_pfafem

contains

!> [a,b] = grad chi . (grad a x grad b), from (R, Z, phi) gradients
pure real*8 function vbr(a, b, chR, chZ, chP, R)
  real*8, intent(in) :: a(3), b(3), chR, chZ, chP, R
  vbr = ( (a(2)*b(3) - a(3)*b(2))*chR + (a(3)*b(1) - a(1)*b(3))*chZ      &
        + (a(1)*b(2) - a(2)*b(1))*chP ) / R
end function vbr

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
  !> v_par's W terms (model600 only): zero here, for the common interface
  real*8, dimension(DFV, DFV, 3), intent(out), optional :: ELM_v

  !> channels (a, b): phi-derivative order on test, trial
  real*8, allocatable :: ELM_c(:,:,:,:,:)
  real*8, allocatable :: ELM_tmp(:,:)

  real*8, dimension(n_plane,n_gauss,n_gauss) :: x_g, x_s, x_t, x_p, y_s, y_t, y_p
  !> rho0 (1) and T0 (2): value, s, t and fixed-(s,t) phi derivatives
  real*8, dimension(n_plane,2,n_gauss,n_gauss) :: eq_g, eq_s, eq_t, eq_p
  integer, parameter :: iv_T  = max(1, var_T)
  integer, parameter :: iv_Ti = max(1, var_Ti)
  integer, parameter :: iv_Te = max(1, var_Te)

  integer :: i, j, ms, mt, mp, in, a, b, q
  integer :: index_ij, index_kl
  real*8  :: wst, BigR, xjac, theta, zeta_t, pref, GAMMA_l, wvol, sz, xp, yp, vr_m, vt_m
  real*8  :: dr0(3), dT0(3), r0, T0, p0, gp0(3)
  real*8  :: ch(0:2,0:2,0:2), chR, chZ, chP, Bv2, gBv2(3)
  real*8  :: dv(3,0:1)                                  !< (R, Z, phi) gradient per channel
  real*8  :: fc(0:1,NFV), fd(0:1,NFV)                   !< [1/Bv2, f], D(f)
  real*8  :: val

  real*8     :: in_fft(1:n_plane)
  complex*16 :: out_fft(1:n_plane)

  allocate(ELM_c(n_plane, NFV, NFV, 0:1, 0:1))
  ELM   = 0.d0
  if (present(ELM_v)) ELM_v = 0.d0
  ELM_c = 0.d0

  ! the model's own element routine overrides GAMMA locally
  GAMMA_l = 5.d0 / 3.d0
  theta  = time_evol_theta
  zeta_t = time_evol_zeta * 2.0d0 * tstep / (tstep + tstep_prev)
  pref   = (theta * tstep)**2 / (1.d0 + zeta_t)

  x_g = 0.d0; x_s = 0.d0; x_t = 0.d0; x_p = 0.d0
  y_s = 0.d0; y_t = 0.d0; y_p = 0.d0
  eq_g = 0.d0; eq_s = 0.d0; eq_t = 0.d0; eq_p = 0.d0
  do i = 1, n_vertex_max
    do j = 1, n_degrees
      sz = element%size(i,j)
      do ms = 1, n_gauss
        do mt = 1, n_gauss
          do mp = 1, n_plane
            do in = 1, n_coord_tor
              x_g(mp,ms,mt) = x_g(mp,ms,mt) + nodes(i)%x(in,j,1)*sz*H  (i,j,ms,mt)*HZ_coord  (in,mp)
              x_s(mp,ms,mt) = x_s(mp,ms,mt) + nodes(i)%x(in,j,1)*sz*H_s(i,j,ms,mt)*HZ_coord  (in,mp)
              x_t(mp,ms,mt) = x_t(mp,ms,mt) + nodes(i)%x(in,j,1)*sz*H_t(i,j,ms,mt)*HZ_coord  (in,mp)
              x_p(mp,ms,mt) = x_p(mp,ms,mt) + nodes(i)%x(in,j,1)*sz*H  (i,j,ms,mt)*HZ_coord_p(in,mp)
              y_s(mp,ms,mt) = y_s(mp,ms,mt) + nodes(i)%x(in,j,2)*sz*H_s(i,j,ms,mt)*HZ_coord  (in,mp)
              y_t(mp,ms,mt) = y_t(mp,ms,mt) + nodes(i)%x(in,j,2)*sz*H_t(i,j,ms,mt)*HZ_coord  (in,mp)
              y_p(mp,ms,mt) = y_p(mp,ms,mt) + nodes(i)%x(in,j,2)*sz*H  (i,j,ms,mt)*HZ_coord_p(in,mp)
            enddo
            do in = 1, n_tor
              vr_m = nodes(i)%values(in,j,var_rho)
              if (with_TiTe) then
                vt_m = nodes(i)%values(in,j,iv_Ti) + nodes(i)%values(in,j,iv_Te)
              else
                vt_m = nodes(i)%values(in,j,iv_T)
              endif
              eq_g(mp,1,ms,mt) = eq_g(mp,1,ms,mt) + vr_m*sz*H  (i,j,ms,mt)*HZ  (in,mp)
              eq_s(mp,1,ms,mt) = eq_s(mp,1,ms,mt) + vr_m*sz*H_s(i,j,ms,mt)*HZ  (in,mp)
              eq_t(mp,1,ms,mt) = eq_t(mp,1,ms,mt) + vr_m*sz*H_t(i,j,ms,mt)*HZ  (in,mp)
              eq_p(mp,1,ms,mt) = eq_p(mp,1,ms,mt) + vr_m*sz*H  (i,j,ms,mt)*HZ_p(in,mp)
              eq_g(mp,2,ms,mt) = eq_g(mp,2,ms,mt) + vt_m*sz*H  (i,j,ms,mt)*HZ  (in,mp)
              eq_s(mp,2,ms,mt) = eq_s(mp,2,ms,mt) + vt_m*sz*H_s(i,j,ms,mt)*HZ  (in,mp)
              eq_t(mp,2,ms,mt) = eq_t(mp,2,ms,mt) + vt_m*sz*H_t(i,j,ms,mt)*HZ  (in,mp)
              eq_p(mp,2,ms,mt) = eq_p(mp,2,ms,mt) + vt_m*sz*H  (i,j,ms,mt)*HZ_p(in,mp)
            enddo
          enddo
        enddo
      enddo
    enddo
  enddo

  do ms = 1, n_gauss
   do mt = 1, n_gauss
    wst = wgauss(ms)*wgauss(mt)

    do mp = 1, n_plane
      xjac = x_s(mp,ms,mt)*y_t(mp,ms,mt) - x_t(mp,ms,mt)*y_s(mp,ms,mt)
      xp   = x_p(mp,ms,mt); yp = y_p(mp,ms,mt)
      BigR = x_g(mp,ms,mt)
      wvol = wst * BigR * xjac                       ! dV = R dR dZ dphi

      dr0 = grad(eq_s(mp,1,ms,mt), eq_t(mp,1,ms,mt), eq_p(mp,1,ms,mt))
      dT0 = grad(eq_s(mp,2,ms,mt), eq_t(mp,2,ms,mt), eq_p(mp,2,ms,mt))
      r0  = eq_g(mp,1,ms,mt)
      T0  = eq_g(mp,2,ms,mt)
      p0  = r0*T0
      gp0 = dr0*T0 + r0*dT0

      !--- vacuum field: chi(mp,ms,mt,i,j,k) = d_R^i d_Z^j d_phi^k chi
      ch  = element%chi(mp,ms,mt,0:2,0:2,0:2)
      chR = ch(1,0,0); chZ = ch(0,1,0); chP = ch(0,0,1)
      Bv2 = chR**2 + chZ**2 + chP**2/BigR**2
      gBv2(1) = 2.d0*(chR*ch(2,0,0) + chZ*ch(1,1,0) + chP*ch(1,0,1)/BigR**2) - 2.d0*chP**2/BigR**3
      gBv2(2) = 2.d0*(chR*ch(1,1,0) + chZ*ch(0,2,0) + chP*ch(0,1,1)/BigR**2)
      gBv2(3) = 2.d0*(chR*ch(1,0,1) + chZ*ch(0,1,1) + chP*ch(0,0,2)/BigR**2)

      ! per basis function (test and trial alike), channel 0: HZ, 1: HZ_p
      do q = 1, NFV
        i = (q - 1) / n_degrees + 1; j = mod(q - 1, n_degrees) + 1
        dv(:,0) = grad(h_s(i,j,ms,mt)*element%size(i,j), h_t(i,j,ms,mt)*element%size(i,j), 0.d0)
        dv(:,1) = grad(0.d0, 0.d0, h(i,j,ms,mt)*element%size(i,j))
        do a = 0, 1
          fc(a,q) = - vbr(gBv2, dv(:,a), chR, chZ, chP, BigR)/Bv2**2
          fd(a,q) = ( vbr(gp0, dv(:,a), chR, chZ, chP, BigR)             &
                      - GAMMA_l*p0*vbr(gBv2, dv(:,a), chR, chZ, chP, BigR)/Bv2 ) / Bv2
        enddo
      enddo

      do index_ij = 1, NFV
        do index_kl = 1, NFV
          do a = 0, 1
            do b = 0, 1
              val = pref * ( - fd(b,index_kl)*fc(a,index_ij) )
              ELM_c(mp,index_ij,index_kl,a,b) = ELM_c(mp,index_ij,index_kl,a,b) + wvol*val
            enddo
          enddo
        enddo
      enddo

    enddo  ! mp
   enddo  ! mt
  enddo   ! ms

  allocate(ELM_tmp(DFV, DFV))
  do a = 0, 1
    do b = 0, 1
      ELM_tmp = 0.d0
      do i = 1, NFV
        do j = 1, NFV
          if (maxval(abs(ELM_c(1:n_plane,i,j,a,b))) == 0.d0) cycle
          in_fft = ELM_c(1:n_plane,i,j,a,b)
#ifdef USE_FFTW
          call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#else
          call pc_my_fft(in_fft, out_fft, n_plane)
#endif
          select case (2*a + b)
          case (0); call scatter_fft_to_elm   (out_fft, i, j, ELM_tmp, DFV)
          case (1); call scatter_fft_to_elm_n (out_fft, i, j, ELM_tmp, DFV)
          case (2); call scatter_fft_to_elm_k (out_fft, i, j, ELM_tmp, DFV)
          case (3); call scatter_fft_to_elm_kn(out_fft, i, j, ELM_tmp, DFV)
          end select
        enddo
      enddo
      ELM = ELM + ELM_tmp
    enddo
  enddo
  deallocate(ELM_tmp, ELM_c)
  ! the scatter accumulates each (row, col) pair twice (as mod_elt_matrix_fft)
  ELM = 0.5d0 * ELM

contains

  !> (R, Z, phi) gradient from the (s, t) derivatives and the fixed-(s,t)
  !! phi derivative f_p, on the current plane
  function grad(f_s, f_t, f_p) result(d)
    real*8, intent(in) :: f_s, f_t, f_p
    real*8 :: d(3)
    d(1) = (  y_t(mp,ms,mt)*f_s - y_s(mp,ms,mt)*f_t ) / xjac
    d(2) = ( -x_t(mp,ms,mt)*f_s + x_s(mp,ms,mt)*f_t ) / xjac
    d(3) = f_p - xp*d(1) - yp*d(2)
  end function grad

end subroutine pc_elt_matrix_force_fft

subroutine pc_elt_matrix_force_pfafem(node_list, element_list, ielm, W_vp, WV_vp)
  use mod_parameters
  use data_structure, only: type_node_list, type_element_list
  use gauss, only: n_gauss, xgauss, wgauss
  use phys_module
  use mod_pfafem, only: pfafem_cache, pfafem_el_lo, pfafem_el_hi, pfafem_ep_dim, &
                        pfafem_ng_tor, pfafem_ng_pol, pfafem_tau_q, pfafem_w_tor, &
                        pfafem_t_pol, pfafem_w_pol, pfafem_dphi, pfafem_chi_q, &
                        pfafem_qp_cached, pfafem_qp_geom, pfafem_qp_ngrp, &
                        pfafem_qp_gu, pfafem_qp_gjb, pfafem_qp_db, &
                        pfafem_eval_elem_geom, pfafem_eval_qp_basis
  implicit none

  type(type_node_list),    intent(in)            :: node_list
  type(type_element_list), intent(in)            :: element_list
  integer,                 intent(in)            :: ielm
  real*8,                  intent(out)           :: W_vp(:, :, :, :, :)
  real*8,                  intent(out), optional :: WV_vp(:, :, :, :, :, :)

  integer, parameter :: iv_T  = max(1, var_T)
  integer, parameter :: iv_Ti = max(1, var_Ti)
  integer, parameter :: iv_Te = max(1, var_Te)
  integer :: eP, ep1, iep_q, iq, ms, mt, iel_loc, a, b, unode, ivp
  integer :: n_grp, g_a, a0, io_a, u_a, jb_a
  integer :: n_raw, raw_u(64), raw_io(64), raw_jb(64)
  integer :: n_act, act_u(64), act_io(64), act_jb(64)
  real*8  :: GAMMA_l, theta, zeta_t, pref, phi_q, t_q, w_t, BigR, y_g, Rs, Rt, Zs, Zt, xjac, wvol
  real*8  :: rv(64), rR(64), rZ(64), rP(64), r_jet4(4)
  real*8  :: bv(64), bR(64), bZ(64), bP(64)
  real*8  :: r0, T0, p0, dr0(3), dT0(3), gp0(3), vr_a, vt_a
  real*8  :: ch(0:2, 0:2, 0:2), chR, chZ, chP, Bv2, gBv2(3), dva(3), fc(64), fd(64)
  logical :: has_axis_u, use_qp_c

  W_vp = 0.d0
  if (present(WV_vp)) WV_vp = 0.d0

  GAMMA_l = 5.d0 / 3.d0
  theta   = time_evol_theta
  zeta_t  = time_evol_zeta * 2.0d0 * tstep / (tstep + tstep_prev)
  pref    = (theta * tstep)**2 / (1.d0 + zeta_t)
  iel_loc = ielm - pfafem_el_lo + 1
  use_qp_c = pfafem_qp_cached .and. (ielm >= pfafem_el_lo) .and. (ielm <= pfafem_el_hi)
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
          if (use_qp_c) then
            BigR = pfafem_qp_geom(1,  ms, mt, iq, ep1, iel_loc)
            y_g  = pfafem_qp_geom(2,  ms, mt, iq, ep1, iel_loc)
            wvol = pfafem_qp_geom(20, ms, mt, iq, ep1, iel_loc)
          else
            if (ielm >= pfafem_el_lo .and. ielm <= pfafem_el_hi) then
              t_q = pfafem_t_pol(ms, mt, iq, iep_q, iel_loc)
              w_t = pfafem_w_pol(ms, mt, iq, iep_q, iel_loc)
            else
              t_q = xgauss(min(n_gauss, mt))
              w_t = wgauss(min(n_gauss, mt))
            end if
            call pfafem_eval_elem_geom(node_list, element_list, ielm, xgauss(ms), t_q, phi_q, &
                                       BigR, y_g, Rs, Rt, Zs, Zt)
            xjac = Rs * Zt - Rt * Zs
            wvol = wgauss(ms) * w_t * pfafem_w_tor(iq) * BigR * xjac
          end if

          r0  = 0.d0; dr0 = 0.d0
          T0  = 0.d0; dT0 = 0.d0
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
                vr_a  = node_list%node(unode)%values(jb_a, io_a, var_rho)
                if (with_TiTe) then
                  vt_a = node_list%node(unode)%values(jb_a, io_a, iv_Ti) &
                       + node_list%node(unode)%values(jb_a, io_a, iv_Te)
                else
                  vt_a = node_list%node(unode)%values(jb_a, io_a, iv_T)
                end if
                r0     = r0     + vr_a * r_jet4(1)
                dr0(1) = dr0(1) + vr_a * r_jet4(2)
                dr0(2) = dr0(2) + vr_a * r_jet4(3)
                dr0(3) = dr0(3) + vr_a * r_jet4(4)
                T0     = T0     + vt_a * r_jet4(1)
                dT0(1) = dT0(1) + vt_a * r_jet4(2)
                dT0(2) = dT0(2) + vt_a * r_jet4(3)
                dT0(3) = dT0(3) + vt_a * r_jet4(4)
              end do
            end do
          else
            call pfafem_eval_qp_basis(node_list, element_list, ielm, eP, iq, ms, mt, &
                                      BigR, y_g, phi_q, .false., n_raw, raw_u, raw_io, raw_jb, &
                                      rv, rR, rZ, rP)
            do a = 1, n_raw
              unode = pfafem_cache(ielm)%unodes(raw_u(a))
              vr_a  = node_list%node(unode)%values(raw_jb(a), raw_io(a), var_rho)
              if (with_TiTe) then
                vt_a = node_list%node(unode)%values(raw_jb(a), raw_io(a), iv_Ti) &
                     + node_list%node(unode)%values(raw_jb(a), raw_io(a), iv_Te)
              else
                vt_a = node_list%node(unode)%values(raw_jb(a), raw_io(a), iv_T)
              end if
              r0     = r0     + vr_a * rv(a)
              dr0(1) = dr0(1) + vr_a * rR(a)
              dr0(2) = dr0(2) + vr_a * rZ(a)
              dr0(3) = dr0(3) + vr_a * rP(a)
              T0     = T0     + vt_a * rv(a)
              dT0(1) = dT0(1) + vt_a * rR(a)
              dT0(2) = dT0(2) + vt_a * rZ(a)
              dT0(3) = dT0(3) + vt_a * rP(a)
            end do
          end if
          p0  = r0 * T0
          gp0 = dr0 * T0 + r0 * dT0

          ch  = pfafem_chi_q(0:2, 0:2, 0:2, ms, mt, iq, ep1, iel_loc)
          chR = ch(1, 0, 0); chZ = ch(0, 1, 0); chP = ch(0, 0, 1)
          Bv2 = chR**2 + chZ**2 + chP**2 / BigR**2
          gBv2(1) = 2.d0 * (chR*ch(2,0,0) + chZ*ch(1,1,0) + chP*ch(1,0,1)/BigR**2) - 2.d0*chP**2/BigR**3
          gBv2(2) = 2.d0 * (chR*ch(1,1,0) + chZ*ch(0,2,0) + chP*ch(0,1,1)/BigR**2)
          gBv2(3) = 2.d0 * (chR*ch(1,0,1) + chZ*ch(0,1,1) + chP*ch(0,0,2)/BigR**2)

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
            dva   = [bR(a), bZ(a), bP(a)]
            fc(a) = -vbr(gBv2, dva, chR, chZ, chP, BigR) / Bv2**2
            fd(a) = (vbr(gp0, dva, chR, chZ, chP, BigR) &
                    - GAMMA_l * p0 * vbr(gBv2, dva, chR, chZ, chP, BigR) / Bv2) / Bv2
          end do

          do a = 1, n_act
            do b = 1, n_act
              ivp = pfafem_cache(ielm)%pair_map(act_u(a), act_u(b))
              W_vp(act_jb(a), act_jb(b), act_io(a), act_io(b), ivp) = &
                W_vp(act_jb(a), act_jb(b), act_io(a), act_io(b), ivp) &
                + wvol * pref * (-fd(b) * fc(a))
            end do
          end do
        end do
      end do
    end do
  end do
end subroutine pc_elt_matrix_force_pfafem

end module mod_pc_elt_matrix_force_fft
