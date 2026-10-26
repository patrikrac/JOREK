p='models/model199/mod_pc_elt_matrix_force_fft.f90'; full=open(p).read()
k=full.index('subroutine pc_elt_matrix_peel_fft(element, nodes, ELM)')
k=full.rfind('!> The scalar operators', 0, k)
head, s = full[:k], full[k:]
def rep(a,b):
    global s
    assert s.count(a)==1, a[:70]
    s=s.replace(a,b)
head=head.replace("integer, parameter, public :: PEEL_NOP = 11   !< pc_elt_matrix_peel_fft's operators",
    "integer, parameter, public :: PEEL_NOP = 12   !< pc_elt_matrix_peel_fft's operators")
rep("""!! 9..11 are v2's products formed per element (-sf_peel_ebe): their sums keep""",
"""!!   ELM(:,:,12) = int R L_w v L_w u, L_w u = div(w grad u): the sandwich in
!!                 strong form (K_w X K_w with X the 1 / R mass inverse, for
!!                 smooth u), the element stencil without element-boundary terms
!! 9..11 are v2's products formed per element (-sf_peel_ebe): their sums keep""")
rep("""  real*8, dimension(n_gauss, n_gauss) :: x_g, x_s, x_t, y_s, y_t
  real*8, dimension(n_plane, 2, n_gauss, n_gauss) :: eq_g
  real*8, dimension(n_plane, n_gauss, n_gauss) :: ps_s, ps_t
  real*8 :: f_s(NFV), f_t(NFV)""","""  real*8, dimension(n_gauss, n_gauss) :: x_g, x_s, x_t, y_s, y_t
  real*8, dimension(n_gauss, n_gauss) :: x_ss, x_st, x_tt, y_ss, y_st, y_tt
  real*8, dimension(n_plane, 2, n_gauss, n_gauss) :: eq_g
  real*8, dimension(n_plane, n_gauss, n_gauss) :: ps_s, ps_t, rs_s, rs_t
  real*8 :: f_s(NFV), f_t(NFV), f_lap(NFV), lw(NFV)
  real*8 :: xjac_x, xjac_y, f_ss, f_st, f_tt, r0_x, r0_y, wh, wh_x, wh_y""")
rep("""  ps_s = 0.d0; ps_t = 0.d0
  thdt = time_evol_theta * tstep""","""  ps_s = 0.d0; ps_t = 0.d0; rs_s = 0.d0; rs_t = 0.d0
  x_ss = 0.d0; x_st = 0.d0; x_tt = 0.d0; y_ss = 0.d0; y_st = 0.d0; y_tt = 0.d0
  thdt = time_evol_theta * tstep""")
rep("""          y_t(ms,mt) = y_t(ms,mt) + nodes(i)%x(1,j,2) * sz * H_t(i,j,ms,mt)
          do mp = 1, n_plane""","""          y_t(ms,mt) = y_t(ms,mt) + nodes(i)%x(1,j,2) * sz * H_t(i,j,ms,mt)
          x_ss(ms,mt) = x_ss(ms,mt) + nodes(i)%x(1,j,1) * sz * H_ss(i,j,ms,mt)
          x_st(ms,mt) = x_st(ms,mt) + nodes(i)%x(1,j,1) * sz * H_st(i,j,ms,mt)
          x_tt(ms,mt) = x_tt(ms,mt) + nodes(i)%x(1,j,1) * sz * H_tt(i,j,ms,mt)
          y_ss(ms,mt) = y_ss(ms,mt) + nodes(i)%x(1,j,2) * sz * H_ss(i,j,ms,mt)
          y_st(ms,mt) = y_st(ms,mt) + nodes(i)%x(1,j,2) * sz * H_st(i,j,ms,mt)
          y_tt(ms,mt) = y_tt(ms,mt) + nodes(i)%x(1,j,2) * sz * H_tt(i,j,ms,mt)
          do mp = 1, n_plane""")
rep("""              ps_t(mp,ms,mt) = ps_t(mp,ms,mt) + nodes(i)%values(in,j,var_psi) * sz * H_t(i,j,ms,mt) * HZ(in,mp)
            enddo""","""              ps_t(mp,ms,mt) = ps_t(mp,ms,mt) + nodes(i)%values(in,j,var_psi) * sz * H_t(i,j,ms,mt) * HZ(in,mp)
              rs_s(mp,ms,mt) = rs_s(mp,ms,mt) + nodes(i)%values(in,j,var_rho) * sz * H_s(i,j,ms,mt) * HZ(in,mp)
              rs_t(mp,ms,mt) = rs_t(mp,ms,mt) + nodes(i)%values(in,j,var_rho) * sz * H_t(i,j,ms,mt) * HZ(in,mp)
            enddo""")
rep("""      f_s(q) = H_s(i,j,ms,mt) * element%size(i,j)
      f_t(q) = H_t(i,j,ms,mt) * element%size(i,j)
""","""      f_s(q) = H_s(i,j,ms,mt) * element%size(i,j)
      f_t(q) = H_t(i,j,ms,mt) * element%size(i,j)
    enddo
    ! the basis' poloidal Laplacian u_xx + u_yy (model199's v_xx, v_yy)
    xjac_x = (x_ss(ms,mt)*y_t(ms,mt)**2 - y_ss(ms,mt)*x_t(ms,mt)*y_t(ms,mt) - 2.d0*x_st(ms,mt)*y_s(ms,mt)*y_t(ms,mt) &
           + y_st(ms,mt)*(x_s(ms,mt)*y_t(ms,mt) + x_t(ms,mt)*y_s(ms,mt)) &
           + x_tt(ms,mt)*y_s(ms,mt)**2 - y_tt(ms,mt)*x_s(ms,mt)*y_s(ms,mt)) / xjac
    xjac_y = (y_tt(ms,mt)*x_s(ms,mt)**2 - x_tt(ms,mt)*y_s(ms,mt)*x_s(ms,mt) - 2.d0*y_st(ms,mt)*x_t(ms,mt)*x_s(ms,mt) &
           + x_st(ms,mt)*(y_t(ms,mt)*x_s(ms,mt) + y_s(ms,mt)*x_t(ms,mt)) &
           + y_ss(ms,mt)*x_t(ms,mt)**2 - x_ss(ms,mt)*y_t(ms,mt)*x_t(ms,mt)) / xjac
    do q = 1, NFV
      i = (q - 1) / n_degrees + 1; j = mod(q - 1, n_degrees) + 1
      f_ss = H_ss(i,j,ms,mt) * element%size(i,j)
      f_st = H_st(i,j,ms,mt) * element%size(i,j)
      f_tt = H_tt(i,j,ms,mt) * element%size(i,j)
      f_lap(q) = (f_ss * y_t(ms,mt)**2 - 2.d0*f_st * y_s(ms,mt)*y_t(ms,mt) + f_tt * y_s(ms,mt)**2 &
               + f_s(q) * (y_st(ms,mt)*y_t(ms,mt) - y_tt(ms,mt)*y_s(ms,mt)) &
               + f_t(q) * (y_st(ms,mt)*y_s(ms,mt) - y_ss(ms,mt)*y_t(ms,mt))) / xjac**2 &
               - xjac_x * (f_s(q) * y_t(ms,mt) - f_t(q) * y_s(ms,mt)) / xjac**2 &
               + (f_ss * x_t(ms,mt)**2 - 2.d0*f_st * x_s(ms,mt)*x_t(ms,mt) + f_tt * x_s(ms,mt)**2 &
               + f_s(q) * (x_st(ms,mt)*x_t(ms,mt) - x_tt(ms,mt)*x_s(ms,mt)) &
               + f_t(q) * (x_st(ms,mt)*x_s(ms,mt) - x_ss(ms,mt)*x_t(ms,mt))) / xjac**2 &
               - xjac_y * (- f_s(q) * x_t(ms,mt) + f_t(q) * x_s(ms,mt)) / xjac**2
""")
rep("""      ! eta_T as model199's element matrix forms it""","""      ! the sandwich's L_w = div(w grad .), w = sqrt(eta_num R^3 rho0)
      r0_x = (  y_t(ms,mt)*rs_s(mp,ms,mt) - y_s(ms,mt)*rs_t(mp,ms,mt) ) / xjac
      r0_y = ( -x_t(ms,mt)*rs_s(mp,ms,mt) + x_s(ms,mt)*rs_t(mp,ms,mt) ) / xjac
      wh = sqrt(eta_num * BigR**3 * max(r0, 1.d-12))
      wh_x = wh * (1.5d0 / BigR + 0.5d0 * r0_x / max(r0, 1.d-12))
      wh_y = wh * 0.5d0 * r0_y / max(r0, 1.d-12)
      lw = wh * f_lap + wh_x * f_x + wh_y * f_y
      ! eta_T as model199's element matrix forms it""")
rep("""          ELM_p(mp,i,j,8) = ELM_p(mp,i,j,8) + wst * f_v(i) * f_v(j) / BigR * xjac
""","""          ELM_p(mp,i,j,8) = ELM_p(mp,i,j,8) + wst * f_v(i) * f_v(j) / BigR * xjac
          ELM_p(mp,i,j,12) = ELM_p(mp,i,j,12) + wst * BigR * lw(i) * lw(j) * xjac
""")
rep("""  do op = 1, 8
    do i = 1, NFV""","""  do op = 1, PEEL_NOP
    if (op >= 9 .and. op <= 11) cycle            ! the element products, below
    do i = 1, NFV""")
open(p,'w').write(head+s)
