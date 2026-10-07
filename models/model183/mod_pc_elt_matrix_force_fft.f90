!> Workstream E for the stellarator model: the reduced-MHD momentum-Schur FORCE
!! OPERATOR W of model183, assembled directly (no matrix triple product).
!!
!! Same role as models/model199/mod_pc_elt_matrix_force_fft.f90 (read its header
!! first): S_uu = B_22 + W, and this routine assembles exactly the matrix to ADD
!! to B_22. What changes is the model: model183 (Nikulsin et al.) writes the
!! field as B = grad(chi) + grad(Psi) x grad(chi) with chi the (Dommaschk) vacuum
!! potential, and the geometry R(s,t,phi), Z(s,t,phi) depends on phi.
!!
!! NOTATION (all at fixed R, Z, phi; Psi, j are the model's Psi/F0, zj/F0, as
!! mod_elt_matrix_fft.f90:321-325 scales them):
!!   Bv2       = |grad chi|^2
!!   [a,b]     = grad chi . (grad a x grad b)                    (Bv_pbrack)
!!   N(f)      = grad chi . grad f + [f, Psi0] = c . (f_R, f_Z, f_phi)
!!               (B0_parderiv; c = the contravariant components of B0)
!!   inprod    = grad a . G . grad b,  G = diag(1,1,1/R^2) - g g^T/Bv2,
!!               g = (chi_R, chi_Z, chi_phi/R^2)                 (perp to grad chi)
!!
!! THE COMPOSITION. Eliminating (Psi, zj, p) from the Phi row of
!! mod_equations.f90 with their time-derivative (mass) parts only:
!!   Psi row:  (1+zeta) dPsi = theta dt N(dPhi)/Bv2
!!   zj  row:  int q Bv2 dj  = - int Bv2 inprod(q, dPsi)         (all q)
!!   p   row:  (1+zeta) dp   = - theta dt ( [p0,dPhi] - gamma p0 [Bv2,dPhi]/Bv2 )/Bv2
!!   Phi row:  - theta dt v N(dj) - theta dt v [j0, dPsi] + theta dt [v, dp]/Bv2
!! N and [.,.] are antisymmetric under int dV because div(grad chi) = 0 and
!! div(grad Psi x grad chi) = 0, and [.,.] is cyclic: int a[b,c] = int c[a,b].
!! With chi^ = N(du)/Bv2, zeta^ = N(v)/Bv2 and pref = (theta dt)^2/(1+zeta):
!!
!!   W(du,v) = pref int [ - Bv2 inprod(zeta^, chi^)              <- bending
!!                        - chi^ [v, j0]                         <- kink
!!                        - D(du) [1/Bv2, v] ] dV                <- curvature
!!   D(du)   = ( [p0,du] - gamma p0 [Bv2,du]/Bv2 ) / Bv2
!!
!! The tokamak limit (chi = -phi, F0 const) gives back model199's three terms:
!! Bv2 ~ 1/R^2, so [1/Bv2, v] ~ 2R v_Z is the toroidal curvature. Bending and
!! the gamma part of the curvature term are symmetric and negative
!! semi-definite; the kink and the grad p0 part are symmetric together exactly
!! at reduced force balance, so the symmetry defect of W at equilibrium measures
!! the force-balance residual of the (3D) equilibrium.
!!
!! THREE TOROIDAL CHANNELS PER SIDE. grad(zeta^) needs the second derivatives of
!! v, INCLUDING v_phi,phi (the phi component of the gradient, and the phi
!! derivative of v_phi inside N). model183 drops those (its FFT has four
!! channels: no / one phi derivative on test and trial). Here every test and
!! trial quantity is split by the order a = 0,1,2 of the phi derivative on its
!! toroidal basis function, HZ, HZ_p, HZ_pp = -n^2 HZ, giving nine channels
!! (a,b). Channels with a <= 1 and b <= 1 go through the shared scatter routines
!! (p, k, n, kn); a or b = 2 is scattered as a = 0 (b = 0) and the rows (cols)
!! scaled by -mode^2.
!!
!! PHYSICAL DERIVATIVES ON A PHI-DEPENDENT GRID. At fixed (s,t) d/dphi is not
!! d/dphi at fixed (R,Z): f_phi = f_p - x_p f_R - y_p f_Z (as in
!! mod_elt_matrix_fft.f90:600-628), and the second derivatives follow by
!! applying that twice; f_phi,phi is the formula model183 has commented out at
!! mod_elt_matrix_fft.f90:613-617 (re-derived, identical).
!!
!! MEASURED AGAINST THE DISCRETE CORRECTION (W7-A 1 Pa, GVEC 41x48, n_tor 3;
!! C = -[B_21 B_23] pair_psi^-1 [B_12; 0] from the dumped blocks):
!!   - on smooth n /= 0 fields that vanish at the wall, v.Wv / v.Cv = 1.002;
!!     on fields that reach the wall 1.4 (C has psi = j = 0 there, W not);
!!   - the generalised eigenvalues of (B_22 + C, B_22 + W) go down to 1/7900
!!     on the st DOFs of the first ring off the axis: the composed operator
!!     carries two derivatives per side, ~ 1/xjac^2, and the GVEC grid's first
!!     ring sits at s = 1/40. B_22 + W ("w" form) therefore does NOT
!!     precondition there (no convergence in 400 its, while the exact Schur in
!!     the same sweep needs 8-14). Use "wpj": its bending and kink go through
!!     the explicit (psi, j) fields, and W keeps the curvature (terms 6) only.
!!
!! TERM SELECTION (terms, i.e. sf_force_terms; -sf_w_terms for the tests):
!!   1 = bending + kink + curvature      2 = bending + kink     3 = bending only
!!   5 = kink + curvature                6 = curvature only
!!   4 = UNIT TEST: theta * inprod(v,u) * R xjac, i.e. exactly A's (w, Phi)
!!       block (amat_semianalytic(var_w, var_Phi)); W_force must equal it to
!!       round-off. Exercises the geometry, channels a,b <= 1 and the scatter.
!!   7 = UNIT TEST: int (v_phi,phi u + v_phi u_phi) dV, zero by integration by
!!       parts in phi; exercises the a = 2 channel. Compare against 8.
!!   8 = UNIT TEST: int v_phi u_phi dV alone, the scale for 7.
!!   9 = UNIT TEST: - theta dt v N(u)/Bv2, i.e. A's (Psi, Phi) block B_12
!!       (checks B0's contravariant components c, Psi0 bracket included)
!!  10 = UNIT TEST: - theta dt v N(u) / F0, i.e. A's (Phi, zj) block B_23
!!  11 = UNIT TEST: theta Bv2 inprod(v,u) / F0, i.e. A's (zj, Psi) block B_31
module mod_pc_elt_matrix_force_fft

  use mod_pc_fft_scatter, only: scatter_fft_to_elm, scatter_fft_to_elm_n, &
                                scatter_fft_to_elm_k, scatter_fft_to_elm_kn, pc_my_fft

  implicit none
  private

  public :: pc_elt_matrix_force_fft

  !> Physical derivatives of one function, in this order
  integer, parameter :: D_R = 1, D_Z = 2, D_P = 3, D_RR = 4, D_RZ = 5, D_ZZ = 6, &
                        D_RP = 7, D_ZP = 8, D_PP = 9, ND = 9

  !> Geometry at one Gauss point on one plane
  type geom_t
    real*8 :: x_s, x_t, x_ss, x_st, x_tt, x_p, x_sp, x_tp, x_pp
    real*8 :: y_s, y_t, y_ss, y_st, y_tt, y_p, y_sp, y_tp, y_pp
    real*8 :: xjac, xjac_x, xjac_y
    real*8 :: x_p_x, x_p_y, y_p_x, y_p_y
  end type geom_t

contains

!> Physical (fixed R, Z, phi) derivatives from the (s,t) and fixed-(s,t) phi
!! derivatives. Linear in the f_* inputs, so it serves both the background
!! fields and each toroidal channel of a basis function.
pure function phys_derivs(g, f_s, f_t, f_ss, f_st, f_tt, f_p, f_sp, f_tp, f_pp) result(d)
  type(geom_t), intent(in) :: g
  real*8, intent(in) :: f_s, f_t, f_ss, f_st, f_tt, f_p, f_sp, f_tp, f_pp
  real*8 :: d(ND)
  real*8 :: fR, fZ, fRR, fZZ, fRZ, fpR, fpZ

  fR  = (  g%y_t*f_s - g%y_s*f_t ) / g%xjac
  fZ  = ( -g%x_t*f_s + g%x_s*f_t ) / g%xjac
  ! The three second derivatives exactly as mod_elt_matrix_fft.f90:290-306
  fRR = ( f_ss*g%y_t**2 - 2.d0*f_st*g%y_s*g%y_t + f_tt*g%y_s**2                 &
        + f_s*(g%y_st*g%y_t - g%y_tt*g%y_s)                                     &
        + f_t*(g%y_st*g%y_s - g%y_ss*g%y_t) ) / g%xjac**2                       &
        - g%xjac_x*( f_s*g%y_t - f_t*g%y_s ) / g%xjac**2
  fZZ = ( f_ss*g%x_t**2 - 2.d0*f_st*g%x_s*g%x_t + f_tt*g%x_s**2                 &
        + f_s*(g%x_st*g%x_t - g%x_tt*g%x_s)                                     &
        + f_t*(g%x_st*g%x_s - g%x_ss*g%x_t) ) / g%xjac**2                       &
        - g%xjac_y*( -f_s*g%x_t + f_t*g%x_s ) / g%xjac**2
  fRZ = ( -f_ss*g%y_t*g%x_t - f_tt*g%x_s*g%y_s                                  &
        + f_st*(g%y_s*g%x_t + g%y_t*g%x_s)                                      &
        - f_s*(g%x_st*g%y_t - g%x_tt*g%y_s)                                     &
        - f_t*(g%x_st*g%y_s - g%x_ss*g%y_t) ) / g%xjac**2                       &
        - g%xjac_x*( -f_s*g%x_t + f_t*g%x_s ) / g%xjac**2
  fpR = (  g%y_t*f_sp - g%y_s*f_tp ) / g%xjac
  fpZ = ( -g%x_t*f_sp + g%x_s*f_tp ) / g%xjac

  d(D_R)  = fR
  d(D_Z)  = fZ
  d(D_RR) = fRR
  d(D_RZ) = fRZ
  d(D_ZZ) = fZZ
  d(D_P)  = f_p - g%x_p*fR - g%y_p*fZ
  d(D_RP) = fpR - g%x_p_x*fR - g%x_p*fRR - g%y_p_x*fZ - g%y_p*fRZ
  d(D_ZP) = fpZ - g%x_p_y*fR - g%x_p*fRZ - g%y_p_y*fZ - g%y_p*fZZ
  d(D_PP) = f_pp - g%x_pp*fR - g%y_pp*fZ - 2.d0*(g%x_p*fpR + g%y_p*fpZ)          &
          + 2.d0*( g%x_p*g%x_p_x*fR + g%x_p*g%y_p_x*fZ                          &
                 + g%y_p*g%x_p_y*fR + g%y_p*g%y_p_y*fZ )                        &
          + g%x_p**2*fRR + 2.d0*g%x_p*g%y_p*fRZ + g%y_p**2*fZZ
end function phys_derivs

!> [a,b] = grad chi . (grad a x grad b), from first derivatives (R, Z, phi)
pure real*8 function vbr(a, b, chR, chZ, chP, R)
  real*8, intent(in) :: a(3), b(3), chR, chZ, chP, R
  vbr = ( (a(2)*b(3) - a(3)*b(2))*chR + (a(3)*b(1) - a(1)*b(3))*chZ      &
        + (a(1)*b(2) - a(2)*b(1))*chP ) / R
end function vbr

subroutine pc_elt_matrix_force_fft(element, nodes, xpoint2, xcase2, &
                                   R_axis, Z_axis, psi_axis, psi_bnd, &
                                   R_xpoint, Z_xpoint, ELM, terms)

  use constants
  use mod_parameters
  use data_structure, only: type_element, type_node
  use gauss
  use basis_at_gaussian
  use phys_module

  implicit none

  type(type_element), intent(in) :: element
  type(type_node),    intent(in) :: nodes(n_vertex_max)
  logical,            intent(in) :: xpoint2
  integer,            intent(in) :: xcase2
  real*8,             intent(in) :: R_axis, Z_axis, psi_axis, psi_bnd
  real*8,             intent(in) :: R_xpoint(2), Z_xpoint(2)

#define NFV (n_vertex_max*n_degrees)
#define DFV (n_tor*n_vertex_max*n_degrees)

  real*8, dimension(DFV, DFV), intent(out) :: ELM
  integer, intent(in) :: terms

  !> Nine toroidal channels (a, b): phi-derivative order on test, trial
  real*8, allocatable :: ELM_c(:,:,:,:,:)
  real*8, allocatable :: ELM_tmp(:,:)

  real*8, dimension(n_plane,n_gauss,n_gauss) :: x_g, x_s, x_t, x_ss, x_st, x_tt, x_p, x_sp, x_tp, x_pp
  real*8, dimension(n_plane,n_gauss,n_gauss) :: y_g, y_s, y_t, y_ss, y_st, y_tt, y_p, y_sp, y_tp, y_pp
  real*8, dimension(n_plane,n_var,n_gauss,n_gauss) :: eq_g, eq_s, eq_t, eq_ss, eq_st, eq_tt
  real*8, dimension(n_plane,n_var,n_gauss,n_gauss) :: eq_p, eq_sp, eq_tp, eq_pp

  type(geom_t) :: g
  integer :: i, j, k, l, ms, mt, mp, in, a, b, q, fo, idx
  integer :: index_ij, index_kl
  real*8  :: wst, BigR, theta, zeta_t, pref, GAMMA_l, wvol

  ! Background (physical derivatives)
  real*8 :: dps(ND), dzj(ND), dr0(ND), dT0(ND)
  real*8 :: r0, T0, p0, gp0(3), gj0(3)
  real*8 :: ch(0:2,0:2,0:2), chR, chZ, chP, Bv2, gBv2(3), gvec(3)
  real*8 :: c(3), dc(3,3)          ! c(k) = B0^k, dc(m,k) = d_m c(k)
  real*8 :: Gm(3,3)                ! perp metric

  ! Basis-function derivatives per channel: dv(:, a)
  real*8 :: dv(ND,0:2)
  ! Per basis function q and channel a (test and trial alike): grad(N/Bv2),
  ! G grad(N/Bv2), N/Bv2, [f,j0], [1/Bv2,f], D(f), grad f, G grad f, f_phiphi, f
  real*8 :: fz(3,0:2,NFV), fgz(3,0:2,NFV), fn(0:2,NFV), fk(0:2,NFV), fc(0:2,NFV), fd(0:2,NFV)
  real*8 :: fg(3,0:2,NFV), fgg(3,0:2,NFV), fpp(0:2,NFV), fval(0:2,NFV)
  real*8 :: Nv, dN(3)
  real*8 :: val

  real*8     :: in_fft(1:n_plane)
  complex*16 :: out_fft(1:n_plane)

  allocate(ELM_c(n_plane, NFV, NFV, 0:2, 0:2))
  ELM   = 0.d0
  ELM_c = 0.d0

  ! Same local overrides as model183's own element routine
  GAMMA_l = 5.d0 / 3.d0
  theta  = time_evol_theta
  zeta_t = time_evol_zeta * 2.0d0 * tstep / (tstep + tstep_prev)
  pref   = (theta * tstep)**2 / (1.d0 + zeta_t)

  fo = terms

  !--------------------------------------------------------------------------
  ! Geometry and background on every plane (as mod_elt_matrix_fft.f90:155-215)
  !--------------------------------------------------------------------------
  x_g = 0.d0; x_s = 0.d0; x_t = 0.d0; x_ss = 0.d0; x_st = 0.d0; x_tt = 0.d0
  x_p = 0.d0; x_sp = 0.d0; x_tp = 0.d0; x_pp = 0.d0
  y_g = 0.d0; y_s = 0.d0; y_t = 0.d0; y_ss = 0.d0; y_st = 0.d0; y_tt = 0.d0
  y_p = 0.d0; y_sp = 0.d0; y_tp = 0.d0; y_pp = 0.d0
  eq_g = 0.d0; eq_s = 0.d0; eq_t = 0.d0; eq_ss = 0.d0; eq_st = 0.d0; eq_tt = 0.d0
  eq_p = 0.d0; eq_sp = 0.d0; eq_tp = 0.d0; eq_pp = 0.d0

  do i = 1, n_vertex_max
    do j = 1, n_degrees
      do ms = 1, n_gauss
        do mt = 1, n_gauss
          do mp = 1, n_plane
            do in = 1, n_coord_tor
              x_g (mp,ms,mt) = x_g (mp,ms,mt) + nodes(i)%x(in,j,1)*element%size(i,j)*H   (i,j,ms,mt)*HZ_coord   (in,mp)
              x_s (mp,ms,mt) = x_s (mp,ms,mt) + nodes(i)%x(in,j,1)*element%size(i,j)*H_s (i,j,ms,mt)*HZ_coord   (in,mp)
              x_t (mp,ms,mt) = x_t (mp,ms,mt) + nodes(i)%x(in,j,1)*element%size(i,j)*H_t (i,j,ms,mt)*HZ_coord   (in,mp)
              x_ss(mp,ms,mt) = x_ss(mp,ms,mt) + nodes(i)%x(in,j,1)*element%size(i,j)*H_ss(i,j,ms,mt)*HZ_coord   (in,mp)
              x_st(mp,ms,mt) = x_st(mp,ms,mt) + nodes(i)%x(in,j,1)*element%size(i,j)*H_st(i,j,ms,mt)*HZ_coord   (in,mp)
              x_tt(mp,ms,mt) = x_tt(mp,ms,mt) + nodes(i)%x(in,j,1)*element%size(i,j)*H_tt(i,j,ms,mt)*HZ_coord   (in,mp)
              x_p (mp,ms,mt) = x_p (mp,ms,mt) + nodes(i)%x(in,j,1)*element%size(i,j)*H   (i,j,ms,mt)*HZ_coord_p (in,mp)
              x_sp(mp,ms,mt) = x_sp(mp,ms,mt) + nodes(i)%x(in,j,1)*element%size(i,j)*H_s (i,j,ms,mt)*HZ_coord_p (in,mp)
              x_tp(mp,ms,mt) = x_tp(mp,ms,mt) + nodes(i)%x(in,j,1)*element%size(i,j)*H_t (i,j,ms,mt)*HZ_coord_p (in,mp)
              x_pp(mp,ms,mt) = x_pp(mp,ms,mt) + nodes(i)%x(in,j,1)*element%size(i,j)*H   (i,j,ms,mt)*HZ_coord_pp(in,mp)

              y_g (mp,ms,mt) = y_g (mp,ms,mt) + nodes(i)%x(in,j,2)*element%size(i,j)*H   (i,j,ms,mt)*HZ_coord   (in,mp)
              y_s (mp,ms,mt) = y_s (mp,ms,mt) + nodes(i)%x(in,j,2)*element%size(i,j)*H_s (i,j,ms,mt)*HZ_coord   (in,mp)
              y_t (mp,ms,mt) = y_t (mp,ms,mt) + nodes(i)%x(in,j,2)*element%size(i,j)*H_t (i,j,ms,mt)*HZ_coord   (in,mp)
              y_ss(mp,ms,mt) = y_ss(mp,ms,mt) + nodes(i)%x(in,j,2)*element%size(i,j)*H_ss(i,j,ms,mt)*HZ_coord   (in,mp)
              y_st(mp,ms,mt) = y_st(mp,ms,mt) + nodes(i)%x(in,j,2)*element%size(i,j)*H_st(i,j,ms,mt)*HZ_coord   (in,mp)
              y_tt(mp,ms,mt) = y_tt(mp,ms,mt) + nodes(i)%x(in,j,2)*element%size(i,j)*H_tt(i,j,ms,mt)*HZ_coord   (in,mp)
              y_p (mp,ms,mt) = y_p (mp,ms,mt) + nodes(i)%x(in,j,2)*element%size(i,j)*H   (i,j,ms,mt)*HZ_coord_p (in,mp)
              y_sp(mp,ms,mt) = y_sp(mp,ms,mt) + nodes(i)%x(in,j,2)*element%size(i,j)*H_s (i,j,ms,mt)*HZ_coord_p (in,mp)
              y_tp(mp,ms,mt) = y_tp(mp,ms,mt) + nodes(i)%x(in,j,2)*element%size(i,j)*H_t (i,j,ms,mt)*HZ_coord_p (in,mp)
              y_pp(mp,ms,mt) = y_pp(mp,ms,mt) + nodes(i)%x(in,j,2)*element%size(i,j)*H   (i,j,ms,mt)*HZ_coord_pp(in,mp)
            enddo
            do k = 1, n_var
              do in = 1, n_tor
                eq_g (mp,k,ms,mt) = eq_g (mp,k,ms,mt) + nodes(i)%values(in,j,k)*element%size(i,j)*H   (i,j,ms,mt)*HZ   (in,mp)
                eq_s (mp,k,ms,mt) = eq_s (mp,k,ms,mt) + nodes(i)%values(in,j,k)*element%size(i,j)*H_s (i,j,ms,mt)*HZ   (in,mp)
                eq_t (mp,k,ms,mt) = eq_t (mp,k,ms,mt) + nodes(i)%values(in,j,k)*element%size(i,j)*H_t (i,j,ms,mt)*HZ   (in,mp)
                eq_ss(mp,k,ms,mt) = eq_ss(mp,k,ms,mt) + nodes(i)%values(in,j,k)*element%size(i,j)*H_ss(i,j,ms,mt)*HZ   (in,mp)
                eq_st(mp,k,ms,mt) = eq_st(mp,k,ms,mt) + nodes(i)%values(in,j,k)*element%size(i,j)*H_st(i,j,ms,mt)*HZ   (in,mp)
                eq_tt(mp,k,ms,mt) = eq_tt(mp,k,ms,mt) + nodes(i)%values(in,j,k)*element%size(i,j)*H_tt(i,j,ms,mt)*HZ   (in,mp)
                eq_p (mp,k,ms,mt) = eq_p (mp,k,ms,mt) + nodes(i)%values(in,j,k)*element%size(i,j)*H   (i,j,ms,mt)*HZ_p (in,mp)
                eq_sp(mp,k,ms,mt) = eq_sp(mp,k,ms,mt) + nodes(i)%values(in,j,k)*element%size(i,j)*H_s (i,j,ms,mt)*HZ_p (in,mp)
                eq_tp(mp,k,ms,mt) = eq_tp(mp,k,ms,mt) + nodes(i)%values(in,j,k)*element%size(i,j)*H_t (i,j,ms,mt)*HZ_p (in,mp)
                eq_pp(mp,k,ms,mt) = eq_pp(mp,k,ms,mt) + nodes(i)%values(in,j,k)*element%size(i,j)*H   (i,j,ms,mt)*HZ_pp(in,mp)
              enddo
            enddo
          enddo
        enddo
      enddo
    enddo
  enddo

  !--------------------------------------------------------------------------
  ! Gauss integration, plane by plane
  !--------------------------------------------------------------------------
  do ms = 1, n_gauss
   do mt = 1, n_gauss
    wst = wgauss(ms)*wgauss(mt)

    do mp = 1, n_plane

      g%x_s  = x_s (mp,ms,mt); g%x_t  = x_t (mp,ms,mt); g%x_ss = x_ss(mp,ms,mt)
      g%x_st = x_st(mp,ms,mt); g%x_tt = x_tt(mp,ms,mt); g%x_p  = x_p (mp,ms,mt)
      g%x_sp = x_sp(mp,ms,mt); g%x_tp = x_tp(mp,ms,mt); g%x_pp = x_pp(mp,ms,mt)
      g%y_s  = y_s (mp,ms,mt); g%y_t  = y_t (mp,ms,mt); g%y_ss = y_ss(mp,ms,mt)
      g%y_st = y_st(mp,ms,mt); g%y_tt = y_tt(mp,ms,mt); g%y_p  = y_p (mp,ms,mt)
      g%y_sp = y_sp(mp,ms,mt); g%y_tp = y_tp(mp,ms,mt); g%y_pp = y_pp(mp,ms,mt)
      g%xjac   = g%x_s*g%y_t - g%x_t*g%y_s
      g%xjac_x = ( g%x_ss*g%y_t**2 - g%y_ss*g%x_t*g%y_t - 2.d0*g%x_st*g%y_s*g%y_t   &
                 + g%y_st*(g%x_s*g%y_t + g%x_t*g%y_s)                               &
                 + g%x_tt*g%y_s**2 - g%y_tt*g%x_s*g%y_s ) / g%xjac
      g%xjac_y = ( g%y_tt*g%x_s**2 - g%x_tt*g%y_s*g%x_s - 2.d0*g%y_st*g%x_t*g%x_s   &
                 + g%x_st*(g%y_t*g%x_s + g%y_s*g%x_t)                               &
                 + g%y_ss*g%x_t**2 - g%x_ss*g%y_t*g%x_t ) / g%xjac
      g%x_p_x = (g%x_sp*g%y_t - g%x_tp*g%y_s)/g%xjac
      g%x_p_y = (g%x_tp*g%x_s - g%x_sp*g%x_t)/g%xjac
      g%y_p_x = (g%y_sp*g%y_t - g%y_tp*g%y_s)/g%xjac
      g%y_p_y = (g%y_tp*g%x_s - g%y_sp*g%x_t)/g%xjac
      BigR = x_g(mp,ms,mt)
      ! dV = R dR dZ dphi; the plane sum and the FFT supply dphi as in
      ! mod_elt_matrix_fft (prefactor wst*BigR*xjac there)
      wvol = wst * BigR * g%xjac

      !--- background. Psi and j in the model's own scaling (/F0).
      dps = phys_derivs(g, eq_s (mp,var_Psi,ms,mt), eq_t (mp,var_Psi,ms,mt), eq_ss(mp,var_Psi,ms,mt), &
                           eq_st(mp,var_Psi,ms,mt), eq_tt(mp,var_Psi,ms,mt), eq_p (mp,var_Psi,ms,mt), &
                           eq_sp(mp,var_Psi,ms,mt), eq_tp(mp,var_Psi,ms,mt), eq_pp(mp,var_Psi,ms,mt)) / F0
      dzj = phys_derivs(g, eq_s (mp,var_zj,ms,mt),  eq_t (mp,var_zj,ms,mt),  eq_ss(mp,var_zj,ms,mt),  &
                           eq_st(mp,var_zj,ms,mt),  eq_tt(mp,var_zj,ms,mt),  eq_p (mp,var_zj,ms,mt),  &
                           eq_sp(mp,var_zj,ms,mt),  eq_tp(mp,var_zj,ms,mt),  eq_pp(mp,var_zj,ms,mt))  / F0
      dr0 = phys_derivs(g, eq_s (mp,var_rho,ms,mt), eq_t (mp,var_rho,ms,mt), eq_ss(mp,var_rho,ms,mt), &
                           eq_st(mp,var_rho,ms,mt), eq_tt(mp,var_rho,ms,mt), eq_p (mp,var_rho,ms,mt), &
                           eq_sp(mp,var_rho,ms,mt), eq_tp(mp,var_rho,ms,mt), eq_pp(mp,var_rho,ms,mt))
      dT0 = phys_derivs(g, eq_s (mp,var_T,ms,mt),   eq_t (mp,var_T,ms,mt),   eq_ss(mp,var_T,ms,mt),   &
                           eq_st(mp,var_T,ms,mt),   eq_tt(mp,var_T,ms,mt),   eq_p (mp,var_T,ms,mt),   &
                           eq_sp(mp,var_T,ms,mt),   eq_tp(mp,var_T,ms,mt),   eq_pp(mp,var_T,ms,mt))
      r0  = eq_g(mp,var_rho,ms,mt)
      T0  = eq_g(mp,var_T,ms,mt)
      p0  = r0*T0
      gp0 = dr0(1:3)*T0 + r0*dT0(1:3)
      gj0 = dzj(1:3)

      !--- vacuum field: chi(mp,ms,mt,i,j,k) = d_R^i d_Z^j d_phi^k chi (mod_chi.f90:516)
      ch  = element%chi(mp,ms,mt,0:2,0:2,0:2)
      chR = ch(1,0,0); chZ = ch(0,1,0); chP = ch(0,0,1)
      Bv2 = chR**2 + chZ**2 + chP**2/BigR**2
      gBv2(1) = 2.d0*(chR*ch(2,0,0) + chZ*ch(1,1,0) + chP*ch(1,0,1)/BigR**2) - 2.d0*chP**2/BigR**3
      gBv2(2) = 2.d0*(chR*ch(1,1,0) + chZ*ch(0,2,0) + chP*ch(0,1,1)/BigR**2)
      gBv2(3) = 2.d0*(chR*ch(1,0,1) + chZ*ch(0,1,1) + chP*ch(0,0,2)/BigR**2)
      gvec = [chR, chZ, chP/BigR**2]
      do a = 1, 3
        do b = 1, 3
          Gm(a,b) = - gvec(a)*gvec(b)/Bv2
        enddo
      enddo
      Gm(1,1) = Gm(1,1) + 1.d0
      Gm(2,2) = Gm(2,2) + 1.d0
      Gm(3,3) = Gm(3,3) + 1.d0/BigR**2

      !--- B0 contravariant: N(f) = c . (f_R, f_Z, f_phi)
      call b0_coeffs(BigR, ch, dps, c, dc)

      !--------------------------------------------------------------------
      ! Per basis function, once per (Gauss point, plane): its physical
      ! derivatives by channel and every test / trial feature built from them.
      ! The same functions serve as test and trial functions.
      !--------------------------------------------------------------------
      do i = 1, n_vertex_max
       do j = 1, n_degrees
        q = n_degrees*(i-1) + j
        call basis_derivs(g, element%size(i,j), h(i,j,ms,mt), h_s(i,j,ms,mt), h_t(i,j,ms,mt), &
                          h_ss(i,j,ms,mt), h_st(i,j,ms,mt), h_tt(i,j,ms,mt), dv)
        do a = 0, 2
          ! N(f) and its gradient; zeta^ = chi^ = N/Bv2 for test / trial
          Nv = c(1)*dv(D_R,a) + c(2)*dv(D_Z,a) + c(3)*dv(D_P,a)
          dN(1) = dc(1,1)*dv(D_R,a) + dc(1,2)*dv(D_Z,a) + dc(1,3)*dv(D_P,a) &
                + c(1)*dv(D_RR,a) + c(2)*dv(D_RZ,a) + c(3)*dv(D_RP,a)
          dN(2) = dc(2,1)*dv(D_R,a) + dc(2,2)*dv(D_Z,a) + dc(2,3)*dv(D_P,a) &
                + c(1)*dv(D_RZ,a) + c(2)*dv(D_ZZ,a) + c(3)*dv(D_ZP,a)
          dN(3) = dc(3,1)*dv(D_R,a) + dc(3,2)*dv(D_Z,a) + dc(3,3)*dv(D_P,a) &
                + c(1)*dv(D_RP,a) + c(2)*dv(D_ZP,a) + c(3)*dv(D_PP,a)
          fz(:,a,q)  = dN/Bv2 - Nv*gBv2/Bv2**2                                   ! grad(N/Bv2)
          fgz(:,a,q) = matmul(Gm, fz(:,a,q))
          fn(a,q)    = Nv/Bv2                                                      ! N/Bv2
          fk(a,q)    = vbr(dv(D_R:D_P,a), gj0, chR, chZ, chP, BigR)               ! [f, j0]
          fc(a,q)    = - vbr(gBv2, dv(D_R:D_P,a), chR, chZ, chP, BigR)/Bv2**2      ! [1/Bv2, f]
          fd(a,q)    = ( vbr(gp0, dv(D_R:D_P,a), chR, chZ, chP, BigR)             &
                       - GAMMA_l*p0*vbr(gBv2, dv(D_R:D_P,a), chR, chZ, chP, BigR)/Bv2 ) / Bv2   ! D(f)
          fg(:,a,q)  = dv(D_R:D_P,a)
          fgg(:,a,q) = matmul(Gm, dv(D_R:D_P,a))
          fpp(a,q)   = dv(D_PP,a)
        enddo
        fval(:,q) = 0.d0
        fval(0,q) = h(i,j,ms,mt)*element%size(i,j)
       enddo
      enddo

      !--------------------------------------------------------------------
      ! Test (index_ij) x trial (index_kl), channels (a, b)
      !--------------------------------------------------------------------
      do index_ij = 1, NFV
        do index_kl = 1, NFV
          do a = 0, 2
            do b = 0, 2
              select case (fo)
              case (4)
                val = theta * dot_product(fgg(:,a,index_ij), fg(:,b,index_kl))
              case (7)
                val = fpp(a,index_ij)*fval(b,index_kl) + fg(3,a,index_ij)*fg(3,b,index_kl)
              case (8)
                val = fg(3,a,index_ij)*fg(3,b,index_kl)
              case (9)
                val = - theta*tstep * fval(a,index_ij) * fn(b,index_kl)
              case (10)
                val = - theta*tstep * fval(a,index_ij) * fn(b,index_kl)*Bv2 / F0
              case (11)
                val = theta * Bv2 * dot_product(fgg(:,a,index_ij), fg(:,b,index_kl)) / F0
              case (2)
                val = pref * ( - Bv2*dot_product(fgz(:,a,index_ij), fz(:,b,index_kl)) &
                               - fn(b,index_kl)*fk(a,index_ij) )
              case (3)
                val = pref * ( - Bv2*dot_product(fgz(:,a,index_ij), fz(:,b,index_kl)) )
              case (5)
                val = pref * ( - fn(b,index_kl)*fk(a,index_ij) - fd(b,index_kl)*fc(a,index_ij) )
              case (6)
                val = pref * ( - fd(b,index_kl)*fc(a,index_ij) )
              case default
                val = pref * ( - Bv2*dot_product(fgz(:,a,index_ij), fz(:,b,index_kl)) &
                               - fn(b,index_kl)*fk(a,index_ij) - fd(b,index_kl)*fc(a,index_ij) )
              end select
              ELM_c(mp,index_ij,index_kl,a,b) = ELM_c(mp,index_ij,index_kl,a,b) + wvol*val
            enddo
          enddo
        enddo
      enddo

    enddo  ! mp
   enddo  ! mt
  enddo   ! ms

  !--------------------------------------------------------------------------
  ! Transform and scatter. Channel order 2 on a side = order 0 times -mode^2
  ! (HZ_pp = -mode^2 HZ, basis_at_gaussian.f90), applied after the scatter.
  !--------------------------------------------------------------------------
  allocate(ELM_tmp(DFV, DFV))
  do a = 0, 2
    do b = 0, 2
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
          ! a (b) = 2 goes through the order-0 scatter on its side
          q = 2*merge(1, 0, a == 1) + merge(1, 0, b == 1)   ! 0 = p, 1 = n, 2 = k, 3 = kn
          select case (q)
          case (0); call scatter_fft_to_elm   (out_fft, i, j, ELM_tmp, DFV)
          case (1); call scatter_fft_to_elm_n (out_fft, i, j, ELM_tmp, DFV)
          case (2); call scatter_fft_to_elm_k (out_fft, i, j, ELM_tmp, DFV)
          case (3); call scatter_fft_to_elm_kn(out_fft, i, j, ELM_tmp, DFV)
          end select
        enddo
      enddo
      ! a = 2 (b = 2) went through the order-0 scatter on that side; the
      ! basis function there is HZ_pp = -mode^2 HZ
      if (a == 2 .or. b == 2) then
        do i = 1, NFV
          do in = 1, n_tor
            idx = n_tor*(i-1) + in
            if (a == 2) ELM_tmp(idx,:) = - float(mode(in))**2 * ELM_tmp(idx,:)
            if (b == 2) ELM_tmp(:,idx) = - float(mode(in))**2 * ELM_tmp(:,idx)
          enddo
        enddo
      endif
      ELM = ELM + ELM_tmp
    enddo
  enddo
  deallocate(ELM_tmp, ELM_c)

  ! The scatter routines accumulate each (row, col) pair twice (as
  ! mod_elt_matrix_fft.f90:944)
  ELM = 0.5d0 * ELM

contains

  !> Physical derivatives of one basis function H*size, split by the order of
  !! the phi derivative on its toroidal factor: d(:,0) for HZ, d(:,1) for HZ_p,
  !! d(:,2) for HZ_pp.
  subroutine basis_derivs(g, sz, h0, hs, ht, hss, hst, htt, d)
    type(geom_t), intent(in) :: g
    real*8, intent(in)  :: sz, h0, hs, ht, hss, hst, htt
    real*8, intent(out) :: d(ND,0:2)
    d(:,0) = phys_derivs(g, hs*sz, ht*sz, hss*sz, hst*sz, htt*sz, 0.d0,  0.d0,  0.d0,  0.d0)
    d(:,1) = phys_derivs(g, 0.d0,  0.d0,  0.d0,   0.d0,   0.d0,   h0*sz, hs*sz, ht*sz, 0.d0)
    d(:,2) = phys_derivs(g, 0.d0,  0.d0,  0.d0,   0.d0,   0.d0,   0.d0,  0.d0,  0.d0,  h0*sz)
  end subroutine basis_derivs

  !> c = (B0^R, B0^Z, B0^phi) with N(f) = c . (f_R, f_Z, f_phi), i.e.
  !! N(f) = Bv_parderiv(f) + Bv_pbrack(f, Psi0) of mod_equations.f90:609-628,
  !! and dc(m,k) = d_m c(k), m = R, Z, phi.
  subroutine b0_coeffs(R, ch, dps, c, dc)
    real*8, intent(in)  :: R, ch(0:2,0:2,0:2), dps(ND)
    real*8, intent(out) :: c(3), dc(3,3)
    real*8 :: xR, xZ, xP, pR, pZ, pP
    real*8 :: dxR(3), dxZ(3), dxP(3), dpR(3), dpZ(3), dpP(3), dRinv(3), dRinv2(3)
    real*8 :: A1, A2, A3, dA1(3), dA2(3), dA3(3)

    xR = ch(1,0,0); xZ = ch(0,1,0); xP = ch(0,0,1)
    pR = dps(D_R);  pZ = dps(D_Z);  pP = dps(D_P)
    ! gradients (d_R, d_Z, d_phi) of the first derivatives
    dxR = [ch(2,0,0), ch(1,1,0), ch(1,0,1)]
    dxZ = [ch(1,1,0), ch(0,2,0), ch(0,1,1)]
    dxP = [ch(1,0,1), ch(0,1,1), ch(0,0,2)]
    dpR = [dps(D_RR), dps(D_RZ), dps(D_RP)]
    dpZ = [dps(D_RZ), dps(D_ZZ), dps(D_ZP)]
    dpP = [dps(D_RP), dps(D_ZP), dps(D_PP)]
    dRinv  = [-1.d0/R**2, 0.d0, 0.d0]
    dRinv2 = [-2.d0/R**3, 0.d0, 0.d0]

    ! [f,Psi0] = (A1 f_R + A2 f_Z + A3 f_phi)/R
    A1 = pZ*xP - pP*xZ
    A2 = pP*xR - pR*xP
    A3 = pR*xZ - pZ*xR
    dA1 = dpZ*xP + pZ*dxP - dpP*xZ - pP*dxZ
    dA2 = dpP*xR + pP*dxR - dpR*xP - pR*dxP
    dA3 = dpR*xZ + pR*dxZ - dpZ*xR - pZ*dxR

    c(1) = xR + A1/R
    c(2) = xZ + A2/R
    c(3) = xP/R**2 + A3/R
    dc(:,1) = dxR + dA1/R + A1*dRinv
    dc(:,2) = dxZ + dA2/R + A2*dRinv
    dc(:,3) = dxP/R**2 + xP*dRinv2 + dA3/R + A3*dRinv
  end subroutine b0_coeffs

end subroutine pc_elt_matrix_force_fft

end module mod_pc_elt_matrix_force_fft
