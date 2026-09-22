!> Workstream E: the reduced-MHD momentum-Schur FORCE OPERATOR, assembled
!! directly instead of formed as a matrix triple product.
!!
!! WHY THIS EXISTS. JOREK builds the (u,u) entry of pair_w as
!!
!!   S_uu = B_22 - Ltil * Shat^-1 * B_12 ,  Ltil = B_21 - B_23 M_j^-1 B_31
!!
!! i.e. by multiplying discrete operators together. Chacon (JCP 526 (2025)
!! 113789, Eqs. 17-19) instead composes the two off-diagonal DIFFERENTIAL
!! operators analytically into one closed-form operator W and discretises that
!! once. Composing at the continuous level never introduces the Riesz map that
!! composing bilinear-form matrices requires, so the mass inverse M_j^-1 --
!! 312 s of PhysPC_MjSolve at np 32, and the reason the fine pair_w operator has
!! to be a MATSHELL at all -- simply does not arise. The stencil drops from
!! 5-ring (318-612 nnz/row) to the plain C1 element stencil (58 nnz/row).
!!
!! THE OPERATOR. Derivation and proofs:
!! docs/physics_pc/note_pair_w_force_operator/force_operator.tex, and
!! docs/physics_pc/workstream_E_pair_w_composition.md S2. With
!! [a,b] := a_R b_Z - a_Z b_R, chi := R[du,psi0], zeta := R[v,psi0], j0 = Delta* psi0
!! and p0 = rho0*T0, the weak form is
!!
!!   a(du,v) = - int (1/R) grad(zeta).grad(chi) dV      <- field-line bending
!!             - int R [du,psi0] [v,j0] dV              <- current-driven (kink)
!!             - 2 int R^2 (dZ v) [p0,du] dV            <- toroidal curvature
!!
!! and S_uu = B_22 + (theta*dt)^2/opz * a(du,v). THIS ROUTINE ASSEMBLES
!! (theta*dt)^2/opz * a, i.e. exactly the matrix to ADD to B_22 -- so the bench
!! can form B_22 + W and compare against the dumped S_uu directly.
!!
!! SELF-ADJOINTNESS, AND WHY ALL THREE TERMS ARE HERE. The continuous operator is
!! symmetric if and only if [psi0,j0] = 2 R dZ p0, which is reduced-MHD force
!! balance j0 x B0 = grad p0. The defect is exactly
!!   a(du,v) - a(v,du) = int R [du,v] ( 2R dZ p0 - [psi0,j0] ) dV .
!! DROPPING THE CURVATURE TERM IS NOT ALLOWED: without it the criterion collapses
!! to the FORCE-FREE condition [psi0,j0] = 0, which a finite-beta Grad-Shafranov
!! equilibrium does not satisfy (j0 = -R^2 p'(psi0) - F F'(psi0) gives
!! [psi0,j0] = 2R p'(psi0) dZ psi0 /= 0). JOREK's default schur_channels = 1 drops
!! exactly this term; the ~1e-4 that justifies it bounds the channels' MAGNITUDE,
!! whereas they carry 100% of the ANTISYMMETRY. So symmetry of the assembled
!! matrix at equilibrium is the correctness gate for this routine.
!!
!! TWO DERIVATIVES ON EACH SIDE. grad(zeta) and grad(chi) each carry two
!! derivatives of the test/trial function -- which is what the C1 Hermite space
!! supplies, and is why a fourth-order operator is assemblable here at all. Both
!! MIXED second derivatives (f_xy) and the background's second derivatives
!! (ps0_xx, ps0_xy, ps0_yy) are therefore needed; eq_ss/eq_st/eq_tt are
!! accumulated below for that reason (the reduced-PDE routine does not need them).
!!
!! PARALLEL GRADIENT. B_12 and B_23 are the SAME operator -- the parallel
!! gradient
!!
!!     Bpar f := [f, psi0] + (eps_cyl * F0 / R) d_phi f      ( = B_0 . grad f )
!!
!! (amat_12 + amat_12_n at mod_elt_matrix_fft.f90:493,495 and amat_23 +
!! amat_23_n at :510,512 -- identical structure). The composition is therefore
!! built on Bpar throughout, which is what makes it Chacon's "dv x B_0" rather
!! than a poloidal special case. Bpar is still ANTISYMMETRIC under int . dV (the
!! bracket by cyclic invariance, d_phi because eps_cyl*F0/R is phi-independent),
!! so the integration-by-parts chain and the weak form are unchanged in form.
!!
!! Since F0 is a constant (phys_module: B_phi = F0/R), R*Bpar f = R[f,psi0] +
!! eps_cyl*F0*d_phi f needs no grad(F0). The bending term is bilinear in Bpar,
!! so it populates all four toroidal channels (p/n/k/kn); the kink term is
!! linear in Bpar and populates p and n; curvature is poloidal (p only).
!!
!! An earlier version of this routine dropped the toroidal parts ("assumption
!! 5", n = 0). Measured per harmonic slot against the discrete correction, that
!! cost 10x on n /= 0 (vs 1.9x on n = 0) and made B_22 + W FAIL as a
!! preconditioner. Do not reintroduce the restriction.
!!
!! PRESSURE. The p-channel solve keeps BOTH halves of Chacon Eq. (19)'s
!! -grad[dv.grad p0 + gamma p0 div dv]. From amat_52/amat_62 (:559,:588) with
!! the R-weighted mass matrices amat_55/amat_66,
!!
!!     drho = theta*dt/opz * ( R[du,rho0] - 2 rho0 dZ du )
!!     dT   = theta*dt/opz * ( R[du,T0]   - 2(gamma-1) T0 dZ du )
!!  => dp   = theta*dt/opz * ( R[du,p0]   - 2 gamma p0 dZ du )
!!
!! The second term (compressional, gamma*p0*div dv) is symmetric and negative
!! semi-definite, so it reinforces rather than disturbs Thm. 4.
module mod_pc_elt_matrix_force_fft

  use mod_pc_fft_scatter, only: scatter_fft_to_elm, scatter_fft_to_elm_n, &
                                scatter_fft_to_elm_k, scatter_fft_to_elm_kn

  implicit none
  private

  public :: pc_elt_matrix_force_fft

contains

subroutine pc_elt_matrix_force_fft(element, nodes, xpoint2, xcase2, &
                                   R_axis, Z_axis, psi_axis, psi_bnd, &
                                   R_xpoint, Z_xpoint, ELM)

  use constants
  use mod_parameters
  use data_structure, only: type_element, type_node
  use gauss
  use basis_at_gaussian
  use phys_module
  use corr_neg

  implicit none

  type(type_element), intent(in) :: element
  type(type_node),    intent(in) :: nodes(n_vertex_max)
  logical,            intent(in) :: xpoint2
  integer,            intent(in) :: xcase2
  real*8,             intent(in) :: R_axis, Z_axis, psi_axis, psi_bnd
  real*8,             intent(in) :: R_xpoint(2), Z_xpoint(2)

  ! One variable (u): poloidal-block and toroidally-expanded dimensions
#define NFV (n_vertex_max*n_degrees)
#define DFV (n_tor*n_vertex_max*n_degrees)

  real*8, dimension(DFV, DFV), intent(out) :: ELM

  !> Four toroidal channels: _n carries d_phi on the trial function, _k on the
  !! test function, _kn on both (see header, PARALLEL GRADIENT).
  real*8, dimension(n_plane, NFV, NFV) :: ELM_p, ELM_n, ELM_k, ELM_kn

  real*8, dimension(n_gauss,n_gauss) :: x_g, x_s, x_t, x_ss, x_st, x_tt
  real*8, dimension(n_gauss,n_gauss) :: y_g, y_s, y_t, y_ss, y_st, y_tt

  !> Background fields. Second (s,t) derivatives ARE needed here, for ps0.
  real*8, dimension(n_plane, n_var, n_gauss, n_gauss) :: eq_g, eq_s, eq_t
  real*8, dimension(n_plane, n_var, n_gauss, n_gauss) :: eq_ss, eq_st, eq_tt

  integer :: i, j, k, l, ms, mt, mp, in
  integer :: index_ij, index_kl

  real*8 :: wst, xjac, xjac_x, xjac_y, BigR
  real*8 :: theta, zeta_t, pref, eps_cyl, GAMMA_l, p0, rw
  integer :: rpow

  ! Test function
  real*8 :: v_x, v_y, v_s, v_t, v_ss, v_st, v_tt, v_xx, v_xy, v_yy
  ! Trial function
  real*8 :: u_x, u_y, u_s, u_t, u_ss, u_st, u_tt, u_xx, u_xy, u_yy

  ! Background at the Gauss point
  real*8 :: ps0_x, ps0_y, ps0_s, ps0_t, ps0_ss, ps0_st, ps0_tt
  real*8 :: ps0_xx, ps0_xy, ps0_yy
  real*8 :: zj0_x, zj0_y
  real*8 :: r0, r0_x, r0_y, T0, T0_x, T0_y, p0_x, p0_y

  ! Brackets and their gradients
  real*8 :: b_tst, b_tri
  real*8 :: bR_tst, bZ_tst, bR_tri, bZ_tri
  real*8 :: zeta_x, zeta_y, chi_x, chi_y
  !> Toroidal halves of R*Bpar: the coefficient multiplying d_phi of the
  !! test (ztor_*) and trial (ctor_*) function.
  real*8 :: ztor_x, ztor_y, ctor_x, ctor_y, bj_tst
  real*8 :: a_bend, a_kink, a_curv, a_uu
  real*8 :: a_uu_n, a_uu_k, a_uu_kn

  real*8     :: in_fft(1:n_plane)
  complex*16 :: out_fft(1:n_plane)

  ELM    = 0.d0
  ELM_p  = 0.d0
  ELM_n  = 0.d0
  ELM_k  = 0.d0
  ELM_kn = 0.d0

  ! Mirrors mod_elt_matrix_fft.f90:111 and :212 -- the same local overrides the
  ! model's own element routine uses, so W composes the identical blocks.
  GAMMA_l = 5.d0 / 3.d0
  eps_cyl = 1.d0

  theta  = time_evol_theta
  zeta_t = time_evol_zeta * 2.0d0 * tstep / (tstep + tstep_prev)
  ! (theta*dt)^2 / opz -- the prefactor that turns a(du,v) into the matrix to be
  ! ADDED to B_22. opz = 1 + zeta, matching build_schur_mixed_prod.
  pref   = (theta * tstep)**2 / (1.d0 + zeta_t)

  ! Workstream E, section 6: R-weight sweep on the BENDING family. Values
  ! 20..24 mean "full operator, bending integrand multiplied by R**(value-22)",
  ! so 22 reproduces force_operator = 1 exactly and is the sweep's own control.
  ! A pointwise weight is a DIFFERENT operator and so can change the direction
  ! of W; a scalar factor cannot, because cos(W,C) is scale-invariant.
  rpow = 0
  if (physics_pc_force_operator >= 20 .and. physics_pc_force_operator <= 24) then
    rpow = physics_pc_force_operator - 22
  endif

  x_g = 0.d0; x_s = 0.d0; x_t = 0.d0; x_ss = 0.d0; x_st = 0.d0; x_tt = 0.d0
  y_g = 0.d0; y_s = 0.d0; y_t = 0.d0; y_ss = 0.d0; y_st = 0.d0; y_tt = 0.d0
  eq_g = 0.d0; eq_s = 0.d0; eq_t = 0.d0
  eq_ss = 0.d0; eq_st = 0.d0; eq_tt = 0.d0

  do i = 1, n_vertex_max
    do j = 1, n_degrees
      do ms = 1, n_gauss
        do mt = 1, n_gauss

          x_g(ms,mt)  = x_g(ms,mt)  + nodes(i)%x(1,j,1) * element%size(i,j) * H(i,j,ms,mt)
          x_s(ms,mt)  = x_s(ms,mt)  + nodes(i)%x(1,j,1) * element%size(i,j) * H_s(i,j,ms,mt)
          x_t(ms,mt)  = x_t(ms,mt)  + nodes(i)%x(1,j,1) * element%size(i,j) * H_t(i,j,ms,mt)
          x_ss(ms,mt) = x_ss(ms,mt) + nodes(i)%x(1,j,1) * element%size(i,j) * H_ss(i,j,ms,mt)
          x_st(ms,mt) = x_st(ms,mt) + nodes(i)%x(1,j,1) * element%size(i,j) * H_st(i,j,ms,mt)
          x_tt(ms,mt) = x_tt(ms,mt) + nodes(i)%x(1,j,1) * element%size(i,j) * H_tt(i,j,ms,mt)

          y_g(ms,mt)  = y_g(ms,mt)  + nodes(i)%x(1,j,2) * element%size(i,j) * H(i,j,ms,mt)
          y_s(ms,mt)  = y_s(ms,mt)  + nodes(i)%x(1,j,2) * element%size(i,j) * H_s(i,j,ms,mt)
          y_t(ms,mt)  = y_t(ms,mt)  + nodes(i)%x(1,j,2) * element%size(i,j) * H_t(i,j,ms,mt)
          y_ss(ms,mt) = y_ss(ms,mt) + nodes(i)%x(1,j,2) * element%size(i,j) * H_ss(i,j,ms,mt)
          y_st(ms,mt) = y_st(ms,mt) + nodes(i)%x(1,j,2) * element%size(i,j) * H_st(i,j,ms,mt)
          y_tt(ms,mt) = y_tt(ms,mt) + nodes(i)%x(1,j,2) * element%size(i,j) * H_tt(i,j,ms,mt)

          do mp = 1, n_plane
            do k = 1, n_var
              do in = 1, n_tor
                eq_g (mp,k,ms,mt) = eq_g (mp,k,ms,mt) &
                     + nodes(i)%values(in,j,k) * element%size(i,j) * H(i,j,ms,mt)    * HZ(in,mp)
                eq_s (mp,k,ms,mt) = eq_s (mp,k,ms,mt) &
                     + nodes(i)%values(in,j,k) * element%size(i,j) * H_s(i,j,ms,mt)  * HZ(in,mp)
                eq_t (mp,k,ms,mt) = eq_t (mp,k,ms,mt) &
                     + nodes(i)%values(in,j,k) * element%size(i,j) * H_t(i,j,ms,mt)  * HZ(in,mp)
                eq_ss(mp,k,ms,mt) = eq_ss(mp,k,ms,mt) &
                     + nodes(i)%values(in,j,k) * element%size(i,j) * H_ss(i,j,ms,mt) * HZ(in,mp)
                eq_st(mp,k,ms,mt) = eq_st(mp,k,ms,mt) &
                     + nodes(i)%values(in,j,k) * element%size(i,j) * H_st(i,j,ms,mt) * HZ(in,mp)
                eq_tt(mp,k,ms,mt) = eq_tt(mp,k,ms,mt) &
                     + nodes(i)%values(in,j,k) * element%size(i,j) * H_tt(i,j,ms,mt) * HZ(in,mp)
              enddo
            enddo
          enddo

        enddo
      enddo
    enddo
  enddo

  !--------------------------------------------------------------------------
  ! Gauss integration
  !--------------------------------------------------------------------------
  do ms = 1, n_gauss
   do mt = 1, n_gauss

    wst  = wgauss(ms)*wgauss(mt)
    xjac = x_s(ms,mt)*y_t(ms,mt) - x_t(ms,mt)*y_s(ms,mt)

    xjac_x = (x_ss(ms,mt)*y_t(ms,mt)**2 - y_ss(ms,mt)*x_t(ms,mt)*y_t(ms,mt)             &
            - 2.d0*x_st(ms,mt)*y_s(ms,mt)*y_t(ms,mt)                                    &
            + y_st(ms,mt)*(x_s(ms,mt)*y_t(ms,mt) + x_t(ms,mt)*y_s(ms,mt))               &
            + x_tt(ms,mt)*y_s(ms,mt)**2 - y_tt(ms,mt)*x_s(ms,mt)*y_s(ms,mt)) / xjac

    xjac_y = (y_tt(ms,mt)*x_s(ms,mt)**2 - x_tt(ms,mt)*y_s(ms,mt)*x_s(ms,mt)             &
            - 2.d0*y_st(ms,mt)*x_t(ms,mt)*x_s(ms,mt)                                    &
            + x_st(ms,mt)*(y_t(ms,mt)*x_s(ms,mt) + y_s(ms,mt)*x_t(ms,mt))               &
            + y_ss(ms,mt)*x_t(ms,mt)**2 - x_ss(ms,mt)*y_t(ms,mt)*x_t(ms,mt)) / xjac

    BigR = x_g(ms,mt)

    do mp = 1, n_plane

      ps0_s  = eq_s (mp,1,ms,mt)
      ps0_t  = eq_t (mp,1,ms,mt)
      ps0_ss = eq_ss(mp,1,ms,mt)
      ps0_st = eq_st(mp,1,ms,mt)
      ps0_tt = eq_tt(mp,1,ms,mt)
      ps0_x  = (  y_t(ms,mt)*ps0_s - y_s(ms,mt)*ps0_t ) / xjac
      ps0_y  = ( -x_t(ms,mt)*ps0_s + x_s(ms,mt)*ps0_t ) / xjac

      ! Second physical derivatives. The xx and yy forms are those of
      ! mod_elt_matrix_fft; xy is the same construction applied as
      ! d_y(f_x) = (-x_t d_s + x_s d_t)(f_x)/xjac.
      ps0_xx = (ps0_ss*y_t(ms,mt)**2 - 2.d0*ps0_st*y_s(ms,mt)*y_t(ms,mt)                &
              + ps0_tt*y_s(ms,mt)**2                                                    &
              + ps0_s*(y_st(ms,mt)*y_t(ms,mt) - y_tt(ms,mt)*y_s(ms,mt))                 &
              + ps0_t*(y_st(ms,mt)*y_s(ms,mt) - y_ss(ms,mt)*y_t(ms,mt)) ) / xjac**2     &
              - xjac_x*ps0_x / xjac

      ps0_yy = (ps0_ss*x_t(ms,mt)**2 - 2.d0*ps0_st*x_s(ms,mt)*x_t(ms,mt)                &
              + ps0_tt*x_s(ms,mt)**2                                                    &
              + ps0_s*(x_st(ms,mt)*x_t(ms,mt) - x_tt(ms,mt)*x_s(ms,mt))                 &
              + ps0_t*(x_st(ms,mt)*x_s(ms,mt) - x_ss(ms,mt)*x_t(ms,mt)) ) / xjac**2     &
              - xjac_y*ps0_y / xjac

      ps0_xy = (-ps0_ss*x_t(ms,mt)*y_t(ms,mt)                                           &
              + ps0_st*(x_t(ms,mt)*y_s(ms,mt) + x_s(ms,mt)*y_t(ms,mt))                  &
              - ps0_tt*x_s(ms,mt)*y_s(ms,mt)                                            &
              + ps0_s*(-x_t(ms,mt)*y_st(ms,mt) + x_s(ms,mt)*y_tt(ms,mt))                &
              + ps0_t*( x_t(ms,mt)*y_ss(ms,mt) - x_s(ms,mt)*y_st(ms,mt)) ) / xjac**2    &
              - xjac_y*ps0_x / xjac

      zj0_x = (  y_t(ms,mt)*eq_s(mp,3,ms,mt) - y_s(ms,mt)*eq_t(mp,3,ms,mt) ) / xjac
      zj0_y = ( -x_t(ms,mt)*eq_s(mp,3,ms,mt) + x_s(ms,mt)*eq_t(mp,3,ms,mt) ) / xjac

      r0   = abs(eq_g(mp,5,ms,mt))
      r0_x = (  y_t(ms,mt)*eq_s(mp,5,ms,mt) - y_s(ms,mt)*eq_t(mp,5,ms,mt) ) / xjac
      r0_y = ( -x_t(ms,mt)*eq_s(mp,5,ms,mt) + x_s(ms,mt)*eq_t(mp,5,ms,mt) ) / xjac
      T0   = abs(eq_g(mp,6,ms,mt))
      T0_x = (  y_t(ms,mt)*eq_s(mp,6,ms,mt) - y_s(ms,mt)*eq_t(mp,6,ms,mt) ) / xjac
      T0_y = ( -x_t(ms,mt)*eq_s(mp,6,ms,mt) + x_s(ms,mt)*eq_t(mp,6,ms,mt) ) / xjac

      ! p0 = rho0 * T0, so the curvature drive needs both channels -- which is
      ! why schur_channels = 1 cannot represent it (see header).
      p0_x = r0_x*T0 + r0*T0_x
      p0_y = r0_y*T0 + r0*T0_y
      ! p0 itself, for the compressional (gamma p0 div dv) half of Eq. (19)
      p0   = r0 * T0

      !--------------------------------------------------------------------
      ! Test function loop
      !--------------------------------------------------------------------
      do i = 1, n_vertex_max
       do j = 1, n_degrees

        index_ij = n_degrees*(i-1) + j

        v_s  = h_s (i,j,ms,mt) * element%size(i,j)
        v_t  = h_t (i,j,ms,mt) * element%size(i,j)
        v_ss = h_ss(i,j,ms,mt) * element%size(i,j)
        v_st = h_st(i,j,ms,mt) * element%size(i,j)
        v_tt = h_tt(i,j,ms,mt) * element%size(i,j)
        v_x  = (  y_t(ms,mt)*v_s - y_s(ms,mt)*v_t ) / xjac
        v_y  = ( -x_t(ms,mt)*v_s + x_s(ms,mt)*v_t ) / xjac

        v_xx = (v_ss*y_t(ms,mt)**2 - 2.d0*v_st*y_s(ms,mt)*y_t(ms,mt) + v_tt*y_s(ms,mt)**2 &
              + v_s*(y_st(ms,mt)*y_t(ms,mt) - y_tt(ms,mt)*y_s(ms,mt))                     &
              + v_t*(y_st(ms,mt)*y_s(ms,mt) - y_ss(ms,mt)*y_t(ms,mt)) ) / xjac**2         &
              - xjac_x*v_x / xjac

        v_yy = (v_ss*x_t(ms,mt)**2 - 2.d0*v_st*x_s(ms,mt)*x_t(ms,mt) + v_tt*x_s(ms,mt)**2 &
              + v_s*(x_st(ms,mt)*x_t(ms,mt) - x_tt(ms,mt)*x_s(ms,mt))                     &
              + v_t*(x_st(ms,mt)*x_s(ms,mt) - x_ss(ms,mt)*x_t(ms,mt)) ) / xjac**2         &
              - xjac_y*v_y / xjac

        v_xy = (-v_ss*x_t(ms,mt)*y_t(ms,mt)                                               &
              + v_st*(x_t(ms,mt)*y_s(ms,mt) + x_s(ms,mt)*y_t(ms,mt))                      &
              - v_tt*x_s(ms,mt)*y_s(ms,mt)                                                &
              + v_s*(-x_t(ms,mt)*y_st(ms,mt) + x_s(ms,mt)*y_tt(ms,mt))                    &
              + v_t*( x_t(ms,mt)*y_ss(ms,mt) - x_s(ms,mt)*y_st(ms,mt)) ) / xjac**2        &
              - xjac_y*v_x / xjac

        ! zeta = R*Bpar v = R[v,psi0] + eps_cyl*F0*d_phi v.
        ! Poloidal half: grad(R[v,psi0]) = [v,psi0] e_R + R grad[v,psi0].
        b_tst  = v_x*ps0_y - v_y*ps0_x
        bR_tst = v_xx*ps0_y + v_x*ps0_xy - v_xy*ps0_x - v_y*ps0_xx
        bZ_tst = v_xy*ps0_y + v_x*ps0_yy - v_yy*ps0_x - v_y*ps0_xy
        zeta_x = b_tst + BigR*bR_tst
        zeta_y =         BigR*bZ_tst
        ! Toroidal half: grad(eps_cyl*F0*d_phi v) = eps_cyl*F0*d_phi(grad v),
        ! so these are the coefficients of d_phi (F0 constant => no grad F0).
        ztor_x = eps_cyl * F0 * v_x
        ztor_y = eps_cyl * F0 * v_y

        ! [v, j0] -- the kink term's test-side factor, trial-independent
        bj_tst = v_x*zj0_y - v_y*zj0_x

        !--------------------------------------------------------------------
        ! Trial function loop
        !--------------------------------------------------------------------
        do k = 1, n_vertex_max
         do l = 1, n_degrees

          index_kl = n_degrees*(k-1) + l

          u_s  = h_s (k,l,ms,mt) * element%size(k,l)
          u_t  = h_t (k,l,ms,mt) * element%size(k,l)
          u_ss = h_ss(k,l,ms,mt) * element%size(k,l)
          u_st = h_st(k,l,ms,mt) * element%size(k,l)
          u_tt = h_tt(k,l,ms,mt) * element%size(k,l)
          u_x  = (  y_t(ms,mt)*u_s - y_s(ms,mt)*u_t ) / xjac
          u_y  = ( -x_t(ms,mt)*u_s + x_s(ms,mt)*u_t ) / xjac

          u_xx = (u_ss*y_t(ms,mt)**2 - 2.d0*u_st*y_s(ms,mt)*y_t(ms,mt) + u_tt*y_s(ms,mt)**2 &
                + u_s*(y_st(ms,mt)*y_t(ms,mt) - y_tt(ms,mt)*y_s(ms,mt))                     &
                + u_t*(y_st(ms,mt)*y_s(ms,mt) - y_ss(ms,mt)*y_t(ms,mt)) ) / xjac**2         &
                - xjac_x*u_x / xjac

          u_yy = (u_ss*x_t(ms,mt)**2 - 2.d0*u_st*x_s(ms,mt)*x_t(ms,mt) + u_tt*x_s(ms,mt)**2 &
                + u_s*(x_st(ms,mt)*x_t(ms,mt) - x_tt(ms,mt)*x_s(ms,mt))                     &
                + u_t*(x_st(ms,mt)*x_s(ms,mt) - x_ss(ms,mt)*x_t(ms,mt)) ) / xjac**2         &
                - xjac_y*u_y / xjac

          u_xy = (-u_ss*x_t(ms,mt)*y_t(ms,mt)                                               &
                + u_st*(x_t(ms,mt)*y_s(ms,mt) + x_s(ms,mt)*y_t(ms,mt))                      &
                - u_tt*x_s(ms,mt)*y_s(ms,mt)                                                &
                + u_s*(-x_t(ms,mt)*y_st(ms,mt) + x_s(ms,mt)*y_tt(ms,mt))                    &
                + u_t*( x_t(ms,mt)*y_ss(ms,mt) - x_s(ms,mt)*y_st(ms,mt)) ) / xjac**2        &
                - xjac_y*u_x / xjac

          ! chi = R*Bpar du, same decomposition as zeta above
          b_tri  = u_x*ps0_y - u_y*ps0_x
          bR_tri = u_xx*ps0_y + u_x*ps0_xy - u_xy*ps0_x - u_y*ps0_xx
          bZ_tri = u_xy*ps0_y + u_x*ps0_yy - u_yy*ps0_x - u_y*ps0_xy
          chi_x  = b_tri + BigR*bR_tri
          chi_y  =         BigR*bZ_tri
          ctor_x = eps_cyl * F0 * u_x
          ctor_y = eps_cyl * F0 * u_y

          !------------------------------------------------------------------
          ! Bending: - int (1/R) grad(zeta).grad(chi), bilinear in Bpar, so it
          ! spreads over all four channels. Symmetric, negative semi-definite.
          !------------------------------------------------------------------
          rw = BigR**rpow      ! = 1 unless the section-6 R sweep is active

          a_bend = - rw * ( zeta_x*chi_x + zeta_y*chi_y ) / BigR

          !------------------------------------------------------------------
          ! Kink: - int (R*Bpar du) [v,j0], linear in Bpar -> p and n only.
          !------------------------------------------------------------------
          a_kink = - BigR * b_tri * bj_tst

          !------------------------------------------------------------------
          ! Curvature/pressure, BOTH halves of Eq. (19):
          !   + 2 R^2 (dZ v) [du,p0]        (dv.grad p0)
          !   - 4 gamma R p0 (dZ v)(dZ du)  (gamma p0 div dv -- symmetric, nsd)
          ! Poloidal only: amat_52/amat_62 have no _n partner.
          !------------------------------------------------------------------
          a_curv = - 2.d0 * BigR**2 * v_y * ( p0_x*u_y - p0_y*u_x ) &
                   - 4.d0 * GAMMA_l * BigR * p0 * v_y * u_y

          ! physics_pc_force_operator selects which terms are assembled, so that
          ! W can be compared LIKE FOR LIKE against a production S_uu whose
          ! channel set differs (physics_pc_schur_channels = 1 carries the psi
          ! channel only, i.e. bending + kink and NO pressure term):
          !   1 = bending + kink + curvature   (the full Eq. (19) operator)
          !   2 = bending + kink               (matches schur_channels = 1)
          !   3 = bending only                 (the dominant term alone)
          if (physics_pc_force_operator == 4) then
            ! UNIT TEST, not physics. Assembles amat_31's integrand
            ! (v_x*u_x + v_y*u_y)/BigR*xjac through THIS routine's Jacobians,
            ! derivative formulas, FFT scatter and 0.5*ELM halving. The dumped
            ! W_force must then equal the dumped B_31 to round-off; any
            ! difference is a defect in the machinery, not in the weak form.
            a_uu  = ( v_x*u_x + v_y*u_y ) / BigR * xjac
          else if (physics_pc_force_operator == 3) then
            a_uu  = pref * ( a_bend ) * xjac
          else if (physics_pc_force_operator == 2) then
            a_uu  = pref * ( a_bend + a_kink ) * xjac
          else
            a_uu  = pref * ( a_bend + a_kink + a_curv ) * xjac
          endif
          ! d_phi on trial: bending cross-term + the kink's toroidal half (the
          ! latter dropped with the kink itself when force_operator = 3)
          if (physics_pc_force_operator >= 3 .and. physics_pc_force_operator <= 4) then
            a_uu_n = pref * ( - rw * ( zeta_x*ctor_x + zeta_y*ctor_y ) / BigR ) * xjac
            if (physics_pc_force_operator == 4) a_uu_n = 0.d0   ! unit test: p only
          else
            a_uu_n = pref * ( - rw * ( zeta_x*ctor_x + zeta_y*ctor_y ) / BigR   &
                              - eps_cyl * F0 * bj_tst ) * xjac
          endif
          ! d_phi on test: bending cross-term only
          a_uu_k  = pref * ( - rw * ( ztor_x*chi_x + ztor_y*chi_y ) / BigR ) * xjac
          ! d_phi on both
          a_uu_kn = pref * ( - rw * ( ztor_x*ctor_x + ztor_y*ctor_y ) / BigR ) * xjac

          ELM_p (mp,index_ij,index_kl) = ELM_p (mp,index_ij,index_kl) + wst * a_uu
          ELM_n (mp,index_ij,index_kl) = ELM_n (mp,index_ij,index_kl) + wst * a_uu_n
          ELM_k (mp,index_ij,index_kl) = ELM_k (mp,index_ij,index_kl) + wst * a_uu_k
          ELM_kn(mp,index_ij,index_kl) = ELM_kn(mp,index_ij,index_kl) + wst * a_uu_kn

         enddo  ! l
        enddo   ! k

       enddo  ! j
      enddo   ! i

    enddo  ! mp

   enddo  ! mt
  enddo   ! ms

  !--------------------------------------------------------------------------
  ! Transform the single toroidal channel and scatter into the mode-space ELM
  !--------------------------------------------------------------------------
  do i = 1, NFV
    do j = 1, NFV

      if (maxval(abs(ELM_p(1:n_plane,i,j))) .ne. 0.d0) then
        in_fft = ELM_p(1:n_plane,i,j)
#ifdef USE_FFTW
        call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
        call scatter_fft_to_elm(out_fft, i, j, ELM, DFV)
      endif

      if (maxval(abs(ELM_n(1:n_plane,i,j))) .ne. 0.d0) then
        in_fft = ELM_n(1:n_plane,i,j)
#ifdef USE_FFTW
        call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
        call scatter_fft_to_elm_n(out_fft, i, j, ELM, DFV)
      endif

      if (maxval(abs(ELM_k(1:n_plane,i,j))) .ne. 0.d0) then
        in_fft = ELM_k(1:n_plane,i,j)
#ifdef USE_FFTW
        call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
        call scatter_fft_to_elm_k(out_fft, i, j, ELM, DFV)
      endif

      if (maxval(abs(ELM_kn(1:n_plane,i,j))) .ne. 0.d0) then
        in_fft = ELM_kn(1:n_plane,i,j)
#ifdef USE_FFTW
        call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#endif
        call scatter_fft_to_elm_kn(out_fft, i, j, ELM, DFV)
      endif

    enddo
  enddo

  ! The scatter routines accumulate each (row, col) pair twice; same convention
  ! as mod_elt_matrix_fft and mod_pc_elt_matrix_reduced_fft.
  ELM = 0.5d0 * ELM

end subroutine pc_elt_matrix_force_fft

end module mod_pc_elt_matrix_force_fft
