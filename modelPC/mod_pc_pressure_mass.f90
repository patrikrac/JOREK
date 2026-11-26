!> The pressure term of the flow's Schur complement as a mass on the vorticity,
!! added to the inertia M_rho of the vorticity-space Schur preconditioners
!! (mod_petsc_pc_vs, -vs_pressure mass; study studies/vs_pressure).
!!
!! Eliminating rho and T by their masses gives the flow row the term that the
!! SF sweep adds to B_22 (mod_pc_elt_matrix_force_fft):
!!   W(u, v) = c int ( - 2 R^2 dZ v [p0, u] - 4 gamma R p0 dZ v dZ u ),
!!   c = (theta dt)^2 / opz,
!! the drive and the compression. The VS preconditioners leave it out, and it
!! acts on the modes the line bending does not control. For grad u along a
!! unit direction d it is -c_d grad v . grad u with
!!   c_d = c ( 4 gamma R p0 d_Z^2 + 2 R^2 d_Z (dR p0 d_Z - dZ p0 d_R) ),
!! which goes into M_rho as int c_d v w, as the inertia does (the weight's
!! gradient left out). The direction (pm_dir): n = grad psi0 / |grad psi0|
!! for the block of n = 0 (zonal flows: exact, no drive for p0 = p0(psi0)),
!! t = (-n_Z, n_R) for the rest (the binormal of the field-aligned modes:
!! compression and ballooning drive), or iso, the mean of the two. The clip
!! c_d >= -pm_clip opz R^3 rho_i (rho_i the density weight of the model's
!! inertia) keeps M_rho >= (1 - pm_clip) times the inertia.
module mod_pc_pressure_mass

  implicit none
  private

  integer, parameter, public :: PM_N = 1, PM_T = 2, PM_ISO = 3
  !> the density weight of the model's inertia: |rho0| (model199) or corr_neg_dens(rho0) (model600)
  integer, parameter, public :: PM_RHO_ABS = 1, PM_RHO_CORR = 2

  logical, save, public :: pm_on    = .false.        !< -vs_pressure mass
  real*8,  save, public :: pm_gamma = 1.d0           !< -vs_pressure_gamma (1: isothermal)
  real*8,  save, public :: pm_clip  = 0.2d0          !< -vs_pressure_clip
  !> the direction of the n = 0 block and of the rest (-vs_pressure_dir0, _dirn)
  integer, save, public :: pm_dir(2) = [PM_N, PM_T]
  !> the Gauss points (all planes) of the last assembly, and those clipped per direction
  integer*8, save, public :: pm_n_pts = 0, pm_n_clip(2) = 0

  public :: pm_weights, pm_element

contains

  !> c_d of the two directions pm_dir at one Gauss point, clipped.
  pure subroutine pm_weights(R, rho_i, p0, p0_x, p0_y, ps0_x, ps0_y, pref, opz, c, clipped)
    real*8,  intent(in)  :: R, rho_i, p0, p0_x, p0_y, ps0_x, ps0_y, pref, opz
    real*8,  intent(out) :: c(2)
    logical, intent(out) :: clipped(2)
    real*8  :: g, nR, nZ, cn, ct, floor
    integer :: k

    g = sqrt(ps0_x**2 + ps0_y**2)
    if (g > 0.d0) then
      nR = ps0_x / g;  nZ = ps0_y / g
    else
      nR = 1.d0;       nZ = 0.d0
    endif
    cn = weight(nR, nZ)
    ct = weight(-nZ, nR)
    floor = -pm_clip * opz * R**3 * rho_i
    do k = 1, 2
      select case (pm_dir(k))
      case (PM_N);  c(k) = cn
      case (PM_T);  c(k) = ct
      case default; c(k) = 0.5d0 * (cn + ct)
      end select
      clipped(k) = c(k) < floor
      c(k) = max(c(k), floor)
    enddo

  contains

    pure real*8 function weight(dR, dZ)
      real*8, intent(in) :: dR, dZ
      weight = pref * (4.d0 * pm_gamma * R * p0 * dZ**2 + 2.d0 * R**2 * dZ * (p0_x * dZ - p0_y * dR))
    end function weight

  end subroutine pm_weights

  !> Add the pressure mass of one element to ELM (the element matrix of M_rho,
  !! one variable, n_tor slots): int c_d v w with pm_dir(1) in the rows and
  !! columns of n = 0 and pm_dir(2) elsewhere. p0 = |rho0| |T0| and its
  !! gradient as the SF sweep's W; all harmonics of the state through the
  !! planes. Counts the Gauss points and the clipped ones (pm_n_pts, pm_n_clip).
  subroutine pm_element(element, nodes, rho_w, ELM)
    use mod_parameters, only: n_tor, n_plane, n_vertex_max, n_degrees, var_psi, var_rho, var_T
    use data_structure, only: type_element, type_node
    use gauss,          only: n_gauss, wgauss
    use basis_at_gaussian, only: H, H_s, H_t, HZ
    use phys_module,    only: time_evol_theta, time_evol_zeta, tstep, tstep_prev, fftw_plan
    use mod_pc_fft_scatter, only: scatter_fft_to_elm, pc_my_fft
    use corr_neg,       only: corr_neg_dens

    type(type_element), intent(in)    :: element
    type(type_node),    intent(in)    :: nodes(n_vertex_max)
    integer,            intent(in)    :: rho_w
    real*8,             intent(inout) :: ELM(n_tor*n_vertex_max*n_degrees, n_tor*n_vertex_max*n_degrees)

    integer, parameter :: NFV = n_vertex_max*n_degrees
    integer, parameter :: vq(3) = [var_psi, var_rho, var_T]
    real*8, dimension(n_gauss, n_gauss) :: x_g, x_s, x_t, y_s, y_t
    !> psi0 (1), rho0 (2), T0 (3) and their s, t derivatives on the planes
    real*8, dimension(n_plane, 3, n_gauss, n_gauss) :: eq_g, eq_s, eq_t
    !> the pressure mass on the planes, and scattered, per direction
    real*8, allocatable :: QM_p(:,:,:,:), QM(:,:,:)
    real*8  :: f_v(NFV), c(2), sz, wst, xjac, opz, pref, ps0_x, ps0_y, r0, r0_x, r0_y, T0, T0_x, T0_y
    real*8  :: rho_i
    logical :: clipped(2)
    integer :: i, j, k, ms, mt, mp, in, q, d
    integer*8 :: n_pts, n_clip(2)
    real*8     :: in_fft(1:n_plane)
    complex*16 :: out_fft(1:n_plane)

    allocate(QM_p(n_plane, NFV, NFV, 2), QM(size(ELM, 1), size(ELM, 2), 2))
    QM_p = 0.d0; QM = 0.d0
    opz  = 1.d0 + time_evol_zeta * 2.0d0 * tstep / (tstep + tstep_prev)
    pref = (time_evol_theta * tstep)**2 / opz

    x_g = 0.d0; x_s = 0.d0; x_t = 0.d0; y_s = 0.d0; y_t = 0.d0
    eq_g = 0.d0; eq_s = 0.d0; eq_t = 0.d0
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
                  eq_g(mp,k,ms,mt) = eq_g(mp,k,ms,mt) + nodes(i)%values(in,j,vq(k)) * sz * H(i,j,ms,mt)   * HZ(in,mp)
                  eq_s(mp,k,ms,mt) = eq_s(mp,k,ms,mt) + nodes(i)%values(in,j,vq(k)) * sz * H_s(i,j,ms,mt) * HZ(in,mp)
                  eq_t(mp,k,ms,mt) = eq_t(mp,k,ms,mt) + nodes(i)%values(in,j,vq(k)) * sz * H_t(i,j,ms,mt) * HZ(in,mp)
                enddo
              enddo
            enddo
          enddo
        enddo
      enddo
    enddo

    n_pts = 0; n_clip = 0
    do ms = 1, n_gauss
      do mt = 1, n_gauss
        wst  = wgauss(ms)*wgauss(mt)
        xjac = x_s(ms,mt)*y_t(ms,mt) - x_t(ms,mt)*y_s(ms,mt)
        do q = 1, NFV
          f_v(q) = H((q - 1) / n_degrees + 1, mod(q - 1, n_degrees) + 1, ms, mt) &
                   * element%size((q - 1) / n_degrees + 1, mod(q - 1, n_degrees) + 1)
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
          rho_i = r0
          if (rho_w == PM_RHO_CORR) rho_i = corr_neg_dens(eq_g(mp,2,ms,mt))
          call pm_weights(x_g(ms,mt), rho_i, r0*T0, r0_x*T0 + r0*T0_x, r0_y*T0 + r0*T0_y, ps0_x, ps0_y, &
                          pref, opz, c, clipped)
          n_pts = n_pts + 1
          n_clip = n_clip + merge(1_8, 0_8, clipped)
          do d = 1, 2
            do i = 1, NFV
              do j = 1, NFV
                QM_p(mp,i,j,d) = QM_p(mp,i,j,d) + wst * c(d) * xjac * f_v(i) * f_v(j)
              enddo
            enddo
          enddo
        enddo
      enddo
    enddo

    do d = 1, 2
      do i = 1, NFV
        do j = 1, NFV
          in_fft = QM_p(1:n_plane,i,j,d)
#ifdef USE_FFTW
          call dfftw_execute_dft_r2c(fftw_plan, in_fft, out_fft)
#else
          call pc_my_fft(in_fft, out_fft, n_plane)
#endif
          call scatter_fft_to_elm(out_fft, i, j, QM(:,:,d), size(ELM, 1))
        enddo
      enddo
    enddo
    ! slot 1 of every node DOF is n = 0; the scatter accumulates each pair twice
    do j = 1, size(ELM, 2)
      do i = 1, size(ELM, 1)
        d = merge(1, 2, mod(i - 1, n_tor) == 0 .and. mod(j - 1, n_tor) == 0)
        ELM(i,j) = ELM(i,j) + 0.5d0 * QM(i,j,d)
      enddo
    enddo
    deallocate(QM_p, QM)

    !$omp atomic
    pm_n_pts = pm_n_pts + n_pts
    !$omp atomic
    pm_n_clip(1) = pm_n_clip(1) + n_clip(1)
    !$omp atomic
    pm_n_clip(2) = pm_n_clip(2) + n_clip(2)
  end subroutine pm_element

end module mod_pc_pressure_mass
