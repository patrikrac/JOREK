!> Toroidal FFT scatter helpers shared by the preconditioner element matrices.
!!
!! JOREK assembles the poloidal element matrix on `n_plane` toroidal planes and
!! then transforms to the Fourier representation. Four channels are needed,
!! depending on where covariant toroidal derivatives act:
!!
!!   ELM_p   : no d/dphi                       -> scatter_fft_to_elm
!!   ELM_n   : d/dphi on the trial  (column)   -> scatter_fft_to_elm_n
!!   ELM_k   : d/dphi on the test   (row)      -> scatter_fft_to_elm_k
!!   ELM_kn  : d/dphi on both                  -> scatter_fft_to_elm_kn
!!
!! Each routine scatters the transform of one (row, col) basis-function pair into
!! the real cos/sin-packed element matrix. On Fourier modes d/dphi -> i*n, which
!! is why the `_n` / `_k` variants carry a single factor `mode(...)` and swap
!! real/imag parts, and the `_kn` variant carries two factors and an extra sign
!! from i^2 = -1.
!!
!! These bodies were previously duplicated inside mod_elt_matrix_elliptic and
!! models/model199/mod_elt_matrix_fft; this module is the single source.
module mod_pc_fft_scatter

  implicit none
  private

  public :: scatter_fft_to_elm
  public :: scatter_fft_to_elm_n
  public :: scatter_fft_to_elm_k
  public :: scatter_fft_to_elm_kn

contains

!-----------------------------------------------------------------
!> Scatter FFT output for row i, col j into ELM (no toroidal derivative).
!-----------------------------------------------------------------
subroutine scatter_fft_to_elm(out_fft, i, j, ELM, ndim)

  use mod_parameters, only: n_tor, n_plane

  implicit none

  complex*16, intent(in)    :: out_fft(1:n_plane)
  integer,    intent(in)    :: i, j, ndim
  real*8,     intent(inout) :: ELM(ndim, ndim)

  integer :: k, m, l, index_k, index_m

  do k = 1, (n_tor+1)/2

    index_k = n_tor*(i-1) + max(2*(k-1), 1)

    do m = 1, (n_tor+1)/2

      index_m = n_tor*(j-1) + max(2*(m-1), 1)

      l = (k-1) + (m-1)
      if (l .ge. 0 .and. l .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) +  real(out_fft(l+1))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) - imag(out_fft(l+1))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) - imag(out_fft(l+1))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) -  real(out_fft(l+1))
      elseif (l .lt. 0 .and. abs(l) .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) +  real(out_fft(abs(l)+1))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) + imag(out_fft(abs(l)+1))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) + imag(out_fft(abs(l)+1))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) -  real(out_fft(abs(l)+1))
      endif

      l = (k-1) - (m-1)
      if (l .ge. 0 .and. l .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) +  real(out_fft(l+1))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) - imag(out_fft(l+1))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) + imag(out_fft(l+1))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) +  real(out_fft(l+1))
      elseif (l .lt. 0 .and. abs(l) .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) +  real(out_fft(abs(l)+1))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) + imag(out_fft(abs(l)+1))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) - imag(out_fft(abs(l)+1))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) +  real(out_fft(abs(l)+1))
      endif

    enddo  ! m
  enddo    ! k

end subroutine scatter_fft_to_elm

!-----------------------------------------------------------------
!> ELM_n channel: d/dphi acts on the trial (column) function.
!! Weight float(mode(im)) with im the toroidal index of the column.
!-----------------------------------------------------------------
subroutine scatter_fft_to_elm_n(out_fft, i, j, ELM, ndim)

  use mod_parameters, only: n_tor, n_plane
  use phys_module,    only: mode

  implicit none

  complex*16, intent(in)    :: out_fft(1:n_plane)
  integer,    intent(in)    :: i, j, ndim
  real*8,     intent(inout) :: ELM(ndim, ndim)

  integer :: k, m, im, l, index_k, index_m

  do k = 1, (n_tor+1)/2

    index_k = n_tor*(i-1) + max(2*(k-1), 1)

    do m = 1, (n_tor+1)/2

      im      = max(2*(m-1), 1)
      index_m = n_tor*(j-1) + im

      l = (k-1) + (m-1)
      if (l .ge. 0 .and. l .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) + imag(out_fft(l+1))        * float(mode(im))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) + real(out_fft(l+1))        * float(mode(im))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) + real(out_fft(l+1))        * float(mode(im))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) - imag(out_fft(l+1))        * float(mode(im))
      elseif (l .lt. 0 .and. abs(l) .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) - imag(out_fft(abs(l)+1))   * float(mode(im))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) + real(out_fft(abs(l)+1))   * float(mode(im))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) + real(out_fft(abs(l)+1))   * float(mode(im))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) + imag(out_fft(abs(l)+1))   * float(mode(im))
      endif

      l = (k-1) - (m-1)
      if (l .ge. 0 .and. l .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) - imag(out_fft(l+1))        * float(mode(im))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) - real(out_fft(l+1))        * float(mode(im))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) + real(out_fft(l+1))        * float(mode(im))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) - imag(out_fft(l+1))        * float(mode(im))
      elseif (l .lt. 0 .and. abs(l) .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) + imag(out_fft(abs(l)+1))   * float(mode(im))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) - real(out_fft(abs(l)+1))   * float(mode(im))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) + real(out_fft(abs(l)+1))   * float(mode(im))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) + imag(out_fft(abs(l)+1))   * float(mode(im))
      endif

    enddo  ! m
  enddo    ! k

end subroutine scatter_fft_to_elm_n

!-----------------------------------------------------------------
!> ELM_k channel: d/dphi acts on the test (row) function.
!! Weight float(mode(ik)) with ik the toroidal index of the row.
!-----------------------------------------------------------------
subroutine scatter_fft_to_elm_k(out_fft, i, j, ELM, ndim)

  use mod_parameters, only: n_tor, n_plane
  use phys_module,    only: mode

  implicit none

  complex*16, intent(in)    :: out_fft(1:n_plane)
  integer,    intent(in)    :: i, j, ndim
  real*8,     intent(inout) :: ELM(ndim, ndim)

  integer :: k, m, ik, l, index_k, index_m

  do k = 1, (n_tor+1)/2

    ik      = max(2*(k-1), 1)
    index_k = n_tor*(i-1) + ik

    do m = 1, (n_tor+1)/2

      index_m = n_tor*(j-1) + max(2*(m-1), 1)

      l = (k-1) + (m-1)
      if (l .ge. 0 .and. l .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) + imag(out_fft(l+1))        * float(mode(ik))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) + real(out_fft(l+1))        * float(mode(ik))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) + real(out_fft(l+1))        * float(mode(ik))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) - imag(out_fft(l+1))        * float(mode(ik))
      elseif (l .lt. 0 .and. abs(l) .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) - imag(out_fft(abs(l)+1))   * float(mode(ik))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) + real(out_fft(abs(l)+1))   * float(mode(ik))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) + real(out_fft(abs(l)+1))   * float(mode(ik))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) + imag(out_fft(abs(l)+1))   * float(mode(ik))
      endif

      l = (k-1) - (m-1)
      if (l .ge. 0 .and. l .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) + imag(out_fft(l+1))        * float(mode(ik))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) + real(out_fft(l+1))        * float(mode(ik))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) - real(out_fft(l+1))        * float(mode(ik))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) + imag(out_fft(l+1))        * float(mode(ik))
      elseif (l .lt. 0 .and. abs(l) .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) - imag(out_fft(abs(l)+1))   * float(mode(ik))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) + real(out_fft(abs(l)+1))   * float(mode(ik))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) - real(out_fft(abs(l)+1))   * float(mode(ik))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) - imag(out_fft(abs(l)+1))   * float(mode(ik))
      endif

    enddo  ! m
  enddo    ! k

end subroutine scatter_fft_to_elm_k

!-----------------------------------------------------------------
!> ELM_kn channel: d/dphi acts on BOTH test and trial functions.
!! Weight float(mode(ik))*float(mode(im)); the extra sign is i^2 = -1.
!-----------------------------------------------------------------
subroutine scatter_fft_to_elm_kn(out_fft, i, j, ELM, ndim)

  use mod_parameters, only: n_tor, n_plane
  use phys_module,    only: mode

  implicit none

  complex*16, intent(in)    :: out_fft(1:n_plane)
  integer,    intent(in)    :: i, j, ndim
  real*8,     intent(inout) :: ELM(ndim, ndim)

  integer :: k, m, ik, im, l, index_k, index_m

  do k = 1, (n_tor+1)/2

    ik      = max(2*(k-1), 1)
    index_k = n_tor*(i-1) + ik

    do m = 1, (n_tor+1)/2

      im      = max(2*(m-1), 1)
      index_m = n_tor*(j-1) + im

      l = (k-1) + (m-1)
      if (l .ge. 0 .and. l .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) - real(out_fft(l+1))        * float(mode(im)) * float(mode(ik))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) + imag(out_fft(l+1))        * float(mode(im)) * float(mode(ik))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) + imag(out_fft(l+1))        * float(mode(im)) * float(mode(ik))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) + real(out_fft(l+1))        * float(mode(im)) * float(mode(ik))
      elseif (l .lt. 0 .and. abs(l) .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) - real(out_fft(abs(l)+1))   * float(mode(im)) * float(mode(ik))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) - imag(out_fft(abs(l)+1))   * float(mode(im)) * float(mode(ik))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) - imag(out_fft(abs(l)+1))   * float(mode(im)) * float(mode(ik))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) + real(out_fft(abs(l)+1))   * float(mode(im)) * float(mode(ik))
      endif

      l = (k-1) - (m-1)
      if (l .ge. 0 .and. l .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) + real(out_fft(l+1))        * float(mode(im)) * float(mode(ik))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) - imag(out_fft(l+1))        * float(mode(im)) * float(mode(ik))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) + imag(out_fft(l+1))        * float(mode(im)) * float(mode(ik))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) + real(out_fft(l+1))        * float(mode(im)) * float(mode(ik))
      elseif (l .lt. 0 .and. abs(l) .le. n_plane/2) then
        ELM(index_k,   index_m  ) = ELM(index_k,   index_m  ) + real(out_fft(abs(l)+1))   * float(mode(im)) * float(mode(ik))
        ELM(index_k+1, index_m  ) = ELM(index_k+1, index_m  ) + imag(out_fft(abs(l)+1))   * float(mode(im)) * float(mode(ik))
        ELM(index_k,   index_m+1) = ELM(index_k,   index_m+1) - imag(out_fft(abs(l)+1))   * float(mode(im)) * float(mode(ik))
        ELM(index_k+1, index_m+1) = ELM(index_k+1, index_m+1) + real(out_fft(abs(l)+1))   * float(mode(im)) * float(mode(ik))
      endif

    enddo  ! m
  enddo    ! k

end subroutine scatter_fft_to_elm_kn

end module mod_pc_fft_scatter
