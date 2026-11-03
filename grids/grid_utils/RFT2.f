      SUBROUTINE RFT2(DATA,NR,KR)
      !*****************************************************************
      !* REAL FOURIER TRANSFORM  (forward, real-to-complex).          *
      !*                                                              *
      !* Self-contained drop-in preserving the original RFT2 name,    *
      !* argument list and coefficient layout.                        *
      !*                                                              *
      !* INPUT : NR real values                                       *
      !*           DATA(1), DATA(1+KR), ..., DATA(1+(NR-1)*KR)        *
      !* OUTPUT: NR/2+1 complex coefficients as consecutive (re,im)   *
      !*         pairs                                                *
      !*           ( DATA(1),       DATA(1+KR)        )   mode 0       *
      !*           ( DATA(1+2*KR),  DATA(1+3*KR)      )   mode 1       *
      !*            .........................................         *
      !*           ( DATA(1+NR*KR), DATA(1+(NR+1)*KR) )   mode NR/2    *
      !*                                                              *
      !* Convention: unnormalised forward transform                   *
      !*   X_k = sum_{n=0}^{NR-1} x_n * exp(-2*pi*i*k*n/NR),           *
      !* identical to FFTW's dfftw_execute_dft_r2c, so the FFTW and    *
      !* non-FFTW code paths yield the same coefficients.             *
      !* DATA must be dimensioned with at least (NR+1)*KR+1 elements   *
      !* (i.e. NR+2 when KR=1).                                        *
      !*****************************************************************

      implicit none
      real*8  :: DATA(*)
      integer :: NR, KR

      complex*16, allocatable :: work(:)
      integer :: n, k

      n = NR
      allocate(work(0:n-1))

      do k = 0, n-1
        work(k) = cmplx(DATA(1+k*KR), 0.d0, kind=8)
      enddo

      call RFT2_DFT(work, n)

      do k = 0, n/2
        DATA(1 + (2*k  )*KR) = real (work(k))
        DATA(1 + (2*k+1)*KR) = aimag(work(k))
      enddo

      deallocate(work)
      RETURN
      END

      SUBROUTINE RFT2_DFT(X,N)
      !*****************************************************************
      !* In-place forward discrete Fourier transform,                 *
      !*   X_k = sum_n X_n * exp(-2*pi*i*k*n/N).                       *
      !* Iterative radix-2 Cooley-Tukey when N is a power of two,      *
      !* otherwise a direct O(N^2) evaluation. Standard textbook       *
      !* (public-domain) algorithm.                                   *
      !*****************************************************************

      implicit none
      integer,    intent(in)    :: N
      complex*16, intent(inout) :: X(0:N-1)

      real*8, parameter :: TWOPI = 6.28318530717958647692528677d0
      complex*16, allocatable :: Y(:)
      complex*16 :: u, t, w, wm
      integer :: i, j, k, m, mh
      real*8  :: ang

      if (N <= 1) return

      if (iand(N, N-1) == 0) then

        ! bit-reversal permutation
        j = 0
        do i = 1, N-1
          k = N/2
          do while (k <= j)
            j = j - k
            k = k/2
          enddo
          j = j + k
          if (i < j) then
            t = X(i)
            X(i) = X(j)
            X(j) = t
          endif
        enddo

        ! Danielson-Lanczos butterflies
        m = 1
        do while (m < N)
          mh = m
          m  = m + m
          ang = -TWOPI / dble(m)
          wm  = cmplx(cos(ang), sin(ang), kind=8)
          do k = 0, N-1, m
            w = (1.d0, 0.d0)
            do j = 0, mh-1
              u = X(k+j)
              t = w * X(k+j+mh)
              X(k+j)    = u + t
              X(k+j+mh) = u - t
              w = w * wm
            enddo
          enddo
        enddo

      else

        ! direct evaluation (any N)
        allocate(Y(0:N-1))
        do k = 0, N-1
          u = (0.d0, 0.d0)
          do j = 0, N-1
            ang = -TWOPI * dble(mod(k*j, N)) / dble(N)
            u = u + X(j) * cmplx(cos(ang), sin(ang), kind=8)
          enddo
          Y(k) = u
        enddo
        X = Y
        deallocate(Y)

      endif

      RETURN
      END
