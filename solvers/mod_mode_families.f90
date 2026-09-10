!> The toroidal mode-family decomposition of the preconditioner, shared by both
!! solver backends.
!!
!! JOREK's preconditioner is block-diagonal in the toroidal harmonics: the modes
!! are partitioned into *families*, each family gives an independent sub-system,
!! and the MPI ranks are partitioned into one disjoint subgroup per family so the
!! blocks are factorized concurrently by smaller communicators.
!!
!! Both the legacy path (mod_preconditioner) and the PETSc path
!! (mod_petsc_pc_modesplit) need exactly the same partition, and a disagreement
!! between them would be a silently wrong preconditioner rather than a crash -
!! hence one implementation, called from both.
!!
!! Everything here is pure bookkeeping over the namelist variables in phys_module
!! (autodistribute_modes, modes_per_family, mode_families_modes,
!! weights_per_family, autodistribute_ranks, ranks_per_family). No MPI, no matrices.
module mod_mode_families

  implicit none
  private
  public :: mode_family_count, mode_family_distribute_modes, &
            mode_family_distribute_ranks, mode_family_weight

contains

  !> Number of mode families this run uses.
  !!
  !! With autodistribute_modes the families are fixed by the toroidal resolution
  !! (one per harmonic, i.e. per sin/cos pair) and the namelist n_mode_families is
  !! ignored. broadcast_phys already overwrites n_mode_families to this value, but
  !! deriving it here keeps the module usable before that point.
  integer function mode_family_count()
    use phys_module,    only: autodistribute_modes, n_mode_families
    use mod_parameters, only: n_tor

    if (autodistribute_modes) then
      mode_family_count = (n_tor + 1)/2
    else
      mode_family_count = n_mode_families
    endif
  end function mode_family_count


  !> Which toroidal modes belong to each family.
  !!
  !! modes_per_fam(f)   number of modes in family f
  !! fam_modes(f, 1:n)  their i_tor indices, 1-based; the second dimension is the
  !!                    largest family, and unused slots are -1.
  !!
  !! Both arguments are allocated here and must be unassociated on entry.
  !!
  !! The autodistribute layout is family 1 = the n=0 harmonic alone (mode 1), and
  !! family i>1 = the sin/cos pair (2(i-1), 2(i-1)+1) of one harmonic. Family 1 is
  !! therefore half the size of every other one, which is what ranks_per_family
  !! exists to compensate for.
  subroutine mode_family_distribute_modes(n_fam, modes_per_fam, fam_modes)
    use phys_module, only: autodistribute_modes, modes_per_family, mode_families_modes

    integer, intent(in) :: n_fam
    integer, dimension(:),   pointer :: modes_per_fam
    integer, dimension(:,:), pointer :: fam_modes

    integer :: i, j, n_fam_max

    allocate(modes_per_fam(n_fam))

    if (autodistribute_modes) then
      modes_per_fam(1) = 1
      if (n_fam > 1) modes_per_fam(2:n_fam) = 2
    else
      do i = 1, n_fam
        modes_per_fam(i) = modes_per_family(i)
      enddo
    endif

    n_fam_max = 1
    do i = 1, n_fam
      n_fam_max = max(n_fam_max, modes_per_fam(i))
    enddo

    allocate(fam_modes(n_fam, n_fam_max))
    fam_modes(:,:) = -1

    if (autodistribute_modes) then
      fam_modes(1,1) = 1
      do i = 2, n_fam
        fam_modes(i,1) = (i - 1)*2
        fam_modes(i,2) = (i - 1)*2 + 1
      enddo
    else
      do i = 1, n_fam
        do j = 1, modes_per_fam(i)
          fam_modes(i,j) = mode_families_modes(i,j)
        enddo
      enddo
    endif

  end subroutine mode_family_distribute_modes


  !> Which MPI ranks belong to each family.
  !!
  !! Families get contiguous global rank ranges, so that rank 0 is always the
  !! master of family 1 - the legacy communicator setup asserts exactly that.
  !!
  !! ranks_per_fam(f)     number of ranks in family f
  !! rank_range(f)        1-based global rank at which family f starts;
  !!                      rank_range(n_fam+1) == n_cpu + 1
  !! fam_ranks(f, 1:n)    the 0-based global ranks of family f, -1 in unused slots
  !! family_of_rank(r)    1-based family of the 0-based global rank r-1
  !! ok                   .false. if the requested ranks_per_family does not sum
  !!                      to n_cpu, or leaves a family empty
  !!
  !! All output arrays are allocated here and must be unassociated on entry, except
  !! when ok comes back .false. - the caller must abort in that case rather than
  !! read them.
  subroutine mode_family_distribute_ranks(n_cpu, n_fam, ranks_per_fam, rank_range, &
                                          fam_ranks, family_of_rank, ok)
    use phys_module, only: autodistribute_ranks, ranks_per_family

    integer, intent(in) :: n_cpu, n_fam
    integer, dimension(:),   pointer :: ranks_per_fam, rank_range, family_of_rank
    integer, dimension(:,:), pointer :: fam_ranks
    logical, intent(out) :: ok

    integer :: mcpu, r, i, j

    allocate(rank_range(n_fam + 1))
    allocate(ranks_per_fam(n_fam))
    allocate(fam_ranks(n_fam, n_cpu))
    allocate(family_of_rank(n_cpu))

    fam_ranks(:,:)    = -1
    family_of_rank(:) = -1

    mcpu = n_cpu/n_fam
    r    = mod(n_cpu, n_fam)

    if (autodistribute_ranks) then
      do i = 1, n_fam
        ranks_per_fam(i) = mcpu
        if ((r > 0) .and. (i <= r)) ranks_per_fam(i) = ranks_per_fam(i) + 1  ! spread the remainder
      enddo
    else
      do i = 1, n_fam
        ranks_per_fam(i) = ranks_per_family(i)
      enddo
    endif

    ! An empty family has nobody to factorize its block, and a total that misses
    ! n_cpu leaves ranks unassigned (or assigns some twice). Both are input errors,
    ! and both would show up much later as a deadlock or a wrong answer.
    ok = .true.
    do i = 1, n_fam
      if (ranks_per_fam(i) < 1) ok = .false.
    enddo
    if (sum(ranks_per_fam(1:n_fam)) /= n_cpu) ok = .false.
    if (.not. ok) return

    rank_range(1) = 1
    do i = 2, n_fam + 1
      rank_range(i) = rank_range(i-1) + ranks_per_fam(i-1)
    enddo

    do i = 1, n_cpu
      do j = 2, n_fam + 1
        if ((i >= rank_range(j-1)) .and. (i < rank_range(j))) then
          family_of_rank(i) = j - 1
          exit
        endif
      enddo
    enddo

    do j = 1, n_fam
      r = 0
      do i = 1, n_cpu
        if (family_of_rank(i) == j) then
          r = r + 1
          fam_ranks(j,r) = i - 1
        endif
      enddo
    enddo

  end subroutine mode_family_distribute_ranks


  !> The weight family f's solution carries when the per-family contributions are
  !! summed back into the global vector.
  !!
  !! Only meaningful when the families overlap - a mode belonging to two families
  !! is then solved for twice, and the weights are what stop it being counted
  !! twice (the documented usage is 0.5 for two-fold overlap). Disjoint families
  !! use 1.0 throughout, which is also the namelist default and the value the
  !! autodistribute layout always gets, since that layout cannot overlap.
  real(kind=8) function mode_family_weight(f)
    use phys_module, only: autodistribute_modes, weights_per_family

    integer, intent(in) :: f

    if (autodistribute_modes) then
      mode_family_weight = 1.0d0
    else
      mode_family_weight = weights_per_family(f)
    endif
  end function mode_family_weight

end module mod_mode_families
