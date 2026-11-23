! Compatibility mappings for PETSc < 3.23 Fortran pointer interfaces.
! Include right after "use petsc" (and any mod_petsc_raw_csr imports).
#if PETSC_VERSION_LT(3,23,0)
#define VecGetArray VecGetArrayF90
#define VecRestoreArray VecRestoreArrayF90
#define VecGetArrayRead VecGetArrayReadF90
#define VecRestoreArrayRead VecRestoreArrayReadF90
#define MatDenseGetArray MatDenseGetArrayF90
#define MatDenseRestoreArray MatDenseRestoreArrayF90
#define MatDenseGetArrayRead MatDenseGetArrayReadF90
#define MatDenseRestoreArrayRead MatDenseRestoreArrayReadF90
#define MatGetRowIJ MatGetRowIJF90
#define MatRestoreRowIJ MatRestoreRowIJF90
#define MatGetRow mat_get_row
#define MatRestoreRow mat_restore_row
#define MatCreateSubMatrices mat_create_submatrices
#define MatDestroySubMatrices mat_destroy_submatrices
#define PCASMGetSubKSP pc_asm_get_sub_ksp
#define PetscObjectGetId petsc_object_get_id
#define MatGetNonzeroState mat_get_nonzero_state
#endif
