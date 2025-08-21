// y = A x
extern "C" void cmatv(double **x_, double **y_, double **a_, int **iptr_, int **jcn_, int **csr_, int *n_, int *nl_, int *block_size_, bool *gpu_, MPI_Fint *commg_);