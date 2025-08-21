// y = A x
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <mpi.h>
#include <omp.h>

#include <csignal>

extern "C" void cmatv(double **x_, double **y_, double **a_, int **iptr_, int **jcn_, int **csr_, int *n_, int *nl_, int *block_size_, bool *gpu_, MPI_Fint *commg_){

  double *x=*x_;
  double *y=*y_;
  double *a=*a_;
  int *iptr=*iptr_;
  int *jcn=*jcn_;
  int *csr=*csr_;
  const int block_size=*block_size_;
  const int n=*n_;
  const int nl = *nl_;
  bool gpu = *gpu_;
  bool offload = false;

  int my_id;
  int n_cpu;
  MPI_Comm commg=MPI_Comm_f2c(*commg_);
  MPI_Comm_rank(commg, &my_id);
  MPI_Comm_size(commg, &n_cpu);

#ifdef USE_GPU
#pragma omp target defaultmap(tofrom:scalar)
  offload = !omp_is_initial_device() && gpu;
#endif

  double *y_tmp = (double *)malloc(nl * sizeof(double));

  std::chrono::steady_clock::time_point t0, t1;
  t0 = std::chrono::steady_clock::now();

#ifdef USE_GPU
  if (offload){
    //if (!my_id){std::cout<<"Using GPU"<<std::endl;}
#pragma omp target enter data map(to: x[:n]) map(alloc: y_tmp[0:nl])
#pragma omp target teams distribute
    for (int i = 0; i < nl; ++i) {
      double ddum=0.0;
#pragma omp parallel for reduction(+:ddum)
      for (int j = iptr[i]-1; j < iptr[i + 1]-1; ++j) {
          int ib = j / block_size;
	        int k = (csr[ib] - 1 - ib)*block_size + j;
          ddum += a[k] * x[jcn[k]-1];
      }
      y_tmp[i] = ddum;
    }
#pragma omp target exit data map(from: y_tmp[0:nl]) map(delete: x[:n])
  }
#endif
  if (!offload){
    //if (!my_id){std::cout<<"Using CPU"<<std::endl;}
#pragma omp parallel for
    for (int i = 0; i < nl; ++i) {
      y_tmp[i] = 0.0;
      for (int j = iptr[i]-1; j < iptr[i + 1]-1; ++j) {
          int ib = j / block_size;
          int k = (csr[ib] - 1 - ib)*block_size + j;
          y_tmp[i] += a[k] * x[jcn[k]-1];
      }	  
    }
  }

  int *rc = new int[n_cpu];
  int *rd = new int[n_cpu];
  MPI_Allgather(&nl, 1, MPI_INT, rc, 1, MPI_INT, commg);

  rd[0] = 0;
  for (int i=1; i<n_cpu; i++){
    rd[i] = rd[i-1] + rc[i-1];
  }

  MPI_Barrier(commg);
  MPI_Allgatherv(y_tmp, nl, MPI_DOUBLE, y, rc, rd, MPI_DOUBLE, commg);

  free(y_tmp);

  t1 = std::chrono::steady_clock::now();
  if (!my_id){
    std::cout<<"Elapsed time cmatv (s) = "<< std::chrono::duration_cast<
	    std::chrono::microseconds>(t1 - t0).count()*1e-6 <<std::endl;
      //std::chrono::microseconds>(t1 - t0).count()*1e-6 <<" "<<blas::nrm2(n, y, 1)<<std::endl;
    }

  return;
}
