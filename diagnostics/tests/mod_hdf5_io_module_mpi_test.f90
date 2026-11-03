module mod_hdf5_io_module_mpi_test
use mpi,           only: MPI_INFO_NULL,MPI_COMM_WORLD
use mpi,           only: MPI_Info_create,MPI_Info_free
use mod_pcg32_rng, only: pcg32_rng
use fruit
use fruit_mpi
use hdf5_io_module
implicit none
private
public :: run_fruit_hdf5_io_module_mpi
!> Variables --------------------------------------------------
integer,parameter          :: master_rank=0
integer,parameter          :: mpi_comm=MPI_COMM_WORLD
integer,parameter          :: mpi_info=MPI_INFO_NULL
integer,parameter          :: ndims_tot=5
integer,parameter          :: n_sol_char=26
integer,parameter          :: len_char=5
integer,parameter          :: n_char=4
integer,parameter          :: n_delta_elements=20
integer(HSIZE_T),parameter :: n1_HSIZE_T=int(-1,kind=HSIZE_T)
integer,dimension(2),parameter                   :: seed_interval=(/1,900000/) 
integer,dimension(ndims_tot),parameter           :: n_elements=[13,20,18,7,4]
character(len=1),dimension(n_sol_char),parameter :: letters=(/'a','b','c','d',&
'e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u',&
'v','w','x','y','z'/)
real*4,parameter  :: tol_r4=1.d-16
real*8,parameter  :: tol_r8=1.d-7
logical,parameter :: access_hdf5_parallel=.true.
logical,parameter :: mpio_collective=.true.
logical,parameter :: use_hdf5_mpio=.true.
character(len=14) :: filename_base="test_hdf5_file"
character(len=3)  :: extension=".h5"
character(len=1)  :: rank_format
integer           :: rank_loc,n_tasks_loc,ifail_loc
integer           :: mpi_comm_loc,mpi_info_loc
real*8,dimension(n_elements(1),n_elements(2),n_elements(3),&
n_elements(4),n_elements(5))              :: array_sol
character(len=len_char),dimension(n_char) :: char_sol 
type(pcg32_rng)   :: rng
!> Interfaces -------------------------------------------------
contains
!> Fruit basket -----------------------------------------------
subroutine run_fruit_hdf5_io_module_mpi(rank,n_tasks,ifail)
  implicit none
  integer,intent(inout) :: ifail
  integer,intent(in)    :: rank,n_tasks
  if(rank.eq.master_rank) write(*,'(/A)') " ... setting-up: hdf5 IO module mpi tests"
  call setup(rank,n_tasks,ifail) 
  if(rank.eq.master_rank) write(*,'(/A)') " ... running: hdf5 IO module mpi tests" 
  call run_test_case(test_create_hdf5_file,'test_create_hdf5_file')
  call run_test_case(test_open_hdf5_file,'test_open_hdf5_file')
  call run_test_case(test_create_open_hdf5_file,'test_create_open_hdf5_file')
  call run_test_case(test_HDF5_saving_char,"test_HDF5_saving_char")
  call run_test_case(test_HDF5_array1D_saving_char,"test_HDF5_array1D_saving_char")
  call run_test_case(test_HDF5_real_saving,"test_HDF5_integer_saving")
  call run_test_case(test_HDF5_array1D_saving_int,"test_HDF5_array1D_saving_int")
  call run_test_case(test_HDF5_array2D_saving_int,"test_HDF5_array2D_saving_int")
  call run_test_case(test_HDF5_array3D_saving_int,"test_HDF5_array3D_saving_int")
  call run_test_case(test_HDF5_array1D_saving_r4,"test_HDF5_array1D_saving_r4")
  call run_test_case(test_HDF5_real_saving,"test_HDF5_real_saving")
  call run_test_case(test_HDF5_array1D_saving_r8,"test_HDF5_array1D_saving_r8")
  call run_test_case(test_HDF5_array2D_saving_r8,"test_HDF5_array2D_saving_r8")
  call run_test_case(test_HDF5_array3D_saving_r8,"test_HDF5_array3D_saving_r8")
  call run_test_case(test_HDF5_array4D_saving_r8,"test_HDF5_array4D_saving_r8")
  call run_test_case(test_HDF5_array5D_saving_r8,"test_HDF5_array5D_saving_r8")
  if(rank.eq.master_rank) write(*,'(/A)') " ... tearing-down: hdf5 IO module mpi tests" 
  call teardown(rank,n_tasks,ifail)
end subroutine run_fruit_hdf5_io_module_mpi

!> Set-up and tear-down ---------------------------------------
subroutine setup(rank,n_tasks,ifail)
  use mod_gnu_rng,     only: set_seed_sys_time
  use mod_gnu_rng,     only: gnu_rng_interval 
  use mod_random_seed, only: random_seed
  implicit none
  integer,intent(in)          :: rank,n_tasks
  integer,intent(inout)       :: ifail
  integer                     :: ii,jj,kk,pp
  integer,dimension(len_char) :: id_array
  character(len=len_char)     :: buffer
  rank_loc = rank; n_tasks_loc = n_tasks; ifail_loc = ifail;
  mpi_info_loc = mpi_info; mpi_comm_loc = mpi_comm;
  if(mpi_info_loc.ne.MPI_INFO_NULL) call MPI_Info_create(mpi_info_loc,ifail_loc)
  rank_format = '1' 
  if(rank.gt.0) write(rank_format,'(I1)') int(log10(real(rank_loc,kind=8)))+1
  !> initialise double array for testing
  call rng%initialize(n_elements(1),random_seed(),n_tasks_loc,rank_loc,ifail_loc)
  do pp=1,n_elements(5)
    do kk=1,n_elements(4)
      do jj=1,n_elements(3)
        do ii=1,n_elements(2)
          call rng%next(array_sol(:,ii,jj,kk,pp))
        enddo
      enddo
    enddo
  enddo
  !> initialise character array for testing
  call set_seed_sys_time(seed_interval,rank_loc); char_sol='';
  do ii=1,n_char
    call gnu_rng_interval(len_char,[1,n_sol_char],id_array)
    buffer = ''
    do jj=1,len_char
      write(buffer,'(A,A)') trim(char_sol(ii)),trim(letters(id_array(jj)))
      char_sol(ii) = trim(buffer) 
    enddo
  enddo   
end subroutine setup

subroutine teardown(rank,n_tasks,ifail)
  implicit none
  integer,intent(in)    :: rank,n_tasks
  integer,intent(inout) :: ifail
  if(mpi_info_loc.ne.MPI_INFO_NULL) call MPI_Info_free(mpi_info_loc,ifail_loc)
  ifail = ifail_loc; rank_loc = -1; n_tasks_loc = -1;
end subroutine teardown
!> Tests ------------------------------------------------------
!> Test procedure for exclusively creating HDF5 files
subroutine test_create_hdf5_file()
  implicit none
  integer(HID_T)     :: file_id
  character(len=100) :: filename
  logical            :: file_exists
  write(filename,'(A,A,I'//trim(rank_format)//',A)') &
  trim(filename_base),'_rank',rank_loc,trim(extension)
  call HDF5_create(trim(filename),file_id,ifail_loc)
  call HDF5_close(file_id)
  file_exists=.false.; inquire(file=trim(filename),exist=file_exists)
  call assert_true(file_exists,"Error test create HDF5 file posix: file "//&
  trim(filename)//" not created!")
  call remove_file(filename,file_exists_in=file_exists)
  filename = ''; filename = trim(filename_base)//trim(extension)
  call HDF5_create(filename,file_id,create_access_plist_in=access_hdf5_parallel,&
  mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_close(file_id)
  file_exists=.false.; inquire(file=trim(filename),exist=file_exists);
  call assert_true(file_exists,"Error test create HDF5 file access FILE_ACCESS: file "//&
  trim(filename)//" not created!")
  call remove_file(filename,file_exists_in=file_exists)
end subroutine test_create_hdf5_file

!> Test procedure for exclusively opening HDF5 file
subroutine test_open_hdf5_file()
  implicit none
  integer(HID_T)     :: file_id
  character(len=100) :: filename
  write(filename,'(A,A,I'//trim(rank_format)//',A)') & 
  trim(filename_base),'_rank',rank_loc,trim(extension)
  ifail_loc=0; call HDF5_create(filename,file_id,ifail_loc); call HDF5_close(file_id); 
  call HDF5_open(filename,file_id,ifail_loc); call HDF5_close(file_id);
  call assert_equals(ifail_loc,0,"Error test open HDF5 file posix: file "//&
  trim(filename)//" not opened!"); call remove_file(filename);
  filename = ''; filename = trim(filename_base)//trim(extension); ifail_loc=0;
  call HDF5_create(filename,file_id,create_access_plist_in=access_hdf5_parallel,&
  mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc); call HDF5_close(file_id); 
  call HDF5_open(filename,file_id,ierr=ifail_loc,create_access_plist_in=access_hdf5_parallel,&
  mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc); call HDF5_close(file_id);
  call assert_equals(ifail_loc,0,"Error test open HDF5 file access FILE_ACCESS: file "//&
  trim(filename)//" not opened!"); call remove_file(filename);
end subroutine test_open_hdf5_file

!> test combined procedure for creating and opening HDF5 files
subroutine test_create_open_hdf5_file()
  implicit none
  integer(HID_T)     :: file_id
  character(len=100) :: filename
  logical            :: file_exists
  write(filename,'(A,A,I'//trim(rank_format)//',A)') &
  trim(filename_base),'_rank',rank_loc,trim(extension)
  ifail_loc=0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc);
  call HDF5_close(file_id); file_exists=.false.; 
  inquire(file=filename,exist=file_exists);
  call assert_true(file_exists,"Error test create-open HDF5 file posix: file "//&
  trim(filename)//" not created!"); ifail_loc=0;
  call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc);
  call HDF5_close(file_id);
  call assert_equals(ifail_loc,0,"Error test create-open HDF5 file posix: file "//&
  trim(filename)//" not opened!"); call remove_file(trim(filename)); 
  filename=''; filename = trim(filename_base)//trim(extension); ifail_loc=0;
  call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_close(file_id); file_exists=.false.; 
  inquire(file=filename,exist=file_exists);
  call assert_true(file_exists,"Error test create-open HDF5 file access FILE_ACCESS: file "//&
  trim(filename)//" not created!");
  ifail_loc=0; call HDF5_open_or_create(filename,file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_close(file_id);
  call assert_equals(ifail_loc,0,"Error test create-open HDF5 file access FILE_ACCESS: file "//&
  trim(filename)//" not opened!"); call remove_file(filename);
end subroutine test_create_open_hdf5_file

!> the the posix and collective writing / reading HDF5 file
!> of a single string
subroutine test_HDF5_saving_char()
  implicit none
  character(len=12),parameter  :: datasetname='char_value'
  character(len=len_char)      :: test_value,result_value
  character(len=:),allocatable :: test_value_alloc
  integer(HID_T)               :: file_id
  character(len=100)           :: filename 
  !> initialise posix test
  test_value = ''; result_value = char_sol(1);
  write(filename,'(A,A,I'//trim(rank_format)//',A)') &
  trim(filename_base),'_rank',rank_loc,trim(extension)
  ifail_loc=0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc);
  call HDF5_char_saving(file_id,result_value,datasetname)
  call HDF5_char_reading(file_id,test_value,datasetname)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals(test_value,result_value,&
  "Error test HDF5 I/O 0D character posix: test and result values mismatch!")
  filename = trim(filename_base)//trim(extension);
  test_value = ''; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_char_saving(file_id,result_value,datasetname,mpi_rank=rank_loc,&
  n_mpi_tasks=n_tasks_loc,mpi_comm_in=mpi_comm_loc,use_hdf5_parallel_in=use_hdf5_mpio,&
  mpio_collective_in=mpio_collective)
  call HDF5_char_reading(file_id,test_value,datasetname,&
  mpi_rank=rank_loc,n_mpi_tasks=n_tasks_loc)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals(test_value,result_value,&
  "Error test HDF5 I/O 0D character MPI collective: test and result values mismatch!")
  call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_char_saving(file_id,result_value,datasetname,mpi_rank=rank_loc,&
  n_mpi_tasks=n_tasks_loc,mpi_comm_in=mpi_comm_loc,use_hdf5_parallel_in=use_hdf5_mpio,&
  mpio_collective_in=mpio_collective)
  call HDF5_allocatable_char_reading(file_id,test_value_alloc,datasetname,&
  mpi_rank=rank_loc,n_mpi_tasks=n_tasks_loc)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals(test_value_alloc,result_value,&
  "Error test HDF5 I/O 0D character allocatable MPI collective: test and result values mismatch!")
  if(allocated(test_value_alloc)) deallocate(test_value_alloc)
end subroutine test_HDF5_saving_char

!> the the posix and collective writing / reading HDF5 file
!> of character 1D array
subroutine test_HDF5_array1D_saving_char()
  implicit none
  character(len=12),parameter               :: datasetname='array1D_char'
  character(len=len_char),dimension(n_char) :: test_array,result_array
  integer(HID_T)                            :: file_id
  integer(HSIZE_T),dimension(1)             :: offset
  character(len=100)                        :: filename 
  !> initialise posix test
  test_array = ''; result_array = char_sol;
  write(filename,'(A,A,I'//trim(rank_format)//',A)') &
  trim(filename_base),'_rank',rank_loc,trim(extension)
  ifail_loc=0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc);
  call HDF5_array1D_saving_char(file_id,result_array,n_char,datasetname)
  call HDF5_array1D_reading_char(file_id,test_array,datasetname)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals(test_array,result_array,n_char,&
  "Error test HDF5 I/O 1D character posix: test and result array mismatch!")
  filename = trim(filename_base)//trim(extension); offset=[rank_loc*n_char];
  test_array = ''; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array1D_saving_char(file_id,result_array,n_tasks_loc*n_char,&
  datasetname,start=offset,mpi_comm_in=mpi_comm_loc,use_hdf5_parallel_in=use_hdf5_mpio,&
  mpio_collective_in=mpio_collective)
  call HDF5_array1D_reading_char(file_id,test_array,datasetname,start=offset)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals(test_array,result_array,n_char,&
  "Error test HDF5 I/O 1D character MPI collective: test and result array mismatch!")
end subroutine test_HDF5_array1D_saving_char

!> the the posix and collective writing / reading HDF5 file of a single integer
subroutine test_HDF5_integer_saving()
  implicit none
  character(len=12),parameter :: datasetname='integer_value'
  integer                     :: test_value,result_value
  integer(HID_T)              :: file_id
  character(len=100)          :: filename 
  !> initialise posix test
  test_value = 0; result_value = int(1d3*array_sol(1,1,1,1,1));
  write(filename,'(A,A,I'//trim(rank_format)//',A)') &
  trim(filename_base),'_rank',rank_loc,trim(extension)
  ifail_loc=0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc);
  call HDF5_integer_saving(file_id,result_value,datasetname)
  call HDF5_integer_reading(file_id,test_value,datasetname)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals(test_value,result_value,&
  "Error test HDF5 I/O 0D integer posix: test and result array mismatch!")
  filename = trim(filename_base)//trim(extension);
  test_value = 0d0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_integer_saving(file_id,result_value,datasetname,mpi_rank=rank_loc,&
  n_mpi_tasks=n_tasks_loc,use_hdf5_parallel_in=use_hdf5_mpio,mpio_collective_in=mpio_collective)
  call HDF5_integer_reading(file_id,test_value,datasetname,&
  mpi_rank=rank_loc,n_mpi_tasks=n_tasks_loc)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals(test_value,result_value,&
  "Error test HDF5 I/O 0D integer MPI collective: test and result array mismatch!")
end subroutine test_HDF5_integer_saving

!> the the posix and collective writing / reading HDF5 file
!> of integer 1D array
subroutine test_HDF5_array1D_saving_int()
  use mod_assert_equals_tools, only: assert_equals_allocatable_arrays
  implicit none
  character(len=11),parameter      :: datasetname='array1D_int'
  integer                          :: ii
  integer,dimension(n_elements(1)) :: test_array,result_array
  integer,dimension(n_tasks_loc)   :: elements_all,displs
  integer,dimension(:),allocatable :: test_array_allocatable
  integer(HID_T)                   :: file_id
  integer(HSIZE_T),dimension(1)    :: offset,reqdim
  character(len=100)               :: filename 
  !> initialise posix test
  test_array = 0; result_array = int(1d3*array_sol(:,1,1,1,1));
  write(filename,'(A,A,I'//trim(rank_format)//',A)') &
  trim(filename_base),'_rank',rank_loc,trim(extension)
  ifail_loc=0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc);
  call HDF5_array1D_saving_int(file_id,result_array,n_elements(1),datasetname)
  call HDF5_array1D_reading_int(file_id,test_array,datasetname)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals(test_array,result_array,n_elements(1),&
  "Error test HDF5 I/O 1D integer posix: test and result array mismatch!")
  filename = trim(filename_base)//trim(extension); elements_all = n_elements(1);
  displs = 0; do ii=2,n_tasks_loc; displs(ii)=sum(elements_all(1:ii-1)); enddo; 
  offset = [rank_loc*n_elements(1)]; test_array = 0; 
  if(rank_loc.eq.master_rank) call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc) 
  call HDF5_array1D_saving_int_native_or_gatherv(file_id,result_array,sum(elements_all),&
  datasetname,.true.,dim1_all_tasks=elements_all,displs=displs,mpi_rank=rank_loc,&
  n_mpi=n_tasks_loc,mpi_comm_loc=mpi_comm_loc,use_hdf5_parallel_in=use_hdf5_mpio,&
  mpio_collective_in=mpio_collective)  
  if(rank_loc.eq.master_rank) call HDF5_close(file_id); call MPI_Barrier(mpi_comm_loc,ifail_loc);
  call HDF5_open(filename,file_id,ifail_loc,create_access_plist_in=access_hdf5_parallel,&
  mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array1D_reading_int(file_id,test_array,datasetname,start=offset)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals(test_array,result_array,n_elements(1),&
  "Error test HDF5 I/O 1D integer MPI Native-Gatherv (Gatherv): test and result array mismatch!")
  filename = trim(filename_base)//trim(extension); offset=[rank_loc*n_elements(1)];
  test_array = 0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array1D_saving_int(file_id,result_array,n_tasks_loc*n_elements(1),&
  datasetname,start=offset,use_hdf5_parallel_in=use_hdf5_mpio,mpio_collective_in=mpio_collective)
  call HDF5_array1D_reading_int(file_id,test_array,datasetname,start=offset);
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals(test_array,result_array,n_elements(1),&
  "Error test HDF5 I/O 1D integer MPI collective: test and result array mismatch!")
  filename = trim(filename_base)//trim(extension); offset=[rank_loc*n_elements(1)];
  test_array = 0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array1D_saving_int_native_or_gatherv(file_id,result_array,n_tasks_loc*n_elements(1),&
  datasetname,.false.,start=offset,use_hdf5_parallel_in=use_hdf5_mpio,mpio_collective_in=mpio_collective)
  call HDF5_array1D_reading_int(file_id,test_array,datasetname,start=offset);
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals(test_array,result_array,n_elements(1),&
  "Error test HDF5 I/O 1D integer MPI Native-Gatherv (Native): test and result array mismatch!")
  write(filename,'(A,A,I'//trim(rank_format)//',A)') &
  trim(filename_base),'_rank',rank_loc,trim(extension)
  call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc);
  call HDF5_array1D_saving_int(file_id,result_array,n_elements(1),datasetname,&
  use_hdf5_parallel_in=use_hdf5_mpio)
  call HDF5_allocatable_array1D_reading_int(file_id,test_array_allocatable,&
  datasetname); call HDF5_close(file_id); call remove_file(filename);
  call assert_equals_allocatable_arrays(n_elements(1),result_array,test_array_allocatable,&
  "Error test HDF5 I/O 1D integer allocatable MPI collective:")
  filename = trim(filename_base)//trim(extension); offset=[rank_loc*n_elements(1)];
  reqdim = [int(n_elements(1),kind=HSIZE_T)]; test_array_allocatable = 0;
  call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array1D_saving_int(file_id,result_array,n_tasks_loc*n_elements(1),&
  datasetname,start=offset,use_hdf5_parallel_in=use_hdf5_mpio,mpio_collective_in=mpio_collective)
  call HDF5_allocatable_array1D_reading_int(file_id,test_array_allocatable,&
  datasetname,start=offset,reqdims_in=reqdim); call HDF5_close(file_id); call remove_file(filename);
  call assert_equals_allocatable_arrays(n_elements(1),result_array,test_array_allocatable,&
  "Error test HDF5 I/O 1D integer allocatable reqdims MPI collective:")
  if(allocated(test_array_allocatable)) deallocate(test_array_allocatable)
end subroutine test_HDF5_array1D_saving_int

!> the the posix and collective writing / reading HDF5 file
!> of integer 2D array
subroutine test_HDF5_array2D_saving_int()
  use mod_assert_equals_tools, only: assert_equals_allocatable_arrays
  implicit none
  character(len=11),parameter                    :: datasetname='array2D_int'
  integer                                        :: ii
  integer,dimension(n_tasks_loc)                 :: elements_all,displs
  integer,dimension(n_elements(1),n_elements(2)) :: test_array,result_array
  integer,dimension(:,:),allocatable             :: test_array_allocatable
  integer(HID_T)                                 :: file_id
  integer(HSIZE_T),dimension(2)                  :: offset,reqdim
  character(len=100)                             :: filename 
  !> initialise posix test
  test_array = 0; result_array = int(1d3*array_sol(:,:,1,1,1));
  write(filename,'(A,A,I'//trim(rank_format)//',A)') &
  trim(filename_base),'_rank',rank_loc,trim(extension)
  ifail_loc=0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc);
  call HDF5_array2D_saving_int(file_id,result_array,n_elements(1),n_elements(2),&
  datasetname); call HDF5_array2D_reading_int(file_id,test_array,datasetname)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals(test_array,result_array,n_elements(1),n_elements(2),&
  "Error test HDF5 I/O 2D integer posix: test and result array mismatch!")
  filename = trim(filename_base)//trim(extension); elements_all = n_elements(2);
  displs = 0; do ii=2,n_tasks_loc; displs(ii)=sum(elements_all(1:ii-1)); enddo;
  offset = [0,rank_loc*n_elements(2)]; test_array = 0; 
  if(rank_loc.eq.master_rank) call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc) 
  call HDF5_array2D_saving_int_native_or_gatherv(file_id,result_array,n_elements(1),&
  sum(elements_all),datasetname,.true.,dim2_all_tasks=elements_all,displs=displs,&
  mpi_rank=rank_loc,n_mpi=n_tasks_loc,mpi_comm_loc=mpi_comm_loc,use_hdf5_parallel_in=use_hdf5_mpio,&
  mpio_collective_in=mpio_collective)
  if(rank_loc.eq.master_rank) call HDF5_close(file_id); call MPI_Barrier(mpi_comm_loc,ifail_loc);
  call HDF5_open(filename,file_id,ifail_loc,create_access_plist_in=access_hdf5_parallel,&
  mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array2D_reading_int(file_id,test_array,datasetname,start=offset)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals(test_array,result_array,n_elements(1),n_elements(2),&
  "Error test HDF5 I/O 2D integer MPI Native-Gatherv (Gatherv): test and result array mismatch!")
  filename = trim(filename_base)//trim(extension); offset=[0,rank_loc*n_elements(2)];
  test_array = 0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array2D_saving_int(file_id,result_array,n_elements(1),n_tasks_loc*n_elements(2),&
  datasetname,start=offset,use_hdf5_parallel_in=use_hdf5_mpio,mpio_collective_in=mpio_collective)
  call HDF5_array2D_reading_int(file_id,test_array,datasetname,start=offset)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals(test_array,result_array,n_elements(1),n_elements(2),&
  "Error test HDF5 I/O 2D integer MPI collective: test and result array mismatch!")
  write(filename,'(A,A,I'//trim(rank_format)//',A)') &
  trim(filename_base),'_rank',rank_loc,trim(extension)
  call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc);
  call HDF5_array2D_saving_int(file_id,result_array,n_elements(1),n_elements(2),datasetname,&
  use_hdf5_parallel_in=use_hdf5_mpio,mpio_collective_in=mpio_collective)
  call HDF5_allocatable_array2D_reading_int(file_id,test_array_allocatable,&
  datasetname); call HDF5_close(file_id); call remove_file(filename);
  call assert_equals_allocatable_arrays(n_elements(1),n_elements(2),result_array,test_array_allocatable,&
  "Error test HDF5 I/O 2D integer allocatable MPI collective: mismatch!")
  filename = trim(filename_base)//trim(extension); offset=[0,rank_loc*n_elements(2)];
  test_array = 0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array2D_saving_int_native_or_gatherv(file_id,result_array,n_elements(1),&
  n_tasks_loc*n_elements(2),datasetname,.false.,start=offset,use_hdf5_parallel_in=use_hdf5_mpio,&
  mpio_collective_in=mpio_collective)
  call HDF5_array2D_reading_int(file_id,test_array,datasetname,start=offset);
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals(test_array,result_array,n_elements(1),n_elements(2),&
  "Error test HDF5 I/O 2D integer MPI Native-Gatherv (Native): test and result array mismatch!")
  filename = trim(filename_base)//trim(extension); offset=[0,rank_loc*n_elements(2)];
  reqdim = [int(n_elements(1),kind=HSIZE_T),int(n_elements(2),kind=HSIZE_T)]; 
  test_array_allocatable = 0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array2D_saving_int(file_id,result_array,n_elements(1),n_tasks_loc*n_elements(2),&
  datasetname,start=offset,use_hdf5_parallel_in=use_hdf5_mpio,mpio_collective_in=mpio_collective)
  call HDF5_allocatable_array2D_reading_int(file_id,test_array_allocatable,&
  datasetname,start=offset,reqdims_in=reqdim); call HDF5_close(file_id); call remove_file(filename);
  call assert_equals_allocatable_arrays(n_elements(1),n_elements(2),result_array,test_array_allocatable,&
  "Error test HDF5 I/O 2D integer allocatable reqdims MPI collective: mismatch!")
  offset=[0,rank_loc*n_elements(2)]; reqdim = [n1_HSIZE_T,int(n_elements(2),kind=HSIZE_T)]; 
  test_array_allocatable = 0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array2D_saving_int(file_id,result_array,n_elements(1),n_tasks_loc*n_elements(2),&
  datasetname,start=offset,use_hdf5_parallel_in=use_hdf5_mpio,mpio_collective_in=mpio_collective)
  call HDF5_allocatable_array2D_reading_int(file_id,test_array_allocatable,&
  datasetname,start=offset,reqdims_in=reqdim); call HDF5_close(file_id); call remove_file(filename);
  call assert_equals_allocatable_arrays(n_elements(1),n_elements(2),result_array,test_array_allocatable,&
  "Error test HDF5 I/O 2D integer allocatable reqdims -1 MPI collective: mismatch!")
  if(allocated(test_array_allocatable)) deallocate(test_array_allocatable)
end subroutine test_HDF5_array2D_saving_int

!> the the posix and collective writing / reading HDF5 file
!> of double 3D array
subroutine test_HDF5_array3D_saving_int()
  use mod_assert_equals_tools, only: assert_equals_extended
  implicit none
  character(len=11),parameter                                  :: datasetname='array3D_int'
  integer,dimension(n_elements(1),n_elements(2),n_elements(3)) :: test_array,result_array
  integer(HID_T)                                               :: file_id
  integer(HSIZE_T),dimension(3)                                :: offset
  character(len=100)                                           :: filename 
  !> initialise posix test
  test_array = 0; result_array = int(1d3*array_sol(:,:,:,1,1));
  write(filename,'(A,A,I'//trim(rank_format)//',A)') &
  trim(filename_base),'_rank',rank_loc,trim(extension)
  ifail_loc=0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc);
  call HDF5_array3D_saving_int(file_id,result_array,n_elements(1),n_elements(2),&
  n_elements(3),datasetname); call HDF5_array3D_reading_int(file_id,test_array,datasetname)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals_extended(n_elements(1),n_elements(2),n_elements(3),test_array,&
  result_array,"Error test HDF5 I/O 3D integer posix: test and result array mismatch!")
  filename = trim(filename_base)//trim(extension); offset=[0,0,rank_loc*n_elements(3)];
  test_array = 0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array3D_saving_int(file_id,result_array,n_elements(1),n_elements(2),&
  n_tasks_loc*n_elements(3),datasetname,start=offset,use_hdf5_parallel_in=use_hdf5_mpio,&
  mpio_collective_in=mpio_collective)
  call HDF5_array3D_reading_int(file_id,test_array,datasetname,start=offset)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals_extended(n_elements(1),n_elements(2),n_elements(3),test_array,&
  result_array,"Error test HDF5 I/O 3D integer MPI collective: test and result array mismatch!")
end subroutine test_HDF5_array3D_saving_int

!> the the posix and collective writing / reading HDF5 file
!> of floats 1D array
subroutine test_HDF5_array1D_saving_r4()
  use mod_assert_equals_tools, only: assert_equals_allocatable_arrays
  implicit none
  character(len=11),parameter      :: datasetname='array1D_r4'
  integer                          :: ii
  integer,dimension(n_tasks_loc)   :: elements_all,displs 
  real*4,dimension(n_elements(1))  :: test_array,result_array
  real*4,dimension(:),allocatable  :: test_array_allocatable
  integer(HID_T)                   :: file_id
  integer(HSIZE_T),dimension(1)    :: reqdim,offset
  character(len=100)               :: filename 
  !> initialise posix test
  test_array = 0.0; result_array = real(array_sol(:,1,1,1,1),kind=4);
  write(filename,'(A,A,I'//trim(rank_format)//',A)') &
  trim(filename_base),'_rank',rank_loc,trim(extension)
  ifail_loc=0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc);
  call HDF5_array1D_saving_r4(file_id,result_array,n_elements(1),datasetname)
  call HDF5_array1D_reading_r4(file_id,test_array,datasetname)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals(test_array,result_array,n_elements(1),&
  "Error test HDF5 I/O 1D float posix: test and result array mismatch!")
  filename = trim(filename_base)//trim(extension); elements_all = n_elements(1);
  displs = 0; do ii=2,n_tasks_loc; displs(ii)=sum(elements_all(1:ii-1)); enddo; 
  offset = [rank_loc*n_elements(1)]; test_array = 0; 
  if(rank_loc.eq.master_rank) call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc) 
  call HDF5_array1D_saving_r4_native_or_gatherv(file_id,result_array,sum(elements_all),&
  datasetname,.true.,dim1_all_tasks=elements_all,displs=displs,mpi_rank=rank_loc,&
  n_mpi=n_tasks_loc,mpi_comm_loc=mpi_comm_loc,use_hdf5_parallel_in=use_hdf5_mpio,&
  mpio_collective_in=mpio_collective)  
  if(rank_loc.eq.master_rank) call HDF5_close(file_id); call MPI_Barrier(mpi_comm_loc,ifail_loc);
  call HDF5_open(filename,file_id,ifail_loc,create_access_plist_in=access_hdf5_parallel,&
  mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array1D_reading_r4(file_id,test_array,datasetname,start=offset)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals(test_array,result_array,n_elements(1),&
  "Error test HDF5 I/O 1D float MPI Native-Gatherv (Gatherv): test and result array mismatch!")
  filename = trim(filename_base)//trim(extension); offset=[rank_loc*n_elements(1)];
  test_array = 0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array1D_saving_r4(file_id,result_array,n_tasks_loc*n_elements(1),&
  datasetname,start=offset,use_hdf5_parallel_in=use_hdf5_mpio,mpio_collective_in=mpio_collective)
  call HDF5_array1D_reading_r4(file_id,test_array,datasetname,start=offset)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals(test_array,result_array,n_elements(1),&
  "Error test HDF5 I/O 1D float MPI collective: test and result array mismatch!")
  filename = trim(filename_base)//trim(extension); offset=[rank_loc*n_elements(1)];
  test_array = 0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array1D_saving_r4_native_or_gatherv(file_id,result_array,n_tasks_loc*n_elements(1),&
  datasetname,.false.,start=offset,use_hdf5_parallel_in=use_hdf5_mpio,mpio_collective_in=mpio_collective)
  call HDF5_array1D_reading_r4(file_id,test_array,datasetname,start=offset);
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals(test_array,result_array,n_elements(1),&
  "Error test HDF5 I/O 1D float MPI Native-Gatherv (Native): test and result array mismatch!")
  write(filename,'(A,A,I'//trim(rank_format)//',A)') &
  trim(filename_base),'_rank',rank_loc,trim(extension)
  call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc);
  call HDF5_array1D_saving_r4(file_id,result_array,n_elements(1),datasetname,&
  use_hdf5_parallel_in=use_hdf5_mpio,mpio_collective_in=mpio_collective)
  call HDF5_allocatable_array1D_reading_r4(file_id,test_array_allocatable,&
  datasetname); call HDF5_close(file_id); call remove_file(filename);
  call assert_equals_allocatable_arrays(n_elements(1),result_array,test_array_allocatable,&
  tol_r4,"Error test HDF5 I/O 1D float allocatable MPI collective:")
  filename = trim(filename_base)//trim(extension); offset=[rank_loc*n_elements(1)];
  reqdim = [int(n_elements(1),kind=HSIZE_T)]; test_array_allocatable = 0.;
  call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array1D_saving_r4(file_id,result_array,n_tasks_loc*n_elements(1),&
  datasetname,start=offset,use_hdf5_parallel_in=use_hdf5_mpio,mpio_collective_in=mpio_collective)
  call HDF5_allocatable_array1D_reading_r4(file_id,test_array_allocatable,&
  datasetname,start=offset,reqdims_in=reqdim); call HDF5_close(file_id); call remove_file(filename);
  call assert_equals_allocatable_arrays(n_elements(1),result_array,test_array_allocatable,&
  tol_r4,"Error test HDF5 I/O 1D float allocatable reqdims MPI collective:")
  if(allocated(test_array_allocatable)) deallocate(test_array_allocatable)
end subroutine test_HDF5_array1D_saving_r4

!> the the posix and collective writing / reading HDF5 file of a single double
subroutine test_HDF5_real_saving()
  implicit none
  character(len=12),parameter :: datasetname='double_value'
  real*8                      :: test_value,result_value
  integer(HID_T)              :: file_id
  character(len=100)          :: filename 
  !> initialise posix test
  test_value = 0d0; result_value = array_sol(1,1,1,1,1);
  write(filename,'(A,A,I'//trim(rank_format)//',A)') &
  trim(filename_base),'_rank',rank_loc,trim(extension)
  ifail_loc=0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc);
  call HDF5_real_saving(file_id,result_value,datasetname)
  call HDF5_real_reading(file_id,test_value,datasetname)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals(test_value,result_value,&
  "Error test HDF5 I/O 0D double posix: test and result array mismatch!")
  filename = trim(filename_base)//trim(extension);
  test_value = 0d0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_real_saving(file_id,result_value,datasetname,mpi_rank=rank_loc,&
  n_mpi_tasks=n_tasks_loc,use_hdf5_parallel_in=use_hdf5_mpio,mpio_collective_in=mpio_collective)
  call HDF5_real_reading(file_id,test_value,datasetname,&
  mpi_rank=rank_loc,n_mpi_tasks=n_tasks_loc)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals(test_value,result_value,&
  "Error test HDF5 I/O 0D double MPI collective: test and result array mismatch!")
end subroutine test_HDF5_real_saving

!> the the posix and collective writing / reading HDF5 file
!> of double 1D array
subroutine test_HDF5_array1D_saving_r8()
  use mod_assert_equals_tools, only: assert_equals_allocatable_arrays
  implicit none
  character(len=11),parameter      :: datasetname='array1D_r8'
  integer,dimension(n_tasks_loc)   :: elements_all,displs
  real*8,dimension(n_elements(1))  :: test_array,result_array
  real*8,dimension(:),allocatable  :: test_array_allocatable
  integer(HID_T)                   :: file_id
  integer(HSIZE_T),dimension(1)    :: reqdim,offset
  character(len=100)               :: filename 
  integer                          :: ii
  !> initialise posix test
  test_array = 0d0; result_array = array_sol(:,1,1,1,1);
  write(filename,'(A,A,I'//trim(rank_format)//',A)') &
  trim(filename_base),'_rank',rank_loc,trim(extension)
  ifail_loc=0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc);
  call HDF5_array1D_saving(file_id,result_array,n_elements(1),datasetname)
  call HDF5_array1D_reading(file_id,test_array,datasetname)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals(test_array,result_array,n_elements(1),&
  "Error test HDF5 I/O 1D double posix: test and result array mismatch!")
  filename = trim(filename_base)//trim(extension); elements_all = n_elements(1);
  displs = 0; do ii=2,n_tasks_loc; displs(ii)=sum(elements_all(1:ii-1)); enddo; 
  offset = [rank_loc*n_elements(1)]; test_array = 0; 
  if(rank_loc.eq.master_rank) call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc) 
  call HDF5_array1D_saving_native_or_gatherv(file_id,result_array,sum(elements_all),&
  datasetname,.true.,dim1_all_tasks=elements_all,displs=displs,mpi_rank=rank_loc,&
  n_mpi=n_tasks_loc,mpi_comm_loc=mpi_comm_loc,use_hdf5_parallel_in=use_hdf5_mpio,&
  mpio_collective_in=mpio_collective)  
  if(rank_loc.eq.master_rank) call HDF5_close(file_id); call MPI_Barrier(mpi_comm_loc,ifail_loc);
  call HDF5_open(filename,file_id,ifail_loc,create_access_plist_in=access_hdf5_parallel,&
  mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array1D_reading(file_id,test_array,datasetname,start=offset)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals(test_array,result_array,n_elements(1),&
  "Error test HDF5 I/O 1D double MPI Native-Gatherv (Gatherv): test and result array mismatch!")
  filename = trim(filename_base)//trim(extension); offset=[rank_loc*n_elements(1)];
  test_array = 0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array1D_saving_native_or_gatherv(file_id,result_array,n_tasks_loc*n_elements(1),&
  datasetname,.false.,start=offset,use_hdf5_parallel_in=use_hdf5_mpio,mpio_collective_in=mpio_collective)
  call HDF5_array1D_reading(file_id,test_array,datasetname,start=offset);
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals(test_array,result_array,n_elements(1),&
  "Error test HDF5 I/O 1D double MPI Native-Gatherv (Native): test and result array mismatch!")
  filename = trim(filename_base)//trim(extension); offset=[rank_loc*n_elements(1)];
  test_array = 0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array1D_saving(file_id,result_array,n_tasks_loc*n_elements(1),&
  datasetname,start=offset,use_hdf5_parallel_in=use_hdf5_mpio,mpio_collective_in=mpio_collective)
  call HDF5_array1D_reading(file_id,test_array,datasetname,start=offset)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals(test_array,result_array,n_elements(1),&
  "Error test HDF5 I/O 1D double MPI collective: test and result array mismatch!")
  write(filename,'(A,A,I'//trim(rank_format)//',A)') &
  trim(filename_base),'_rank',rank_loc,trim(extension)
  call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc);
  call HDF5_array1D_saving(file_id,result_array,n_elements(1),datasetname,&
  use_hdf5_parallel_in=use_hdf5_mpio,mpio_collective_in=mpio_collective)
  call HDF5_allocatable_array1D_reading(file_id,test_array_allocatable,&
  datasetname); call HDF5_close(file_id); call remove_file(filename);
  call assert_equals_allocatable_arrays(n_elements(1),result_array,test_array_allocatable,&
  tol_r8,"Error test HDF5 I/O 1D double allocatable MPI collective:")
  filename = trim(filename_base)//trim(extension); offset=[rank_loc*n_elements(1)];
  reqdim = [int(n_elements(1),kind=HSIZE_T)]; test_array_allocatable = 0d0;
  call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array1D_saving(file_id,result_array,n_tasks_loc*n_elements(1),&
  datasetname,start=offset,use_hdf5_parallel_in=use_hdf5_mpio,mpio_collective_in=mpio_collective)
  call HDF5_allocatable_array1D_reading(file_id,test_array_allocatable,&
  datasetname,start=offset,reqdims_in=reqdim); call HDF5_close(file_id); call remove_file(filename);
  call assert_equals_allocatable_arrays(n_elements(1),result_array,test_array_allocatable,&
  tol_r8,"Error test HDF5 I/O 1D double allocatable reqdims MPI collective:")
  if(allocated(test_array_allocatable)) deallocate(test_array_allocatable)
end subroutine test_HDF5_array1D_saving_r8

!> the the posix and collective writing / reading HDF5 file
!> of double 2D array
subroutine test_HDF5_array2D_saving_r8()
  use mod_assert_equals_tools, only: assert_equals_allocatable_arrays
  implicit none
  character(len=11),parameter                   :: datasetname='array2D_r8'
  integer,dimension(n_tasks_loc)                :: elements_all,displs
  real*8,dimension(n_elements(1),n_elements(2)) :: test_array,result_array
  real*8,dimension(:,:),allocatable             :: test_array_allocatable
  integer(HID_T)                                :: file_id
  integer(HSIZE_T),dimension(2)                 :: offset,reqdim
  character(len=100)                            :: filename 
  integer                                       :: ii
  !> initialise posix test
  test_array = 0d0; result_array = array_sol(:,:,1,1,1);
  write(filename,'(A,A,I'//trim(rank_format)//',A)') &
  trim(filename_base),'_rank',rank_loc,trim(extension)
  ifail_loc=0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc);
  call HDF5_array2D_saving(file_id,result_array,n_elements(1),n_elements(2),&
  datasetname); call HDF5_array2D_reading(file_id,test_array,datasetname)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals(test_array,result_array,n_elements(1),n_elements(2),&
  "Error test HDF5 I/O 2D double posix: test and result array mismatch!")
  filename = trim(filename_base)//trim(extension); elements_all = n_elements(2);
  displs = 0; do ii=2,n_tasks_loc; displs(ii)=sum(elements_all(1:ii-1)); enddo;
  offset = [0,rank_loc*n_elements(2)]; test_array = 0; 
  if(rank_loc.eq.master_rank) call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc) 
  call HDF5_array2D_saving_native_or_gatherv(file_id,result_array,n_elements(1),&
  sum(elements_all),datasetname,.true.,dim2_all_tasks=elements_all,displs=displs,&
  mpi_rank=rank_loc,n_mpi=n_tasks_loc,mpi_comm_loc=mpi_comm_loc,&
  use_hdf5_parallel_in=use_hdf5_mpio,mpio_collective_in=mpio_collective)
  if(rank_loc.eq.master_rank) call HDF5_close(file_id); call MPI_Barrier(mpi_comm_loc,ifail_loc);
  call HDF5_open(filename,file_id,ifail_loc,create_access_plist_in=access_hdf5_parallel,&
  mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array2D_reading(file_id,test_array,datasetname,start=offset)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals(test_array,result_array,n_elements(1),n_elements(2),&
  "Error test HDF5 I/O 2D double MPI Native-Gatherv (Gatherv): test and result array mismatch!")
  filename = trim(filename_base)//trim(extension); offset=[0,rank_loc*n_elements(2)];
  test_array = 0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array2D_saving(file_id,result_array,n_elements(1),n_tasks_loc*n_elements(2),&
  datasetname,start=offset,use_hdf5_parallel_in=use_hdf5_mpio,mpio_collective_in=mpio_collective)
  call HDF5_array2D_reading(file_id,test_array,datasetname,start=offset)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals(test_array,result_array,n_elements(1),n_elements(2),&
  "Error test HDF5 I/O 2D double MPI collective: test and result array mismatch!")
  write(filename,'(A,A,I'//trim(rank_format)//',A)') &
  trim(filename_base),'_rank',rank_loc,trim(extension)
  call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc);
  call HDF5_array2D_saving(file_id,result_array,n_elements(1),n_elements(2),datasetname,&
  use_hdf5_parallel_in=use_hdf5_mpio,mpio_collective_in=mpio_collective)
  call HDF5_allocatable_array2D_reading(file_id,test_array_allocatable,&
  datasetname); call HDF5_close(file_id); call remove_file(filename);
  call assert_equals_allocatable_arrays(n_elements(1),n_elements(2),result_array,&
  test_array_allocatable,tol_r8,"Error test HDF5 I/O 2D double allocatable MPI collective: mismatch!")
  filename = trim(filename_base)//trim(extension); offset=[0,rank_loc*n_elements(2)];
  test_array = 0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array2D_saving_native_or_gatherv(file_id,result_array,n_elements(1),&
  n_tasks_loc*n_elements(2),datasetname,.false.,start=offset,use_hdf5_parallel_in=use_hdf5_mpio,&
  mpio_collective_in=mpio_collective)
  call HDF5_array2D_reading(file_id,test_array,datasetname,start=offset);
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals(test_array,result_array,n_elements(1),n_elements(2),&
  "Error test HDF5 I/O 2D double MPI Native-Gatherv (Native): test and result array mismatch!")
  filename = trim(filename_base)//trim(extension); offset=[0,rank_loc*n_elements(2)];
  reqdim = [int(n_elements(1),kind=HSIZE_T),int(n_elements(2),kind=HSIZE_T)]; 
  test_array_allocatable = 0d0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array2D_saving(file_id,result_array,n_elements(1),n_tasks_loc*n_elements(2),&
  datasetname,start=offset,use_hdf5_parallel_in=use_hdf5_mpio,mpio_collective_in=mpio_collective)
  call HDF5_allocatable_array2D_reading(file_id,test_array_allocatable,&
  datasetname,start=offset,reqdims_in=reqdim); call HDF5_close(file_id); call remove_file(filename);
  call assert_equals_allocatable_arrays(n_elements(1),n_elements(2),result_array,&
  test_array_allocatable,tol_r8,"Error test HDF5 I/O 2D double allocatable reqdims MPI collective: mismatch!")
  offset=[0,rank_loc*n_elements(2)]; reqdim = [n1_HSIZE_T,int(n_elements(2),kind=HSIZE_T)]; 
  test_array_allocatable = 0d0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array2D_saving(file_id,result_array,n_elements(1),n_tasks_loc*n_elements(2),&
  datasetname,start=offset,use_hdf5_parallel_in=use_hdf5_mpio,mpio_collective_in=mpio_collective)
  call HDF5_allocatable_array2D_reading(file_id,test_array_allocatable,&
  datasetname,start=offset,reqdims_in=reqdim); call HDF5_close(file_id); call remove_file(filename);
  call assert_equals_allocatable_arrays(n_elements(1),n_elements(2),result_array,&
  test_array_allocatable,tol_r8,"Error test HDF5 I/O 2D double allocatable reqdims -1 MPI collective: mismatch!")
  if(allocated(test_array_allocatable)) deallocate(test_array_allocatable)
end subroutine test_HDF5_array2D_saving_r8

!> the the posix and collective writing / reading HDF5 file
!> of double 3D array
subroutine test_HDF5_array3D_saving_r8()
  use mod_assert_equals_tools, only: assert_equals_extended
  use mod_assert_equals_tools, only: assert_equals_allocatable_arrays
  implicit none
  character(len=11),parameter                                 :: datasetname='array3D_r8'
  integer,dimension(n_tasks_loc)                              :: elements_all,displs
  real*8,dimension(n_elements(1),n_elements(2),n_elements(3)) :: test_array,result_array
  real*8,dimension(:,:,:),allocatable                         :: test_array_allocatable
  integer(HID_T)                                              :: file_id
  integer(HSIZE_T),dimension(3)                               :: offset,reqdim
  character(len=100)                                          :: filename 
  integer                                                     :: ii
  !> initialise posix test
  test_array = 0d0; result_array = array_sol(:,:,:,1,1);
  write(filename,'(A,A,I'//trim(rank_format)//',A)') &
  trim(filename_base),'_rank',rank_loc,trim(extension)
  ifail_loc=0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc);
  call HDF5_array3D_saving(file_id,result_array,n_elements(1),n_elements(2),&
  n_elements(3),datasetname); call HDF5_array3D_reading(file_id,test_array,datasetname)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals_extended(n_elements(1),n_elements(2),n_elements(3),test_array,&
  result_array,tol_r8,"Error test HDF5 I/O 3D double posix: test and result array mismatch!")
  filename = trim(filename_base)//trim(extension); elements_all = n_elements(3);
  displs = 0; do ii=2,n_tasks_loc; displs(ii)=sum(elements_all(1:ii-1)); enddo;
  offset = [0,0,rank_loc*n_elements(3)]; test_array = 0; 
  if(rank_loc.eq.master_rank) call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc) 
  call HDF5_array3D_saving_native_or_gatherv(file_id,result_array,n_elements(1),n_elements(2),&
  sum(elements_all),datasetname,.true.,dim3_all_tasks=elements_all,&
  displs=displs,mpi_rank=rank_loc,n_mpi=n_tasks_loc,mpi_comm_loc=mpi_comm_loc,&
  use_hdf5_parallel_in=use_hdf5_mpio,mpio_collective_in=mpio_collective)
  if(rank_loc.eq.master_rank) call HDF5_close(file_id); call MPI_Barrier(mpi_comm_loc,ifail_loc);
  call HDF5_open(filename,file_id,ifail_loc,create_access_plist_in=access_hdf5_parallel,&
  mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array3D_reading(file_id,test_array,datasetname,start=offset)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals_extended(n_elements(1),n_elements(2),n_elements(3),test_array,&
  result_array,tol_r8,"Error test HDF5 I/O 3D double Native-Gatherv (Gatherv): test and result array mismatch!")
  filename = trim(filename_base)//trim(extension); offset=[0,0,rank_loc*n_elements(3)];
  test_array = 0d0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array3D_saving(file_id,result_array,n_elements(1),n_elements(2),&
  n_tasks_loc*n_elements(3),datasetname,start=offset,use_hdf5_parallel_in=use_hdf5_mpio,&
  mpio_collective_in=mpio_collective)
  call HDF5_array3D_reading(file_id,test_array,datasetname,start=offset)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals_extended(n_elements(1),n_elements(2),n_elements(3),test_array,&
  result_array,tol_r8,"Error test HDF5 I/O 3D double MPI collective: test and result array mismatch!")
  filename = trim(filename_base)//trim(extension); offset=[0,0,rank_loc*n_elements(3)];
  test_array = 0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array3D_saving_native_or_gatherv(file_id,result_array,n_elements(1),n_elements(2),&
  n_tasks_loc*n_elements(3),datasetname,.false.,start=offset,use_hdf5_parallel_in=use_hdf5_mpio,&
  mpio_collective_in=mpio_collective)
  call HDF5_array3D_reading(file_id,test_array,datasetname,start=offset);
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals_extended(n_elements(1),n_elements(2),n_elements(3),test_array,&
  result_array,tol_r8,"Error test HDF5 I/O 3D double Native-Gatherv (Native): test and result array mismatch!")
  write(filename,'(A,A,I'//trim(rank_format)//',A)') &
  trim(filename_base),'_rank',rank_loc,trim(extension)
  call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc);
  call HDF5_array3D_saving(file_id,result_array,n_elements(1),n_elements(2),&
  n_elements(3),datasetname,use_hdf5_parallel_in=use_hdf5_mpio,mpio_collective_in=mpio_collective); 
  call HDF5_allocatable_array3D_reading(file_id,test_array_allocatable,&
  datasetname); call HDF5_close(file_id); call remove_file(filename);
  call assert_equals_allocatable_arrays(n_elements(1),n_elements(2),n_elements(3),result_array,&
  test_array_allocatable,tol_r8,"Error test HDF5 I/O 3D double allocatable MPI collective: mismatch!")
  filename = trim(filename_base)//trim(extension); offset=[0,0,rank_loc*n_elements(3)];
  reqdim = [int(n_elements(1),kind=HSIZE_T),int(n_elements(2),kind=HSIZE_T),&
  int(n_elements(3),kind=HSIZE_T)]; test_array_allocatable = 0d0; 
  call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array3D_saving(file_id,result_array,n_elements(1),n_elements(2),&
  n_tasks_loc*n_elements(3),datasetname,start=offset,use_hdf5_parallel_in=use_hdf5_mpio,&
  mpio_collective_in=mpio_collective)
  call HDF5_allocatable_array3D_reading(file_id,test_array_allocatable,&
  datasetname,start=offset,reqdims_in=reqdim); call HDF5_close(file_id); call remove_file(filename);
  call assert_equals_allocatable_arrays(n_elements(1),n_elements(2),n_elements(3),result_array,&
  test_array_allocatable,tol_r8,"Error test HDF5 I/O 3D double allocatable reqdims MPI collective: mismatch!")
  offset = [0,0,rank_loc*n_elements(3)]; reqdim = [n1_HSIZE_T,n1_HSIZE_T,int(n_elements(3),kind=HSIZE_T)]; 
  test_array_allocatable = 0d0; 
  call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array3D_saving(file_id,result_array,n_elements(1),n_elements(2),&
  n_tasks_loc*n_elements(3),datasetname,start=offset,use_hdf5_parallel_in=use_hdf5_mpio,&
  mpio_collective_in=mpio_collective)
  call HDF5_allocatable_array3D_reading(file_id,test_array_allocatable,&
  datasetname,start=offset,reqdims_in=reqdim); call HDF5_close(file_id); call remove_file(filename);
  call assert_equals_allocatable_arrays(n_elements(1),n_elements(2),n_elements(3),result_array,&
  test_array_allocatable,tol_r8,"Error test HDF5 I/O 3D double allocatable reqdims -1 MPI collective: mismatch!")
  if(allocated(test_array_allocatable)) deallocate(test_array_allocatable)
end subroutine test_HDF5_array3D_saving_r8

!> the the posix and collective writing / reading HDF5 file
!> of double 4D array
subroutine test_HDF5_array4D_saving_r8()
  use mod_assert_equals_tools, only: assert_equals_extended
  use mod_assert_equals_tools, only: assert_equals_allocatable_arrays
  implicit none
  character(len=11),parameter                     :: datasetname='array4D_r8'
  real*8,dimension(n_elements(1),n_elements(2),&
  n_elements(3),n_elements(4))                    :: test_array,result_array
  real*8,dimension(:,:,:,:),allocatable           :: test_array_allocatable
  integer(HID_T)                                  :: file_id
  integer(HSIZE_T),dimension(4)                   :: offset,reqdim
  character(len=100)                              :: filename 
  !> initialise posix test
  test_array = 0d0; result_array = array_sol(:,:,:,:,1);
  write(filename,'(A,A,I'//trim(rank_format)//',A)') &
  trim(filename_base),'_rank',rank_loc,trim(extension)
  ifail_loc=0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc);
  call HDF5_array4D_saving(file_id,result_array,n_elements(1),n_elements(2),&
  n_elements(3),n_elements(4),datasetname); 
  call HDF5_array4D_reading(file_id,test_array,datasetname)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals_extended(n_elements(1),n_elements(2),n_elements(3),n_elements(4),test_array,&
  result_array,tol_r8,"Error test HDF5 I/O 4D double posix: test and result array mismatch!")
  filename = trim(filename_base)//trim(extension); offset=[0,0,0,rank_loc*n_elements(4)];
  test_array = 0d0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array4D_saving(file_id,result_array,n_elements(1),n_elements(2),&
  n_elements(3),n_tasks_loc*n_elements(4),datasetname,start=offset,&
  use_hdf5_parallel_in=use_hdf5_mpio,mpio_collective_in=mpio_collective)
  call HDF5_array4D_reading(file_id,test_array,datasetname,start=offset)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals_extended(n_elements(1),n_elements(2),n_elements(3),n_elements(4),test_array,&
  result_array,tol_r8,"Error test HDF5 I/O 4D double MPI collective: test and result array mismatch!")
  write(filename,'(A,A,I'//trim(rank_format)//',A)') &
  trim(filename_base),'_rank',rank_loc,trim(extension)
  call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc);
  call HDF5_array4D_saving(file_id,result_array,n_elements(1),n_elements(2),n_elements(3),&
  n_elements(4),datasetname,use_hdf5_parallel_in=use_hdf5_mpio,mpio_collective_in=mpio_collective); 
  call HDF5_allocatable_array4D_reading(file_id,test_array_allocatable,&
  datasetname); call HDF5_close(file_id); call remove_file(filename);
  call assert_equals_allocatable_arrays(n_elements(1),n_elements(2),n_elements(3),n_elements(4),&
  result_array,test_array_allocatable,tol_r8,"Error test HDF5 I/O 4D double allocatable MPI collective: mismatch!")
  filename = trim(filename_base)//trim(extension); offset=[0,0,0,rank_loc*n_elements(4)];
  reqdim = [int(n_elements(1),kind=HSIZE_T),int(n_elements(2),kind=HSIZE_T),&
  int(n_elements(3),kind=HSIZE_T),int(n_elements(4),kind=HSIZE_T)]; test_array_allocatable = 0d0; 
  call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array4D_saving(file_id,result_array,n_elements(1),n_elements(2),n_elements(3),&
  n_tasks_loc*n_elements(4),datasetname,start=offset,use_hdf5_parallel_in=use_hdf5_mpio,&
  mpio_collective_in=mpio_collective)
  call HDF5_allocatable_array4D_reading(file_id,test_array_allocatable,&
  datasetname,start=offset,reqdims_in=reqdim); call HDF5_close(file_id); call remove_file(filename);
  call assert_equals_allocatable_arrays(n_elements(1),n_elements(2),n_elements(3),n_elements(4),result_array,&
  test_array_allocatable,tol_r8,"Error test HDF5 I/O 4D double allocatable reqdims MPI collective: mismatch!")
  offset=[0,0,0,rank_loc*n_elements(4)]; reqdim = [n1_HSIZE_T,n1_HSIZE_T,n1_HSIZE_T,&
  int(n_elements(4),kind=HSIZE_T)]; test_array_allocatable = 0d0; 
  call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array4D_saving(file_id,result_array,n_elements(1),n_elements(2),n_elements(3),&
  n_tasks_loc*n_elements(4),datasetname,start=offset,use_hdf5_parallel_in=use_hdf5_mpio,&
  mpio_collective_in=mpio_collective)
  call HDF5_allocatable_array4D_reading(file_id,test_array_allocatable,&
  datasetname,start=offset,reqdims_in=reqdim); call HDF5_close(file_id); call remove_file(filename);
  call assert_equals_allocatable_arrays(n_elements(1),n_elements(2),n_elements(3),n_elements(4),result_array,&
  test_array_allocatable,tol_r8,"Error test HDF5 I/O 4D double allocatable reqdims MPI collective: mismatch!")
  if(allocated(test_array_allocatable)) deallocate(test_array_allocatable)
end subroutine test_HDF5_array4D_saving_r8

!> the the posix and collective writing / reading HDF5 file
!> of double 5D array
subroutine test_HDF5_array5D_saving_r8()
  use mod_assert_equals_tools, only: assert_equals_extended
  use mod_assert_equals_tools, only: assert_equals_allocatable_arrays
  implicit none
  character(len=11),parameter                     :: datasetname='array5D_r8'
  real*8,dimension(n_elements(1),n_elements(2),&
  n_elements(3),n_elements(4),n_elements(5))      :: test_array,result_array
  real*8,dimension(:,:,:,:,:),allocatable         :: test_array_allocatable
  integer(HID_T)                                  :: file_id
  integer(HSIZE_T),dimension(5)                   :: offset,reqdim
  character(len=100)                              :: filename 
  !> initialise posix test
  test_array = 0d0; result_array = array_sol;
  write(filename,'(A,A,I'//trim(rank_format)//',A)') &
  trim(filename_base),'_rank',rank_loc,trim(extension)
  ifail_loc=0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc);
  call HDF5_array5D_saving(file_id,result_array,n_elements(1),n_elements(2),&
  n_elements(3),n_elements(4),n_elements(5),datasetname); 
  call HDF5_array5D_reading(file_id,test_array,datasetname)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals_extended(n_elements(1),n_elements(2),n_elements(3),&
  n_elements(4),n_elements(5),test_array,result_array,tol_r8,&
  "Error test HDF5 I/O 5D double posix: test and result array mismatch!")
  filename = trim(filename_base)//trim(extension); offset=[0,0,0,0,rank_loc*n_elements(5)];
  test_array = 0d0; call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array5D_saving(file_id,result_array,n_elements(1),n_elements(2),&
  n_elements(3),n_elements(4),n_tasks_loc*n_elements(5),datasetname,start=offset,&
  use_hdf5_parallel_in=use_hdf5_mpio,mpio_collective_in=mpio_collective)
  call HDF5_array5D_reading(file_id,test_array,datasetname,start=offset)
  call HDF5_close(file_id); call remove_file(filename);
  call assert_equals_extended(n_elements(1),n_elements(2),n_elements(3),&
  n_elements(4),n_elements(5),test_array,result_array,tol_r8,&
  "Error test HDF5 I/O 5D double MPI collective: test and result array mismatch!")
  write(filename,'(A,A,I'//trim(rank_format)//',A)') &
  trim(filename_base),'_rank',rank_loc,trim(extension)
  call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc);
  call HDF5_array5D_saving(file_id,result_array,n_elements(1),n_elements(2),n_elements(3),n_elements(4),&
  n_elements(5),datasetname,use_hdf5_parallel_in=use_hdf5_mpio,mpio_collective_in=mpio_collective); 
  call HDF5_allocatable_array5D_reading(file_id,test_array_allocatable,&
  datasetname); call HDF5_close(file_id); call remove_file(filename);
  call assert_equals_allocatable_arrays(n_elements(1),n_elements(2),n_elements(3),n_elements(4),n_elements(5),&
  result_array,test_array_allocatable,tol_r8,"Error test HDF5 I/O 5D double allocatable MPI collective: mismatch!")
  filename = trim(filename_base)//trim(extension); offset=[0,0,0,0,rank_loc*n_elements(5)];
  reqdim = [int(n_elements(1),kind=HSIZE_T),int(n_elements(2),kind=HSIZE_T),int(n_elements(3),kind=HSIZE_T),&
  int(n_elements(4),kind=HSIZE_T),int(n_elements(5),kind=HSIZE_T)]; test_array_allocatable = 0d0; 
  call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array5D_saving(file_id,result_array,n_elements(1),n_elements(2),n_elements(3),n_elements(4),&
  n_tasks_loc*n_elements(5),datasetname,start=offset,use_hdf5_parallel_in=use_hdf5_mpio,&
  mpio_collective_in=mpio_collective)
  call HDF5_allocatable_array5D_reading(file_id,test_array_allocatable,&
  datasetname,start=offset,reqdims_in=reqdim); call HDF5_close(file_id); call remove_file(filename);
  call assert_equals_allocatable_arrays(n_elements(1),n_elements(2),n_elements(3),n_elements(4),n_elements(5),&
  result_array,test_array_allocatable,tol_r8,&
  "Error test HDF5 I/O 5D double allocatable reqdims MPI collective: mismatch!")
  offset=[0,0,0,0,rank_loc*n_elements(5)]; reqdim = [n1_HSIZE_T,n1_HSIZE_T,n1_HSIZE_T,&
  n1_HSIZE_T,int(n_elements(5),kind=HSIZE_T)]; test_array_allocatable = 0d0; 
  call HDF5_open_or_create(trim(filename),file_id,ierr=ifail_loc,&
  create_access_plist_in=access_hdf5_parallel,mpi_comm_in=mpi_comm_loc,mpi_info=mpi_info_loc)
  call HDF5_array5D_saving(file_id,result_array,n_elements(1),n_elements(2),n_elements(3),n_elements(4),&
  n_tasks_loc*n_elements(5),datasetname,start=offset,use_hdf5_parallel_in=use_hdf5_mpio,&
  mpio_collective_in=mpio_collective)
  call HDF5_allocatable_array5D_reading(file_id,test_array_allocatable,&
  datasetname,start=offset,reqdims_in=reqdim); call HDF5_close(file_id); call remove_file(filename);
  call assert_equals_allocatable_arrays(n_elements(1),n_elements(2),n_elements(3),n_elements(4),n_elements(5),&
  result_array,test_array_allocatable,tol_r8,&
  "Error test HDF5 I/O 5D double allocatable reqdims -1 MPI collective: mismatch!")
  if(allocated(test_array_allocatable)) deallocate(test_array_allocatable)
end subroutine test_HDF5_array5D_saving_r8

!> Tools ------------------------------------------------------
subroutine remove_file(filename,file_exists_in)
  implicit none
  character(len=*),intent(in) :: filename
  logical,optional,intent(in) :: file_exists_in
  logical                     :: file_exists
  if(present(file_exists_in)) then
    file_exists = file_exists_in
  else
    inquire(file=filename,exist=file_exists)
  endif
  if(file_exists) call system("rm -rf "//trim(filename))
end subroutine remove_file

!> ------------------------------------------------------------
end module mod_hdf5_io_module_mpi_test
