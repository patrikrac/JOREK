subroutine Broadcast_nodes(my_id,node_list)
!----------------------------------------------------------
! subroutine to broadcast all the nodes in the node_list
!----------------------------------------------------------
use tr_module 
use data_structure
use mpi_mod
implicit none

type (type_node_list)    :: node_list

type (type_node)         :: anode
integer                  :: i, ierr, my_id, position, bufsize, IDBL_EXT, INT_EXT, ILOG_EXT
character, allocatable   :: buffer(:)

!  type type_node                                      ! type definition of a node (i.e. a vertex)
!    real*8     :: x(n_degrees,n_dim)                  ! x,y,z coordinates of points and additional nodal geometry
!    real*8     :: values(n_tor,n_degrees,n_var)
!    integer    :: index(n_degrees)                    ! the index in the main matrix
!    integer    :: boundary                            ! = 1, 2 or 3 for boundary nodes
!  endtype type_node                                   ! x(:,1) : position, x(:,2) : vector u, x(:,3) : vector v, x(4) : vector w

!  type type_node_list                                 ! type definition of a list of nodes
!    integer :: n_nodes                                ! the number of nodes in the list
!    type (type_node)     :: node(n_nodes_max)         ! an allocatable list of nodes
!  endtype type_node_list


call MPI_PACK_SIZE(1,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,IDBL_EXT,ierr)
call MPI_PACK_SIZE(1,MPI_INTEGER,MPI_COMM_WORLD,INT_EXT,ierr)
call MPI_PACK_SIZE(1,MPI_LOGICAL,MPI_COMM_WORLD,ILOG_EXT,ierr)

call MPI_BCAST(node_list%n_nodes,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
call MPI_BCAST(node_list%n_dof,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)

#ifdef STELLARATOR_MODEL
#ifndef USE_DOMM
bufsize = node_list%n_nodes * ((n_coord_tor*n_degrees*(n_dim+3*(n_dim+1)+1) + 2*n_tor*n_degrees*n_var + n_tor*n_degrees + 2 + 2*n_degrees)*IDBL_EXT + (n_degrees + 1+3+1+1)*INT_EXT + (2)*ILOG_EXT)
#else
bufsize = node_list%n_nodes * ((n_coord_tor*n_degrees*(n_dim+3*(n_dim+1)) + 2*n_tor*n_degrees*n_var + n_tor*n_degrees + 2 + 2*n_degrees)*IDBL_EXT + (n_degrees + 1+3+1+1)*INT_EXT + (2)*ILOG_EXT)
#endif
#elif fullmhd
bufsize = node_list%n_nodes * ((n_coord_tor*n_degrees*n_dim + 2*n_tor*n_degrees*n_var+2*n_degrees+2)*IDBL_EXT + (n_degrees +1+3+1+1)*INT_EXT + (2)*ILOG_EXT)
#elif altcs                          
bufsize = node_list%n_nodes * ((n_coord_tor*n_degrees*n_dim + 2*n_tor*n_degrees*n_var+2*n_degrees+2)*IDBL_EXT + (n_degrees +1+3+1+1)*INT_EXT + (2)*ILOG_EXT)
#else                                
bufsize = node_list%n_nodes * ((n_coord_tor*n_degrees*n_dim + 2*n_tor*n_degrees*n_var+2)*IDBL_EXT + (n_degrees + 1+3+1+1)*INT_EXT + (2)*ILOG_EXT)
#endif

call init_node(anode, n_var)
allocate(buffer(bufsize))
call tr_register_mem(bufsize,"bcastn_buffer")

if (my_id .eq. 0) then

  position = 0

  do i=1,node_list%n_nodes

    call make_deep_copy_node(node_list%node(i), anode)

    call MPI_PACK(anode%x              ,n_coord_tor*n_degrees*n_dim      ,MPI_DOUBLE_PRECISION,buffer,bufsize,position,MPI_COMM_WORLD,ierr)
    call MPI_PACK(anode%values         ,n_tor*n_degrees*n_var,MPI_DOUBLE_PRECISION,buffer,bufsize,position,MPI_COMM_WORLD,ierr)
    call MPI_PACK(anode%deltas         ,n_tor*n_degrees*n_var,MPI_DOUBLE_PRECISION,buffer,bufsize,position,MPI_COMM_WORLD,ierr)
#ifdef STELLARATOR_MODEL
    call MPI_PACK(anode%r_tor_eq       ,n_degrees,MPI_DOUBLE_PRECISION,buffer,bufsize,position,MPI_COMM_WORLD,ierr)
#if JOREK_MODEL == 180
    call MPI_PACK(anode%pressure       ,n_degrees,MPI_DOUBLE_PRECISION,buffer,bufsize,position,MPI_COMM_WORLD,ierr)
    call MPI_PACK(anode%j_field        ,n_coord_tor*n_degrees*(n_dim+1),MPI_DOUBLE_PRECISION,buffer,bufsize,position,MPI_COMM_WORLD,ierr)
    call MPI_PACK(anode%b_field        ,n_coord_tor*n_degrees*(n_dim+1),MPI_DOUBLE_PRECISION,buffer,bufsize,position,MPI_COMM_WORLD,ierr)
#endif
#ifndef USE_DOMM
#ifdef USE_EXT_FIELD
    call MPI_PACK(anode%b_vac_field    ,n_coord_tor*n_degrees*(n_dim+1),MPI_DOUBLE_PRECISION,buffer,bufsize,position,MPI_COMM_WORLD,ierr)
#else
    call MPI_PACK(anode%chi_correction ,n_coord_tor*n_degrees          ,MPI_DOUBLE_PRECISION,buffer,bufsize,position,MPI_COMM_WORLD,ierr)
#endif
#endif
    call MPI_PACK(anode%j_source       ,n_tor*n_degrees,MPI_DOUBLE_PRECISION,buffer,bufsize,position,MPI_COMM_WORLD,ierr)
#elif fullmhd
    call MPI_PACK(anode%Fprof_eq       ,n_degrees,MPI_DOUBLE_PRECISION,buffer,bufsize,position,MPI_COMM_WORLD,ierr)
    call MPI_PACK(anode%psi_eq         ,n_degrees,MPI_DOUBLE_PRECISION,buffer,bufsize,position,MPI_COMM_WORLD,ierr)
#endif
    call MPI_PACK(anode%index          ,n_degrees,MPI_INTEGER,buffer,bufsize,position,MPI_COMM_WORLD,ierr)
    call MPI_PACK(anode%boundary       ,1        ,MPI_INTEGER,buffer,bufsize,position,MPI_COMM_WORLD,ierr)
    call MPI_PACK(anode%boundary_index ,1        ,MPI_INTEGER,buffer,bufsize,position,MPI_COMM_WORLD,ierr)
    call MPI_PACK(anode%axis_node      ,1        ,MPI_LOGICAL,buffer,bufsize,position,MPI_COMM_WORLD,ierr)
    call MPI_PACK(anode%axis_dof       ,1        ,MPI_INTEGER,buffer,bufsize,position,MPI_COMM_WORLD,ierr)
    call MPI_PACK(anode%constrained    ,1        ,MPI_LOGICAL,buffer,bufsize,position,MPI_COMM_WORLD,ierr)
    call MPI_PACK(anode%parents(1:2)   ,2        ,MPI_INTEGER,buffer,bufsize,position,MPI_COMM_WORLD,ierr)
    call MPI_PACK(anode%parent_elem    ,1        ,MPI_INTEGER,buffer,bufsize,position,MPI_COMM_WORLD,ierr)
    call MPI_PACK(anode%ref_lambda     ,1        ,MPI_DOUBLE_PRECISION,buffer,bufsize,position,MPI_COMM_WORLD,ierr)
    call MPI_PACK(anode%ref_mu         ,1        ,MPI_DOUBLE_PRECISION,buffer,bufsize,position,MPI_COMM_WORLD,ierr)
  
   
  enddo

  if (position > bufsize) then
    write(*,*) 'FATAL broadcast_nodes: MPI pack buffer overflow: position=', position, ', bufsize=', bufsize
    write(*,*) 'Bufsize formula is too small - update broadcast_nodes.f90'
    stop
  end if

endif

call MPI_BCAST(buffer,bufsize,MPI_PACKED,0,MPI_COMM_WORLD,ierr)

if (my_id .ne. 0) then

    if (.not. allocated(node_list%node)) call init_node_list(node_list, node_list%n_nodes, node_list%n_dof, n_var)

  position = 0
  do i=1,node_list%n_nodes

    call MPI_UNPACK(buffer,bufsize,position,anode%x              ,n_coord_tor*n_degrees*n_dim      ,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,ierr)
    call MPI_UNPACK(buffer,bufsize,position,anode%values         ,n_tor*n_degrees*n_var,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,ierr)
    call MPI_UNPACK(buffer,bufsize,position,anode%deltas         ,n_tor*n_degrees*n_var,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,ierr)
#ifdef STELLARATOR_MODEL
    call MPI_UNPACK(buffer,bufsize,position,anode%r_tor_eq       ,n_degrees                        ,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,ierr)
#if JOREK_MODEL == 180
    call MPI_UNPACK(buffer,bufsize,position,anode%pressure       ,n_degrees                        ,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,ierr)
    call MPI_UNPACK(buffer,bufsize,position,anode%j_field        ,n_coord_tor*n_degrees*(n_dim+1)  ,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,ierr)
    call MPI_UNPACK(buffer,bufsize,position,anode%b_field        ,n_coord_tor*n_degrees*(n_dim+1)  ,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,ierr)
#endif
#ifndef USE_DOMM
#ifdef USE_EXT_FIELD
    call MPI_UNPACK(buffer,bufsize,position,anode%b_vac_field    ,n_coord_tor*n_degrees*(n_dim+1)  ,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,ierr)
#else
    call MPI_UNPACK(buffer,bufsize,position,anode%chi_correction ,n_coord_tor*n_degrees            ,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,ierr)
#endif
#endif
    call MPI_UNPACK(buffer,bufsize,position,anode%j_source       ,n_tor*n_degrees                  ,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,ierr)
#elif fullmhd
    call MPI_UNPACK(buffer,bufsize,position,anode%Fprof_eq       ,n_degrees,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,ierr)
    call MPI_UNPACK(buffer,bufsize,position,anode%psi_eq         ,n_degrees,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,ierr)
#endif
    call MPI_UNPACK(buffer,bufsize,position,anode%index          ,n_degrees,MPI_INTEGER,MPI_COMM_WORLD,ierr)
    call MPI_UNPACK(buffer,bufsize,position,anode%boundary       ,1        ,MPI_INTEGER,MPI_COMM_WORLD,ierr)
    call MPI_UNPACK(buffer,bufsize,position,anode%boundary_index ,1        ,MPI_INTEGER,MPI_COMM_WORLD,ierr)
    call MPI_UNPACK(buffer,bufsize,position,anode%axis_node      ,1        ,MPI_LOGICAL,MPI_COMM_WORLD,ierr)
    call MPI_UNPACK(buffer,bufsize,position,anode%axis_dof       ,1        ,MPI_INTEGER,MPI_COMM_WORLD,ierr)
    call MPI_UNPACK(buffer,bufsize,position,anode%constrained    ,1        ,MPI_LOGICAL,MPI_COMM_WORLD,ierr)
    call MPI_UNPACK(buffer,bufsize,position,anode%parents(1:2)   ,2        ,MPI_INTEGER,MPI_COMM_WORLD,ierr)
    call MPI_UNPACK(buffer,bufsize,position,anode%parent_elem    ,1        ,MPI_INTEGER,MPI_COMM_WORLD,ierr)
    call MPI_UNPACK(buffer,bufsize,position,anode%ref_lambda     ,1        ,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,ierr)
    call MPI_UNPACK(buffer,bufsize,position,anode%ref_mu         ,1        ,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,ierr)

    call make_deep_copy_node(anode, node_list%node(i))

  enddo

endif

call tr_unregister_mem(bufsize,"bcastn_buffer")
call dealloc_node(anode)
deallocate(buffer)

return
end subroutine Broadcast_nodes
