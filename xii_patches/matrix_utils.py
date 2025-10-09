from dolfin import (PETScMatrix, Matrix, IndexMap, PETScVector, Vector,
                    as_backend_type, FunctionSpace, MPI)
from block import block_mat, block_vec
from scipy.sparse import csr_matrix
from contextlib import contextmanager
from petsc4py import PETSc
import numpy as np


def is_petsc_vec(v):
    
    return isinstance(v, (PETScVector, Vector))


def is_petsc_mat(A):
    
    return isinstance(A, (PETScMatrix, Matrix))


def is_number(i):
    
    return isinstance(i, (float, int))


def as_petsc(A):
    
    if is_petsc_vec(A):
        return as_backend_type(A).vec()
    if is_petsc_mat(A):
        return as_backend_type(A).mat()

    raise ValueError('%r is not matrix/vector.' % type(A))


def transpose_matrix(A):
    
    if isinstance(A, PETSc.Mat):
        At = PETSc.Mat()  
        A.transpose(At)  
        return At

    At = transpose_matrix(as_backend_type(A).mat())
    return PETScMatrix(At)


def diagonal_matrix(size, A):
    
    if isinstance(A, (int, float)):
        d = PETSc.Vec().createWithArray(A*np.ones(size))
    else:
        d = as_backend_type(A).vec()
    I = PETSc.Mat().createAIJ(size=size, nnz=1)
    I.setDiagonal(d)
    I.assemble()

    return PETScMatrix(I)


def identity_matrix(V):
    
    if isinstance(V, FunctionSpace):
        return diagonal_matrix(V.dim(), 1)

    if isinstance(V, int):
        return diagonal_matrix(V, 1)

    mat = block_mat([[0]*len(V) for _ in range(len(V))])
    for i in range(len(mat)):
        mat[i][i] = identity_matrix(V[i])
    return mat


def block_reshape(AA, offsets):
    
    nblocks = len(offsets)
    mat = block_mat([[0]*nblocks for _ in range(nblocks)])

    offsets = [0] + list(offsets)
    AA = AA.blocks
    for row, (ri, rj) in enumerate(zip(offsets[:-1], offsets[1:])):
        for col, (ci, cj) in enumerate(zip(offsets[:-1], offsets[1:])):
            if rj-ri == 1 and cj -ci == 1:
                mat[row][col] = AA[ri, ci]
            else:
                mat[row][col] = block_mat(AA[ri:rj, ci:cj])

    return mat


def zero_matrix(nrows, ncols):
    
    mat = csr_matrix((np.zeros(nrows, dtype=float),  
                      
                      (np.arange(nrows), np.zeros(nrows, dtype=int))),  
                     shape=(nrows, ncols))

    A = PETSc.Mat().createAIJ(size=[[nrows, nrows], [ncols, ncols]],
                              csr=(mat.indptr, mat.indices, mat.data))
    A.assemble()

    return PETScMatrix(A)


def row_matrix(rows):
    
    ncols, = set(row.size() for row in rows)
    nrows = len(rows)

    indptr = np.cumsum(np.array([0]+[ncols]*nrows))
    indices = np.tile(np.arange(ncols), nrows)
    data = np.hstack([row.get_local() for row in rows])

    mat = csr_matrix((data, indices, indptr), shape=(nrows, ncols))
    
    A = PETSc.Mat().createAIJ(size=[[nrows, nrows], [ncols, ncols]],
                              csr=(mat.indptr, mat.indices, mat.data))
    A.assemble()

    return PETScMatrix(A)


@contextmanager
def petsc_serial_matrix(test_space, trial_space, nnz=None):
    
    
    
    if is_number(test_space) and is_number(trial_space):
        
        
        comm = MPI.comm_world
        if comm.size != 1:
            raise RuntimeError("petsc_serial_matrix(int,int) is serial-only; "
                               "call it with FunctionSpaces under MPI.")
        sizes = [[test_space, test_space], [trial_space, trial_space]]
        row_map = PETSc.IS().createStride(test_space, 0, 1, comm)
        col_map = PETSc.IS().createStride(trial_space, 0, 1, comm)
    
    else:
        mesh = test_space.mesh()
        comm = mesh.mpi_comm()
        
        row_dm = test_space.dofmap()
        col_dm = trial_space.dofmap()

        nrow_owned = row_dm.index_map().size(IndexMap.MapSize.OWNED)
        ncol_owned = col_dm.index_map().size(IndexMap.MapSize.OWNED)

        sizes = [[nrow_owned, row_dm.index_map().size(IndexMap.MapSize.GLOBAL)],
                 [ncol_owned, col_dm.index_map().size(IndexMap.MapSize.GLOBAL)]]

        
        row_map = list(map(int, row_dm.tabulate_local_to_global_dofs()[:nrow_owned]))
        col_map = list(map(int, col_dm.tabulate_local_to_global_dofs()[:ncol_owned]))
        
    lgmap = lambda indices: (PETSc.LGMap().create(indices, comm=comm)
                             if isinstance(indices, list)
                             else
                             PETSc.LGMap().createIS(indices))
    
    row_lgmap, col_lgmap = list(map(lgmap, (row_map, col_map)))


    
    mat = PETSc.Mat().createAIJ(sizes, nnz=nnz, comm=comm)
    mat.setUp()
    
    mat.setLGMap(row_lgmap, col_lgmap)

    mat.assemblyBegin()
    
    yield mat
    
    mat.assemblyEnd()
