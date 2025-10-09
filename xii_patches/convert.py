from xii.linalg.matrix_utils import (is_petsc_vec, is_petsc_mat, diagonal_matrix,
                                     is_number, as_petsc, petsc_serial_matrix,
                                     zero_matrix)
import xii

from block.block_compose import block_mul, block_add, block_sub, block_transpose
from block import block_mat, block_vec
from dolfin import PETScVector, PETScMatrix
from dolfin import Vector, GenericVector, Matrix, MPI
from scipy.sparse import bmat as numpy_block_mat
from scipy.sparse import csr_matrix, vstack as sp_vstack
from petsc4py import PETSc
import numpy as np
import itertools
import operator
from functools import reduce

COMM = PETSc.COMM_WORLD


def convert(bmat, algorithm='numpy'):
    
    
    if isinstance(bmat, block_vec):
        
        comm = MPI.comm_world
        
        if comm.size == 1:
            array = block_vec_to_numpy(bmat)
            vec = PETSc.Vec().createWithArray(array)
            vec.assemble()
            return PETScVector(vec)
        
        comp = [as_petsc(v) for v in bmat]            
        sizes = [int(v.getSize()) for v in comp]      
        N = int(sum(sizes))
        
        nloc = int(sum((v.getOwnershipRange()[1] - v.getOwnershipRange()[0]) for v in comp))
        out = PETSc.Vec().create(comm=COMM)
        out.setSizes((nloc, N))                       
        out.setUp()
        
        offset = 0
        for v, sz in zip(comp, sizes):
            r0, r1 = v.getOwnershipRange()                         
            comp_idx = np.arange(r0, r1, dtype=PETSc.IntType)
            vals = v.getValues(comp_idx)                           
            tgt_idx = comp_idx + offset                            
            out.setValues(tgt_idx, vals, PETSc.InsertMode.INSERT_VALUES)
            offset += sz
        out.assemblyBegin(); out.assemblyEnd()
        return PETScVector(out)
    
    
    
    if isinstance(bmat, block_mat):
        
        row_sizes, col_sizes = bmat_sizes(bmat)
        nrows, ncols = len(row_sizes), len(col_sizes)
        indices = itertools.product(list(range(nrows)), list(range(ncols)))

        blocks = np.zeros((nrows, ncols), dtype='object')
        for block, (i, j) in zip(bmat.blocks.flatten(), indices):
            
            A = collapse(block)

            if is_number(A):
                
                if i == j and row_sizes[i] == col_sizes[j]:
                    A = diagonal_matrix(row_sizes[i], A)
                else:
                    
                    A = zero_matrix(row_sizes[i], col_sizes[j])
                
                
            
            blocks[i, j] = A
        
        bmat = block_mat(blocks)

        assert all(is_petsc_mat(block) or is_number(block)
                   for block in bmat.blocks.flatten())
        
        
        if not algorithm:
            set_lg_map(bmat)
            return bmat

        comm = MPI.comm_world
        if comm.size > 1:
            
            petsc_blocks = [[as_petsc(bmat[i][j]) if is_petsc_mat(bmat[i][j]) else None
                             for j in range(ncols)]
                            for i in range(nrows)]
            Nest = PETSc.Mat().createNest(petsc_blocks, comm=COMM)
            Nest.assemble()
            Aij = Nest.convert('aij')   
            return PETScMatrix(Aij)
        else:
            array = block_mat_to_numpy(bmat)
            bmat = numpy_to_petsc(array)
            set_lg_map(bmat)
            return bmat

    
    return collapse(bmat)


def collapse(bmat):
    
    
    
    if is_petsc_mat(bmat) or is_number(bmat) or is_petsc_vec(bmat):
        return bmat

    if isinstance(bmat, (Vector, Matrix, GenericVector)):
        return bmat

    
    if isinstance(bmat, block_mul):
        return collapse_mul(bmat)
    
    elif isinstance(bmat, block_add):
        return collapse_add(bmat)
    
    elif isinstance(bmat, block_sub):
        return collapse_sub(bmat)
    
    elif isinstance(bmat, block_transpose):
        return collapse_tr(bmat)

    
    
    elif hasattr(bmat, 'v'):
        
        diagonal = bmat.v

        n = diagonal.size
        mat = PETSc.Mat().createAIJ(comm=COMM, size=[[n, n], [n, n]], nnz=1)
        mat.assemblyBegin()
        mat.setDiagonal(diagonal)
        mat.assemblyEnd()

        return PETScMatrix(mat)
    
    elif hasattr(bmat, 'matrix'):
        return bmat.matrix

    
    elif hasattr(bmat, 'collapse'):
        return bmat.collapse()
    
    elif hasattr(bmat, 'create_vec'):
        x = bmat.create_vec()
        columns = []
        for ei in Rn_basis(x):
            y = bmat*ei
            columns.append(csr_matrix(convert(y).get_local()))
        bmat = (sp_vstack(columns).T).tocsr()

        return numpy_to_petsc(bmat)
    
    raise ValueError('Do not know how to collapse %r' % type(bmat))


def collapse_tr(bmat):
    
    A = bmat.A
    if is_petsc_mat(A):
        A_ = as_petsc(A)
        C_ = PETSc.Mat()
        A_.transpose(C_)
        
        try:
            rA, cA = A_.getLGMap()
            if rA is not None or cA is not None:
                C_.setLGMap(cA, rA)
        except Exception:
            pass
        return PETScMatrix(C_)


def collapse_add(bmat):
    
    A, B = bmat.A, bmat.B
    if is_petsc_mat(A) and is_number(B) and abs(B) < 1E-14:
        return A
    if is_petsc_mat(B) and is_number(A) and abs(A) < 1E-14:
        return B    
    
    if is_petsc_mat(A) and is_petsc_mat(B):
        A_ = as_petsc(A)
        B_ = as_petsc(B)
        assert A_.size == B_.size
        C_ = A_.copy()
        
        C_.axpy(1., B_, PETSc.Mat.Structure.DIFFERENT)
        
        try:
            rA, cA = A_.getLGMap()
            if rA is not None or cA is not None:
                C_.setLGMap(rA, cA)
        except Exception:
            pass
        return PETScMatrix(C_)
    
    return collapse_add(collapse(A) + collapse(B))


def collapse_sub(bmat):
    
    A, B = bmat.A, bmat.B
    
    if is_petsc_mat(A) and is_petsc_mat(B):
        A_ = as_petsc(A)
        B_ = as_petsc(B)
        assert A_.size == B_.size
        C_ = A_.copy()
        
        C_.axpy(-1., B_, PETSc.Mat.Structure.DIFFERENT)
        
        try:
            rA, cA = A_.getLGMap()
            if rA is not None or cA is not None:
                C_.setLGMap(rA, cA)
        except Exception:
            pass
        return PETScMatrix(C_)
    
    return collapse_sub(collapse(A) - collapse(B))


def collapse_mul(bmat):
    
    
    A, B = bmat.chain[0], bmat.chain[1:]

    if len(B) == 1:
        B = B[0]
        
        if is_petsc_mat(A) and is_petsc_mat(B):
            A_ = as_petsc(A)
            B_ = as_petsc(B)
            assert A_.size[1] == B_.size[0]
            C_ = PETSc.Mat()
            A_.matMult(B_, C_)
            
            try:
                rA, _ = A_.getLGMap()
            except Exception:
                rA = None
            try:
                _, cB = B_.getLGMap()
            except Exception:
                cB = None
            try:
                if rA is not None or cB is not None:
                    
                    C_.setLGMap(rA, cB)
            except Exception:
                pass
            return PETScMatrix(C_)
        
        elif is_petsc_mat(A) and is_number(B):
            A_ = as_petsc(A)
            C_ = A_.copy()
            C_.scale(B)
            
            try:
                rA, cA = A_.getLGMap()
                if rA is not None or cA is not None:
                    C_.setLGMap(rA, cA)
            except Exception:
                pass
            return PETScMatrix(C_)

        elif is_petsc_mat(B) and is_number(A):
            B_ = as_petsc(B)
            C_ = B_.copy()
            C_.scale(A)
            
            try:
                rB, cB = B_.getLGMap()
                if rB is not None or cB is not None:
                    C_.setLGMap(rB, cB)
            except Exception:
                pass
            return PETScMatrix(C_)
        
        else:
            return collapse(collapse(A)*collapse(B))
    
    else:
        return collapse_mul(collapse(A)*collapse(reduce(operator.mul, B)))                                    

    

def block_vec_to_numpy(bvec):
    
    return np.hstack([v.get_local() for v in bvec])


def block_mat_to_numpy(bmat):
    
    
    if is_petsc_mat(bmat):
        bmat = as_petsc(bmat)
        return csr_matrix(bmat.getValuesCSR()[::-1], shape=bmat.size)
    
    if is_number(bmat):
        return None  
    
    blocks = np.array(list(map(block_mat_to_numpy, bmat.blocks.flatten())))
    blocks = blocks.reshape(bmat.blocks.shape)
    
    return numpy_block_mat(blocks).tocsr()


def numpy_to_petsc(mat):
    
    
    if isinstance(mat, np.ndarray):
        if mat.ndim == 1:
            vec = PETSc.Vec().createWithArray(mat)
            vec.assemble()
            return PETScVector(vec)

        return numpy_to_petsc(csr_matrix(mat))
    
    A = PETSc.Mat().createAIJ(comm=COMM,
                              size=mat.shape,
                              csr=(mat.indptr, mat.indices, mat.data))
    
    A.assemble()
    return PETScMatrix(A)


def block_mat_to_petsc(bmat):
    
    
    def iter_rows(matrix):
        for i in range(matrix.size(0)):
            yield matrix.getrow(i)

    row_sizes, col_sizes = get_sizes(bmat)
    row_offsets = np.cumsum([0] + list(row_sizes))
    col_offsets = np.cumsum([0] + list(col_sizes))

    with petsc_serial_matrix(row_offsets[-1], col_offsets[-1]) as mat:
        row = 0
        for row_blocks in bmat.blocks:
            
            for indices_values in zip(*list(map(iter_rows, row_blocks))):
                indices, values = list(zip(*indices_values))

                indices = [list(index+offset) for index, offset in zip(indices, col_offsets)]
                indices = sum(indices, [])
            
                row_values = np.hstack(values)

                mat.setValues([row], indices, row_values, PETSc.InsertMode.INSERT_VALUES)

                row += 1
    return PETScMatrix(mat)


def get_dims(thing):
    
    if is_petsc_vec(thing): return thing.size()

    if is_petsc_mat(thing): return (thing.size(0), thing.size(1))
    
    if is_number(thing): return None
    
    
    
    if isinstance(thing, block_mul):
        A, B = thing.chain[0], thing.chain[1:]

        dims_A, dims_B = get_dims(A), get_dims(B[0])
        
        if dims_A is None:
            return dims_B
        if dims_B is None:
            return dims_A
        
        if len(B) == 1:
            assert len(dims_A) == len(dims_B) 
            assert dims_A[1] == dims_B[0], (dims_A, dims_B) 
            return (dims_A[0], dims_B[1])
        else:
            dims_B = get_dims(reduce(operator.mul, B))
            
            assert len(dims_A) == len(dims_B) 
            assert dims_A[1] == dims_B[0], (dims_A, dims_B)
            return (dims_A[0], dims_B[1])
    
    if isinstance(thing, (block_add, block_sub)):
        A, B = thing.A, thing.B
        if is_number(A):
            return get_dims(B)

        if is_number(B):
            return get_dims(A)

        dims = get_dims(A)
        assert dims == get_dims(B), (dims, get_dims(B))
        return dims
    
    if isinstance(thing, block_transpose):
        dims = get_dims(thing.A)
        return (dims[1], dims[0])
    
    
    if hasattr(thing, 'A'):
        assert is_petsc_mat(thing.A)
        return get_dims(thing.A)

    if hasattr(thing, '__sizes__'):
        return thing.__sizes__
    
    if hasattr(thing, 'create_vec'):
        return (thing.create_vec(0).size(), thing.create_vec(1).size())

    raise ValueError('Cannot get_dims of %r, %s' % (type(thing), thing))

    
def bmat_sizes(bmat):
    
    if isinstance(bmat, block_vec):
        return tuple(map(get_dims, block_vec.blocks))

    if isinstance(bmat, block_mat):
        vec = bmat.create_vec(0)
        vecs = (vec, ) if not hasattr(vec, 'blocks') else vec.blocks
        row_sizes = tuple(vec.size() for vec in vecs)

        vec = bmat.create_vec(1)
        vecs = (vec, ) if not hasattr(vec, 'blocks') else vec.blocks
        col_sizes = tuple(vec.size() for vec in vecs)
        
        return row_sizes, col_sizes
    
    raise ValueError('Cannot bmat_sizes of %r, %s' % (type(bmat), bmat))


def set_lg_map(mat):
    
    
    

    if is_number(mat): return mat

    assert is_petsc_mat(mat) or isinstance(mat, block_mat), (type(mat))

    if isinstance(mat, block_mat):
        blocks = np.array(list(map(set_lg_map, mat.blocks.flatten()))).reshape(mat.blocks.shape)
        return block_mat(blocks)

    
    
    try:
        comm = as_petsc(mat).getComm()
    except Exception:
        comm = MPI.comm_world
    if comm.size > 1:
        return mat

    
    rowmap, colmap = list(range(mat.size(0))), list(range(mat.size(1)))

    row_lgmap = PETSc.LGMap().create(rowmap, comm=comm)
    col_lgmap = PETSc.LGMap().create(colmap, comm=comm)

    as_petsc(mat).setLGMap(row_lgmap, col_lgmap)

    return mat


def Rn_basis(vec):
    if not isinstance(vec, block_vec):
        vec = Vector(MPI.comm_world, vec.local_size())
        values = np.zeros(vec.local_size())
        for i in range(len(values)):
            values[i] = 1.
            
            vec.set_local(values)
            yield vec

            values[:] *= 0.
    else:
        assert False




if __name__ == '__main__':
    from dolfin import *

    mesh = UnitSquareMesh(32, 32)
    V = FunctionSpace(mesh, 'CG', 1)
    Q = FunctionSpace(mesh, 'DG', 0)
    W = [V, Q]
    
    u, p = list(map(TrialFunction, W))
    v, q = list(map(TestFunction, W))

    [[A00, A01],
     [A10, A11]] = [[assemble(inner(u, v)*dx), assemble(inner(v, p)*dx)],
                    [assemble(inner(u, q)*dx), assemble(inner(p, q)*dx)]]
    blocks = [[A00*A00, A01+A01],
              [2*A10 - A10, A11*A11*A11]]
    
    AA = block_mat(blocks)

    t = Timer('x'); t.start()
    X = convert(AA)
    print((t.stop()))

    t = Timer('x'); t.start()
    Y = convert(AA, 'foo')
    print((t.stop()))

    X_ = X.array()
    X_[:] -= Y.array()
    print((np.linalg.norm(X_, np.inf)))
