
from block import block_bc, block_assemble
from dolfin import *

from xii.linalg.convert import convert 
from xii.linalg.convert import numpy_to_petsc
from scipy.sparse import csr_matrix
from block import block_mat, block_vec
from block.block_bc import block_rhs_bc
from petsc4py import PETSc
import numpy as np


def apply_bc(A, b, bcs, diag_val=1., symmetric=True, return_apply_b=False):
    
    
    if bcs is None or not bcs or (isinstance(bcs, list) and not any(bc for bc in bcs)):
        return A, b
    
    
    
    has_wrapped_A = False
    if not isinstance(A, block_mat):
        A = block_mat([[A]])
        has_wrapped_A = True

    if b is None:
        b = A.create_vec()
    
    if not isinstance(b, block_vec):
        assert has_wrapped_A
        b = block_vec([b])

    if isinstance(bcs, dict):
        assert all(0 <= k < len(A) for k in list(bcs.keys()))
        assert all(isinstance(v, list) for v in list(bcs.values()))
        bcs = [bcs.get(i, []) for i in range(len(A))]
    else:
        if has_wrapped_A:
            bcs = [bcs]

    
    assert len(A) == len(b) == len(bcs), (len(A), len(b), len(bcs))

    bcs_ = []
    for bc in bcs:
        assert isinstance(bc, (DirichletBC, list, dict)), (type(bc), )
        if isinstance(bc, DirichletBC):
            bcs_.append([bc])
        else:
            bcs_.append(bc)
    bcs = bcs_

    if not bcs or not any(bcs): return A, b

    
    
    comm = MPI.comm_world
    if comm.size > 1:
        
        A = convert(A, algorithm=None)

        
        
        
        for i, row_bcs in enumerate(bcs):
            
            if not row_bcs:
                continue
            
            if isinstance(row_bcs, dict):
                row_bcs = sum(row_bcs.values(), [])
            if isinstance(row_bcs, DirichletBC):
                row_bcs = [row_bcs]
            
            row_V = None
            for bc in row_bcs:
                if isinstance(bc, DirichletBC):
                    row_V = bc.function_space()
                    break
            if row_V is None:
                continue

            
            dm = row_V.dofmap()
            n_owned = dm.index_map().size(IndexMap.MapSize.OWNED)
            row_l2g = list(map(int, dm.tabulate_local_to_global_dofs()[:n_owned]))
            row_lgmap = PETSc.LGMap().create(row_l2g, comm=comm)

            
            for j in range(len(A[i])):
                Aij = A[i][j]
                
                mat = as_backend_type(Aij).mat()
                
                try:
                    col_lgmap = mat.getLGMap()[1]
                except Exception:
                    col_lgmap = None
                if col_lgmap is None:
                    
                    cstart, cend = mat.getOwnershipRangeColumn()
                    col_is = PETSc.IS().createStride(cend - cstart, cstart, 1, comm=comm)
                    col_lgmap = PETSc.LGMap().createIS(col_is)
                mat.setLGMap(row_lgmap, col_lgmap)

        op = block_bc(bcs, symmetric)
        op.apply(A)
        if b is not None:
            
            block_rhs_bc(bcs, A).apply(b)

        if has_wrapped_A:
            A, b = A[0][0], b[0]

        if return_apply_b and symmetric:
            def apply_b(bb, A=A, bcs=bcs):
                block_rhs_bc(bcs, A).apply(bb)
                return bb
            return A, b, apply_b
        return A, b

    
    AA, bb = list(map(convert, (A, b)))

    
    AA, bb = as_backend_type(AA).mat(), as_backend_type(bb).vec()

    
    
    
    
    offsets = [0]
    for bi in b:
        offsets.append(offsets[-1] + bi.size())

    assert AA.size[0] == AA.size[1] == offsets[-1], (AA.size[0], AA.size[1], offsets)
        
    rows = []
    x_values = []
    
    
    for shift, bcs_sub in zip(offsets, bcs): 
        for bc in bcs_sub:
            
            
            if isinstance(bc, DirichletBC): bc = bc.get_boundary_values()
            
            rows.extend(shift + np.array(list(bc.keys()), dtype='int32'))
            x_values.extend(list(bc.values()))
            
    rows = np.hstack(rows)
    x_values = np.array(x_values)

    x = bb.copy()
    x.zeroEntries()
    x.setValues(rows, x_values)
    x.assemble()

    blocks = []
    for first, last in zip(offsets[:-1], offsets[1:]):
        blocks.append(PETSc.IS().createStride(last-first, first, 1))

        
    if return_apply_b and symmetric:
        
        
        
        
        
        def apply_b(bb, AA=AA.copy(), dofs=rows, bcs=bcs, blocks=blocks):
            
            values = np.array(sum((list(bc.get_boundary_values().values()) if isinstance(bc, DirichletBC) else []
                                   for bcs_sub in bcs for bc in bcs_sub),
                                  []))
            
            c, Ac = AA.createVecs()

            c.zeroEntries()
            c.setValues(dofs, values)  
            c.assemble()

            AA.mult(c, Ac)  
            
            b = convert(bb)  
            b_vec = as_backend_type(b).vec()
            b_vec.axpy(-1, Ac)  
            b_vec.setValues(dofs, values)
            b_vec.assemble()

            bb *= 0  
            bb += as_block(b, blocks)
                
            return bb

    
    if symmetric:
        len(rows) and AA.zeroRowsColumns(rows, diag=diag_val, x=x, b=bb)
    else:
        len(rows) and AA.zeroRows(rows, diag=diag_val, x=x, b=bb)
    
    b = as_block(bb, blocks)
    A = as_block(AA, blocks)
    
    if has_wrapped_A: return A[0][0], b[0]

    if return_apply_b:
        return A, b, apply_b
    
    return A, b


def element_types(iterable):
    
    return set(map(type, iterable))


def as_block(monolithic, blocks):
    
    comm = MPI.comm_world

    
    elm_type, = element_types(blocks)
    if elm_type is FunctionSpace:
        offsets = np.cumsum(np.r_[0, [Wi.dim() for Wi in blocks]])

        idx = []
        for first, last in zip(offsets[:-1], offsets[1:]):
            idx.append(PETSc.IS().createStride(last-first, first, 1))
        return as_block(monolithic, idx)
            
    elif elm_type in (list, np.ndarray):
        return as_block(monolithic, [PETSc.IS().createGeneral(np.asarray(block, dtype='int32')) for block in blocks])

    assert elm_type is PETSc.IS

    
    if isinstance(monolithic, (GenericVector, Vector)):
        return as_block(as_backend_type(monolithic).vec(), blocks)

    if isinstance(monolithic, (Matrix, )):
        return as_block(as_backend_type(monolithic).mat(), blocks)

    if isinstance(monolithic, PETSc.Vec):
        b = [PETSc.Vec().createWithArray(monolithic.getValues(block), comm=comm) for block in blocks]
        for bi in b:
            bi.assemblyBegin()
            bi.assemblyEnd()
        return block_vec(list(map(PETScVector, b)))

    
    try:
        monolithic.getSubMatrix
        A = [[PETScMatrix(monolithic.getSubMatrix(block_row, block_col)) for block_col in blocks]
             for block_row in blocks]
    
    
    
    except AttributeError:
        monolithic = csr_matrix(monolithic.getValuesCSR()[::-1], shape=monolithic.size)

        A = [[numpy_to_petsc(monolithic[block_row.array, :][:, block_col.array]) for block_col in blocks]
             for block_row in blocks]
    
    return block_mat(A)


def identity(ncells):
    mesh = UnitSquareMesh(ncells, ncells)

    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    W = [V, Q]

    u, p = list(map(TrialFunction, W))
    v, q = list(map(TestFunction, W))

    a = [[0]*len(W) for _ in range(len(W))]
    a[0][0] = inner(grad(u), grad(v))*dx
    a[0][1] = inner(p, div(v))*dx
    a[1][0] = inner(q, div(u))*dx

    x, y = SpatialCoordinate(mesh)
    L = [inner(as_vector((y*x**2, sin(pi*(x+y)))), v)*dx,
         inner(x+y, q)*dx]

    A0 = block_assemble(a)
    b0 = block_assemble(L)

    
    bcs = [[], []]

    A, b = apply_bc(A0, b0, bcs)

    b0_ = np.hstack([bi.get_local() for bi in b0])
    b_ = np.hstack([bi.get_local() for bi in b])

    eb = (b - b0).norm()

    for i in range(len(W)):
        for j in range(len(W)):
            Aij = A[i][j]  
            
            x = Vector(MPI.comm_world, Aij.size(1))
            x.set_local(np.random.rand(x.local_size()))

            y = Aij*x
            y0 = A0[i][j]*x

            print(i, j, '>>>', (y - y0).norm('linf'))
    return eb

    
def speed(ncells):
    mesh = UnitSquareMesh(ncells, ncells)

    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    W = [V, Q]

    u, p = list(map(TrialFunction, W))
    v, q = list(map(TestFunction, W))

    a = [[0]*len(W) for _ in range(len(W))]
    a[0][0] = inner(grad(u), grad(v))*dx
    a[0][1] = inner(p, div(v))*dx
    a[1][0] = inner(q, div(u))*dx

    x, y = SpatialCoordinate(mesh)
    L = [inner(as_vector((y*x**2, sin(pi*(x+y)))), v)*dx,
         inner(x+y, q)*dx]

    A0 = block_assemble(a)
    b0 = block_assemble(L)

    
    bc = DirichletBC(V, Constant((0, 0)), 'on_boundary')
    bcs = [[bc], []]
    t = Timer('first')
    block_bc(bcs, True).apply(A0).apply(b0)
    dt0 = t.stop()

    dimW = sum(Wi.dim() for Wi in W)

    A, b = list(map(block_assemble, (a, L)))
    
    t = Timer('second')
    A, b = apply_bc(A, b, bcs)
    dt1 = t.stop()

    print('>>>', (b - b0).norm())

    
    A, c = list(map(block_assemble, (a, L)))    
    block_rhs_bc(bcs, A).apply(c)

    print('>>>', (b - c).norm())

    return dimW, dt0, dt1, dt0/dt1



if __name__ == '__main__':
    parameters['reorder_dofs_serial'] = False
    
    


    mesh = UnitSquareMesh(3, 3)
    W_elm = MixedElement([FiniteElement('Lagrange', triangle, 1),
                          VectorElement('Lagrange', triangle, 2)])
    W = FunctionSpace(mesh, W_elm)

    f = interpolate(Expression(('x[0]', 'x[1]', 'x[1]+x[0]'), degree=1), W)
    b = f.vector()

    W0 = FunctionSpace(mesh, 'CG', 1)
    W1 = VectorFunctionSpace(mesh, 'CG', 2)

    
    bb = as_block(b, [W.sub(0).dofmap().dofs(), W.sub(1).dofmap().dofs()])
    f0 = Function(W0, bb[0])
    f1 = Function(W1, bb[1])

    f00, f10 = f.split()
    print((sqrt(abs(assemble(inner(f0 - f00, f0 - f00)*dx)))))
    print((sqrt(abs(assemble(inner(f1 - f10, f1 - f10)*dx)))))

    
    if False:
        msg = 'dim(W) = %d, block = %g s, petsc = %g s, speedup %.2f'
        for ncells in [4, 8, 16, 32, 64, 128]:
            print(msg % speed(ncells)) 

    V = FunctionSpace(mesh, 'CG', 1)
    u, v, = TrialFunction(V), TestFunction(V)

    a = [[inner(u, v)*dx, 0], [0, inner(u, v)*dx]]
    L = [inner(Constant(0), v)*dx, inner(Constant(0), v)*dx]
    from xii import ii_assemble

    A, b = list(map(ii_assemble, (a, L)))

    foo = Constant(1)
    
    bcs = [[DirichletBC(V, foo, 'on_boundary')],
           [DirichletBC(V, Constant(2), 'on_boundary')]]

    A, b, apply_b = apply_bc(A, b, bcs, return_apply_b=True)
    
    c = apply_b(b)
    d = c.copy()

    foo.assign(Constant(2))
    b = apply_b(b)
    print((d - b).norm())
    
    
