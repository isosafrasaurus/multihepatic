from xii.linalg.matrix_utils import petsc_serial_matrix, is_number
from xii.assembler.average_form import average_cell, average_space

from numpy.polynomial.legendre import leggauss
from dolfin import PETScMatrix, cells, Point, Cell, Function
import scipy.sparse as sp
from petsc4py import PETSc
import numpy as np
import tqdm
import dolfin as df


def memoize_average(average_mat):
    
    cache = {}
    def cached_average_mat(V, TV, reduced_mesh, data):
        
        
        comm_size = TV.mesh().mpi_comm().size
        key = ((V.ufl_element(), V.mesh().id()),
               (TV.ufl_element(), TV.mesh().id()),
               data['shape'],
               ('comm', comm_size))

        if key not in cache:
            cache[key] = average_mat(V, TV, reduced_mesh, data)
        return cache[key]
    
    return cached_average_mat


@memoize_average
def avg_mat(V, TV, reduced_mesh, data):
    
    assert TV.mesh().id() == reduced_mesh.id()
    
    
    assert TV.ufl_element().family() == 'Discontinuous Lagrange'
    
    
    assert V.ufl_element().value_shape() == TV.ufl_element().value_shape()
    assert average_cell(V) == TV.mesh().ufl_cell()
    assert V.mesh().geometry().dim() == TV.mesh().geometry().dim()

    shape = data['shape']
    
    if shape is None:
        return PETScMatrix(trace_3d1d_matrix(V, TV, reduced_mesh))

    
    Rmat = average_matrix(V, TV, shape)
        
    return PETScMatrix(Rmat)
                

def average_matrix(V, TV, shape):
    
    
    
    
    
    
    
    mesh_x = TV.mesh().coordinates()
    
    
    
    value_size = TV.ufl_element().value_size()

    if value_size == 1:
        return scalar_average_matrix(V, TV, shape)
    
    mesh = V.mesh()
    
    tree = mesh.bounding_box_tree()
    limit = mesh.num_cells()

    
    
    
    
    gdim = TV.mesh().geometry().dim()
    TV_coordinates = TV.tabulate_dof_coordinates().reshape((-1, gdim))
    line_mesh = TV.mesh()
    
    
    TV_dm_full = TV.dofmap()
    V_dm = V.dofmap()
    l2g_rows = TV_dm_full.tabulate_local_to_global_dofs()
    l2g_cols = V_dm.tabulate_local_to_global_dofs()
    nrow_owned = TV_dm_full.index_map().size(df.IndexMap.MapSize.OWNED)

    
    
    TV_dm_scalar = TV.sub(0).dofmap() if value_size > 1 else TV_dm_full

    Vel = V.element()               
    basis_values = np.zeros(V.element().space_dimension()*value_size)
    with petsc_serial_matrix(TV, V) as mat:

        for line_cell in tqdm.tqdm(cells(line_mesh), desc=f'Averaging over {line_mesh.num_cells()} cells',
                                   total=line_mesh.num_cells()):
            
            
            v0, v1 = mesh_x[line_cell.entities(0)]
            n = v0 - v1

            
            scalar_dofs = TV_dm_scalar.cell_dofs(line_cell.index())
            scalar_dofs_x = TV_coordinates[scalar_dofs]
            for scalar_row, avg_point in zip(scalar_dofs, scalar_dofs_x):
                
                quadrature = shape.quadrature(avg_point, n)
                integration_points = quadrature.points
                wq = quadrature.weights

                curve_measure = sum(wq)

                data = {}
                for index, ip in enumerate(integration_points):
                    c = tree.compute_first_entity_collision(Point(*ip))
                    if c >= limit:
                        c = None
                        continue

                    if c is None:
                        cs = tree.compute_entity_collisions(Point(*ip))[:1]
                    else:
                        cs = (c, )
                    
                    for c in cs:
                        Vcell = Cell(mesh, c)
                        vertex_coordinates = Vcell.get_vertex_coordinates()
                        cell_orientation = Vcell.orientation()
                        basis_values[:] = Vel.evaluate_basis_all(ip, vertex_coordinates, cell_orientation)

                        cols_ip = V_dm.cell_dofs(c)
                        values_ip = basis_values*wq[index]
                        
                        for col, value in zip(cols_ip, values_ip.reshape((-1, value_size))):
                            if col in data:
                                data[col] += value/curve_measure
                            else:
                                data[col] = value/curve_measure
                            
                
                
                column_indices = np.array(list(data.keys()), dtype='int32')
                for shift in range(value_size):
                    row_local = scalar_row + shift
                    
                    if row_local >= nrow_owned:
                        continue
                    
                    row_global = np.array([l2g_rows[row_local]], dtype=PETSc.IntType)
                    column_values = np.array([data[col][shift] for col in column_indices])
                    cols_global = np.asarray(l2g_cols[column_indices], dtype=PETSc.IntType)
                    
                    mat.setValues(row_global, cols_global, column_values,
                                  PETSc.InsertMode.ADD_VALUES)
            
        
    return mat


def scalar_average_matrix(V, TV, shape):
    
    
    
    
    
    
    
    
    mesh_x = TV.mesh().coordinates()
    
    
    
    value_size = TV.ufl_element().value_size()

    mesh = V.mesh()
    
    tree = mesh.bounding_box_tree()
    limit = mesh.num_cells()

    gdim = TV.mesh().geometry().dim()
    TV_coordinates = TV.tabulate_dof_coordinates().reshape((-1, gdim))  
    line_mesh = TV.mesh()
    
    TV_dm = TV.dofmap()
    V_dm = V.dofmap()

    Vel = V.element()               
    basis_values = np.zeros(V.element().space_dimension()*value_size)


    
    l2g_rows = TV_dm.tabulate_local_to_global_dofs()
    l2g_cols = V_dm.tabulate_local_to_global_dofs()
    nrow_owned = TV_dm.index_map().size(df.IndexMap.MapSize.OWNED)

    with petsc_serial_matrix(TV, V) as mat:
        for line_cell in tqdm.tqdm(cells(line_mesh), desc=f'Averaging over {line_mesh.num_cells()} cells',
                                   total=line_mesh.num_cells()):
            
            
            v0, v1 = mesh_x[line_cell.entities(0)]
            n = v0 - v1

            scalar_dofs = TV_dm.cell_dofs(line_cell.index())
            scalar_dofs_x = TV_coordinates[scalar_dofs]
            for scalar_row, avg_point in zip(scalar_dofs, scalar_dofs_x):
                quadrature = shape.quadrature(avg_point, n)
                integration_points = quadrature.points
                wq = quadrature.weights
                curve_measure = sum(wq)

                data = {}
                for index, ip in enumerate(integration_points):
                    c = tree.compute_first_entity_collision(Point(*ip))
                    if c >= limit:
                        c = None
                        continue
                    if c is None:
                        cs = tree.compute_entity_collisions(Point(*ip))[:1]
                    else:
                        cs = (c, )
                    c = cs[0]
                    Vcell = Cell(mesh, c)
                    vertex_coordinates = Vcell.get_vertex_coordinates()
                    cell_orientation = Vcell.orientation()
                    basis_values[:] = Vel.evaluate_basis_all(ip, vertex_coordinates, cell_orientation)

                    cols_ip = V_dm.cell_dofs(c)
                    values_ip = basis_values*wq[index]
                    for col, value in zip(cols_ip, values_ip):
                        data[col] = data.get(col, 0.0) + value/curve_measure

                if not data:
                    continue

                column_indices = np.array(list(data.keys()), dtype='int32')
                column_values = np.array([data[col] for col in column_indices])
                
                if scalar_row < nrow_owned:
                    row_global = np.array([l2g_rows[scalar_row]], dtype=PETSc.IntType)
                    cols_global = np.asarray(l2g_cols[column_indices], dtype=PETSc.IntType)
                    mat.setValues(row_global, cols_global, column_values,
                                  PETSc.InsertMode.ADD_VALUES)
    return mat

def trace_3d1d_matrix(V, TV, reduced_mesh):
    
    assert reduced_mesh.id() == TV.mesh().id()
    assert any((V.ufl_element().family() == 'Lagrange',
                (V.ufl_element().family() == 'Discontinuous Lagrange' and V.ufl_element().degree() == 0)
                ))
    
    mesh = V.mesh()
    line_mesh = TV.mesh()
    
    
    
    
    value_size = TV.ufl_element().value_size()

    
    if hasattr(reduced_mesh, 'parent_entity_map'):
        
        mapping = reduced_mesh.parent_entity_map[mesh.id()][1]
        
        mesh.init(1)
        mesh.init(1, 3)
        e2c = mesh.topology()(1, 3)
        
        get_cell3d = lambda c, d1d3=mapping, d3d3=e2c: d3d3(d1d3[c.index()])[0]
    
    else:
        tree = mesh.bounding_box_tree()
        limit = mesh.num_cells()

        get_cell3d = lambda c, tree=tree, bound=limit: (
            lambda index: index if index<bound else None
        )(tree.compute_first_entity_collision(c.midpoint()))
  
    gdim = TV.mesh().geometry().dim()
    TV_coordinates = TV.tabulate_dof_coordinates().reshape((-1, gdim))  
    TV_dm = TV.dofmap()
    V_dm = V.dofmap()
    
    if value_size > 1:
        TV_dm = TV.sub(0).dofmap()

    Vel = V.element()               
    basis_values = np.zeros(V.element().space_dimension()*value_size)
    l2g_rows = TV.dofmap().tabulate_local_to_global_dofs()
    l2g_cols = V.dofmap().tabulate_local_to_global_dofs()
    nrow_owned = TV.dofmap().index_map().size(df.IndexMap.MapSize.OWNED)

    with petsc_serial_matrix(TV, V) as mat:

        for line_cell in tqdm.tqdm(cells(line_mesh), desc=f'Averaging over {line_mesh.num_cells()} cells',
                                   total=line_mesh.num_cells()):
            
            
            scalar_dofs = TV_dm.cell_dofs(line_cell.index())
            scalar_dofs_x = TV_coordinates[scalar_dofs]

            
            
            tet_cell = get_cell3d(line_cell)
            if tet_cell is None: continue
            
            Vcell = Cell(mesh, tet_cell)
            vertex_coordinates = Vcell.get_vertex_coordinates()
            cell_orientation = 0
            
            
            
            column_indices = np.array(V_dm.cell_dofs(tet_cell), dtype='int32')

            for scalar_row, avg_point in zip(scalar_dofs, scalar_dofs_x):
                
                basis_values[:] = Vel.evaluate_basis_all(avg_point, vertex_coordinates, cell_orientation)
                
                
                
                data = basis_values.reshape((-1, value_size)).T
                for shift, column_values in enumerate(data):
                    row_local = scalar_row + shift
                    if row_local >= nrow_owned:
                        continue
                    row_global = np.array([l2g_rows[row_local]], dtype=PETSc.IntType)
                    cols_global = np.asarray(l2g_cols[column_indices], dtype=PETSc.IntType)
                    mat.setValues(row_global, cols_global, column_values,
                                  PETSc.InsertMode.ADD_VALUES)
            
        
    return mat


def MeasureFunction(averaged):
    
    
    V = averaged.function_space()  
    
    if V.ufl_element().value_shape(): V = V.sub(0).collapse()

    mesh_1d = averaged.average_['mesh']
    
    TV = average_space(V, mesh_1d)

    gdim = TV.mesh().geometry().dim()
    TV_coordinates = TV.tabulate_dof_coordinates().reshape((-1, gdim))  
    TV_dm = TV.dofmap()
    imap = TV_dm.index_map()
    n_owned = imap.size(df.IndexMap.MapSize.OWNED)
    visited = np.zeros(n_owned, dtype=bool)
    mesh_x = mesh_1d.coordinates()
    shape = averaged.average_['shape']
    
    values = np.empty(n_owned, dtype=float)
    for cell in cells(mesh_1d):
        
        
        v0, v1 = mesh_x[cell.entities(0)]
        n = v0 - v1

        
        dofs = TV_dm.cell_dofs(cell.index())
        for dof in dofs:
            if dof < n_owned and not visited[dof]:
                x = TV_coordinates[dof]
                
                values[dof] = sum(shape.quadrature(x, n).weights)
                visited[dof] = True
                
    assert np.all(visited)
    
    
    m = Function(TV)
    m.vector().set_local(values)
    m.vector().apply('insert')

    return m




if __name__ == '__main__':
    from dolfin import *
    from xii import EmbeddedMesh
    from xii.assembler.average_shape import Circle

    
    def is_close(a, b=0): return abs(a - b) < 1E-13
    
    
    
    mesh = UnitCubeMesh(10, 10, 10)

    f = MeshFunction('size_t', mesh, 1, 0)
    CompiledSubDomain('near(x[0], 0.5) && near(x[1], 0.5)').mark(f, 1)

    bmesh = EmbeddedMesh(f, 1)

    
    V = FunctionSpace(mesh, 'CG', 2)
    TV = FunctionSpace(bmesh, 'DG', 1)
    
    f = interpolate(Expression('x[0]+x[1]+x[2]', degree=1), V)
    Tf0 = interpolate(f, TV)

    Trace = avg_mat(V, TV, bmesh, {'shape': None})
    Tf = Function(TV)
    Trace.mult(f.vector(), Tf.vector())
    Tf0.vector().axpy(-1, Tf.vector())
    assert is_close(Tf0.vector().norm('linf'))

    V = VectorFunctionSpace(mesh, 'CG', 2)
    TV = VectorFunctionSpace(bmesh, 'DG', 1)
    
    f = interpolate(Expression(('x[0]+x[1]+x[2]',
                                'x[0]-x[1]',
                                'x[1]+x[2]'), degree=1), V)
    Tf0 = interpolate(f, TV)

    Trace = avg_mat(V, TV, bmesh, {'shape': None})
    Tf = Function(TV)
    Trace.mult(f.vector(), Tf.vector())
    Tf0.vector().axpy(-1, Tf.vector())
    assert is_close(Tf0.vector().norm('linf'))

    radius = 0.01
    quad_degree = 10
    
    shape = Circle(radius=radius, degree=quad_degree)

    
    V = FunctionSpace(mesh, 'CG', 3)
    Q = FunctionSpace(bmesh, 'DG', 3)

    f = Expression('x[2]*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5))', degree=3)
    Pif = Expression('x[2]*A*A', A=radius, degree=1)
    
    f = interpolate(f, V)
    Pi_f0 = interpolate(Pif, Q)

    Pi_f = Function(Q)

    Pi = avg_mat(V, Q, bmesh, {'shape': shape})
    Pi.mult(f.vector(), Pi_f.vector())

    Pi_f0.vector().axpy(-1, Pi_f.vector())
    assert is_close(Pi_f0.vector().norm('linf'))

    
    V = VectorFunctionSpace(mesh, 'CG', 3)
    Q = VectorFunctionSpace(bmesh, 'DG', 3)

    f = Expression(('x[2]*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5))',
                    '2*x[2]*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5))',
                    '-3*x[2]*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5))'),
                    degree=3)
    Pif = Expression(('x[2]*A*A',
                      '2*x[2]*A*A',
                      '-3*x[2]*A*A'), A=radius, degree=1)
    
    f = interpolate(f, V)
    Pi_f0 = interpolate(Pif, Q)

    Pi_f = Function(Q)

    Pi = avg_mat(V, Q, bmesh, {'shape': shape})
    Pi.mult(f.vector(), Pi_f.vector())

    Pi_f0.vector().axpy(-1, Pi_f.vector())
    assert is_close(Pi_f0.vector().norm('linf'))
