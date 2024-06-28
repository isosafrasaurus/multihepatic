

import networkx as nx
from .fenics_graph import *
from vtk import *
import os

'''
Overloaded File class for writing .vtp files for functions defined on the graph
This allows for using the TubeFilter in paraview
'''

class TubeRadius(UserExpression):
    
    
    def __init__(self, G, **kwargs):
        self.G = G
        super().__init__(**kwargs)
    def eval_cell(self, value, x, cell):
        edge_ix = self.G.mf[cell.index]
        edge = list(self.G.edges())[edge_ix]
        value[0] = self.G.edges()[edge]['radius']


class TubeFile(File):
    def __init__(self, G, fname, **kwargs):
        
        
        super().__init__(fname, 'ascii')
        
        f_name, f_ext = os.path.splitext(fname)
        assert f_ext == '.pvd', 'TubeFile must have .pvd file ending'
        
        self.fname = f_name
        self.G = G
        
        assert self.G.geom_dim==3, f'Coordinates are {self.G.geom_dim}d, they need to be 3d'
        assert len(nx.get_edge_attributes(self.G, 'radius'))>0, 'Graph must have radius attribute'
        
        
        pvdfile = open(fname, "w")
        pvdfile.write(pvd_header + pvd_footer) 
        pvdfile.close()
        
        
        
    def __lshift__(self, func):
        
        
        if type(func) is tuple:
            func, i = func
        else:
            i = 0    

        
        
        
        radius_dict = nx.get_edge_attributes(self.G, 'radius')
        mesh0, foo = self.G.get_mesh(0)
        DG = FunctionSpace(mesh0, 'DG', 0)
        radius = Function(DG)
        radius.vector()[:] = list(radius_dict.values())
        radius.set_allow_extrapolation(True)
                
        
        
        
        coords = self.G.mesh.coordinates()
        points = vtkPoints()
        for c in coords:
            points.InsertNextPoint(list(c))

        
        lines = vtkCellArray()
        edge_to_vertices = self.G.mesh.cells()

        for vs in edge_to_vertices: 
            line = vtkLine()
            line.GetPointIds().SetId(0, vs[0])
            line.GetPointIds().SetId(1, vs[1])
            lines.InsertNextCell(line)

        
        linesPolyData = vtkPolyData()
        linesPolyData.SetPoints(points)
        linesPolyData.SetLines(lines)


        
        data = vtkDoubleArray()
        
        data.SetName(func.name())
        data.SetNumberOfComponents(1)
        
        
        for c in coords:
            data.InsertNextTuple([func(c)])

        linesPolyData.GetPointData().AddArray(data)

        
        
        data = vtkDoubleArray()
        data.SetName('radius')
        data.SetNumberOfComponents(1)
        
        
        for c in coords:
            data.InsertNextTuple([radius(c)])

        linesPolyData.GetPointData().AddArray(data)

        
        writer = vtkXMLPolyDataWriter()
        writer.SetFileName(f"{self.fname}{int(i):06d}.vtp")
        writer.SetInputData(linesPolyData)
        writer.Update()
        writer.Write()
        
        
        
        pvdfile = open(self.fname+ ".pvd", "r")
        content = pvdfile.read().splitlines()
        
        
        short_fname = self.fname.split('/')[-1]
        
        pvd_entry = f"<DataSet timestep=\"{i}\" part=\"0\" file=\"{short_fname}{int(i):06d}.vtp\" />"
        updated_content = content[:-2] + [pvd_entry] + content[-2:] 
        updated_content = "\n".join(updated_content)
        
        pvdfile = open(self.fname + '.pvd', "w")
        pvdfile.write(updated_content)
        pvdfile.close()
        
        
    
pvd_header = """<?xml version=\"1.0\"?>
<VTKFile type="Collection" version=\"0.1\">
  <Collection>\n"""

pvd_footer= """</Collection>
</VTKFile>"""