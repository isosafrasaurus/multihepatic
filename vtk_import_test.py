# @title .vtk import test
import vtk

reader = vtk.vtkPolyDataReader()
reader.SetFileName(WD_PATH + 'data/vtk/sortedDomain.vtk')
reader.Update()

polydata = reader.GetOutput()
cells = polydata.GetLines()  # or GetPolys() for general cells

for cellId in range(cells.GetNumberOfCells()):
    cell = vtk.vtkIdList()
    cells.GetCell(cellId, cell)
    print("Cell", cellId, ":", [cell.GetId(i) for i in range(cell.GetNumberOfIds())])