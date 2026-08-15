#ifndef VTKMeshWriter_h
#define VTKMeshWriter_h

#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkCellArray.h>
#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkSmartPointer.h>

#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridWriter.h>

#include <vtkHexagonalPrism.h>
#include <vtkHexahedron.h>
#include <vtkLine.h>
#include <vtkPentagonalPrism.h>
#include <vtkPixel.h>
#include <vtkPolyLine.h>
#include <vtkPolyVertex.h>
#include <vtkPolygon.h>
#include <vtkPyramid.h>
#include <vtkQuad.h>
#include <vtkTetra.h>
#include <vtkTriangle.h>
#include <vtkTriangleStrip.h>
#include <vtkVertex.h>
#include <vtkVoxel.h>
#include <vtkWedge.h>

#include <string>
#include <memory>

namespace OF {
namespace FEM_EXAMPLE {

class VTKMeshWriter
{
private:
    vtkSmartPointer<vtkUnstructuredGrid> m_ugrid;
    vtkSmartPointer<vtkXMLUnstructuredGridWriter> m_writer;

public:
    VTKMeshWriter()
    {
        m_ugrid = vtkSmartPointer<vtkUnstructuredGrid>::New();
        m_writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
    }

    template<class Mesh>
    void set_mesh(Mesh & mesh)
    {
      m_ugrid->Initialize(); //把网格清空
      set_points(mesh);
      set_cells(mesh);
    }


    template<class Mesh>
    void set_points(Mesh & mesh)
    {
      auto NN = mesh.number_of_nodes();
      auto points = vtkSmartPointer<vtkPoints>::New();
      points->Allocate(NN);
      auto GD = mesh.geo_dimension();

      if(GD == 3)
      {
        auto func = [&](int n)->void
        {
          double * pp = mesh.node(n);
          points->InsertNextPoint(pp[0], pp[1], pp[2]); // (*it) 括号是必须的
        };
        mesh.forch_node(func);
      }
      else if(GD == 2)
      {
        auto func = [&](int n)->void
        {
          double * pp = mesh.node(n);
          points->InsertNextPoint(pp[0], pp[1], 0.0); // (*it) 括号是必须的
        };
        mesh.forch_node(func);
      }
      m_ugrid->SetPoints(points);
    }

    template<class Mesh>
    void set_cells(Mesh & mesh)
    {
        auto NC = mesh.number_of_cells();
        auto nn = mesh.number_of_nodes_of_each_cell();

        auto cells = vtkSmartPointer<vtkCellArray>::New();
        cells->Allocate(NC * nn);

        const int * idx = mesh.vtk_write_cell_index();
        auto func = [&](int c)->void
        {
          cells->InsertNextCell(nn);
          for(int i = 0; i < nn; i++)
          {
            cells->InsertCellPoint(mesh.cell(c)[idx[i]]);
          }
        };
        mesh.forch_cell(func);
        m_ugrid->SetCells(mesh.vtk_cell_type(), cells);
    }

    template<typename T>
    void set_point_data(std::vector<T> & data, int ncomponents, const std::string name)
    {
        int n = data.size()/ncomponents;
        auto vtkdata = vtkSmartPointer<vtkAOSDataArrayTemplate<T>>::New();
        vtkdata->SetNumberOfComponents(ncomponents);
        vtkdata->SetNumberOfTuples(n);
        vtkdata->SetName(name.c_str());
        for(int i = 0; i < n; i++)
        {
            for(int j = 0; j < ncomponents; j ++)
                vtkdata->SetComponent(i, j, data[i*ncomponents + j]);
        }
        m_ugrid->GetPointData()->AddArray(vtkdata);
    }

    template<typename T>
    void set_cell_data(std::vector<T> & data, int ncomponents, const std::string name)
    {
        int n = data.size()/ncomponents;
        auto vtkdata = vtkSmartPointer<vtkAOSDataArrayTemplate<T>>::New();
        vtkdata->SetNumberOfComponents(ncomponents);
        vtkdata->SetNumberOfTuples(n);
        vtkdata->SetName(name.c_str());
        for(int i = 0; i < n; i++)
        {
            for(int j = 0; j < ncomponents; j ++)
                vtkdata->SetComponent(i, j, data[i*ncomponents + j]);
        }
        m_ugrid->GetCellData()->AddArray(vtkdata);
    }

    void write(const std::string & fname)
    {
        m_writer->SetFileName(fname.c_str());
        m_writer->SetInputData(m_ugrid);
        m_writer->Write();

        //m_ugrid->Initialize();
    }

};

} // end of namespace Mesh

} // end of namespace OF
#endif // end of VTKMeshWriter_h
