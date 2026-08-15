#ifndef EM_MeshGen_h
#define EM_MeshGen_h

#include <gmsh.h>
#include <unordered_map>
#include <iostream>

template<typename Mesh>
void make_uniform_mesh(double size, Mesh* mesh)
{
	std::vector<double> nodes;
	std::vector<double> cells;

	gmsh::initialize();
	gmsh::option::setNumber("General.Terminal", 0);
	gmsh::model::add("modle0");
	gmsh::option::setNumber("Mesh.MeshSizeMax", 1.2 * size);
	gmsh::option::setNumber("Mesh.MeshSizeMin", 0.8 * size);

	//gmsh::model::occ::addBox(-1, -1, -1, 2, 2, 2);
	gmsh::model::occ::addBox(0, 0, 0, 1, 1, 1, 1);
	//gmsh::model::occ::addBox(-0.5, -0.5, -0.5, 1, 1, 1);
	gmsh::model::occ::synchronize();

	gmsh::option::setNumber("Mesh.Optimize", 1);
	gmsh::option::setNumber("Mesh.OptimizeNetgen", 1);

	gmsh::model::mesh::generate(3);

	std::vector<size_t> nodeTag;
	std::vector<double> _nodeParaCoord;
	gmsh::model::mesh::getNodes(nodeTag, nodes, _nodeParaCoord, -1, -1, false, false);

	int NN = nodeTag.size();
	std::unordered_map<int, int> nTag2Nid;
	for (int i = 0; i < NN; i++)
		nTag2Nid[nodeTag[i]] = i;

	gmsh::vectorpair dim3d_tags;
	gmsh::model::getEntities(dim3d_tags, 3);
	for (const auto& it : dim3d_tags)
	{
		const int tag = it.second;
		std::vector<int> cellType;
		std::vector<std::vector<std::size_t> > cellTag;
		std::vector<std::vector<std::size_t> > c2nList;
		gmsh::model::mesh::getElements(cellType, cellTag, c2nList, 3, tag);

		std::vector<std::size_t>& cellTag1 = cellTag[0];
		std::vector<std::size_t>& c2nList1 = c2nList[0];
		int N = cells.size();
		int NC_new = cellTag1.size();
		cells.reserve(N + NC_new * 4);
		for (int i = 0; i < NC_new * 4; i++)
			cells.push_back(nTag2Nid[c2nList1[i]]);
	}

	//gmsh::fltk::run();
	gmsh::finalize();

	int NC = cells.size() / 4;
	mesh->nodes().resize(NN, 3);
	mesh->cells().resize(NC, 4);
	for (int i = 0; i < NN; i++)
	{
		mesh->node(i)[0] = nodes[i * 3];
		mesh->node(i)[1] = nodes[i * 3 + 1];
		mesh->node(i)[2] = nodes[i * 3 + 2];
	}
	for (int i = 0; i < NC; i++)
	{
		mesh->cell(i)[0] = cells[i * 4];
		mesh->cell(i)[1] = cells[i * 4 + 1];
		mesh->cell(i)[2] = cells[i * 4 + 2];
		mesh->cell(i)[3] = cells[i * 4 + 3];
	}

	mesh->init_top();
}

template<typename Mesh>
void make_mesh(std::function<double(int, int, double, double, double, double)>& size_field, Mesh* mesh)
{
	std::vector<double> nodes;
	std::vector<double> cells;

	gmsh::initialize();
	gmsh::option::setNumber("General.Terminal", 0);
	gmsh::model::add("modle0");

	//gmsh::model::occ::addBox(-1, -1, -1, 2, 2, 2);
	gmsh::model::occ::addBox(0, 0, 0, 1, 1, 1, 1);
	//gmsh::model::occ::addBox(-0.5, -0.5, -0.5, 1, 1, 1);
	gmsh::model::occ::synchronize();

	gmsh::option::setNumber("Mesh.Optimize", 1);
	gmsh::option::setNumber("Mesh.OptimizeNetgen", 1);
	gmsh::model::mesh::setSizeCallback(size_field);
	gmsh::model::mesh::generate(3);

	std::vector<size_t> nodeTag;
	std::vector<double> _nodeParaCoord;
	gmsh::model::mesh::getNodes(nodeTag, nodes, _nodeParaCoord, -1, -1, false, false);

	int NN = nodeTag.size();
	std::unordered_map<int, int> nTag2Nid;
	for (int i = 0; i < NN; i++)
		nTag2Nid[nodeTag[i]] = i;

	gmsh::vectorpair dim3d_tags;
	gmsh::model::getEntities(dim3d_tags, 3);
	for (const auto& it : dim3d_tags)
	{
		const int tag = it.second;
		std::vector<int> cellType;
		std::vector<std::vector<std::size_t> > cellTag;
		std::vector<std::vector<std::size_t> > c2nList;
		gmsh::model::mesh::getElements(cellType, cellTag, c2nList, 3, tag);

		std::vector<std::size_t>& cellTag1 = cellTag[0];
		std::vector<std::size_t>& c2nList1 = c2nList[0];
		int N = cells.size();
		int NC_new = cellTag1.size();
		cells.reserve(N + NC_new * 4);
		for (int i = 0; i < NC_new * 4; i++)
			cells.push_back(nTag2Nid[c2nList1[i]]); //单元
	}

	//gmsh::fltk::run();
	gmsh::finalize();

	int NC = cells.size() / 4;
	mesh->clear();
	mesh->nodes().resize(NN, 3);
	mesh->cells().resize(NC, 4);
	for (int i = 0; i < NN; i++)
	{
		mesh->node(i)[0] = nodes[i * 3];
		mesh->node(i)[1] = nodes[i * 3 + 1];
		mesh->node(i)[2] = nodes[i * 3 + 2];
	}
	for (int i = 0; i < NC; i++)
	{
		mesh->cell(i)[0] = cells[i * 4];
		mesh->cell(i)[1] = cells[i * 4 + 1];
		mesh->cell(i)[2] = cells[i * 4 + 2];
		mesh->cell(i)[3] = cells[i * 4 + 3];
	}
	mesh->init_top();
}

#endif // end of EM_MeshGen_h
