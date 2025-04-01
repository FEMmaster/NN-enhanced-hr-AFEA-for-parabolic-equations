#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <cmath>
#include <list>
#include <vector>
#include <fstream>
#include "LinearAlgebro.h"

using LVector = OF::FEM_EXAMPLE::LVector;
using CSRMatrix = OF::FEM_EXAMPLE::CSRMatrix;
using Triplet = OF::FEM_EXAMPLE::Triplet<double>;

template<typename M, typename F>
void ComputeDensity(const M& mesh, const LVector& error, F& size_field, int ite)
{
	auto& node = mesh.nodes();
	auto& cell = mesh.cells();
	int NN = mesh.number_of_nodes();
	int NC = mesh.number_of_cells();

	CSRMatrix p2t(NN, NC);
	std::vector<Triplet> tls;
	tls.reserve(NC * 4);
	for (int i = 0; i < NC; ++i) {
		for (int j = 0; j < 4; ++j)
			tls.push_back({ cell(i, j), i, 1.0 });
	}
	p2t.setFromTriplets(tls.begin(), tls.end());

	LVector volume(NC);
	LVector havg(NC);
	havg.setZero();
	LVector edge_lengths(NC);
	for (int i = 0; i < NC; ++i) {
		volume[i] = mesh.cell_volume(i);
		edge_lengths[i] = std::cbrt(6 * std::sqrt(2) * volume[i]);
		for (int j = 0; j < 6; ++j) {
			havg[i] += mesh.edge_measure(mesh.cell_to_edge(i)[j]);
		}
		havg[i] /= 6.0;
	}

	LVector nodeError = (p2t * error).array() / (p2t * LVector::Ones(NC)).array();
	//LVector nodeEdge = (p2t * havg).array() / (p2t * LVector::Ones(NC)).array();
	LVector nodeEdge = (p2t * edge_lengths).array() / (p2t * LVector::Ones(NC)).array();

	LVector rho = nodeError.array().pow(1) / nodeEdge.array().pow(1.5);
	std::vector<std::pair<double, size_t>> valueIndexPairs;
	for (size_t i = 0; i < rho.size(); ++i) {
		valueIndexPairs.push_back(std::make_pair(rho[i], i));
	}

	std::sort(valueIndexPairs.begin(), valueIndexPairs.end(), [](const std::pair<double, size_t>& a, const std::pair<double, size_t>& b) {
		return a.first > b.first;
		});

	// Mark strategy 3
	double threshold = 0.999 * rho.array().pow(2.0).sum();
	LVector labels = LVector::Constant(rho.size(), 1.2);
	double cumulativeErrorSquared = 0.0;
	int N = 0;
	for (int i = 0; i < valueIndexPairs.size(); ++i) {
		auto pair = valueIndexPairs[i];
		cumulativeErrorSquared += std::pow(pair.first, 2);
		if (cumulativeErrorSquared <= threshold) {
			N++;
		}
		else {
			std::cout << "Marked " << N << " points that will be refine." << std::endl;
			break;
		}
	}

	double value = 0.0;
	value = std::pow(std::sqrt(1.0 / 2.0), std::log(double(NN) / N + 2) / std::log(2.0));

	for (int i = 0; i < N; ++i) {
		auto pair = valueIndexPairs[i];
		labels[pair.second] = std::pow(value, ite);
	}

	size_field.get_data() = nodeEdge.array() * labels.array();
}


template<typename MESH, typename Function>
void ComputeSize(const MESH& mesh, Function& size_field)
{
	auto& node = mesh.nodes();
	auto& cell = mesh.cells();
	int NN = mesh.number_of_nodes();
	int NC = mesh.number_of_cells();

	CSRMatrix p2t(NN, NC);
	std::vector<Triplet> tls;
	tls.reserve(NC * 4);
	for (int i = 0; i < NC; ++i) {
		for (int j = 0; j < 4; ++j)
			tls.push_back({ cell(i, j), i, 1.0 });
	}
	p2t.setFromTriplets(tls.begin(), tls.end());

	LVector volume(NC);
	LVector havg(NC);
	LVector edge_lengths(NC);
	for (int i = 0; i < NC; ++i) {
		volume[i] = mesh.cell_volume(i);
		edge_lengths[i] = std::cbrt(6.0 * std::sqrt(2) * volume[i]);
		havg[i] = 0;
		for (int j = 0; j < 6; ++j) {
			havg[i] += mesh.edge_measure(mesh.cell_to_edge(i)[j]);
		}
		havg[i] /= 6.0;
	}

	//LVector nodeEdge = (p2t * havg).array() / (p2t * LVector::Ones(NC)).array();
	LVector nodeEdge = (p2t * edge_lengths).array() / (p2t * LVector::Ones(NC)).array();

	LVector labels = LVector::Constant(nodeEdge.size(), 0.85);
	size_field.get_data() = nodeEdge.array() * labels.array();
}