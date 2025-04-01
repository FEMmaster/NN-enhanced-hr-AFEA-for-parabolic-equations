#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <cmath>
#include <list>
#include <vector>
#include <fstream>
#include "LinearAlgebro.h"
#include "commons.h"

using namespace OF::COMMONS;

template<typename M, typename F>
void ComputeDensity(const M& mesh, const LVector& error, F& size_field, int ite)
{
	auto& node = mesh.nodes();
	auto& cell = mesh.cells();
	int NN = mesh.number_of_nodes();
	int NC = mesh.number_of_cells();

	CSRMatrix p2t(NN, NC);
	std::vector<Triplet<double>> tls;
	tls.reserve(NC * 3);
	for (int i = 0; i < NC; ++i) {
		for (int j = 0; j < 3; ++j)
			tls.push_back({ cell(i, j), i, 1.0 });
	}
	p2t.setFromTriplets(tls.begin(), tls.end());

	LVector volume(NC);
	LVector havg(NC);
	havg.setZero();
	LVector edge_lengths(NC);
	for (int i = 0; i < NC; ++i) {
		volume[i] = mesh.cell_volume(i);
		edge_lengths[i] = std::sqrt((4.0 * volume[i]) / std::sqrt(3));
		for (int j = 0; j < 3; ++j) {
			havg[i] += mesh.edge_measure(mesh.cell_to_edge(i)[j]);
		}
		havg[i] /= 3.0;
	}

	LVector nodeError = (p2t * error).array() / (p2t * LVector::Ones(NC)).array();
	//LVector nodeEdge = (p2t * havg).array() / (p2t * LVector::Ones(NC)).array();
	LVector nodeEdge = (p2t * edge_lengths).array() / (p2t * LVector::Ones(NC)).array();

	LVector rho = nodeError.array().pow(1) / nodeEdge.array().pow(1);
	std::vector<std::pair<double, size_t>> valueIndexPairs;
	for (size_t i = 0; i < rho.size(); ++i) {
		valueIndexPairs.push_back(std::make_pair(rho[i], i));
	}

	std::sort(valueIndexPairs.begin(), valueIndexPairs.end(), [](const std::pair<double, size_t>& a, const std::pair<double, size_t>& b) {
		return a.first > b.first;
		});

	//// Mark strategy 1
	//double threshold = std::pow(0.92, std::pow(2, iter)) * rho.array().pow(2.0).sum();
	//LVector labels = LVector::Constant(rho.size(), 1.1);
	//double cumulativeErrorSquared = 0.0;
	//for (int i = 0; i < valueIndexPairs.size(); ++i) {
	//	auto pair = valueIndexPairs[i];
	//	cumulativeErrorSquared += std::pow(pair.first, 2);
	//	if (cumulativeErrorSquared <= threshold) {
	//		labels[pair.second] = 1.0 / 3.0;
	//	}
	//	else {
	//		std::cout << "Fine " << i << " points." << std::endl;
	//		break;
	//	}
	//}

	//// Mark strategy 2
	//int N = 3;
	//LVector labels = LVector::Constant(rho.size(), 1.1);
	//for (int i = 0; i < std::min(N, static_cast<int>(valueIndexPairs.size())); ++i) {
	//	auto pair = valueIndexPairs[i];
	//	labels[pair.second] = std::pow(1.0 / 2.0, 4.0 / 2.0);
	//	//labels[pair.second] = std::pow(std::sqrt(1.0 / 2.0),log(double(NN) / N + 1) / log(2.0));
	//}
	//std::cout << "Marked " << std::min(N, static_cast<int>(valueIndexPairs.size())) << " points." << std::endl;

	// Mark strategy 3
	//double threshold = std::pow(0.92, 2*iter) * rho.array().pow(2.0).sum();
	double threshold = 0.99 * rho.array().pow(2.0).sum();
	LVector labels = LVector::Constant(rho.size(), 1.1);
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

	//for (const auto& pair : valueIndexPairs) {
	//	std::cout << pair.second << "\t" << pair.first << "\n";
	//}

	double value = std::pow(std::sqrt(1.0 / 2.0), std::log(double(NN) / N + 1) / std::log(2.0));
	for (int i = 0; i < N; ++i) {
		auto pair = valueIndexPairs[i];
		labels[pair.second] = std::pow(value, ite);
	}

	size_field.get_data() = nodeEdge.array() * labels.array();
}

//template<typename M, typename F>
//void ComputeSize(const M& mesh, F& size_field)
//{
//	auto& node = mesh.nodes();
//	auto& cell = mesh.cells();
//	int NN = mesh.number_of_nodes();
//	int NC = mesh.number_of_cells();
//
//	CSRMatrix p2t(NN, NC);
//	std::vector<Triplet<double>> tls;
//	tls.reserve(NC * 3);
//	for (int i = 0; i < NC; ++i) {
//		for (int j = 0; j < 3; ++j)
//			tls.push_back({ cell(i, j), i, 1.0 });
//	}
//	p2t.setFromTriplets(tls.begin(), tls.end());
//
//	LVector volume(NC);
//	LVector havg(NC);
//	havg.setZero();
//	LVector edge_lengths(NC);
//	for (int i = 0; i < NC; ++i) {
//		volume[i] = mesh.cell_volume(i);
//		edge_lengths[i] = std::sqrt((4.0 * volume[i]) / std::sqrt(3));
//		for (int j = 0; j < 3; ++j) {
//			havg[i] += mesh.edge_measure(mesh.cell_to_edge(i)[j]);
//		}
//		havg[i] /= 3.0;
//	}
//	//LVector nodeEdge = (p2t * havg).array() / (p2t * LVector::Ones(NC)).array();
//	LVector nodeEdge = (p2t * edge_lengths).array() / (p2t * LVector::Ones(NC)).array();
//	
//	LVector labels = LVector::Constant(nodeEdge.size(), 1.0);
//	size_field.get_data() = nodeEdge;
//}