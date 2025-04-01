#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>
#include <typeinfo>
#include <Eigen/Sparse>
#include <gmsh.h>
#include <unordered_map>
#include <chrono>

#include "LinearAlgebro.h"
#include "LinearLagrangeFE.h"
#include "LinearLagrangeFESpace.h"
#include "TetrahedronMeshWithTree.h"

#include "MeshGen0.h"
#include "Dens.h"
#include "NN.h"
#include "error_estimator.h"

#include "VTKMeshWriter.h"

typedef OF::FEM_EXAMPLE::VTKMeshWriter Writer;
typedef OF::FEM_EXAMPLE::TetrahedronMeshWithTree Mesh;
typedef OF::FEM_EXAMPLE::CSRMatrix CSRMatrix;
typedef OF::FEM_EXAMPLE::LVector LVector;

template<typename F>
void Solve(CSRMatrix& A, LVector& b, F& uh) {
	Eigen::BiCGSTAB<CSRMatrix> solver;
	std::cout << "number of unknow : " << b.size() << std::endl;
	solver.compute(A);
	std::cout << "solved over" << std::endl;
	uh.get_data() = solver.solve(b);
}

double u(double x, double y, double z, double t) {
	//// Example1_AC1 solve in [-1,1]
	//const double epsilon = 0.05;
	//double c = std::cos(1.5 * M_PI * x) * std::cos(1.5 * M_PI * y) * (std::sin(M_PI * z) + std::sin(2 * M_PI * z));
	//return epsilon * c;

	//// Example1_AC2 solve in [0,1]
	//const double epsilon = 0.02;
	//double absX = std::pow(x-0.5, 2);
	//double absY = std::pow(y-0.5, 2);
	//double absZ = std::pow(z-0.5, 2);
	//return tanh((0.4 - std::sqrt(absX + absY + absZ)) / std::sqrt(2.0)/ epsilon);

	//// Example2_Possion_Rotation
	//const int beta = -500;
	//double temp1 = x -0.5 - 0.3 * std::cos(2 * M_PI * t);
	//double temp2 = y -0.5 - 0.3 * std::sin(2 * M_PI * t);
	//double temp3 = z -0.5;
	//return std::exp(beta * (temp1 * temp1 + temp2 * temp2 + temp3 * temp3));

	//// Example3_Possion_splitting
	//double temp = -100;
	//double temp1 = std::exp(temp * (std::pow(x - 0.5 + 0.3 * t, 2) + std::pow(y - 0.5, 2) + std::pow(z - 0.5, 2)));
	//double temp2 = std::exp(temp * (std::pow(x - 0.5 - 0.3 * t, 2) + std::pow(y - 0.5, 2) + std::pow(z - 0.5, 2)));
	//return temp1 + temp2;

	// Example4_Possion_Diffusion
	double temp = -5000;
	double R = std::sqrt(std::pow(x - 0.5, 2) + std::pow(y - 0.5, 2) + std::pow(z - 0.5, 2));
	double r = R + 0.2 * t - 0.3;
	return std::exp(temp * std::pow(r, 2.0));
}

void grad_u(double x, double y, double z, double t, std::array<double, 3>& val) {
	//// Example1_AC
	//val[0] = 0.0;
	//val[1] = 0.0;
	//val[2] = 0.0;

	//// Example2_Possion_Rotation
	//const int beta = -500;
	//double temp1 = x - 0.5 - 0.3 * std::cos(2 * M_PI * t);
	//double temp2 = y - 0.5 - 0.3 * std::sin(2 * M_PI * t);
	//double temp3 = z - 0.5;
	//val[0] = 2*beta * temp1 * u(x, y, z, t);
	//val[1] = 2*beta * temp2 * u(x, y, z, t);
	//val[2] = 2*beta * temp3 * u(x, y, z, t);

	//// Example3_Possion_splitting
	//double temp = -100;
	//double temp1 = std::exp(temp * (std::pow(x - 0.5 + 0.3 * t, 2) + std::pow(y - 0.5, 2) + std::pow(z - 0.5, 2)));
	//double temp2 = std::exp(temp * (std::pow(x - 0.5 - 0.3 * t, 2) + std::pow(y - 0.5, 2) + std::pow(z - 0.5, 2)));
	//val[0] = temp * 2 * (x - 0.5 + 0.3 * t) * temp1 + temp * 2 * (x - 0.5 - 0.3 * t) * temp2;
	//val[1] = temp * 2 * (y - 0.5) * temp1 + temp * 2 * (y - 0.5) * temp2;
	//val[2] = temp * 2 * (z - 0.5) * temp1 + temp * 2 * (z - 0.5) * temp2;

	// Example4_Possion_Diffusion
	double temp = -5000;
	double R = std::sqrt(std::pow(x - 0.5, 2) + std::pow(y - 0.5, 2) + std::pow(z - 0.5, 2));
	double r = R + 0.2 * t - 0.3;
	val[0] = temp * 2 * r * (x - 0.5) / R * u(x, y, z, t);
	val[1] = temp * 2 * r * (y - 0.5) / R * u(x, y, z, t);
	val[2] = temp * 2 * r * (z - 0.5) / R * u(x, y, z, t);
}

double f(double x, double y, double z, double t) {
	//// Example1_AC
	//return 0.0;

	//// Example2_Possion_Rotation
	//const int beta = -500;
	//double temp1 = x -0.5 - 0.3 * std::cos(2 * M_PI * t);
	//double temp2 = y -0.5 - 0.3 * std::sin(2 * M_PI * t);
	//double temp3 = z - 0.5;
	//double u_t = -4 * beta * M_PI * 0.3 * ((y - 0.5) * std::cos(2 * M_PI * t) - (x - 0.5) * std::sin(2 * M_PI * t)) * u(x, y, z, t);
	//double u_xx = 2 * beta * (2 * beta * temp1 * temp1 + 1) * u(x, y, z, t);
	//double u_yy = 2 * beta * (2 * beta * temp2 * temp2 + 1) * u(x, y, z, t);
	//double u_zz = 2 * beta * (2 * beta * temp3 * temp3 + 1) * u(x, y, z, t);
	//return u_t - u_xx - u_yy - u_zz;

	//// Example3_Possion_splitting
	//double temp = -100;
	//double temp1 = std::exp(temp * (std::pow(x - 0.5 + 0.3 * t, 2) + std::pow(y - 0.5, 2) + std::pow(z - 0.5, 2)));
	//double temp2 = std::exp(temp * (std::pow(x - 0.5 - 0.3 * t, 2) + std::pow(y - 0.5, 2) + std::pow(z - 0.5, 2)));
	//double u_t = temp * 2 * (x - 0.5 + 0.3 * t) * 0.3 * temp1 - temp * 2 * (x - 0.5 - 0.3 * t) * 0.3 * temp2;
	//double u_xx = (temp * 2 + std::pow(temp * 2 * (x - 0.5 + 0.3 * t), 2)) * temp1 + (temp * 2 + (std::pow(temp * 2 * (x - 0.5 - 0.3 * t), 2))) * temp2;
	//double u_yy = (temp * 2 + std::pow(temp * 2 * (y - 0.5), 2)) * temp1 + (temp * 2 + std::pow(temp * 2 * (y - 0.5), 2)) * temp2;
	//double u_zz = (temp * 2 + std::pow(temp * 2 * (z - 0.5), 2)) * temp1 + (temp * 2 + std::pow(temp * 2 * (z - 0.5), 2)) * temp2;
	//return u_t - u_xx - u_yy - u_zz;

	// Example4_Possion_Diffusion
	double temp = -5000;
	double R = std::sqrt(std::pow(x - 0.5, 2) + std::pow(y - 0.5, 2) + std::pow(z - 0.5, 2));
	double r = R + 0.2 * t - 0.3;
	double u_t = temp * 2 * r * 0.2 * u(x, y, z, t);
	double u_xx = temp * 2 * u(x, y, z, t) * (std::pow((x - 0.5) / R, 2) + r * (R - (x - 0.5)) * (R + x - 0.5) / std::pow(R, 3) + temp * 2 * std::pow(r * (x - 0.5) / R, 2));
	double u_yy = temp * 2 * u(x, y, z, t) * (std::pow((y - 0.5) / R, 2) + r * (R - (y - 0.5)) * (R + y - 0.5) / std::pow(R, 3) + temp * 2 * std::pow(r * (y - 0.5) / R, 2));
	double u_zz = temp * 2 * u(x, y, z, t) * (std::pow((z - 0.5) / R, 2) + r * (R - (z - 0.5)) * (R + z - 0.5) / std::pow(R, 3) + temp * 2 * std::pow(r * (z - 0.5) / R, 2));
	return u_t - u_xx - u_yy - u_zz;
}

void equation_nn() {
	typedef OF::FEM_EXAMPLE::LinearLagrangeFE FE;
	typedef OF::FEM_EXAMPLE::LinearLagrangeFESpace<Mesh, FE> Space;

	int NT = 50;
	double dt = 1.0 / NT;
	double eTol = 0.05;
	double t;

	Net net({ 3, 32, 32, 32, 32, 1});
	const int train_epoch = 10000;
	const double learn_rate = 1.0e-3;

	Mesh mesht;
	Space spacet(&mesht);
	auto uht = spacet.function();
	auto size_field = spacet.function();
	auto interpola_uht = spacet.function();

	Mesh meshtm1;
	Space spacetm1(&meshtm1);
	auto uhtm1 = spacetm1.function();

	std::function<double(int, int, double x, double y, double z, double)>
		back_size_field = [&size_field](int, int, double x, double y, double z, double)->double {
		std::array<double, 3> p = { x, y, z };
		return size_field(p);
		};

	auto _uhtm1 = [&uhtm1](double x, double y, double z)->double {
		std::array<double, 3> p = { x, y, z };
		return uhtm1(p);
		};

	auto uxt = [&t](double x, double y, double z)->double { return u(x, y, z, t); };
	auto fxt = [&t](double x, double y, double z)->double { return f(x, y, z, t); };
	auto grad_uxt = [&t](double x, double y, double z, std::array<double, 3>& val)->void
		{grad_u(x, y, z, t, val); };

	std::vector<double> last_NN_values;
	std::vector<double> last_H1_values;
	std::vector<double> last_RC_values;
	for (int times = 0; times < NT + 1; times++) {
		t = times * dt;
		std::cout << "****************************************************************** timese iteration " << t << std::endl;

		make_uniform_mesh(0.2, &mesht);
		uht.update();

		std::vector<double> NN_values;
		std::vector<double> H1_values;
		std::vector<double> RC_values;
		int iter = 0;
		while (1) {
			std::cout << "**********************************************************Spatial adaptive " << iter << std::endl;

			if (times == 0) {
				spacet.interpola(uxt, uht);
				std::cout << "number of unknow : " << mesht.number_of_nodes() << std::endl;
			}
			else {
				CSRMatrix S;
				spacet.stiff_matrix(S);

				CSRMatrix M;
				spacet.mass_matrix(M);
				CSRMatrix A = M + dt * S;

				LVector source;
				std::vector<std::array<std::array<double, 3>, 4>> all_points(mesht.number_of_cells());
				spacet.source_vector(fxt, source, all_points);
				torch::Tensor all_points_tensor = torch::from_blob(all_points.data(), { mesht.number_of_cells(), 4, 3 }, torch::kDouble).to(torch::kFloat);
				torch::Tensor output = net.forward(all_points_tensor).reshape({ mesht.number_of_cells(), 4 });

				LVector source0;
				spacet.source_vector_batch(output, source0);
				source = source * dt + source0;

				spacet.set_dirichlet_boundary(uxt, A, source);
				Solve(A, source, uht);
			}

			NN_values.push_back(mesht.number_of_nodes());

			LVector real_eta;
			double grad_u_norm = spacet.H1_error(grad_uxt, uht, real_eta);
			double ETA = std::sqrt(real_eta.squaredNorm());
			std::cout << "H1 error : " << ETA << std::endl;
			H1_values.push_back(ETA);

			//LVector L2nume_eta(mesht.number_of_cells());
			//L2nume_eta.setZero();
			//if (times > 0) {
			//	interpola_uht.update();
			//	spacet.interpola(_uhtm1, interpola_uht);
			//	spacet.L2_error(_uhtm1, interpola_uht, L2nume_eta);
			//}

			LVector nume_eta;
			error_estimator(uht, nume_eta);
			ETA = std::sqrt(nume_eta.squaredNorm());
			std::cout << "Recovered H1 error : " << ETA << std::endl;
			RC_values.push_back(ETA);

			std::stringstream ss;
			ss << "mesh_" << times << iter << ".vtu";
			Writer writer;
			writer.set_points(mesht);
			writer.set_cells(mesht);
			std::vector<double> real_eta_copy(real_eta.data(), real_eta.data() + real_eta.size());
			writer.set_cell_data(real_eta_copy, 1, "real_eta");
			std::vector<double> nume_eta_copy(nume_eta.data(), nume_eta.data() + nume_eta.size());
			writer.set_cell_data(nume_eta_copy, 1, "nume_eta");
			std::vector<double> uh_copy(uht.get_data().data(), uht.get_data().data() + uht.get_data().size());
			writer.set_point_data(uh_copy, 1, "uh");
			writer.write(ss.str());

			double relative_error = ETA / grad_u_norm;
			std::cout << "relative_error : " << relative_error << std::endl;
			if (relative_error < eTol || iter > 5+3) {

				last_NN_values.push_back(NN_values[NN_values.size() - 1]);
				last_H1_values.push_back(H1_values[H1_values.size() - 1]);
				last_RC_values.push_back(RC_values[RC_values.size() - 1]);
				break;
			}

			int flag;
			if (iter == 4 + 3) {
				Eigen::MatrixXd A = Eigen::MatrixXd::Ones(3, 2);
				Eigen::VectorXd b(3);
				for (int i = 0; i < 3; ++i) {
					A(i, 1) = log(NN_values[i + 2 + 3]);
					b(i) = log(RC_values[i + 2 + 3]/ grad_u_norm);
				}
				Eigen::VectorXd p = (A.transpose() * A).ldlt().solve(A.transpose() * b);
				int ideaN = std::floor(std::exp((std::log(eTol) - p(0)) / p(1)));
				flag = std::max(std::floor(std::log(ideaN / NN_values[iter]) / std::log(2)), 1.0);
				std::cout << "Expect " << ideaN << " nodes which require " << flag << " iterations." << std::endl;
			}
			else
				flag = 1;

			iter++;

			//nume_eta = nume_eta.cwiseMax(L2nume_eta / (dt * dt));
			ComputeDensity(mesht, nume_eta, size_field, flag);
			make_mesh(back_size_field, &mesht);
			size_field.update();
			uht.update();
		}

		// save to csv
		std::ofstream nodeFile("result.csv", std::ios_base::app);
		for (size_t i = 0; i < H1_values.size(); ++i) {
			nodeFile << t << "," << i << "," << NN_values[i] << "," << H1_values[i] << "," << RC_values[i];
			nodeFile << "\n";
		}
		nodeFile.close();

		if (times != NT) {
			meshtm1 = mesht;
			uhtm1 = uht;

			auto start = std::chrono::high_resolution_clock::now();
			std::cout << "Transfer Train Points to Tensor and Start Training Neural Network " << std::endl;
			auto nodes = mesht.nodes();
			auto values = uht.get_data();

			torch::Tensor x_train = torch::from_blob(nodes.row_data(), { nodes.shape(0), nodes.shape(1) }, torch::kDouble).to(torch::kFloat32);
			torch::Tensor y_train = torch::from_blob(values.data(), { values.size(), 1 }, torch::kDouble).to(torch::kFloat32);

			if (torch::cuda::is_available()) {
				std::cout << "CUDA is available! Training on GPU." << std::endl;
				net.to(torch::kCUDA);
				x_train = x_train.to(torch::kCUDA);
				y_train = y_train.to(torch::kCUDA);
			}
            net.train(x_train, y_train, train_epoch, learn_rate);
			//net.train_minibatch(x_train, y_train, train_epoch, learn_rate, mesht.number_of_nodes()/9);
			auto stop = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
			std::cout << "Train Net has expend : " << duration.count() << " seconds" << std::endl;
		}

		// print result output
		std::cout << "*********************************************************************************************" << std::endl;
		std::cout << "MESH iteration : "; for (double value : NN_values)  std::cout << value << " ";  std::cout << std::endl;
		std::cout << "ER_H1 iteration : ";  for (double value : H1_values) std::cout << value << " ";  std::cout << std::endl;
		std::cout << "ER_RC iteration : ";  for (double value : RC_values) std::cout << value << " ";  std::cout << std::endl;
		std::cout << "*********************************************************************************************" << std::endl;
	}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       

	// print result output
	std::cout << "*********************************************************************************************" << std::endl;
	std::cout << "MESH iteration : "; for (double value : last_NN_values)  std::cout << value << " ";  std::cout << std::endl;
	std::cout << "ER_H1 iteration : ";  for (double value : last_H1_values) std::cout << value << " ";  std::cout << std::endl;
	std::cout << "ER_RC iteration : ";  for (double value : last_RC_values) std::cout << value << " ";  std::cout << std::endl;
	std::cout << "*********************************************************************************************" << std::endl;
}

int main() {
	auto start = std::chrono::high_resolution_clock::now();
	equation_nn();
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
	std::cout << "times taken by function: " << duration.count() << " seconds" << std::endl;

	return 0;
}
