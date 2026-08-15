#include <vector>
#include "LinearAlgebro.h"

namespace OF {
	namespace FEM_EXAMPLE {

		// Weighted Average Method
		template<typename MESH>
		void WAV(MESH* mesh, const Matrix& Guh, Matrix& Ghuh) {
			auto& cell = mesh->cells();
			int NN = mesh->number_of_nodes();
			int NC = mesh->number_of_cells();
			std::vector<double> cellvol;
			mesh->cell_volume(cellvol);

			std::vector<double> dudxVolume(NN, 0.0), dudyVolume(NN, 0.0), dudzVolume(NN, 0.0);
			std::vector<double> patchVolume(NN, 0.0);

			for (int i = 0; i < NC; ++i) {
				for (int j = 0; j < 4; ++j) {
					patchVolume[cell(i, j)] += 1 / cellvol[i];
					dudxVolume[cell(i, j)] += Guh(i, 0) / cellvol[i];
					dudyVolume[cell(i, j)] += Guh(i, 1) / cellvol[i];
					dudzVolume[cell(i, j)] += Guh(i, 2) / cellvol[i];
				}
			}

			Ghuh.resize(NN, 3);
			for (int i = 0; i < NN; ++i) {
				Ghuh(i, 0) = dudxVolume[i] / patchVolume[i];
				Ghuh(i, 1) = dudyVolume[i] / patchVolume[i];
				Ghuh(i, 2) = dudzVolume[i] / patchVolume[i];
			}
		}

		// Gradient Reconstruction Method
		template<typename MESH>
		void SPR(MESH* mesh, const Matrix& Guh, Matrix& Ghuh) {
			auto & mesh_node = mesh->nodes();
			auto & cell = mesh->cells();
			int NN = mesh->number_of_nodes();
			int NC = mesh->number_of_cells();
			std::vector<bool> bdflag;
			mesh->boundary_node_flag(bdflag);

			Matrix node(NN, 3);
			for(int i = 0; i < NN; i++)
			{
			  node(i, 0) = mesh_node[i][0];
			  node(i, 1) = mesh_node[i][1];
			  node(i, 2) = mesh_node[i][2];
			}

			// Assembly p2t matrix
			CSRMatrix p2t(NN, NC);
			std::vector<Triplet<double>> tls;
			tls.reserve(NC * 4);
			for (int i = 0; i < NC; ++i) {
			    for (int j = 0; j < 4; ++j)
			        tls.push_back({ cell(i, j), i, 1.0 });
			}
			p2t.setFromTriplets(tls.begin(), tls.end());
			CSRMatrix p2p = p2t * p2t.transpose();

			// Calculate the barycentric coordinates of each cell
			Matrix xnode(NC, 3);
			for (int i = 0; i < NC; ++i) {
				xnode(i, 0) = (node(cell(i, 0), 0) + node(cell(i, 1), 0) + node(cell(i, 2), 0) + node(cell(i, 3), 0)) / 4;
				xnode(i, 1) = (node(cell(i, 0), 1) + node(cell(i, 1), 1) + node(cell(i, 2), 1) + node(cell(i, 3), 1)) / 4;
				xnode(i, 2) = (node(cell(i, 0), 2) + node(cell(i, 1), 2) + node(cell(i, 2), 2) + node(cell(i, 3), 2)) / 4;
			}			        

			Ghuh.resize(NN, 3);
			Ghuh.setZero();
			for (int i = 0; i < NN; ++i) {
			    // Find all cells that contain the inner point i
			    std::vector<int> ne;
			    for (CSRMatrix::InnerIterator it(p2t, i); it; ++it) {
			        ne.push_back(it.col());
			    }

				Matrix tempp(ne.size(), Guh.cols());	// Find the gradient values of these cells
				Matrix tempx0(ne.size(), xnode.cols());	// Find the barycentric coordinates of these cells
				for (size_t kk = 0; kk < ne.size(); ++kk) {
					tempp.row(kk) = Guh.row(ne[kk]);
					tempx0.row(kk) = xnode.row(ne[kk]);
				}

			    if (bdflag[i]) {
			        // Find all point index with point i
			        std::vector<int> np;
			        for (CSRMatrix::InnerIterator it(p2p, i); it; ++it) {
			            np.push_back(it.col());
			        }
			        // Find inner point index with point i
			        std::vector<int> ip;
			        for (int idx : np) {
			            if (!bdflag[idx]) {
			                ip.push_back(idx);
			            }
			        }

			        int ipn = ip.size();
			        if (ipn == 0)
			            Ghuh.row(i) = tempp.colwise().mean();   // No inner point connected
			        else {
			            for (int k = 0; k < ipn; ++k) {
			                // Find all cells that contain the inner point ip[k]
			                std::vector<int> ne;
			                for (CSRMatrix::InnerIterator it(p2t, ip[k]); it; ++it)
			                    ne.push_back(it.col());

							Matrix tempp(ne.size(), Guh.cols());	// Find the gradient values of these cells
							Matrix tempx0(ne.size(), xnode.cols());	// Find the barycentric coordinates of these cells
							for (size_t kk = 0; kk < ne.size(); ++kk) {
								tempp.row(kk) = Guh.row(ne[kk]);
								tempx0.row(kk) = xnode.row(ne[kk]);
							}

			                // Linear mapping
			                LVector center = tempx0.colwise().mean();
			                Matrix diff = tempx0.rowwise() - center.transpose();
			                double h = 0.1 * std::sqrt((diff.array().square().colwise().sum()).maxCoeff());
			                Matrix tempx = diff / h;

			                // Least square linear regression
			                Matrix X = Matrix::Ones(ne.size(), 4);
			                X.block(0, 1, ne.size(), 3) = tempx;
			                LVector coefficient1 = (X.transpose() * X).ldlt().solve(X.transpose() * tempp.col(0));
			                Ghuh(0, 0) += (node.row(i) - center.transpose()) / h * coefficient1.segment(1, 3) + coefficient1(0);
			                LVector coefficient2 = (X.transpose() * X).ldlt().solve(X.transpose() * tempp.col(1));
			                Ghuh(0, 1) += (node.row(i) - center.transpose()) / h * coefficient2.segment(1, 3) + coefficient2(0);
			                LVector coefficient3 = (X.transpose() * X).ldlt().solve(X.transpose() * tempp.col(2));
			                Ghuh(0, 2) += (node.row(i) - center.transpose()) / h * coefficient3.segment(1, 3) + coefficient3(0);
			            }
			            Ghuh.row(i) = Ghuh.row(i) / ipn;
			        }
			    }
			    else {
			        // Find the barycentric coordinates of these cells
			        Matrix tempx0(ne.size(), xnode.cols());
			        for (size_t kk = 0; kk < ne.size(); ++kk)
			            tempx0.row(kk) = xnode.row(ne[kk]);

			        // Linear mapping
			        LVector center = tempx0.colwise().mean();
			        Matrix diff = tempx0.rowwise() - center.transpose();
			        double h = 0.1 * std::sqrt((diff.array().square().colwise().sum()).maxCoeff());
			        Matrix tempx = diff / h;

			        // Least square linear regression
			        Matrix X = Matrix::Ones(ne.size(), 4);
			        X.block(0, 1, ne.size(), 3) = tempx;
			        LVector coefficient1 = (X.transpose() * X).ldlt().solve(X.transpose() * tempp.col(0));
			        Ghuh(i, 0) = (node.row(i) - center.transpose()) / h * coefficient1.segment(1, 3) + coefficient1(0);
			        LVector coefficient2 = (X.transpose() * X).ldlt().solve(X.transpose() * tempp.col(1));
			        Ghuh(i, 1) = (node.row(i) - center.transpose()) / h * coefficient2.segment(1, 3) + coefficient2(0);
			        LVector coefficient3 = (X.transpose() * X).ldlt().solve(X.transpose() * tempp.col(2));
			        Ghuh(i, 2) = (node.row(i) - center.transpose()) / h * coefficient3.segment(1, 3) + coefficient3(0);
			    }
			}
		}

		/**
		 * @brief 对 uh 梯度重构，得到 ghuh, ghuh 是三个函数组成，分别是 ghuh
		 * 的三个坐标分量。然后计算 ghuh 和 uh 之间的误差得到 eta
		 * @return eta : 每个单元上的重构型后验误差
		 */
		template<typename FUNC>
		void error_estimator(FUNC& uh, LVector& eta)
		{
			auto space = uh.get_space();
			auto mesh = space->get_mesh();

			LVector guhx, guhy, guhz;
			space->grad_value(uh, guhx, guhy, guhz);

			Matrix guh, ghuh;
			guh.resize(guhx.size(), 3);
			guh.col(0) = guhx;
			guh.col(1) = guhy;
			guh.col(2) = guhz;

			WAV(mesh, guh, ghuh);

			auto ghuhx = space->function();
			auto ghuhy = space->function();
			auto ghuhz = space->function();
			ghuhx.get_data() = ghuh.col(0);
			ghuhy.get_data() = ghuh.col(1);
			ghuhz.get_data() = ghuh.col(2);

			std::function<void(double, double, double, std::array<double, 3>&)> ff =
				[&ghuhx, &ghuhy, ghuhz](double x, double y, double z, std::array<double, 3>& gval)->void
				{
					std::array<double, 3> p = { x, y, z };
					gval[0] = ghuhx(p);
					gval[1] = ghuhy(p);
					gval[2] = ghuhz(p);
				};
			space->H1_error(ff, uh, eta);
		}
	}
}