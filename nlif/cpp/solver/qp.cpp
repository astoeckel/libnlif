/*
 *  libnlif -- Multi-compartment LIF simulator and weight solver
 *  Copyright (C) 2017-2021  Andreas Stöckel
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "osqp/osqp.h"
#include "qp.hpp"

using namespace Eigen;

namespace {
class CSCMatrix {
private:
	csc m_csc;

public:
	CSCMatrix(SpMatrixXd &mat)
	{
		mat.makeCompressed();
		m_csc.m = mat.rows();
		m_csc.n = mat.cols();
		m_csc.p = mat.outerIndexPtr();
		m_csc.i = mat.innerIndexPtr();
		m_csc.x = mat.valuePtr();
		m_csc.nzmax = mat.nonZeros();
		m_csc.nz = -1;
	}

	operator csc *() { return &m_csc; }
};
}

QPResult solve_qp(SpMatrixXd &P, VectorXd &q, SpMatrixXd &G, VectorXd &h,
                  double tol, int max_iter)
{
	// Convert the P and G matrix into sparse CSC matrices
	CSCMatrix Pcsc(P), Gcsc(G);

#ifdef BQP_DEBUG
	std::cout << "P =\n" << MatrixXd(P).format(CleanFmt) << std::endl;
	std::cout << "q =\n" << MatrixXd(q).format(CleanFmt) << std::endl;
#endif

	// Generate a lower bound matrix
	VectorXd l =
	    VectorXd::Ones(G.rows()) * std::numeric_limits<c_float>::lowest();

	// Populate data
	OSQPData data;
	data.n = P.rows();
	data.m = G.rows();
	data.P = Pcsc;
	data.q = const_cast<c_float *>(q.data());
	data.A = Gcsc;
	data.l = const_cast<c_float *>(l.data());
	data.u = const_cast<c_float *>(h.data());

	// Define solver settings as default
	OSQPSettings settings;
	osqp_set_default_settings(&settings);
	settings.scaling = 0;
	settings.scaled_termination = 0;
	settings.rho = 1e-1;  // Default value
	settings.eps_rel = tol;
	settings.eps_abs = tol;
	settings.polish = true;
	settings.polish_refine_iter = 3;  // Default value
	if (max_iter > 0) {
		settings.max_iter = max_iter;
	}

	// Setup workspace
	QPResult res;
	OSQPWorkspace *work = nullptr;
	res.status = osqp_setup(&work, &data, &settings);
	if (res.status != 0) {
		return res;
	}

	// Solve the problem
	res.status = osqp_solve(work);
	if (res.status == 0 && work->info->status_val < 0) {
		res.status = work->info->status_val;
	}
#ifdef BQP_DEBUG
	std::cout << "res.status = " << work->info->status
	          << " res.status_polish = " << work->info->status_polish
	          << std::endl;
#endif

	// Copy the results to the output arrays
	res.x = Map<VectorXd>(work->solution->x, P.rows());
	res.objective_val = work->info->obj_val;

	// Cleanup the workspace
	osqp_cleanup(work);

	return res;
}

