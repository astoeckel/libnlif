/*
 *  libbioneuronqp -- Library solving for synaptic weights
 *  Copyright (C) 2020  Andreas Stöckel
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Affero General Public License as
 *  published by the Free Software Foundation, either version 3 of the
 *  License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Affero General Public License for more details.
 *
 *  You should have received a copy of the GNU Affero General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

/**
 * @file bioneuron.c
 *
 * Actual implementation of libioneuron.
 *
 * @author Andreas Stöckel
 */

#include "bioneuronqp.h"

#include <osqp/osqp.h>

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

#include "threadpool.hpp"

using namespace Eigen;

/******************************************************************************
 * INTERNAL C++ CODE                                                          *
 ******************************************************************************/

using MatrixMap = Map<Matrix<double, Dynamic, Dynamic, Eigen::RowMajor>>;
using BoolMatrixMap =
    Map<Matrix<unsigned char, Dynamic, Dynamic, Eigen::RowMajor>>;
using BoolVector = Matrix<unsigned char, Dynamic, 1>;

namespace {
/**
 * This class represents a matrix in the Compressed Sparse Column (CSC) format
 * using zero-based indexing.
 *
 * See https://people.sc.fsu.edu/~jburkardt/data/cc/cc.html for a description of
 * the format.
 */
struct CSCMatrix {
private:
	csc m_csc;
	std::vector<c_int> m_col;
	std::vector<c_int> m_row;
	std::vector<c_float> m_x;

public:
	CSCMatrix(const MatrixXd &mat, bool upper_triangle = false)
	{
		c_int m = mat.rows(), n = mat.cols();

		m_col.push_back(0);
		for (c_int j = 0; j < n; j++) {
			for (c_int i = 0; i < (upper_triangle ? (j + 1) : m); i++) {
				if (mat(i, j) != 0.0) {
					m_row.push_back(i);
					m_x.push_back(mat(i, j));
				}
			}
			m_col.push_back(m_x.size());
		}

		m_csc.nzmax = m * n;
		m_csc.m = m;
		m_csc.n = n;
		m_csc.p = m_col.data();
		m_csc.i = m_row.data();
		m_csc.x = m_x.data();
		m_csc.nzmax = m_x.size();
		m_csc.nz = -1;
	}

	operator csc *() { return &m_csc; }
};

VectorXd _solve_qp(const MatrixXd &P, const VectorXd &q, const MatrixXd &G,
                   const VectorXd &h, double tol)
{
	// Convert the P and G matrix into sparse CSC matrices
	CSCMatrix Pcsc(P, true), Gcsc(G);

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
	settings.scaling = 1;
	settings.scaled_termination = 1;
	settings.eps_rel = tol;
	settings.eps_abs = tol;

	// Setup workspace
	OSQPWorkspace *work = nullptr;
	/*int err =*/osqp_setup(&work, &data, &settings);
	// XXX: Handle errors?

	// Solve the problem
	osqp_solve(work);

	// Copy the results to the output arrays
	VectorXd res = Map<VectorXd>(work->solution->x, P.rows());

	// Cleanup the workspace
	osqp_cleanup(work);

	return res;
}

VectorXd _solve_linearly_constrained_quadratic_loss(const MatrixXd &C,
                                                    const VectorXd &d,
                                                    const MatrixXd &G,
                                                    const VectorXd &h,
                                                    double tol)
{
	// Compute the symmetric "P" matrix for the QP problem
	MatrixXd Pqp = C.transpose() * C;
	VectorXd qqp = -C.transpose() * d;

	// Solve the QP problem
	return _solve_qp(Pqp, qqp, G, h, tol);
}

VectorXd _solve_weights_qp(const MatrixXd &A, const VectorXd &b,
                           const BoolVector &valid, double i_th, double reg,
                           double tol, bool nonneg)
{
	//
	// Step 1: Count stuff and setup indices used to partition the matrices
	//         into smaller parts.
	//

	// Compute the number of slack variables required to solve this problem
	const size_t n_cstr = A.rows();
	const size_t n_vars = A.cols();
	const size_t n_cstr_valid = valid.sum();
	const size_t n_cstr_invalid = n_cstr - n_cstr_valid;
	const size_t n_slack = n_cstr_invalid;

	// Variables
	const size_t v0 = 0;
	const size_t v1 = v0 + n_vars;
	const size_t v2 = v1 + n_slack;

	// Quadratic constraints
	const size_t a0 = 0;
	const size_t a1 = a0 + n_cstr_valid;
	const size_t a2 = a1 + n_vars;
	const size_t a3 = a2 + n_slack;

	// Inequality constraints
	const size_t g0 = 0;
	const size_t g1 = g0 + n_cstr_invalid;
	const size_t g2 = g1 + (nonneg ? n_vars : 0);

	//
	// Step 2: Assemble the QP matrices
	//

	// We need balance the regularisation error for the super- (valid) and
	// sub-threshold (invalid) constraints. This is done by dividing by the
	// number of valid/invalid constraints. We need to multiply with the number
	// of constraints since the regularisation factor has been chosen in such a
	// way that the errors are implicitly divided by the number of constraints.
	const double m1 =
	    std::sqrt(double(n_cstr) / std::max<double>(1.0, n_cstr_valid));
	const double m2 =
	    std::sqrt(double(n_cstr) / std::max<double>(1.0, n_cstr_invalid));

	// Copy the valid constraints to Aext
	MatrixXd Aext = MatrixXd::Zero(a3, v2);
	VectorXd bext = VectorXd::Zero(a3);
	size_t i_valid = 0;
	for (size_t i = 0; i < n_cstr; i++) {
		if (valid[i]) {
			for (size_t j = 0; j < n_vars; j++) {
				Aext(a0 + i_valid, v0 + j) = A(i, j) * m1;
			}
			bext[i_valid] = b[i] * m1;
			i_valid++;
		}
	}

	// Regularise the weights
	reg = std::sqrt(reg);
	for (size_t i = 0; i < n_vars; i++) {
		Aext(a1 + i, v0 + i) = reg;
	}

	// Penalise slack variables
	for (size_t i = 0; i < n_slack; i++) {
		Aext(a2 + i, v1 + i) = m2;
	}

	// Form the inequality constraints
	MatrixXd G = MatrixXd::Zero(g2, v2);
	VectorXd h = VectorXd::Zero(g2);
	i_valid = 0;
	for (size_t i = 0; i < n_cstr; i++) {
		if (!valid[i]) {
			for (size_t j = 0; j < n_vars; j++) {
				G(g0 + i_valid, v0 + j) = A(i, j);
			}
			h[i_valid] = i_th;
			i_valid++;
		}
	}
	for (size_t i = 0; i < n_slack; i++) {
		G(g0 + i, v1 + i) = -1;
	}
	if (nonneg) {
		for (size_t i = 0; i < n_vars; i++) {
			G(g1 + i, v0 + i) = -1;
		}
	}

	//
	// Step 3: Sovle the QP
	//
	return _solve_linearly_constrained_quadratic_loss(Aext, bext, G, h, tol);
}

void _bioneuronqp_solve_single(BioneuronWeightProblem *problem,
                               BioneuronSolverParameters *params, size_t j)
{
	// Copy some input parameters as convenient aliases
	size_t Npre = problem->n_pre;
	size_t Nsamples = problem->n_samples;

	// Copy some relevant input matrices
	MatrixMap APre(problem->a_pre, problem->n_samples, problem->n_pre);

	MatrixMap JPost(problem->j_post, problem->n_samples, problem->n_post);

	MatrixMap Ws(problem->model_weights, problem->n_post, 6);
	Matrix<double, 6, 1> ws = Ws.row(j);

	BoolMatrixMap ConExc(problem->connection_matrix_exc, problem->n_pre,
	                     problem->n_post);
	BoolMatrixMap ConInh(problem->connection_matrix_inh, problem->n_pre,
	                     problem->n_post);

	// Fetch the output weights
	MatrixMap WExc(problem->synaptic_weights_exc, problem->n_pre,
	               problem->n_post);
	MatrixMap WInh(problem->synaptic_weights_inh, problem->n_pre,
	               problem->n_post);

	// Count the number of excitatory and inhibitory pre-neurons; also reset the
	// output weights for this post neuron
	size_t Npre_exc = 0, Npre_inh = 0;
	for (size_t i = 0; i < Npre; i++) {
		if (ConExc(i, j)) {
			Npre_exc++;
		}
		if (ConInh(i, j)) {
			Npre_inh++;
		}
		WExc(i, j) = 0.0;
		WInh(i, j) = 0.0;
	}

	// Compute the total number of pre neurons. We're done if there are no pre
	// neurons.
	const size_t Npre_tot = Npre_exc + Npre_inh;
	if (Npre_tot == 0) {
		return;
	}

	// Renormalise the target currents
	double Wscale = 1.0, LambdaScale = 1.0;
	if (params->renormalise) {
		// Need to scale the regularisation factor as well
		LambdaScale = 1.0 / (ws[1] * ws[1]);

		// Compute synaptic weights in nS
		Wscale = 1e-9;
		ws[1] *= Wscale;
		ws[2] *= Wscale;
		ws[4] *= Wscale;
		ws[5] *= Wscale;

		// Set ws[1]=1 for better numerical stability/conditioning
		ws /= ws[1];
	}

	// Account for the number of samples in the regularisation factor
	LambdaScale *= double(Nsamples);

	// Demangle the weight vector
	double a0 = ws[0], a1 = ws[1], a2 = ws[2], b0 = ws[3], b1 = ws[4],
	       b2 = ws[5];

	// Warn if some weights are out of range
	if (params->warn && std::abs(b2) > 0.0 && std::abs(b1) > 0.0) {
		const double jPostMax = JPost.col(j).array().maxCoeff();
		if ((a1 / b1) < jPostMax) {
			std::stringstream ss;
			ss << "Target currents for neuron " << j << " cannot be reached! "
			   << jPostMax << " ∉ [" << (a2 / b2) << ", " << (a1 / b2) << "]";
			params->warn(ss.str().c_str(), j);
		}
	}

	// Assemble the "A" matrix for the least squares problem
	MatrixXd A(Nsamples, Npre_tot);
	size_t i_pre_exc = 0, i_pre_inh = Npre_exc;
	for (size_t i = 0; i < Npre; i++) {
		if (ConExc(i, j)) {
			for (size_t k = 0; k < Nsamples; k++) {
				A(k, i_pre_exc) = (a1 - b1 * JPost(k, j)) * APre(k, i);
			}
			i_pre_exc++;
		}
		if (ConInh(i, j)) {
			for (size_t k = 0; k < Nsamples; k++) {
				A(k, i_pre_inh) = (a2 - b2 * JPost(k, j)) * APre(k, i);
			}
			i_pre_inh++;
		}
	}

	// Assemble the "b" matrix for the least squares problem
	MatrixXd b = JPost.col(j).array() * b0 - a0;

	// Determine which target currents are valid and which target currents are
	// not
	Matrix<unsigned char, Dynamic, 1> valid =
	    Matrix<unsigned char, Dynamic, 1>::Ones(Nsamples);
	if (problem->relax_subthreshold) {
		for (size_t i = 0; i < Nsamples; i++) {
			if (JPost(i, j) < problem->j_threshold) {
				valid(i) = 0;
			}
		}
	}

	// Solve the quadratic programing problem
	const double i_th = problem->j_threshold * b0 - a0;
	const double reg = problem->regularisation * LambdaScale;
	const double tol = params->tolerance;
	VectorXd W =
	    _solve_weights_qp(A, b, valid, i_th, reg, tol, problem->non_negative);

	// Distribute the resulting weights back to their correct locations
	i_pre_exc = 0, i_pre_inh = Npre_exc;
	for (size_t i = 0; i < Npre; i++) {
		if (ConExc(i, j)) {
			WExc(i, j) = W[i_pre_exc++] * Wscale;
		}
		if (ConInh(i, j)) {
			WInh(i, j) = W[i_pre_inh++] * Wscale;
		}
	}
}

BioneuronError _bioneuronqp_solve(BioneuronWeightProblem *problem,
                                  BioneuronSolverParameters *params)
{
	// Construct the kernal that is being executed -- here, we're solving the
	// weights for a single post-neuron.
	auto kernel = [&](size_t idx) {
		_bioneuronqp_solve_single(problem, params, idx);
	};

	// Construct the progress callback
	bool did_cancel = false;
	auto progress = [&](size_t cur, size_t max) {
		if (params->progress) {
			if (!params->progress(cur, max)) {
				did_cancel = true;
			}
			return !did_cancel;
		}
		return true;
	};

	// Create a threadpool and solve the weights for all neurons
	Threadpool pool(params->n_threads);
	pool.run(problem->n_post, kernel, progress);

	return did_cancel ? BN_ERR_CANCEL : BN_ERR_OK;
}

}  // namespace

/******************************************************************************
 * EXTERNAL C API                                                             *
 ******************************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 * Enum BioneuronError                                                        *
 ******************************************************************************/

const char *bioneuronqp_strerr(BioneuronError err)
{
	switch (err) {
		case BN_ERR_OK:
			return "no error";
		case BN_ERR_INVALID_N_PRE:
			return "n_pre is invalid";
		case BN_ERR_INVALID_N_POST:
			return "n_post is invalid";
		case BN_ERR_INVALID_N_SAMPLES:
			return "n_samples is invalid";
		case BN_ERR_INVALID_A_PRE:
			return "a_pre is invalid";
		case BN_ERR_INVALID_J_POST:
			return "j_post is invalid";
		case BN_ERR_INVALID_MODEL_WEIGHTS:
			return "model_weights is invalid";
		case BN_ERR_INVALID_CONNECTION_MATRIX_EXC:
			return "connection_matrix_exc is invalid";
		case BN_ERR_INVALID_CONNECTION_MATRIX_INH:
			return "connection_matrix_inh is invalid";
		case BN_ERR_INVALID_REGULARISATION:
			return "regularisation is invalid";
		case BN_ERR_INVALID_SYNAPTIC_WEIGHTS_EXC:
			return "synaptic_weights_exc is invalid";
		case BN_ERR_INVALID_SYNAPTIC_WEIGHTS_INH:
			return "synaptic_weights_inh is invalid";
		case BN_ERR_INVALID_TOLERANCE:
			return "tolerance is invalid";
		case BN_ERR_CANCEL:
			return "canceled by user";
	}
	return "unknown error code";
}

/******************************************************************************
 * Struct BioneuronWeightProblem                                              *
 ******************************************************************************/

#define DEFAULT_REGULARISATION 1e-1
#define DEFAULT_TOLERANCE 1e-6;

BioneuronWeightProblem *bioneuronqp_problem_create()
{
	BioneuronWeightProblem *problem = new BioneuronWeightProblem;
	problem->n_pre = 0;
	problem->n_post = 0;
	problem->n_samples = 0;
	problem->a_pre = nullptr;
	problem->j_post = nullptr;
	problem->model_weights = nullptr;
	problem->connection_matrix_exc = nullptr;
	problem->connection_matrix_inh = nullptr;
	problem->regularisation = DEFAULT_REGULARISATION;
	problem->j_threshold = 0.0;
	problem->relax_subthreshold = 0;
	problem->non_negative = 0;
	problem->synaptic_weights_exc = 0;
	problem->synaptic_weights_inh = 0;
	return problem;
}

void bioneuronqp_problem_free(BioneuronWeightProblem *problem)
{
	delete problem;
}

/******************************************************************************
 * Struct BioneuronSolverParameters                                           *
 ******************************************************************************/

BioneuronSolverParameters *bioneuronqp_solver_parameters_create()
{
	BioneuronSolverParameters *params = new BioneuronSolverParameters;
	params->renormalise = 1;
	params->tolerance = DEFAULT_TOLERANCE;
	params->progress = nullptr;
	params->warn = nullptr;
	params->n_threads = 0;
	return params;
}

void bioneuronqp_solver_parameters_free(BioneuronSolverParameters *params)
{
	delete params;
}

/******************************************************************************
 * Actual solver code                                                         *
 ******************************************************************************/

static BioneuronError _check_problem_is_valid(BioneuronWeightProblem *problem)
{
	if (problem->n_pre <= 0) {
		return BN_ERR_INVALID_N_PRE;
	}
	if (problem->n_post <= 0) {
		return BN_ERR_INVALID_N_PRE;
	}
	if (problem->n_samples <= 0) {
		return BN_ERR_INVALID_N_SAMPLES;
	}
	if (problem->a_pre == nullptr) {
		return BN_ERR_INVALID_A_PRE;
	}
	if (problem->j_post == nullptr) {
		return BN_ERR_INVALID_J_POST;
	}
	if (problem->model_weights == nullptr) {
		return BN_ERR_INVALID_MODEL_WEIGHTS;
	}
	if (problem->connection_matrix_exc == nullptr) {
		return BN_ERR_INVALID_CONNECTION_MATRIX_EXC;
	}
	if (problem->connection_matrix_inh == nullptr) {
		return BN_ERR_INVALID_CONNECTION_MATRIX_INH;
	}
	if (problem->regularisation < 0.0) {
		return BN_ERR_INVALID_REGULARISATION;
	}
	if (problem->synaptic_weights_exc == nullptr) {
		return BN_ERR_INVALID_SYNAPTIC_WEIGHTS_EXC;
	}
	if (problem->synaptic_weights_inh == nullptr) {
		return BN_ERR_INVALID_SYNAPTIC_WEIGHTS_INH;
	}
	return BN_ERR_OK;
}

static BioneuronError _check_parameters_is_valid(
    BioneuronSolverParameters *params)
{
	if (params->tolerance <= 0.0) {
		return BN_ERR_INVALID_TOLERANCE;
	}
	return BN_ERR_OK;
}

BioneuronError bioneuronqp_solve(BioneuronWeightProblem *problem,
                                 BioneuronSolverParameters *params)
{
	// Make sure the given pointers point at valid problem and parameter
	// descriptors
	BioneuronError err;
	if ((err = _check_problem_is_valid(problem)) < 0) {
		return err;
	}
	if ((err = _check_parameters_is_valid(params)) < 0) {
		return err;
	}

	// Forward the parameters and the problem description to the internal C++
	// code.
	return _bioneuronqp_solve(problem, params);
}

#ifdef __cplusplus
}
#endif