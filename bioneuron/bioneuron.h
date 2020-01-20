/*
 *  libbioneuron -- Library solving for synaptic weights
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
 * @file bioneuron.h
 *
 * Main header file providing the libbioneuron interface.
 *
 * @author Andreas Stöckel
 */

#ifndef BIONEURON_BIONEURON_H
#define BIONEURON_BIONEURON_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
	BN_ERR_OK = 0,
	BN_ERR_INVALID_N_PRE = -1,
	BN_ERR_INVALID_N_POST = -2,
	BN_ERR_INVALID_N_SAMPLES = -3,
	BN_ERR_INVALID_A_PRE = -4,
	BN_ERR_INVALID_J_POST = -5,
	BN_ERR_INVALID_MODEL_WEIGHTS = -6,
	BN_ERR_INVALID_CONNECTION_MATRIX_EXC = -7,
	BN_ERR_INVALID_CONNECTION_MATRIX_INH = -8,
	BN_ERR_INVALID_REGULARISATION = -9,
	BN_ERR_INVALID_SYNAPTIC_WEIGHTS_EXC = -10,
	BN_ERR_INVALID_SYNAPTIC_WEIGHTS_INH = -11,
	BN_ERR_INVALID_TOLERANCE = -12,
	BN_ERR_CANCEL = -13
} BioneuronError;

const char *bioneuron_strerr(BioneuronError err);

/**
 * Internally used Boolean type; this should be compatible with the C++ "bool"
 * type.
 */
typedef unsigned char BioneuronBool;

/**
 * The BioneuronWeightProblem structure describes the optimization problem for
 * computing the weights between two neuron populations.
 */
typedef struct {
	/**
	 * Number of neurons in the pre-population.
	 */
	int n_pre;

	/**
	 * Number of neurons in the post-population.
	 */
	int n_post;

	/**
	 * Number of samples.
	 */
	int n_samples;

	/**
	 * Pre-population activites for each sample as a n_samples x n_pre matrix.
	 * Should be in row-major order (i.e. the activtivities for all neurons for
	 * one sample are stored consecutively in memory).
	 */
	double *a_pre;

	/**
	 * Desired input currents for each sample as a n_samples x n_post matrix.
	 * Should be in row-major order (i.e. the desired input currents for all
	 * neurons for one sample are stored consecutively in memory).
	 */
	double *j_post;

	/**
	 * Neuron model weights as a n_post x 6 matrix (should be irow-major
	 * format).
	 */
	double *model_weights;

	/**
	 * Matrix determining the connectivity between pre- and post-neurons. This
	 * is a n_pre x n_post matrix.
	 */
	BioneuronBool *connection_matrix_exc;

	/**
	 * Matrix determining the connectivity between pre- and post-neurons. This
	 * is a n_pre x n_post matrix.
	 */
	BioneuronBool *connection_matrix_inh;

	/**
	 * Regularisation factor.
	 */
	double regularisation;

	/**
	 * Target neuron threshold current. Only valid if relax_sub_threshold is
	 * true.
	 */
	double j_threshold;

	/**
	 * Relax the optimization problem for subthreshold neurons.
	 */
	BioneuronBool relax_subthreshold;

	/**
	 * Ensure that synaptic weights are non-negative.
	 */
	BioneuronBool non_negative;

	/**
	 * Output memory region in which the resulting excitatory neural weights
	 * should be stored. This is a n_pre x n_post matrix.
	 */
	double *synaptic_weights_exc;

	/**
	 * Output memory region in which the resulting inhibitory neural weights
	 * should be stored. This is a n_pre x n_post matrix.
	 */
	double *synaptic_weights_inh;
} BioneuronWeightProblem;

/**
 * Creates a new BioneuronWeightProblem struct and fills it with some sane
 * default values.
 */
BioneuronWeightProblem *bioneuron_problem_create();

/**
 * Frees a previously created BioneuronWeightProblem struct. Does not free the
 * memory regions "problem" is pointing at.
 */
void bioneuron_problem_free(BioneuronWeightProblem *problem);

/**
 * Callback function type used to inform the caller about the current progress
 * of the weight solving process. Must return "true" if the computation should
 * continue; returning "false" cancels the computation and bioneuron_solve
 * will return BN_ERR_CANCEL.
 */
typedef bool (*BioneuronProgressCallback)(int n_done, int n_total);

/**
 * Callback function type used to inform the caller about any warnings that
 * pop up during the weight solving process.
 */
typedef void (*BioneuronWarningCallback)(const char *msg, int i_post);

/**
 * Parameters used when solving for weights.
 */
typedef struct {
	BioneuronBool renormalise;
	double tolerance;
	BioneuronProgressCallback progress;
	BioneuronWarningCallback warn;
	int n_threads;
} BioneuronSolverParameters;

BioneuronSolverParameters *bioneuron_solver_parameters_create();

void bioneuron_solver_parameters_free(BioneuronSolverParameters *params);

BioneuronError bioneuron_solve(BioneuronWeightProblem *problem,
                               BioneuronSolverParameters *params);

#ifdef __cplusplus
}
#endif

#endif /* BIONEURON_BIONEURON_H */
