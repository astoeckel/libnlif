#!/usr/bin/env python3

#  libbioneuronqp -- Library solving for synaptic weights
#  Copyright (C) 2020  Andreas Stöckel
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as
#  published by the Free Software Foundation, either version 3 of the
#  License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import ctypes
import numpy as np
import sys
import signal

class BioneuronWeightProblem(ctypes.Structure):
    _fields_ = [
        ("n_pre", ctypes.c_int),
        ("n_post", ctypes.c_int),
        ("n_samples", ctypes.c_int),
        ("a_pre", ctypes.POINTER(ctypes.c_double)),
        ("j_post", ctypes.POINTER(ctypes.c_double)),
        ("model_weights", ctypes.POINTER(ctypes.c_double)),
        ("connection_matrix_exc", ctypes.POINTER(ctypes.c_ubyte)),
        ("connection_matrix_inh", ctypes.POINTER(ctypes.c_ubyte)),
        ("regularisation", ctypes.c_double),
        ("j_threshold", ctypes.c_double),
        ("relax_subthreshold", ctypes.c_ubyte),
        ("non_negative", ctypes.c_ubyte),
        ("synaptic_weights_exc", ctypes.POINTER(ctypes.c_double)),
        ("synaptic_weights_inh", ctypes.POINTER(ctypes.c_double)),
        ("objective_vals", ctypes.POINTER(ctypes.c_double)),
    ]


PBioneuronWeightProblem = ctypes.POINTER(BioneuronWeightProblem)

BioneuronProgressCallback = ctypes.CFUNCTYPE(ctypes.c_ubyte, ctypes.c_int,
                                             ctypes.c_int)

BioneuronWarningCallback = ctypes.CFUNCTYPE(None, ctypes.c_char_p,
                                            ctypes.c_int)


def default_progress_callback(n_done, n_total):
    sys.stderr.write("\rSolved {}/{} neuron weights".format(
        n_done, n_total))
    sys.stderr.flush()
    return True

def default_warning_callback(msg, idx):
    print("WARN: " + str(msg, "utf-8"))


class BioneuronSolverParameters(ctypes.Structure):
    _fields_ = [
        ("renormalise", ctypes.c_ubyte),
        ("tolerance", ctypes.c_double),
        ("max_iter", ctypes.c_int),
        ("progress", BioneuronProgressCallback),
        ("warn", BioneuronWarningCallback),
        ("n_threads", ctypes.c_int),
    ]


PBioneuronSolverParameters = ctypes.POINTER(BioneuronSolverParameters)

DLLFound = None
DLL = None


def _load_dll():
    """
    Helper function used internally to load the C shared library.
    """

    # Load the DLL if this has not been done yet
    global DLL, DLLFound
    if DLLFound is None:
        DLL = None
        try:
            DLL = ctypes.CDLL("libbioneuronqp.so")
        except OSError:
            pass
        if DLL is None:
            DLLFound = False
            return None
        else:
            DLLFound = True

        DLL.bioneuronqp_strerr.argtypes = [ctypes.c_int]
        DLL.bioneuronqp_strerr.restype = ctypes.c_char_p

        DLL.bioneuronqp_solve.argtypes = [
            PBioneuronWeightProblem, PBioneuronSolverParameters
        ]
        DLL.bioneuronqp_solve.restype = ctypes.c_int

    return DLL

class SigIntHandler:
    """
    Class responsible for canceling the neuron weight solving process whenever
    the SIGINT event is received.
    """

    def __init__(self):
        self._old_handler = None
        self._args = None
        self.triggered = False

    def __enter__(self):
        def handler(*args):
            self._args = args
            self.triggered = True
        try:
            self._old_handler = signal.signal(signal.SIGINT, handler)
        except ValueError:
            # Ignore errors -- this is most likely triggered when
            # signal.signal is not called from the main thread. This is e.g.
            # what happens in Nengo GUI.
            pass
        return self

    def __exit__(self, type, value, traceback):
        if self._old_handler:
            signal.signal(signal.SIGINT, self._old_handler)
            if self.triggered:
                self._old_handler(*self._args)
            self._old_handler = None

def solve(Apre,
          Jpost,
          ws,
          connection_matrix=None,
          iTh=None,
          nonneg=True,
          renormalise=True,
          tol=None,
          reg=None,
          use_lstsq=False,
          return_objective_vals=False,
          progress_callback=default_progress_callback,
          warning_callback=default_warning_callback,
          n_threads=0,
          max_iter=0):
    """
    Solves a synaptic weight qp problem.


    Parameters
    ==========

    Apre:
        Pre-synaptic activities. Must be a n_samples x n_pre matrix.
    Jpost:
        Desired post-synaptic currents. Must be a n_samples x n_post matrix.
    ws:
        Neuron model parameters of the form
                    b0 + b1 * gExc + b2 * gInh
            Jequiv = --------------------------
                    a0 + a1 * gExc + a2 * gInh
        where ws = [b0, b1, b2, a0, a1, a2]. Use [0, 1, -1, 1, 0, 0] for a
        standard current-based LIF neuron.
    connection_matrix:
        A binary 2 x n_pre x n_post matrix determining which neurons are
        excitatorily and which are inhibitorily connected.
    iTh:
        Threshold current. The optimization problem is relaxed for currents
        below the given value. If "None", relaxation is deactivated.
    nonneg:
        Whether synaptic weights should be nonnegative
    renormalise:
        Whether the problem should be renormalised. Only set this to true if the
        target currents are in plausible biological scales (i.e. currents in pA
        to nA).
    tol:
        Solver tolerance. Default is 1e-6
    reg:
        Regularisation. Default is 1e-1
    use_lstsq:
        Ignored. For compatibility with the nengo_bio internal solver.
    return_objective_vals:
        If true, returns the achieved objective values.
    progress_callback:
        Function that is regularly being called with the current progress. May
        return "False" to cancel the solving process, must return "True"
        otherwise. Set to "None" to use no progress callback.
    warning_callback:
        Function that is being called whenever a warning is triggered. Set to
        "None" to use no progress callback.
    n_threads:
        Maximum number of threads to use when solving for weights. Uses all
        available CPU cores if set to zero.
    max_iter:
        Maximum number of iterations to take. The default (zero) means that no
        such limit exists.


    Returns
    =======

    Two matrices, WE, WI containing the weights for all post-neurons. If
    "return_objective_vals" is set to true, a third return value is returned,
    a vector of all objective values for the post-neurons.
    """

    # Load the solver library
    _dll = _load_dll()
    if _dll is None:
        raise OSError(
            "libbioneuronqp.so not found; make sure the library is located withing the library path"
        )

    # Set some default values
    if tol is None:
        tol = 1e-6
    if reg is None:
        reg = 1e-1

    # Fetch some counts
    assert Apre.shape[0] == Jpost.shape[0]
    Nsamples = Apre.shape[0]
    Npre = Apre.shape[1]
    Npost = Jpost.shape[1]

    # Use an all-to-all connection if connection_matrix is set to None
    if connection_matrix is None:
        connection_matrix = np.ones((2, Npre, Npost), dtype=np.bool)

    # Create a neuron model parameter vector for each neuron, if the parameters
    # are not already in this format
    assert ws.size == 6 or ws.ndim == 2, "Model weight vector must either be 6-element one-dimensional or a 2D matrix"
    if (ws.size == 6):
        ws = np.repeat(ws.reshape(1, -1), Npost, axis=0)
    else:
        assert ws.shape[0] == Npost and ws.shape[1] == 6, \
            "Invalid model weight matrix shape"

    # Make sure the connection matrix has the correct size
    assert connection_matrix.shape[0] == 2
    assert connection_matrix.shape[1] == Npre
    assert connection_matrix.shape[2] == Npost

    # Make sure all matrices are in the correct format
    c_a_pre = Apre.astype(dtype=np.float64, order='C', copy=False)
    c_j_post = Jpost.astype(dtype=np.float64, order='C', copy=False)
    c_connection_matrix_exc = connection_matrix[0].astype(dtype=np.uint8,
                                                          order='C',
                                                          copy=False)
    c_connection_matrix_inh = connection_matrix[1].astype(dtype=np.uint8,
                                                          order='C',
                                                          copy=False)
    c_model_weights = ws.astype(dtype=np.float64, order='C', copy=False)
    c_we = np.zeros((Npre, Npost), dtype=np.float, order='C')
    c_wi = np.zeros((Npre, Npost), dtype=np.float, order='C')
    if return_objective_vals:
        c_objective_vals = np.zeros((Npost,), dtype=np.float, order='C')

    # Matrix conversion helper functions
    def PDouble(mat):
        return mat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def PBool(mat):
        return mat.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    # Assemble the weight solver problem
    problem = BioneuronWeightProblem()
    problem.n_pre = Npre
    problem.n_post = Npost
    problem.n_samples = Nsamples
    problem.a_pre = PDouble(c_a_pre)
    problem.j_post = PDouble(c_j_post)
    problem.model_weights = PDouble(c_model_weights)
    problem.connection_matrix_exc = PBool(c_connection_matrix_exc)
    problem.connection_matrix_inh = PBool(c_connection_matrix_inh)
    problem.regularisation = reg
    problem.j_threshold = 0.0 if iTh is None else iTh
    problem.relax_subthreshold = not (iTh is None)
    problem.non_negative = nonneg
    problem.synaptic_weights_exc = PDouble(c_we)
    problem.synaptic_weights_inh = PDouble(c_wi)
    problem.objective_vals = PDouble(c_objective_vals) if return_objective_vals else None

    # Progress wrapper
    with SigIntHandler() as sig_int_handler:
        def progress_callback_wrapper(n_done, n_total):
            if sig_int_handler.triggered:
                return False
            if progress_callback:
                return progress_callback(n_done, n_total)
            return True

        params = BioneuronSolverParameters()
        params.renormalise = renormalise
        params.tolerance = tol
        params.max_iter = max_iter
        params.progress = BioneuronProgressCallback(progress_callback_wrapper)
        params.warn = BioneuronWarningCallback(
            0 if warning_callback is None else warning_callback)
        params.n_threads = n_threads

        # Actually run the solver
        err = _dll.bioneuronqp_solve(ctypes.pointer(problem),
                                    ctypes.pointer(params))

    if err != 0:
        raise RuntimeError(_dll.bioneuronqp_strerr(err))

    if return_objective_vals:
        return c_we, c_wi, c_objective_vals
    return c_we, c_wi


###############################################################################
# REFERENCE IMPLEMENTATION OF THE LOSS FUNCTION                               #
###############################################################################


def make_loss_function(
        A_pre_exc,
        A_pre_inh,
        j_tar,
        j_th=None,
        ws=None,
        lambda_=0.0,
        grad=False,
        pre_mult_ws=False):
    """
    A reference implementation of the loss-function that is being optimized by
    libbioneuronqp.

    This function returns the loss function L(w+, w-) for the optimization
    problem specified by the parameters to this function. If desired, a function
    computing the Jacobi matrix is returned instead (this required autograd to
    be installed).

    This function is not explicitly used anywhere in the code, but the
    unit-tests shipping with libbioneuronqp ensure that the solution returned by
    libbioneuronqp correspond to a global minimum of this function by making
    sure that the Jacobi matrix is zero (this is sufficient, since this loss
    function is convex).


    Parameters
    ==========

    A_pre_exc:
        A (N x n+) matrix containing the excitatory pre-population activities.
    A_pre_inh:
        A (N x n-) matrix containing the inhibitory pre-population activities.
    j_tar:
        A N-dimensional vector containing the desired target currents.
    j_th:
        The threshold current of the post-neuron. Currents below this threshold
        should not evoke any neural activity in the post-neuron. If set to None,
        j_th is assumed to be -Infinity, which is equivalent to saying that all
        currents evoke some activity in the target.
    ws:
        A vector (a0, a1, a2, b0, b1, b2) containing the post synapse/neuron
        model parameters. Set to None or (0, 1, -1, 1, 0, 0) for a standard
        current-based LIF neuron.
    lambda_:
        The L2 regularisation factor.
    grad:
        If set to true, a function returning the Jacobi matrix (i.e. gradient
        for each input weight) is returned. The Python package "autograd" must
        be installed for this to work.
    pre_mult_ws:
        If true, directly applies the model parameters to the inputs A, a_tar,
        and j_th, instead of evaluating H in the returned loss function. This
        parameter mostly exists for debugging purposes.


    Returns
    =======

    A function L(w_exc, w_inh) that maps the given excitatory and inhibitory
    weights onto a loss value. The goal of weight optimization is to minimize
    the loss.

    If grad=False (the default), w_exc and w_inh may be matrices, in which case
    the loss is computed for multiple weight vectors at a time. The first matrix
    dimension must be equal to the number of pre-neurons.

    If grad=True, two matrices corresponding to the gradient of each coefficient
    of w_exc, w_inh is returned.
    """

    # Import autograd, if the gradient is requested
    if grad:
        import autograd
        import autograd.numpy as np
    else:
        import numpy as np

    # Make sure all given matrices have the right dimensionality
    assert A_pre_exc.ndim == 2
    assert A_pre_inh.ndim == 2
    assert j_tar.ndim == 1
    assert A_pre_exc.shape[0] == A_pre_inh.shape[0] == j_tar.shape[0]
    n_samples = A_pre_exc.shape[0] # = A_pre_inh.shape[0] = J_tar.shape[0]
    n_pre_exc = A_pre_exc.shape[1]
    n_pre_inh = A_pre_inh.shape[1]

    # Sume handy aliases for subscripting
    iw0, iw1, iw2 = 0, n_pre_exc, n_pre_exc + n_pre_inh

    # Destructure the given model weights
    if ws is None:
        ws = (0, 1, -1, 1, 0, 0)
    a0, a1, a2, b0, b1, b2 = ws

    # Apply the model parameters to the input arrays if pre_mult_ws is set
    # to true. Then set the model parameters to a neutral value.
    if pre_mult_ws:
        A_pre_exc = ((a1 - j_tar * b1).T * A_pre_exc.T).T
        A_pre_inh = ((a2 - j_tar * b2).T * A_pre_inh.T).T
        j_tar = j_tar * b0 - a0
        if not j_th is None:
            j_th = j_th * b0 - a0
        a0, a1, a2, b0, b1, b2 = (0, 1, 1, 1, 0, 0)

    # Split the pre-population activities, as well as the target currents
    # into super- and sub-threshold samples
    if j_th is None: # Same as j_th = -np.inf
        j_th = -np.inf
    is_subth = j_tar < j_th
    is_supth = j_tar >= j_th

    # Samples with a super-threshold target current, also called "valid"
    # constraints in the C++ code.
    A_pre_exc_supth = A_pre_exc[is_supth]
    A_pre_inh_supth = A_pre_inh[is_supth]
    j_tar_supth = j_tar[is_supth]

    # Samples with a sub-threshold target current, also called "invalid"
    # constraints in the C++ code.
    A_pre_exc_subth = A_pre_exc[is_subth]
    A_pre_inh_subth = A_pre_inh[is_subth]
    j_tar_subth = j_tar[is_subth]

    # Implementation of the synaptic nonlinearity model H[., .]
    def H(A_pre_exc, A_pre_inh, w_exc, w_inh):
        return (
            (a0 + a1 * A_pre_exc @ w_exc + a2 * A_pre_inh @ w_inh) /
            (b0 + b1 * A_pre_exc @ w_exc + b2 * A_pre_inh @ w_inh))

    # Actual implementation of the loss-function
    def _L(w_combined):
        # Split ws into w_exc and w_inh (this is for autograd)
        w_exc, w_inh = w_combined[iw0:iw1], w_combined[iw1:iw2]

        # Make sure the input arrays are 2D
        w_exc = np.atleast_1d(w_exc)
        w_inh = np.atleast_1d(w_inh)

        # Make sure the given matrices have the right shape
        assert w_exc.shape[0] == n_pre_exc
        assert w_inh.shape[0] == n_pre_inh

        # Compute j_hat, i.e., the actually decoded output current
        j_hat_supth = H(A_pre_exc_supth, A_pre_inh_supth, w_exc, w_inh)
        j_hat_subth = H(A_pre_exc_subth, A_pre_inh_subth, w_exc, w_inh)

        # Compute the quadratic super-threshold loss
        L_quad_sup = np.sum(
            np.square(j_hat_supth.T - j_tar_supth.T).T, axis=0)

        # Compute the quadratic sub-threshold loss
        L_quad_sub = np.sum(np.square(
            np.maximum(j_th, j_hat_subth).T - j_th).T, axis=0)

        # Compute the regularisation error
        L_reg = lambda_ * n_samples * (np.sum(np.square(w_exc), axis=0) +
                                       np.sum(np.square(w_inh), axis=0))

        return L_quad_sup + L_quad_sub + L_reg

    # Turns _L into a function that receives independent excitatory
    # and inhibitory weights
    def wrap_L(_L):
        def L(w_exc, w_inh):
            res = _L(np.concatenate((w_exc, w_inh)))
            return (res[iw0:iw1], res[iw1:iw2]) if grad else res
        return L

    # If "grad" is set to "true", use autograd to compute the Lagrangian
    # of the above function, otherwise just return L
    if grad:
        return wrap_L(autograd.grad(_L))
    else:
        return wrap_L(_L)


###############################################################################
# MAIN PROGRAM -- RUNS THE ABOVE CODE                                         #
###############################################################################

if __name__ == "__main__":
    ###########################################################################
    # Imports                                                                 #
    ###########################################################################

    # Used for measuring the ellapsed time
    import time

    # Well-tested reference implementation from nengo_bio
    try:
        from nengo_bio.internal.qp_solver import solve as solve_ref
    except ImportError:
        solve_ref = None

    # Plot the results if matplotlib is installed
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    ###########################################################################
    # Micro implementation of the NEF                                         #
    ###########################################################################

    class LIF:
        slope = 2.0 / 3.0

        @staticmethod
        def inverse(a):
            valid = a > 0
            return 1.0 / (1.0 - np.exp(LIF.slope - (1.0 / (valid * a + 1e-6))))

        @staticmethod
        def activity(x):
            valid = x > (1.0 + 1e-6)
            return valid / (LIF.slope - np.log(1.0 - valid * (1.0 / x)))

    class Ensemble:
        def __init__(self, n_neurons, n_dimensions, neuron_type=LIF):
            self.neuron_type = neuron_type

            # Randomly select the intercepts and the maximum rates
            self.intercepts = np.random.uniform(-0.95, 0.95, n_neurons)
            self.max_rates = np.random.uniform(0.5, 1.0, n_neurons)

            # Randomly select the encoders
            self.encoders = np.random.normal(0, 1, (n_neurons, n_dimensions))
            self.encoders /= np.linalg.norm(self.encoders, axis=1)[:, None]

            # Compute the current causing the maximum rate/the intercept
            J_0 = self.neuron_type.inverse(0)
            J_max_rates = self.neuron_type.inverse(self.max_rates)

            # Compute the gain and bias
            self.gain = (J_0 - J_max_rates) / (self.intercepts - 1.0)
            self.bias = J_max_rates - self.gain

        def __call__(self, x):
            return self.neuron_type.activity(self.J(x))

        def J(self, x):
            return self.gain[:, None] * self.encoders @ x + self.bias[:, None]

    ###########################################################################
    # Actual test code                                                        #
    ###########################################################################

    def compute_error(J_tar, J_dec, i_th):
        if i_th is None:
            e_invalid = 0
            e_valid = np.sum(np.square(J_tar - J_dec))
        else:
            valid = J_tar > i_th
            invalid_violated = np.logical_and(J_tar < i_th, J_dec > i_th)
            e_invalid = np.sum(np.square(i_th - J_dec[invalid_violated]))
            e_valid = np.sum(np.square(J_tar[valid] - J_dec[valid]))

        return np.sqrt((e_valid + e_invalid) / J_tar.size)

    def E(Apre, Jpost, WE, WI, iTh):
        return compute_error(Jpost.T, Apre.T @ WE - Apre.T @ WI, iTh)

    np.random.seed(34812)

    ens1 = Ensemble(101, 1)
    ens2 = Ensemble(102, 1)

    xs = np.linspace(-1, 1, 100).reshape(1, -1)
    Apre = ens1(xs)
    Jpost = ens2.J(xs)

    ws = np.array([0, 1, -1, 1, 0, 0], dtype=np.float64)

    kwargs = {
        "Apre": Apre.T,
        "Jpost": Jpost.T,
        "ws": ws,
        "iTh": None,#1.0,
        "tol": 1e-1,
        "reg": 1e-2,
        "renormalise": False,
    }

    print("Solving weights using libbioneuronqp...")
    t1 = time.perf_counter()
    WE, WI = solve(**kwargs)
    sys.stderr.write('\n')
    t2 = time.perf_counter()
    print("Time : ", t2 - t1)
    print("Error: ", E(Apre, Jpost, WE, WI, kwargs['iTh']))
    print()

    if not solve_ref is None:
        print("Solving weights using nengobio.qp_solver (cvxopt)")
        t1 = time.perf_counter()
        WE_ref, WI_ref = solve_ref(**kwargs)
        t2 = time.perf_counter()
        print("Time : ", t2 - t1)
        print("Error: ", E(Apre, Jpost, WE_ref, WI_ref, kwargs['iTh']))
        print()

    ###########################################################################
    # Plotting                                                                #
    ###########################################################################

    if not plt is None:
        if solve_ref is None:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14))
        else:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 18))

        ax1.plot(xs[0], Apre.T)
        ax1.set_xlabel("Stimulus $x$")
        ax1.set_ylabel("Response rate")
        ax1.set_title("Ens 1 Tuning curves")

        ax2.plot(xs[0], Jpost.T)
        ax2.set_ylim(-1, 4)
        ax2.set_xlabel("Stimulus $x$")
        ax2.set_ylabel("Input current $J$")
        ax2.set_title("Ens 2 Desired Input Currents")

        ax3.plot(Apre.T @ WE - Apre.T @ WI)
        ax3.set_ylim(-1, 4)
        ax3.set_xlabel("Stimulus $x$")
        ax3.set_ylabel("Input current $J$")
        ax3.set_title("Ens 2 Desired Input Currents (libbioneuronqp)")

        if not solve_ref is None:
            ax4.plot(Apre.T @ WE_ref - Apre.T @ WI_ref)
            ax4.set_ylim(-1, 4)
            ax4.set_xlabel("Stimulus $x$")
            ax4.set_ylabel("Input current $J$")
            ax4.set_title("Ens 2 Desired Input Currents (nengo_bio.qp_solver)")

        plt.tight_layout()
        plt.show()
