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
    ]


PBioneuronWeightProblem = ctypes.POINTER(BioneuronWeightProblem)

BioneuronProgressCallback = ctypes.CFUNCTYPE(ctypes.c_ubyte, ctypes.c_int,
                                             ctypes.c_int)
BioneuronWarningCallback = ctypes.CFUNCTYPE(None, ctypes.c_char_p,
                                            ctypes.c_int)


def default_progress_callback(n_done, n_total):
    try:
        sys.stderr.write("\rSolved {}/{} neuron weights".format(
            n_done, n_total))
        sys.stderr.flush()
    except KeyboardInterrupt:
        return False
    return True


def default_warning_callback(msg, idx):
    print("WARN: " + msg)


class BioneuronSolverParameters(ctypes.Structure):
    _fields_ = [
        ("renormalise", ctypes.c_ubyte),
        ("tolerance", ctypes.c_double),
        ("progress", BioneuronProgressCallback),
        ("warn", BioneuronWarningCallback),
        ("n_threads", ctypes.c_int),
    ]


PBioneuronSolverParameters = ctypes.POINTER(BioneuronSolverParameters)

DLLFound = None
DLL = None


def _load_dll():
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

        DLL.bioneuronqp_problem_create.argtypes = []
        DLL.bioneuronqp_problem_create.restype = PBioneuronWeightProblem

        DLL.bioneuronqp_problem_free.argtypes = [PBioneuronWeightProblem]
        DLL.bioneuronqp_problem_free.restype = None

        DLL.bioneuronqp_solver_parameters_create.argtypes = []
        DLL.bioneuronqp_solver_parameters_create.restype = PBioneuronSolverParameters

        DLL.bioneuronqp_solver_parameters_free.argtypes = [
            PBioneuronSolverParameters
        ]
        DLL.bioneuronqp_solver_parameters_free.restype = None

        DLL.bioneuronqp_solve.argtypes = [
            PBioneuronWeightProblem, PBioneuronSolverParameters
        ]
        DLL.bioneuronqp_solve.restype = ctypes.c_int

    return DLL


def solve(Apre,
          Jpost,
          ws,
          connection_matrix=None,
          iTh=None,
          nonneg=True,
          renormalise=True,
          tol=None,
          reg=None,
          progress_callback=default_progress_callback,
          warning_callback=default_warning_callback):
    # Load the solver library
    _dll = _load_dll()
    if _dll is None:
        raise OSError("libbioneuronqp.so not found; make sure the library is located withing the library path")

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
        assert ws.shape[0] == Npost and ws.shape[
            1] == 6, "Invalid model weight matrix shape"

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

    params = BioneuronSolverParameters()
    params.renormalise = renormalise
    params.tolerance = tol
    params.progress = BioneuronProgressCallback(0 if progress_callback is None else progress_callback)
    params.warn = BioneuronWarningCallback(0 if warning_callback is None else warning_callback)
    params.n_threads = 0

    # Actually run the solver
    err = _dll.bioneuronqp_solve(ctypes.pointer(problem), ctypes.pointer(params))
    if err != 0:
        raise RuntimeError(_dll.bioneuronqp_strerr(err))

    return c_we, c_wi


###############################################################################
# MAIN PROGRAM -- RUNS THE ABOVE CODE                                         #
###############################################################################

if __name__ == "__main__":
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

    # Micro implementation of the NEF
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

    def RMS(e):
        return np.sqrt(np.mean(np.square(e)))

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
        "iTh": None,
        "renormalise": False,
        "tol": 1e-4,
        "reg": 1e-1,
    }

    print("Solving weights using libbioneuronqp...")
    t1 = time.perf_counter()
    WE, WI = solve(**kwargs)
    sys.stderr.write('\n')
    t2 = time.perf_counter()
    print("Time : ", t2 - t1)
    print("Error: ", RMS(Jpost.T - Apre.T @ WE - Apre.T @ WI))
    print()

    if not solve_ref is None:
        print("Solving weights using nengobio.qp_solver (cvxopt)")
        t1 = time.perf_counter()
        WE_ref, WI_ref = solve_ref(**kwargs)
        t2 = time.perf_counter()
        print("Time : ", t2 - t1)
        print("Error: ", RMS(Jpost.T - Apre.T @ WE_ref - Apre.T @ WI_ref))
        print()


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