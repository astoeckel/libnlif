#  libnlif -- Multi-compartment LIF simulator and weight solver
#  Copyright (C) 2017-2021  Andreas Stöckel
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import copy
import numpy as np
import tqdm

from .internal.adam import Adam


def _check_shape(reduced_sys, gs, Js):
    # Makse sure that gs and Js are arrays
    gs, Js = np.asarray(gs), np.asarray(Js)

    # Make sure that the input dimensions match
    if gs.shape[-1] != reduced_sys.n_inputs:
        raise RuntimeError(
            "Last dimension of gs must be equal to the number of inputs")
    if Js.shape != gs.shape[:-1]:
        raise RuntimeError("Dimensionality mismatch between Js and gs")

    # Reshape stuff for easier iteration
    output_shape = Js.shape
    Js = Js.reshape(-1)
    gs = gs.reshape(-1, reduced_sys.n_inputs)
    N = len(Js)  # Number of samples

    return gs, Js, N, output_shape


def loss(reduced_sys, gs, Js):
    """
    Computes the loss function for the reduced system reduced_sys and the given
    inputs and target currents.
    """
    gs, Js, N, output_shape = _check_shape(reduced_sys, gs, Js)

    # Compute the squared error
    res = np.zeros(N)
    for i in range(len(Js)):
        res[i] = np.square((reduced_sys.i_som(gs[i]) - Js[i]))
    return res.reshape(output_shape)


def loss_gradient_numerical(reduced_sys, gs, Js, eta=1e-9):
    """
    Computes the parameter gradient for the reduced system reduced_sys
    numerically. This function is slow, numerically unstable, and mainly for
    testing. Use `loss_gradient` instead.

    Returns matrices dA, da_const, dB, db_const containing the gradients for
    each input sample.
    """
    gs, Js, N, output_shape = _check_shape(reduced_sys, gs, Js)

    def dE(rsys1, rsys2):
        return (loss(rsys2, gs, Js) - loss(rsys1, gs, Js)) / eta

    n, k = reduced_sys.n_compartments, reduced_sys.n_inputs
    dA, da_const = np.zeros((N, n, k)), np.zeros((N, n))
    dB, db_const = np.zeros((N, n, k)), np.zeros((N, n))

    for i, j in zip(*np.where(reduced_sys.A_mask)):
        rsys1, rsys2 = copy.deepcopy(reduced_sys), copy.deepcopy(reduced_sys)
        rsys1.A[i, j] -= 0.5 * eta
        rsys2.A[i, j] += 0.5 * eta
        dA[:, i, j] = dE(rsys1, rsys2)

    for i in np.where(reduced_sys.a_const_mask)[0]:
        rsys1, rsys2 = copy.deepcopy(reduced_sys), copy.deepcopy(reduced_sys)
        rsys1.a_const[i] -= 0.5 * eta
        rsys2.a_const[i] += 0.5 * eta
        da_const[:, i] = dE(rsys1, rsys2)

    for i, j in zip(*np.where(reduced_sys.B_mask)):
        rsys1, rsys2 = copy.deepcopy(reduced_sys), copy.deepcopy(reduced_sys)
        rsys1.B[i, j] -= 0.5 * eta
        rsys2.B[i, j] += 0.5 * eta
        dB[:, i, j] = dE(rsys1, rsys2)

    for i in np.where(reduced_sys.b_const_mask)[0]:
        rsys1, rsys2 = copy.deepcopy(reduced_sys), copy.deepcopy(reduced_sys)
        rsys1.b_const[i] -= 0.5 * eta
        rsys2.b_const[i] += 0.5 * eta
        db_const[:, i] = dE(rsys1, rsys2)

    return (
        dA.reshape((*output_shape, n, k)),
        da_const.reshape((*output_shape, n)),
        dB.reshape((*output_shape, n, k)),
        db_const.reshape((*output_shape, n)),
    )


def loss_gradient(reduced_sys, gs, Js):
    """
    Computes the parameter gradient for the reduced system reduced_sys
    numerically.

    Returns matrices dA, da_const, dB, db_const containing the mean gradient of
    the loss function over the entire batch.
    """
    gs, Js, N, output_shape = _check_shape(reduced_sys, gs, Js)

    n, k = reduced_sys.n_compartments, reduced_sys.n_inputs
    dA, da_const = np.zeros((N, n, k)), np.zeros((N, n))
    dB, db_const = np.zeros((N, n, k)), np.zeros((N, n))

    for i in range(N):
        # Fetch some matrices that we'll need throughout the computation
        A_inv = np.linalg.inv(reduced_sys.A_dyn(gs[i]))
        b = reduced_sys.b_dyn(gs[i])
        v_eq = reduced_sys.v_eq(gs[i])
        c = reduced_sys.c
        J, g = Js[i], gs[i]

        # Compute the outer derivative
        E = -2 * (np.inner(reduced_sys.c, v_eq - reduced_sys.v_som) - J)

        # Compute the final derivative
        da_const[i] = E * np.einsum('ir,rj,j,i->r', A_inv, A_inv, b, c) * reduced_sys.a_const_mask
        dA[i] = E * np.einsum('ir,rj,s,j,i->rs', A_inv, A_inv, g, b,
                              c) * reduced_sys.A_mask
        db_const[i] = E * np.einsum('ir,i->r', A_inv, c) * reduced_sys.b_const_mask
        dB[i] = E * np.einsum('ir,s,i->rs', A_inv, g, c) * reduced_sys.B_mask

    return (
        dA.reshape((*output_shape, n, k)),
        da_const.reshape((*output_shape, n)),
        dB.reshape((*output_shape, n, k)),
        db_const.reshape((*output_shape, n)),
    )


def optimise_sgd(reduced_sys,
                 gs_train,
                 Js_train,
                 gs_test=None,
                 Js_test=None,
                 N_epochs=100,
                 N_batch=100,
                 alpha=1e-9,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-7, # From TensorFlow
                 rng=np.random):
    # Clone the system
    reduced_sys = copy.deepcopy(reduced_sys)

    # Prepare the input data
    if (gs_test is None) != (Js_test is None):
        raise RuntimeError(
            "Noth gs_test and Js_test must either be None or not None.")
    gs_train, Js, N_train, _ = _check_shape(reduced_sys, gs_train, Js_train)
    if not Js_test is None:
        gs_test, Js_test, N_test, _ = _check_shape(reduced_sys, gs_test,
                                                   Js_test)

    # Fetch references at the parameter matrices
    p = (
        reduced_sys.A,
        reduced_sys.a_const,
        reduced_sys.B,
        reduced_sys.b_const,
    )

    # Initialise the errors
    errs_train, errs_test = np.zeros((2, N_epochs + 1))

    def update_err(i):
        errs_train[i] = np.sqrt(
            np.mean(loss(reduced_sys, gs_train, Js_train)))
        if not Js_test is None:
            errs_test[i] = np.sqrt(
                np.mean(loss(reduced_sys, gs_test, Js_test)))

    update_err(0)

    # Instantiate the optimiser and perform the actual optmisation
    optimiser = Adam(alpha=alpha, beta1=beta1, beta2=beta2, epsilon=epsilon)
    for i in tqdm.tqdm(range(N_epochs)):
        # Divide the input into batches
        sample_idcs = list(range(N_train))
        rng.shuffle(sample_idcs)
        batch_idcs = np.linspace(0, N_train, int(np.ceil(N_train / N_batch)) + 1, dtype=int)
        batch_idcs0, batch_idcs1 = batch_idcs[:-1], batch_idcs[1:]

        # Iterate over each batch
        for i0, i1 in zip(batch_idcs0, batch_idcs1):
            # Compute the mean gradient
            dp = tuple(
                map(lambda x: np.mean(x, axis=0),
                loss_gradient(reduced_sys, gs_train[i0:i1], Js_train[i0:i1])))

            # Update the parameters
            optimiser.step(p, dp)
            #for j in range(len(dp)):
            #    p[j][...] = p[j] - alpha * dp[j]

            # Make sure that the non-negative portions of the parameters do not
            # become negative
            p[0][...] = np.maximum(p[0], 0.0)
            p[1][...] = np.maximum(p[1], 0.0)

        # Compute the error
        update_err(i + 1)

    # Return the updated system as well as the recorded errors
    if Js_test is None:
        return reduced_sys, errs_train
    else:
        return reduced_sys, errs_train, errs_test


#def optimise_lstsq_trust_region(reduced_sys, v_som, gs_train, Js_train, gs_test, Js_test, N_epochs, alpha=0.0):
#    
