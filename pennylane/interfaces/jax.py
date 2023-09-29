# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains functions for adding the JAX interface
to a PennyLane Device class.
"""
# pylint: disable=too-many-arguments
import inspect
import logging

import jax
import jax.numpy as jnp

import pennylane as qml
from pennylane.transforms import convert_to_numpy_parameters

dtype = jnp.float64

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _set_copy_and_unwrap_tape(t, a, unwrap=True):
    """Copy a given tape with operations and set parameters"""
    tc = t.bind_new_parameters(a, t.trainable_params)
    return convert_to_numpy_parameters(tc) if unwrap else tc


def set_parameters_on_copy_and_unwrap(tapes, params, unwrap=True):
    """Copy a set of tapes with operations and set parameters"""
    return tuple(_set_copy_and_unwrap_tape(t, a, unwrap=unwrap) for t, a in zip(tapes, params))


def get_jax_interface_name(tapes):
    """Check all parameters in each tape and output the name of the suitable
    JAX interface.

    This function checks each tape and determines if any of the gate parameters
    was transformed by a JAX transform such as ``jax.jit``. If so, it outputs
    the name of the JAX interface with jit support.

    Note that determining if jit support should be turned on is done by
    checking if parameters are abstract. Parameters can be abstract not just
    for ``jax.jit``, but for other JAX transforms (vmap, pmap, etc.) too. The
    reason is that JAX doesn't have a public API for checking whether or not
    the execution is within the jit transform.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute

    Returns:
        str: name of JAX interface that fits the tape parameters, "jax" or
        "jax-jit"
    """
    for t in tapes:
        for op in t:
            # Unwrap the observable from a MeasurementProcess
            op = op.obs if hasattr(op, "obs") else op
            if op is not None:
                # Some MeasurementProcess objects have op.obs=None
                for param in op.data:
                    if qml.math.is_abstract(param):
                        return "jax-jit"

    return "jax"


def execute(tapes, execute_fn, jpc):
    """Execute a batch of tapes with JAX parameters on a device.

    Args:
        tapes (Sequence[.QuantumTape]): batch of tapes to execute
        device (pennylane.Device): Device to use for the shots vectors.
        execute_fn (callable): The execution function used to execute the tapes
            during the forward pass. This function must return a tuple ``(results, jacobians)``.
            If ``jacobians`` is an empty list, then ``gradient_fn`` is used to
            compute the gradients during the backwards pass.
        gradient_kwargs (dict): dictionary of keyword arguments to pass when
            determining the gradients of tapes
        gradient_fn (callable): the gradient function to use to compute quantum gradients
        _n (int): a positive integer used to track nesting of derivatives, for example
            if the nth-order derivative is requested.
        max_diff (int): If ``gradient_fn`` is a gradient transform, this option specifies
            the maximum order of derivatives to support. Increasing this value allows
            for higher order derivatives to be extracted, at the cost of additional
            (classical) computational overhead during the backwards pass.

    Returns:
        list[list[float]]: A nested list of tape results. Each element in
        the returned list corresponds in order to the provided tapes.
    """
    # Set the trainable parameters
    for tape in tapes:
        params = tape.get_parameters(trainable_only=False)
        tape.trainable_params = qml.math.get_trainable_indices(params)

    parameters = tuple(list(t.get_parameters()) for t in tapes)

    # PennyLane backward execution
    return _execute_bwd(parameters, tapes, execute_fn, jpc)


def _execute_bwd(params, tapes, execute_fn, jpc):
    """The main interface execution function where jacobians of the execute
    function are computed by the registered backward function."""

    # pylint: disable=unused-variable
    # Copy a given tape with operations and set parameters

    # assumes all tapes have the same shot vector
    has_partitioned_shots = tapes[0].shots.has_partitioned_shots

    @jax.custom_jvp
    def execute_wrapper(params):
        new_tapes = set_parameters_on_copy_and_unwrap(tapes, params)
        res = execute_fn(new_tapes)
        return _to_jax_shot_vector(res) if has_partitioned_shots else _to_jax(res)

    @execute_wrapper.defjvp
    def execute_wrapper_jvp(primals, tangents):
        """Primals[0] are parameters as Jax tracers and tangents[0] is a list of tangent vectors as Jax tracers."""
        new_tapes = set_parameters_on_copy_and_unwrap(tapes, primals[0], unwrap=False)
        res, jvps = jpc.execute_and_compute_jvp(new_tapes, tangents[0])

        return res, jvps

    return execute_wrapper(params)


def _execute_fwd(
    params,
    tapes,
    execute_fn,
    gradient_kwargs,
    _n=1,
):
    """The auxiliary execute function for cases when the user requested
    jacobians to be computed in forward mode (e.g. adjoint) or when no gradient function was
    provided. This function does not allow multiple derivatives. It currently does not support shot vectors
    because adjoint jacobian for default qubit does not support it.."""

    # pylint: disable=unused-variable
    @jax.custom_jvp
    def execute_wrapper(params):
        new_tapes = set_parameters_on_copy_and_unwrap(tapes, params, unwrap=False)
        res, jacs = execute_fn(new_tapes, **gradient_kwargs)
        res = _to_jax(res)

        return res, jacs

    @execute_wrapper.defjvp
    def execute_wrapper_jvp(primals, tangents):
        """Primals[0] are parameters as Jax tracers and tangents[0] is a list of tangent vectors as Jax tracers."""
        res, jacs = execute_wrapper(primals[0])
        multi_measurements = [len(tape.measurements) > 1 for tape in tapes]

        jvps = _compute_jvps(jacs, tangents[0], multi_measurements)
        return (res, jacs), (jvps, jacs)

    res, _jacs = execute_wrapper(params)
    return res


def _compute_jvps(jacs, tangents, multi_measurements):
    """Compute the jvps of multiple tapes, directly for a Jacobian and tangents."""
    jvps = []
    for i, multi in enumerate(multi_measurements):
        compute_func = (
            qml.gradients.compute_jvp_multi if multi else qml.gradients.compute_jvp_single
        )
        jvps.append(compute_func(tangents[i], jacs[i]))
    return tuple(jvps)


def _is_count_result(r):
    """Checks if ``r`` is a single count (or broadcasted count) result"""
    return isinstance(r, dict) or isinstance(r, list) and all(isinstance(i, dict) for i in r)


def _to_jax(res):
    """From a list of tapes results (each result is either a np.array or tuple), transform it to a list of Jax
    results (structure stay the same)."""
    res_ = []
    for r in res:
        if _is_count_result(r):
            res_.append(r)
        elif not isinstance(r, tuple):
            res_.append(jnp.array(r))
        else:
            sub_r = []
            for r_i in r:
                if _is_count_result(r_i):
                    sub_r.append(r_i)
                else:
                    sub_r.append(jnp.array(r_i))
            res_.append(tuple(sub_r))
    return tuple(res_)


def _to_jax_shot_vector(res):
    """Convert the results obtained by executing a list of tapes on a device with a shot vector to JAX objects while preserving the input structure.

    The expected structure of the inputs is a list of tape results with each element in the list being a tuple due to execution using shot vectors.
    """
    return tuple(tuple(_to_jax([r_])[0] for r_ in r) for r in res)
