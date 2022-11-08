# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
This module contains the qml.equal function.
"""
# pylint: disable=too-many-arguments,too-many-return-statements
from typing import Union
from functools import singledispatch

import pennylane as qml
from pennylane.measurements import MeasurementProcess, ShadowMeasurementProcess
from pennylane.operation import Operator, Operation, Observable
from pennylane.ops.qubit.hamiltonian import Hamiltonian


@singledispatch
def equal(
    op1: Union[Operator, MeasurementProcess, ShadowMeasurementProcess],
    op2: Union[Operator, MeasurementProcess, ShadowMeasurementProcess],
    **kwargs
):
    r"""Function for determining operator or measurement equality.

    Args:
        op1 (.Operator, .MeasurementProcess, or .ShadowMeasurementProcess): First object to compare
        op2 (.Operator, .MeasurementProcess, or .ShadowMeasurementProcess): Second object to compare
        check_interface (bool, optional): Whether to compare interfaces. Default: ``True``
        check_trainability (bool, optional): Whether to compare trainability status. Default: ``True``
        rtol (float, optional): Relative tolerance for parameters
        atol (float, optional): Absolute tolerance for parameters

    Returns:
        bool: ``True`` if the operators or measurement processes are equal, else ``False``

    **Example**

    Given two operators or measurement processes, ``qml.equal`` determines their equality:

    >>> op1 = qml.RX(np.array(.12), wires=0)
    >>> op2 = qml.RY(np.array(1.23), wires=0)
    >>> qml.equal(op1, op1), qml.equal(op1, op2)
    True False

    >>> qml.equal(qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(0)) )
    >>> True
    >>> qml.equal(qml.probs(wires=(0,1)), qml.probs(wires=(1,2)) )
    >>> False
    >>> qml.equal(qml.classical_shadow(wires=[0,1]), qml.classical_shadow(wires=[0,1]) )
    >>> True

    .. details::
        :title: Usage Details

        You can use the optional arguments when comparing Operations to get more specific results. Note this is not
        implemented for comparisons of Observables, MeasurementProcesses or ShadowMeasurementProcesses.

        Consider the following comparisons:

        >>> op1 = qml.RX(torch.tensor(1.2), wires=0)
        >>> op2 = qml.RX(jax.numpy.array(1.2), wires=0)
        >>> qml.equal(op1, op2)
        False

        >>> qml.equal(op1, op2, check_interface=False, check_trainability=False)
        True

        >>> op3 = qml.RX(np.array(1.2, requires_grad=True), wires=0)
        >>> op4 = qml.RX(np.array(1.2, requires_grad=False), wires=0)
        >>> qml.equal(op3, op4)
        False

        >>> qml.equal(op3, op4, check_trainability=False)
        True
    """
    raise NotImplementedError("Cannot compare {type(op1)} and {type(op2)}")


def _obs_comparison_data(obs):
    if isinstance(obs, Hamiltonian):
        obs.simplify()
        obs = obs._obs_data()  # pylint: disable=protected-access
    else:
        obs = {(1, frozenset(obs._obs_data()))}  # pylint: disable=protected-access
    return obs


@equal.register
def equal_operator(
    op1: Operation,
    op2: Operation,
    check_interface: bool = True,
    check_trainability: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-9,
):
    """Determine whether two Operations objects are equal"""
    if [op1.name, op1.arithmetic_depth, op1.wires] != [
        op2.name,
        op2.arithmetic_depth,
        op2.wires,
    ]:
        return False

    if op1.arithmetic_depth > 0:
        raise NotImplementedError(
            "Comparison of Operations with an arithmetic depth larger than 0 is not yet implemented."
        )

    if not all(
        qml.math.allclose(d1, d2, rtol=rtol, atol=atol) for d1, d2 in zip(op1.data, op2.data)
    ):
        return False

    if op1.hyperparameters != op2.hyperparameters:
        return False

    if check_trainability:
        for params_1, params_2 in zip(op1.data, op2.data):
            if qml.math.requires_grad(params_1) != qml.math.requires_grad(params_2):
                return False

    if check_interface:
        for params_1, params_2 in zip(op1.data, op2.data):
            if qml.math.get_interface(params_1) != qml.math.get_interface(params_2):
                return False

    return getattr(op1, "inverse", False) == getattr(op2, "inverse", False)


@equal.register
def equal_observable(op1: Observable, op2: Observable):
    """Determine whether two Observables objects are equal"""
    return _obs_comparison_data(op1) != _obs_comparison_data(op2)


@equal.register
def equal_measurements(op1: MeasurementProcess, op2: MeasurementProcess):
    """Determine whether two MeasurementProcess objects are equal"""
    return_types_match = op1.return_type == op2.return_type
    if op1.obs is not None and op2.obs is not None:
        observables_match = equal(op1.obs, op2.obs)
    # check obs equality when either one is None (False) or both are None (True)
    else:
        observables_match = op1.obs == op2.obs
    wires_match = op1.wires == op2.wires
    eigvals_match = qml.math.allequal(op1.eigvals(), op2.eigvals())
    log_base_match = op1.log_base == op2.log_base

    return (
        return_types_match
        and observables_match
        and wires_match
        and eigvals_match
        and log_base_match
    )


@equal.register
def equal_shadow_measurements(op1: ShadowMeasurementProcess, op2: ShadowMeasurementProcess):
    """Determine whether two ShadowMeasurementProcess objects are equal"""
    return_types_match = op1.return_type == op2.return_type
    wires_match = op1.wires == op2.wires
    H_match = op1.H == op2.H
    k_match = op1.k == op2.k

    return return_types_match and wires_match and H_match and k_match
