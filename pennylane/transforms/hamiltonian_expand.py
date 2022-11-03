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
Contains the hamiltonian expand tape transform
"""
# pylint: disable=protected-access
from typing import Sequence

import pennylane as qml
from pennylane.measurements import Expectation
from pennylane.ops import Hamiltonian, Sum
from pennylane.tape import QuantumScript


def hamiltonian_expand(tape: QuantumScript, group=True):
    r"""
    Splits a tape measuring a Hamiltonian expectation into mutliple tapes of Pauli expectations,
    and provides a function to recombine the results.

    Args:
        tape (.QuantumTape): the tape used when calculating the expectation value
            of the Hamiltonian
        group (bool): Whether to compute disjoint groups of commuting Pauli observables, leading to fewer tapes.
            If grouping information can be found in the Hamiltonian, it will be used even if group=False.

    Returns:
        tuple[list[.QuantumTape], function]: Returns a tuple containing a list of
        quantum tapes to be evaluated, and a function to be applied to these
        tape executions to compute the expectation value.

    **Example**

    Given a Hamiltonian,

    .. code-block:: python3

        H = qml.PauliY(2) @ qml.PauliZ(1) + 0.5 * qml.PauliZ(2) + qml.PauliZ(1)

    and a tape of the form,

    .. code-block:: python3

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)

            qml.expval(H)

    We can use the ``hamiltonian_expand`` transform to generate new tapes and a classical
    post-processing function for computing the expectation value of the Hamiltonian.

    >>> tapes, fn = qml.transforms.hamiltonian_expand(tape)

    We can evaluate these tapes on a device:

    >>> dev = qml.device("default.qubit", wires=3)
    >>> res = dev.batch_execute(tapes)

    Applying the processing function results in the expectation value of the Hamiltonian:

    >>> fn(res)
    -0.5

    Fewer tapes can be constructed by grouping commuting observables. This can be achieved
    by the ``group`` keyword argument:

    .. code-block:: python3

        H = qml.Hamiltonian([1., 2., 3.], [qml.PauliZ(0), qml.PauliX(1), qml.PauliX(0)])

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)
            qml.expval(H)

    With grouping, the Hamiltonian gets split into two groups of observables (here ``[qml.PauliZ(0)]`` and
    ``[qml.PauliX(1), qml.PauliX(0)]``):

    >>> tapes, fn = qml.transforms.hamiltonian_expand(tape)
    >>> len(tapes)
    2

    Without grouping it gets split into three groups (``[qml.PauliZ(0)]``, ``[qml.PauliX(1)]`` and ``[qml.PauliX(0)]``):

    >>> tapes, fn = qml.transforms.hamiltonian_expand(tape, group=False)
    >>> len(tapes)
    3

    Alternatively, if the Hamiltonian has already computed groups, they are used even if ``group=False``:

    .. code-block:: python3

        obs = [qml.PauliZ(0), qml.PauliX(1), qml.PauliX(0)]
        coeffs = [1., 2., 3.]
        H = qml.Hamiltonian(coeffs, obs, grouping_type='qwc')

        # the initialisation already computes grouping information and stores it in the Hamiltonian
        assert H.grouping_indices is not None

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)
            qml.expval(H)

    Grouping information has been used to reduce the number of tapes from 3 to 2:

    >>> tapes, fn = qml.transforms.hamiltonian_expand(tape, group=False)
    >>> len(tapes)
    2
    """
    non_expanded_measurements = []
    non_expanded_measurement_idxs = []
    expanded_tapes = []
    expanded_measurement_idxs = []
    expanded_coeffs = []
    expanded_grouping = []
    for idx, m in enumerate(tape.measurements):
        if isinstance(m.obs, Hamiltonian) and m.return_type is Expectation:
            hamiltonian = m.obs
            # note: for backward passes of some frameworks
            # it is crucial to use the hamiltonian.data attribute,
            # and not hamiltonian.coeffs when recombining the results
            inner_grouping = False
            if group or hamiltonian.grouping_indices is not None:
                inner_grouping = True

                if hamiltonian.grouping_indices is None:
                    # explicitly selected grouping, but indices not yet computed
                    hamiltonian.compute_grouping()

                inner_coeffs = [
                    qml.math.stack([hamiltonian.data[i] for i in indices])
                    for indices in hamiltonian.grouping_indices
                ]

                obs_groupings = [
                    [hamiltonian.ops[i] for i in indices]
                    for indices in hamiltonian.grouping_indices
                ]

                # make one tape per grouping, measuring the
                # observables in that grouping
                inner_tapes = []
                for obs in obs_groupings:
                    new_tape = tape.__class__(tape._ops, (qml.expval(o) for o in obs), tape._prep)

                    new_tape = new_tape.expand(stop_at=lambda obj: True)
                    inner_tapes.append(new_tape)

            else:
                inner_coeffs = hamiltonian.data

                # make one tape per observable
                inner_tapes = []
                for o in hamiltonian.ops:
                    # pylint: disable=protected-access
                    new_tape = tape.__class__(tape._ops, [qml.expval(o)], tape._prep)
                    inner_tapes.append(new_tape)

            expanded_tapes.extend(inner_tapes)
            expanded_coeffs.append(inner_coeffs)
            expanded_measurement_idxs.append(idx)
            expanded_grouping.append(inner_grouping)

        else:
            non_expanded_measurements.append(m)
            non_expanded_measurement_idxs.append(idx)

    non_expanded_tape = (
        [QuantumScript(ops=tape._ops, measurements=non_expanded_measurements, prep=tape._prep)]
        if non_expanded_measurements
        else []
    )
    tapes = expanded_tapes + non_expanded_tape
    measurement_idxs = expanded_measurement_idxs + non_expanded_measurement_idxs

    def processing_fn(res):
        processed_results = []
        # process results of all tapes except the last one
        for idx, (coeff, grouping) in enumerate(zip(expanded_coeffs, expanded_grouping)):
            expanded_results = res[idx : idx + len(coeff)]
            processed_results.extend(
                [_process_hamiltonian_results(expanded_results, coeff, grouping)]
            )
        # add results of tape containing all the non-sum observables
        if non_expanded_tape:
            non_expanded_res = [res[-1]] if len(non_expanded_measurement_idxs) == 1 else res[-1]
            processed_results.extend(non_expanded_res)
        # sort results
        sorted_results = sorted(zip(processed_results, measurement_idxs), key=lambda x: x[1])
        if len(sorted_results) > 1:
            return [res[0] for res in sorted_results]
        return sorted_results[0][0]

    return tapes, processing_fn


def _process_hamiltonian_results(res: Sequence[float], coeff: Sequence[float], grouping: bool):
    """Process results obtained from the expansion of a single Hamiltonian.

    Args:
        res (Sequence[float]): results from the expanded tapes
        coeff (Sequence[float]): coefficient of each tape
        grouping (bool): wether grouping was used during the expansion

    Returns:
        float: post-processed result
    """
    if grouping:
        if qml.active_return():
            dot_products = [
                qml.math.dot(
                    qml.math.reshape(
                        qml.math.convert_like(r_group, c_group), qml.math.shape(c_group)
                    ),
                    c_group,
                )
                for c_group, r_group in zip(coeff, res)
            ]
        else:
            dot_products = [qml.math.dot(r_group, c_group) for c_group, r_group in zip(coeff, res)]
    else:
        dot_products = [qml.math.dot(qml.math.squeeze(r), c) for c, r in zip(coeff, res)]
    return qml.math.sum(qml.math.stack(dot_products), axis=0)


def sum_expand(tape: QuantumScript, group=True):
    """Splits a tape measuring a Sum expectation into mutliple tapes of summand expectations,
    and provides a function to recombine the results.

    Args:
        tape (.QuantumTape): the tape used when calculating the expectation value
            of the Hamiltonian
        group (bool): Whether to compute disjoint groups of commuting Pauli observables, leading to fewer tapes.
            If grouping information can be found in the Hamiltonian, it will be used even if group=False.

    Returns:
        tuple[list[.QuantumTape], function]: Returns a tuple containing a list of
        quantum tapes to be evaluated, and a function to be applied to these
        tape executions to compute the expectation value.

    **Example**

    Given a Sum operator,

    .. code-block:: python3

        S = qml.op_sum(qml.prod(qml.PauliY(2), qml.PauliZ(1)), qml.s_prod(0.5, qml.PauliZ(2)), qml.PauliZ(1))

    and a tape of the form,

    .. code-block:: python3

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)

            qml.expval(qml.PauliZ(0))
            qml.expval(S)
            qml.expval(qml.PauliX(1))
            qml.expval(qml.PauliZ(2))

    We can use the ``sum_expand`` transform to generate new tapes and a classical
    post-processing function to speed-up the computation of the expectation value of the `Sum`.

    >>> tapes, fn = qml.transforms.sum_expand(tape, group=False)
    >>> for tape in tapes:
    ...     print(tape.measurements)
    [expval(PauliY(wires=[2]) @ PauliZ(wires=[1]))]
    [expval(0.5*(PauliZ(wires=[2])))]
    [expval(PauliZ(wires=[1]))]
    [expval(PauliZ(wires=[0])), expval(PauliX(wires=[1])), expval(PauliZ(wires=[2]))]

    Four tapes are generated: the first three contain the summands of the `Sum` operator,
    and the last tape contains the remaining observables.

    We can evaluate these tapes on a device:

    >>> dev = qml.device("default.qubit", wires=3)
    >>> res = dev.batch_execute(tapes)

    Applying the processing function results in the expectation value of the Hamiltonian:

        >>> fn(res)
    [0.0, -0.5, 0.0, -0.9999999999999996]

    Fewer tapes can be constructed by grouping commuting observables. This can be achieved
    by the ``group`` keyword argument:

    .. code-block:: python3

        S = qml.op_sum(qml.PauliZ(0), qml.s_prod(2, qml.PauliX(1)), qml.s_prod(3, qml.PauliX(0)))

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)
            qml.expval(S)

    With grouping, the Sum gets split into two groups of observables (here
    ``[qml.PauliZ(0), qml.s_prod(2, qml.PauliX(1))]`` and ``[qml.s_prod(3, qml.PauliX(0))]``):

    >>> tapes, fn = qml.transforms.sum_expand(tape, group=True)
    >>> for tape in tapes:
    ...     print(tape.measurements)
    [expval(PauliZ(wires=[0])), expval(2*(PauliX(wires=[1])))]
    [expval(3*(PauliX(wires=[0])))]
    """
    non_expanded_measurements = []
    non_expanded_measurement_idxs = []
    expanded_tapes = []
    expanded_measurement_idxs = []
    num_tapes = []
    for idx, m in enumerate(tape.measurements):
        if isinstance(m.obs, Sum) and m.return_type is Expectation:
            sum_op = m.obs
            if group:
                # make one tape per group of qwc observables
                obs_groupings = _group_summands(sum_op)
                tapes = [
                    QuantumScript(tape._ops, [qml.expval(o) for o in obs], tape._prep)
                    for obs in obs_groupings
                ]
            else:
                # make one tape per summand
                tapes = [
                    QuantumScript(ops=tape.operations, measurements=[qml.expval(summand)])
                    for summand in sum_op.operands
                ]

            expanded_tapes.extend(tapes)
            expanded_measurement_idxs.append(idx)
            num_tapes.append(len(tapes))

        else:
            non_expanded_measurements.append(m)
            non_expanded_measurement_idxs.append(idx)

    non_expanded_tape = (
        [QuantumScript(ops=tape._ops, measurements=non_expanded_measurements, prep=tape._prep)]
        if non_expanded_measurements
        else []
    )
    tapes = expanded_tapes + non_expanded_tape
    measurement_idxs = expanded_measurement_idxs + non_expanded_measurement_idxs

    def inner_processing_fn(res):
        if group:
            res = [qml.math.sum(c_group) for c_group in res]
        return qml.math.sum(qml.math.stack(res), axis=0)

    def outer_processing_fn(res):
        processed_results = []
        # process results of all tapes except the last one
        for idx, n_tapes in enumerate(num_tapes):
            processed_results.extend([inner_processing_fn(res[idx : idx + n_tapes])])
        # add results of tape containing all the non-sum observables
        if non_expanded_tape:
            non_expanded_res = [res[-1]] if len(non_expanded_measurement_idxs) == 1 else res[-1]
            processed_results.extend(non_expanded_res)
        # sort results
        sorted_results = sorted(zip(processed_results, measurement_idxs), key=lambda x: x[1])
        if len(sorted_results) > 1:
            return [res[0] for res in sorted_results]
        return sorted_results[0][0]

    return tapes, outer_processing_fn


def _group_summands(sum: Sum):
    """Group summands of Sum operator into qubit-wise commuting groups.

    Args:
        sum (Sum): sum operator

    Returns:
        list[list[Operator]]: list of lists of qubit-wise commuting operators
    """
    qwc_groups = []
    for summand in sum.operands:
        op_added = False
        for idx, (wires, group) in enumerate(qwc_groups):
            if all(wire not in summand.wires for wire in wires):
                qwc_groups[idx] = (wires + summand.wires, group + [summand])
                op_added = True
                break

        if not op_added:
            qwc_groups.append((summand.wires, [summand]))

    return [group[1] for group in qwc_groups]
