# Copyright 2024 Xanadu Quantum Technologies Inc.

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
This file contains the implementation of the commutator function in PennyLane
"""
import pennylane as qml
from pennylane.pauli import PauliWord, PauliSentence


def comm(op1, op2, pauli=False):
    r"""Alias for :func:`~commutator`"""
    return commutator(op1, op2, pauli)


def commutator(op1, op2, pauli=False):
    r"""Compute commutator between two operators in PennyLane

    .. math:: [O_1, O_2] = O_1 O_2 - O_2 O_1

    Args:
        op1 (Union[Operator, PauliWord, PauliSentence]): First operator
        op2 (Union[Operator, PauliWord, PauliSentence]): Second operator
        pauli (bool): When ``True``, all results are passed as a ``PauliSentence`` instance. Else, results are always returned as ``Operator`` instances.

    Returns:
        ~Operator or ~PauliSentence: The comm

    **Examples**

    You can compute commutators between operators in PennyLane.

    >>> qml.commutator(qml.PauliX(0), qml.PauliY(0))
    2j*(PauliZ(wires=[0]))

    >>> op1 = qml.PauliX(0) @ qml.PauliX(1)
    >>> op2 = qml.PauliY(0) @ qml.PauliY(1)
    >>> qml.commutator(op1, op2)
    0*(Identity(wires=[0, 1]))

    We can return a :class:`~PauliSentence` instance by setting `pauli=True`.

    >>> op1 = qml.PauliX(0) @ qml.PauliX(1)
    >>> op2 = qml.PauliY(0) + qml.PauliY(1)
    >>> qml.commutator(op1, op2, pauli=True)
    2j * X(1) @ Z(0)
    + 2j * Z(1) @ X(0)

    We can also input :class:`~PauliWord` and :class:`~PauliSentence` instances.
    >>> op1 = PauliWord({0:"X", 1:"X"})
    >>> op2 = PauliWord({0:"Y"}) + PauliWord({1:"Y"})
    >>> qml.commutator(op1, op2, pauli=True)
    2j * Z(0) @ X(1)
    + 2j * X(0) @ Z(1)

    Note that when `pauli=False`, even if Pauli operators are used
    as inputs, `qml.comm` returns Operators.

    >>> qml.commutator(op1, op2, pauli=True)
    (2j*(PauliX(wires=[1]) @ PauliZ(wires=[0]))) + (2j*(PauliZ(wires=[1]) @ PauliX(wires=[0])))

    """
    if pauli:
        if not isinstance(op1, PauliSentence):
            op1 = qml.pauli.pauli_sentence(op1)
        if not isinstance(op2, PauliSentence):
            op2 = qml.pauli.pauli_sentence(op2)
        return op1 @ op2 - op2 @ op1

    if isinstance(op1, (PauliWord, PauliSentence)):
        op1 = op1.operation()
    if isinstance(op2, (PauliWord, PauliSentence)):
        op2 = op2.operation()
    with qml.QueuingManager.stop_recording():
        res = qml.sum(qml.prod(op1, op2), qml.s_prod(-1.0, qml.prod(op2, op1)))
        res = res.simplify()
    return res
