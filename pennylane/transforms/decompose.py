# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A transform for performing decompositions to a target gateset.

Design priorities
-----------------

Single Responsibility: this function is only responsible for decomposing operations
Versatility: the user should be able to control both the target gateset and the pathway.


"""

from typing import Tuple, Callable, Generator, List, Union

import pennylane as qml
from pennylane.operation import Operator

from .core import transform


def _decomposition_gen(
    op: Operator,
    acceptance_function: Callable[[Operator], bool],
    decomposer: Callable[[Operator], List[Operator]],
) -> Generator[Operator, None, None]:
    """A generator that yields the next operation that is accepted by the acceptance function."""
    if acceptance_function(op):
        yield op
    else:
        decomp = decomposer(op)
        for sub_op in decomp:
            yield from _decomposition_gen(sub_op, acceptance_function, decomposer)


def _stop_at_from_set(stop_at_set):
    """Convert a set of names and types into a boolean function."""

    def stop_at_function(obj):
        """A stopping condition that checks if the type or name is in the provided set.

        Closure: ``stop_at_set``.
        """
        return type(obj) in stop_at_set or obj.name in stop_at_set

    return stop_at_function


def null_postprocessing(results: qml.typing.ResultBatch) -> qml.typing.Result:
    """A postprocesing function returned by a transform that only converts the batch of results
    into a result for a single ``QuantumTape``.
    """
    return results[0]


def default_decomposer(obj: Operator) -> bool:
    """The default decomposition function. Returns :meth:`~.Operator.decomposition`."""
    return obj.decomposition()


@transform
def decompose(
    tape: qml.tape.QuantumTape,
    stop_at: Union[set[str, type], Callable[[Operator], bool]],
    decomposer: Callable[[Operator], List[Operator]] = default_decomposer,
    skip_initial_state_prep: bool = False,
) -> (Tuple[qml.tape.QuantumTape], Callable):
    """Decompose all the operations until a target gateset is reached via a customizable decomposition strategy.

    Args:
        tape (qml.tape.QuantumTape): the initial circuit.
        stop_at (set[str, type], Callable[[Operator], bool]): A description of the target gateset.  Accepts either a set
            names and types, or a function of an operator that returns a boolean.
        decomposer (Callable[[Operator], List[Operator]]): a function from an arbitrary :class:`~.Operator` to a list of
            operators. Defaults to using :meth:`~.Operator.decomposition`.
        skip_initial_state_prep (bool) = False: whether or not to skip any :class:`~.operation.StatePrepBase` operations
            that occur in the beginning of the circuit.

    Returns:
        pennylane.QNode or qfunc or tuple[List[.QuantumTape], function]: If a QNode is passed,
        it returns a QNode with the transform added to its transform program.
        If a tape is passed, returns a tuple containing a list of
        quantum tapes to be evaluated, and a function to be applied to these
        tape executions.

    Raises:
        ``pennylane.operation.DecompositionUndefinedError``: if an operator is not accepted by ``stop_at`` and
        does not define a decomposition.

        ``RecursionError``: If the decomposer enters an infinte loop

    .. details::
        :title: Usage Details

        **Stop at:**

        ``stop_at`` can be either a set of names and types:

        >>> stop_at = {"CNOT", qml.RX}
        >>> tape = qml.tape.QuantumScript([qml.IsingXX(1.23, wires=(0,1))], [qml.state()])
        >>> batch, fn = decompose(tape, stop_at=stop_at)
        >>> batch[0].operations
        [CNOT(wires=[0, 1]), RX(1.23, wires=[0]), CNOT(wires=[0, 1])]

        Or a function from operators to a boolean:

        >>> def stop_at(obj: qml.operation.Operator) -> bool:
        ...     return obj.num_params < 2
        >>> tape = qml.tape.QuantumTape([qml.Rot(1.2, 2.3, 3.4, wires=0)], [qml.state()])
        >>> batch, postprocessing = decompose(tape, stop_at=stop_at)
        >>> batch[0].operations
        [RZ(1.2, wires=[0]), RY(2.3, wires=[0]), RZ(3.4, wires=[0])]

        :class:`~.BooleanFn` may provide assistance in writing the ``stop_at`` function.

        If an :class:`~.Operator` is not accepted and does not have a decompositon, a
        ``DecompositionUndefinedError`` will be raised.

        >>> decompose(tape, stop_at=lambda obj: False)
        DecompositionUndefinedError

        **Decomposer:**

        Custom decomposition strategies can be used by providing a ``decomposer`` function.
        ``functools.singledispatch`` may help in the construction of such a function.

        >>> @functools.singledispatch
        ... def decomposer(op):
        ...     return op.decomposition()
        >>> @decomposer.register
        ... def custom_cnot(op: qml.CNOT):
        ...     return [qml.Hadamard(op.wires[1]), qml.CZ(op.wires), qml.Hadamard(op.wires[1])]
        >>> tape = qml.tape.QuantumTape([qml.CNOT(("a", "b"))], [qml.state()])
        >>> batch, postprocessing = decompose(tape, stop_at={qml.Hadamard, qml.CZ}, decomposer=decomposer)
        >>> batch[0].operations
        [Hadamard(wires=['b']),
        Controlled(PauliZ(wires=['b']), control_wires=['a']),
        Hadamard(wires=['b'])]

        Note that even if an operator has a custom decomposition, it will not be decomposed unless ``stop_at`` is false:

        >>> tape = qml.tape.QuantumTape([qml.CNOT(("a", "b"))], [qml.state()])
        >>> batch, postprocessing = decompose(tape, stop_at={qml.CNOT}, decomposer=decomposer)
        >>> batch[0].operations
        [CNOT(wires=['a', 'b'])]

    """

    if isinstance(stop_at, set):
        stop_at = _stop_at_from_set(stop_at)

    if skip_initial_state_prep and isinstance(tape[0], qml.operation.StatePrepBase):
        prep_op = tape[0]
    else:
        prep_op = []

    new_ops = [
        final_op
        for op in tape.operations[bool(prep_op) :]
        for final_op in _decomposition_gen(op, stop_at, decomposer)
    ]

    new_tape = qml.tape.QuantumScript(prep_op + new_ops, tape.measurements, shots=tape.shots)
    new_tape._qfunc_output = tape._qfunc_output  # pylint: disable=protected-access

    return (new_tape,), null_postprocessing
