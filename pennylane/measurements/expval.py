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
This module contains the qml.expval measurement.
"""
import warnings
from typing import Sequence, Tuple

import pennylane as qml
from pennylane.operation import Operator
from pennylane.wires import Wires

from .measurements import Expectation, SampleMeasurement, StateMeasurement
from .mid_measure import MeasurementValue
from .sample import SampleMP


def expval(*args, **kwargs) -> "ExpectationMP":
    r"""Expectation value of the supplied observable.

    **Example:**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(0))

    Executing this QNode:

    >>> circuit(0.5)
    -0.4794255386042029

    Args:
        op (Observable): a quantum observable object
        mv (MeasurementValue): ``MeasurementValue`` corresponding to mid-circuit
            measurements. To get the variance for more than one ``MeasurementValue``,
            they can be composed using arithmetic operators.

    Returns:
        ExpectationMP: measurement process instance
    """
    if (n_args := len(args) + len(kwargs)) != 1:
        raise TypeError(f"qml.expval() takes 1 argument, but {n_args} were given.")

    if args:
        arg_name = None
        obj = args[0]
    else:
        arg_name, obj = list(kwargs.items())[0]

    if isinstance(obj, Operator) and arg_name in [None, "op"]:
        if not obj.is_hermitian:
            warnings.warn(f"{obj.name} might not be hermitian.")

        return ExpectationMP(obs=obj)

    if arg_name in [None, "mv"]:
        if isinstance(obj, MeasurementValue):
            return ExpectationMP(mv=obj)

    raise ValueError(
        "Invalid argument provided to qml.expval(). Valid 'op' must be of type "
        "qml.operation.Operator. Valid 'mv' must be a MeasurementValue or a collection "
        "of multiple MeasurementValues using arithmetic operators."
    )


class ExpectationMP(SampleMeasurement, StateMeasurement):
    """Measurement process that computes the expectation value of the supplied observable.

    Please refer to :func:`expval` for detailed documentation.

    Args:
        obs (.Operator): The observable that is to be measured
            as part of the measurement process. Not all measurement processes require observables
            (for example ``Probability``); this argument is optional.
        wires (.Wires): The wires the measurement process applies to.
            This can only be specified if an observable was not provided.
        mv (Union[.MeasurementValue, Sequence[.MeasurementValue]]): One or more ``MeasurementValue``'s
            corresponding to mid-circuit measurements. To get statistics for more than one
            ``MeasurementValue``, they can be passed in a list or tuple or composed using arithmetic
            operators.
        eigvals (array): A flat array representing the eigenvalues of the measurement.
            This can only be specified if an observable was not provided.
        id (str): custom label given to a measurement instance, can be useful for some applications
            where the instance has to be identified
    """

    @property
    def return_type(self):
        return Expectation

    @property
    def numeric_type(self):
        return float

    def shape(self, device, shots):
        if not shots.has_partitioned_shots:
            return ()
        num_shot_elements = sum(s.copies for s in shots.shot_vector)
        return tuple(() for _ in range(num_shot_elements))

    def process_samples(
        self,
        samples: Sequence[complex],
        wire_order: Wires,
        shot_range: Tuple[int] = None,
        bin_size: int = None,
    ):
        # estimate the ev
        with qml.queuing.QueuingManager.stop_recording():
            samples = SampleMP(obs=self.obs, mv=self.mv).process_samples(
                samples=samples, wire_order=wire_order, shot_range=shot_range, bin_size=bin_size
            )

        # With broadcasting, we want to take the mean over axis 1, which is the -1st/-2nd with/
        # without bin_size. Without broadcasting, axis 0 is the -1st/-2nd with/without bin_size
        axis = -1 if bin_size is None else -2
        # TODO: do we need to squeeze here? Maybe remove with new return types
        return qml.math.squeeze(qml.math.mean(samples, axis=axis))

    def process_state(self, state: Sequence[complex], wire_order: Wires):
        # This also covers statistics for mid-circuit measurements manipulated using
        # arithmetic operators
        eigvals = qml.math.asarray(self.eigvals(), dtype="float64")

        # we use ``self.wires`` instead of ``self.obs`` because the observable was
        # already applied to the state
        with qml.queuing.QueuingManager.stop_recording():
            prob = qml.probs(wires=self.wires).process_state(state=state, wire_order=wire_order)
        # In case of broadcasting, `prob` has two axes and this is a matrix-vector product
        return qml.math.dot(prob, eigvals)
