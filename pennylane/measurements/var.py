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
# pylint: disable=protected-access
"""
This module contains the qml.var measurement.
"""
import warnings
from typing import Sequence, Tuple

import pennylane as qml
from pennylane.operation import Operator
from pennylane.ops.qubit.observables import BasisStateProjector
from pennylane.wires import Wires

from .measurements import SampleMeasurement, StateMeasurement, Variance
from .mid_measure import MeasurementValue


def var(*args, **kwargs) -> "VarianceMP":
    r"""Variance of the supplied observable.

    Args:
        op (Operator): a quantum observable object.
        mv (MeasurementValue): ``MeasurementValue`` corresponding to mid-circuit
            measurements. To get the variance for more than one ``MeasurementValue``,
            they can be composed using arithmetic operators.

    Returns:
        VarianceMP: Measurement process instance

    **Example:**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.var(qml.PauliY(0))

    Executing this QNode:

    >>> circuit(0.5)
    0.7701511529340698
    """
    if (n_args := len(args) + len(kwargs)) != 1:
        raise TypeError(f"qml.var() takes 1 argument, but {n_args} were given.")

    if args:
        arg_name = None
        obj = args[0]
    else:
        arg_name, obj = list(kwargs.items())[0]

    if isinstance(obj, Operator) and arg_name in [None, "op"]:
        if not obj.is_hermitian:
            warnings.warn(f"{obj.name} might not be hermitian.")

        return VarianceMP(obs=obj)

    if arg_name in [None, "mv"]:
        if isinstance(obj, MeasurementValue):
            return VarianceMP(mv=obj)

    raise ValueError(
        "Invalid argument provided to qml.var(). Valid 'op' must be of type "
        "qml.operation.Operator. Valid 'mv' must be a MeasurementValue or a collection "
        "of multiple MeasurementValues using arithmetic operators."
    )


class VarianceMP(SampleMeasurement, StateMeasurement):
    """Measurement process that computes the variance of the supplied observable.

    Please refer to :func:`var` for detailed documentation.

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
        return Variance

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
        if isinstance(self.obs, BasisStateProjector):
            # branch specifically to handle the basis state projector observable
            idx = int("".join(str(i) for i in self.obs.parameters[0]), 2)
            # we use ``self.wires`` instead of ``self.obs`` because the observable was
            # already applied before the sampling
            probs = qml.probs(wires=self.wires).process_samples(
                samples=samples, wire_order=wire_order, shot_range=shot_range, bin_size=bin_size
            )
            return probs[idx] - probs[idx] ** 2

        # estimate the variance
        with qml.queuing.QueuingManager.stop_recording():
            samples = qml.sample(op=self.obs, mv=self.mv).process_samples(
                samples=samples, wire_order=wire_order, shot_range=shot_range, bin_size=bin_size
            )

        # With broadcasting, we want to take the variance over axis 1, which is the -1st/-2nd with/
        # without bin_size. Without broadcasting, axis 0 is the -1st/-2nd with/without bin_size
        axis = -1 if bin_size is None else -2
        # TODO: do we need to squeeze here? Maybe remove with new return types
        return qml.math.squeeze(qml.math.var(samples, axis=axis))

    def process_state(self, state: Sequence[complex], wire_order: Wires):
        if isinstance(self.obs, BasisStateProjector):
            # branch specifically to handle the basis state projector observable
            idx = int("".join(str(i) for i in self.obs.parameters[0]), 2)
            # we use ``self.wires`` instead of ``self.obs`` because the observable was
            # already applied to the state
            with qml.queuing.QueuingManager.stop_recording():
                probs = qml.probs(wires=self.wires).process_state(
                    state=state, wire_order=wire_order
                )
            return probs[idx] - probs[idx] ** 2

        # This also covers statistics for mid-circuit measurements manipulated using
        # arithmetic operators
        eigvals = qml.math.asarray(self.eigvals(), dtype="float64")

        # we use ``wires`` instead of ``op`` because the observable was
        # already applied to the state
        with qml.queuing.QueuingManager.stop_recording():
            prob = qml.probs(wires=self.wires).process_state(state=state, wire_order=wire_order)
        # In case of broadcasting, `prob` has two axes and these are a matrix-vector products
        return qml.math.dot(prob, (eigvals**2)) - qml.math.dot(prob, eigvals) ** 2
