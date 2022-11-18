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
This module contains the qml.expval measurement.
"""
import warnings
from typing import Sequence, Tuple

import pennylane as qml
from pennylane.operation import Operator
from pennylane.ops import Projector
from pennylane.wires import Wires

from .measurements import Expectation, SampleMeasurement, StateMeasurement


def expval(op: Operator):
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

    Raises:
        QuantumFunctionError: `op` is not an instance of :class:`~.Observable`
    """
    if not op.is_hermitian:
        warnings.warn(f"{op.name} might not be hermitian.")

    return _Expectation(obs=op)


# TODO: Make public when removing the ObservableReturnTypes enum
class _Expectation(SampleMeasurement, StateMeasurement):
    """Measurement process that computes the probability of each computational basis state."""

    @property
    def numeric_type(self):
        return float

    def shape(self, device):
        if qml.active_return():
            return self._shape_new(device)
        if device.shot_vector is None:
            return (1,)
        num_shot_elements = sum(s.copies for s in device.shot_vector)
        return (num_shot_elements,)

    def _shape_new(self, device):
        if device.shot_vector is None:
            return ()
        num_shot_elements = sum(s.copies for s in device.shot_vector)
        return tuple(() for _ in range(num_shot_elements))

    @property
    def return_type(self):
        return Expectation

    def process_samples(
        self, samples: Sequence[complex], shot_range: Tuple[int] = None, bin_size: int = None
    ):
        if isinstance(self.obs, Projector):
            # branch specifically to handle the projector observable
            idx = int("".join(str(i) for i in self.obs.parameters[0]), 2)
            probs = qml.probs(wires=self.wires).process_samples(
                samples=samples, shot_range=shot_range, bin_size=bin_size
            )
            return probs[idx]
        # estimate the ev
        samples = qml.sample(op=self.obs).process_samples(
            samples=samples, shot_range=shot_range, bin_size=bin_size
        )
        # With broadcasting, we want to take the mean over axis 1, which is the -1st/-2nd with/
        # without bin_size. Without broadcasting, axis 0 is the -1st/-2nd with/without bin_size
        axis = -1 if bin_size is None else -2
        # TODO: do we need to squeeze here? Maybe remove with new return types
        return qml.math.squeeze(qml.math.mean(samples, axis=axis))

    def process_state(self, state: Sequence[complex], wires: Wires):
        if isinstance(self.obs, Projector):
            # branch specifically to handle the projector observable
            idx = int("".join(str(i) for i in self.obs.parameters[0]), 2)
            probs = qml.probs(wires=self.wires).process_state(state=state, wires=wires)
            return probs[idx]
        eigvals = qml.math.asarray(self.obs.eigvals(), dtype="float64")

        # the probability vector must be permuted to account for the permuted
        # wire order of the observable
        old_obs = self.obs
        permuted_wires = self._permute_wires(self.obs.wires)
        self.obs = qml.map_wires(self.obs, dict(zip(self.obs.wires, permuted_wires)))

        # we use ``self.wires`` instead of ``self.obs`` because the observable was
        # already applied to the state
        prob = qml.probs(wires=self.wires).process_state(state=state, wires=wires)
        self.obs = old_obs
        # In case of broadcasting, `prob` has two axes and this is a matrix-vector product
        return qml.math.dot(prob, eigvals)

    def _permute_wires(self, wires: Wires):
        wire_map = dict(zip(wires, range(len(wires))))

        ordered_obs_wire_lst = sorted(self.wires.tolist(), key=lambda label: wire_map[label])

        mapped_wires = [wire_map[w] for w in self.wires]

        permutation = qml.math.argsort(mapped_wires)  # extract permutation via argsort

        return Wires([ordered_obs_wire_lst[index] for index in permutation])
