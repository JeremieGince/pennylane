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
"""
Defines classes that take the vjps, jvps, and jacobians of circuits.

"""
import abc
from functools import partial
from typing import Tuple, Callable, Optional

import pennylane as qml
from pennylane.tape import QuantumScript
from pennylane.typing import ResultBatch, TensorLike

Batch = Tuple[QuantumScript]


def _compute_vjps(jacs, dy, multi_measurements, has_partitioned_shots):
    """Compute the vjps of multiple tapes, directly for a Jacobian and co-tangents dys."""
    vjps = []
    for i, multi in enumerate(multi_measurements):
        dy_ = dy[i] if has_partitioned_shots else (dy[i],)
        jac_ = jacs[i] if has_partitioned_shots else (jacs[i],)

        shot_vjps = []
        for d, j in zip(dy_, jac_):
            if multi:
                shot_vjps.append(qml.gradients.compute_vjp_multi(d, j))
            else:
                shot_vjps.append(qml.gradients.compute_vjp_single(d, j))

        vjps.append(qml.math.sum(qml.math.stack(shot_vjps), axis=0))

    return tuple(vjps)


def _compute_jvps(jacs, tangents, multi_measurements):
    """Compute the jvps of multiple tapes, directly for a Jacobian and tangents."""
    jvps = []
    for i, multi in enumerate(multi_measurements):
        compute_func = (
            qml.gradients.compute_jvp_multi if multi else qml.gradients.compute_jvp_single
        )
        jvps.append(compute_func(tangents[i], jacs[i]))
    return tuple(jvps)


class JacobianProductCalculator(abc.ABC):
    """Provides methods for calculating the jvp and vjps for tapes and tangents/ cotangents."""

    @abc.abstractmethod
    def execute_and_compute_jvp(
        self, tapes: Batch, tangents: TensorLike
    ) -> Tuple[ResultBatch, Tuple]:
        """Calculate both the results for a batch of tapes and the jvp.

        This method is required for the jax interface.

        Args:
            tapes: The batch of tapes to take the derivatives of
            tangents (Sequence[Sequence[TensorLike]]): the tangents for the parameters of the tape

        Returns:
            ResultBatch, TensorLike

        **Examples:**

        >>> tape0 = qml.tape.QuantumScript([qml.RX(0.1, wires=0)], [qml.expval(qml.PauliZ(0))])
        >>> tape1 = qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.expval(qml.PauliZ(0))])
        >>> batch = (tape0, tape1)
        >>> tangents0 = (1.5, )
        >>> tangents1 = (2.0, )
        >>> tangents = (tangents0, tangents1)
        >>> results, jvps = jp_method.execute_and_compute_jvp(batch, tangents)
        >>> results
        (0.9950041652780258, 0.9800665778412417)
        >>> np.cos(0.1), np.cos(0.2)
        (0.9950041652780258, 0.9800665778412416)
        >>> jvps
        (array(-0.14975012), array(-0.39733866))
        >>> 1.5 * -np.sin(0.1), 2.0 * -np.sin(0.2)
        (-0.14975012497024223, -0.39733866159012243)

        """

    @abc.abstractmethod
    def compute_vjp(self, tapes: Batch, dy: Tuple[Tuple[TensorLike]]) -> Tuple:
        """Compute the vjp for a given batch of tapes.

        This method is used by autograd, torch, and tensorflow.

        Args:
            tapes: the batch of tapes to the the derivatives of
            dy: the derivatives of the results of an execution

        Returns:
            TensorLike

        >>> tape0 = qml.tape.QuantumScript([qml.RX(0.1, wires=0)], [qml.expval(qml.PauliZ(0))])
        >>> tape1 = qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(0))])
        >>> batch = (tape0, tape1)
        >>> dy0 = (0.5, )
        >>> dy1 = (2.0, 3.0)
        >>> dys = (dy0, dy1)
        >>> jp_method.compute_vjp(batch, dys)
        (array([-0.04991671]), array([2.54286107]))
        >>> 0.5 * -np.sin(0.1)
        -0.04991670832341408
        >>> 2.0 * -np.sin(0.2) + 3.0 * np.cos(0.2)
        2.5428610719336024

        """

    @abc.abstractmethod
    def compute_jacobian(self, tapes: Batch) -> Tuple:
        """Compute the full jacobian for a batch of tapes.

        This method is required for jax-jit.

        Args:
            tapes: the batch of tapes to take the jacobian of

        >>> tape0 = qml.tape.QuantumScript([qml.RX(0.1, wires=0)], [qml.expval(qml.PauliZ(0))])
        >>> tape1 = qml.tape.QuantumScript([qml.RY(0.2, wires=0)], [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(0))])
        >>> batch = (tape0, tape1)
        >>> jp_method.compute_jacobian(batch)
        (array(-0.09983342), (array(-0.19866933), array(0.98006658)))

        """


class TransformDerivatives(JacobianProductCalculator):
    """Compute vjp, jvps, and jacobians via a gradient transform.

    Args:
        inner_execute (Callable[[Tuple[QuantumTape]], ResultBatch]): a function that
            turns the batch of circuits into results.
        gradient_transform (qml.gradients.gradient_transform): the gradient transform to use.
        gradient_kwargs (dict): Any keyword arguments for the gradient transform.

    >>> inner_execute = qml.device('default.qubit').execute
    >>> gradient_transform = qml.gradients.param_shift
    >>> kwargs = {"broadcast": True}
    >>> jp_method = TransformDerivatives(inner_execute, gradient_transform, kwargs)

    """

    def __repr__(self):
        return f"TransformDerivatives({self._inner_execute}, gradient_transform={self._gradient_transform}, gradient_kwargs={self._gradient_kwargs})"

    def __init__(
        self,
        inner_execute: Callable,
        gradient_transform: "qml.gradients.gradient_transform",
        gradient_kwargs: Optional[dict] = None,
    ):
        self._inner_execute = inner_execute
        self._gradient_transform = gradient_transform
        self._gradient_kwargs = gradient_kwargs or {}

    def execute_and_compute_jvp(self, tapes: Batch, tangents):
        num_result_tapes = len(tapes)

        jvp_tapes, jvp_processing_fn = qml.gradients.batch_jvp(
            tapes, tangents, self._gradient_transform, gradient_kwargs=self._gradient_kwargs
        )

        full_batch = tapes + tuple(jvp_tapes)

        full_results = self._inner_execute(full_batch)

        results = full_results[:num_result_tapes]
        jvp_results = full_results[num_result_tapes:]
        jvps = jvp_processing_fn(jvp_results)
        return tuple(results), tuple(jvps)

    def compute_vjp(self, tapes, dy):
        vjp_tapes, processing_fn = qml.gradients.batch_vjp(
            tapes, dy, self._gradient_transform, gradient_kwargs=self._gradient_kwargs
        )

        vjp_results = self._inner_execute(vjp_tapes)
        # potentially need to squeeze out a singleton dimension here
        return tuple(processing_fn(vjp_results))

    def compute_jacobian(self, tapes):
        partial_gradient_fn = partial(self._gradient_transform, **self._gradient_kwargs)
        jac_tapes, batch_post_processing = qml.transforms.map_batch_transform(
            partial_gradient_fn, tapes
        )
        results = self._inner_execute(jac_tapes)
        return tuple(batch_post_processing(results))


class ExperimentalDeviceDerivatives(JacobianProductCalculator):
    """Calculate jacobian products via an experimental device.

    Args:
        device (pennylane.devices.experimental.Device):
        execution_config (pennylane.devices.experimental.ExecutionConfig):

    """

    def __init__(self, device: qml.devices.experimental.Device, execution_config):
        self._device = device
        self._execution_config = execution_config
        self._jacobian_cache = {}
        self._results_cache = {}

    def __call__(self, tapes: Batch):
        tapes = tuple(qml.transforms.convert_to_numpy_parameters(t) for t in tapes)
        results, jac = self._device.execute_and_compute_derivatives(tapes, self._execution_config)
        self._jacobian_cache[id(tapes)] = jac
        self._results_cache[id(tapes)] = results
        return results

    def execute_and_compute_jvp(self, tapes: Batch, tangents):
        if id(tapes) in self._results_cache:
            results = self._results_cache[id(tapes)]
            jacs = self._jacobian_cache[id(tapes)]
        else:
            tapes = tuple(qml.transforms.convert_to_numpy_parameters(t) for t in tapes)
            results, jacs = self._device.execute_and_compute_derivatives(
                tapes, self._execution_config
            )

        multi_measurements = (len(t.measurments) > 1 for t in tapes)
        jvps = _compute_jvps(jacs, tangents, multi_measurements)
        return results, jvps

    def compute_vjp(self, tapes, dy):
        if id(tapes) in self._jacobian_cache:
            jacs = self._jacobian_cache[id(tapes)]
        else:
            tapes = tuple(qml.transforms.convert_to_numpy_parameters(t) for t in tapes)
            jacs = self._device.compute_derivatives(tapes, self._execution_config)

        multi_measurements = (len(t.measurments) > 1 for t in tapes)
        has_partitioned_shots = tapes[0].shots.has_partitioned_shots
        return _compute_vjps(
            jacs, dy, multi_measurements, has_partitioned_shots=has_partitioned_shots
        )

    def compute_jacobian(self, tapes):
        if id(tapes) in self._jacobian_cache:
            return self._jacobian_cache[id(tapes)]
        tapes = tuple(qml.transforms.convert_to_numpy_parameters(t) for t in tapes)
        return self._device.compute_derivatives(tapes, self._execution_config)