from functools import reduce
from string import ascii_letters as ABC
from pennylane.tape import QuantumScript

import numpy as np

import pennylane as qml


def _get_slice(index, axis, num_axes):
    idx = [slice(None)] * num_axes
    idx[axis] = index
    return tuple(idx)


class PlainNumpySimulator:
    """

    Current Restrictions:
    * No batching

    * No support for state preparation yet
    * No sampling yet
    * restricted measurement types

    Preprocessing restrictions:
    * Quantum Script wires must be adjacent integers starting from zero
    * All operations must have matrices



    """

    name = "PlainNumpy"

    def __init__(self):
        pass

    @classmethod
    def execute(cls, qs: QuantumScript, dtype=np.complex128):
        num_indices = len(qs.wires)
        state = cls.create_zeroes_state(num_indices, dtype=dtype)
        for op in qs._ops:
            state = cls.apply_operation(state, op)

        measurements = tuple(cls.measure_state(state, m) for m in qs.measurements)
        return measurements[0] if len(measurements) == 1 else measurements

    @staticmethod
    def create_zeroes_state(num_indices, dtype=np.complex128):

        state = np.zeros(2**num_indices, dtype=dtype)
        state[0] = 1
        state.shape = [2] * num_indices
        return state

    @staticmethod
    def create_state_vector_state(num_indices, statevector, indices):
        if list(range(num_indices)) == indices:
            statevector.shape = [2] * num_indices
            return statevector
        raise NotImplementedError

    @classmethod
    def apply_operation(cls, state, operation):
        """ """
        matrix = operation.matrix()
        if len(operation.wires) < 3:
            return cls.apply_matrix_einsum(state, matrix, operation.wires)
        return cls.apply_matrix_tensordot(state, matrix, operation.wires)

    @classmethod
    def apply_matrix(cls, state, matrix, indices):
        if len(indices) < 3:
            return cls.apply_matrix_einsum(state, matrix, indices)
        return cls.apply_matrix_tensordot(state, matrix, indices)

    @staticmethod
    def apply_matrix_tensordot(state, matrix, indices):
        """ """
        total_indices = len(state.shape)
        num_indices = len(indices)
        reshaped_mat = np.reshape(matrix, [2] * (num_indices * 2))
        axes = (tuple(range(num_indices, 2 * num_indices)), indices)

        tdot = np.tensordot(reshaped_mat, state, axes=axes)

        unused_idxs = [i for i in range(total_indices) if i not in indices]
        perm = list(indices) + unused_idxs
        inv_perm = np.argsort(perm)

        return np.transpose(tdot, inv_perm)

    @staticmethod
    def apply_matrix_einsum(state, matrix, indices):
        """
        Args:
            state (array[complex]): input state
            mat (array): matrix to multiply
            indices (Iterable[integer]): indices to apply the matrix on

        Returns:
            array[complex]: output_state
        """
        total_indices = len(state.shape)
        num_indices = len(indices)

        state_indices = ABC[:total_indices]
        affected_indices = "".join(ABC[i] for i in indices)

        new_indices = ABC[total_indices : total_indices + num_indices]

        new_state_indices = state_indices
        for old, new in zip(affected_indices, new_indices):
            new_state_indices = new_state_indices.replace(old, new)

        einsum_indices = f"{new_indices}{affected_indices},{state_indices}->{new_state_indices}"

        reshaped_mat = np.reshape(matrix, [2] * (num_indices * 2))

        return np.einsum(einsum_indices, reshaped_mat, state)

    @classmethod
    def measure_state(cls, state, measurementprocess):
        if isinstance(measurementprocess, qml.measurements.StateMeasurement):
            total_indices = len(state.shape)
            wires = qml.wires.Wires(range(total_indices))
            if (
                measurementprocess.obs is not None
                and measurementprocess.obs.has_diagonalizing_gates
            ):
                for op in measurementprocess.obs.diagonalizing_gates():
                    state = cls.apply_operation(state, op)
            return measurementprocess.process_state(state.flatten(), wires)
        return state

    @classmethod
    def generate_samples(cls, state, rng, shots=1):
        total_indices = len(state.shape)
        probs = np.real(state) ** 2 + np.imag(state) ** 2
        basis_states = np.arange(2**total_indices)
        samples = rng.choice(basis_states, shots, p=probs.flatten())

        powers_of_two = 1 << np.arange(total_indices, dtype=np.int64)
        # `samples` typically is one-dimensional, but can be two-dimensional with broadcasting.
        # In any case we want to append a new axis at the *end* of the shape.
        states_sampled_base_ten = samples[..., None] & powers_of_two
        # `states_sampled_base_ten` can be two- or three-dimensional. We revert the *last* axis.
        return (states_sampled_base_ten > 0).astype(np.int64)[..., ::-1]