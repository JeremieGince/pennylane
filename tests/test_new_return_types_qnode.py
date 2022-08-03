# Copyright 2022 Xanadu Quantum Technologies Inc.

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
Unit tests for the new return types with QNode.
"""
import pytest

import numpy as np
import pennylane as qml

wires = [2, 3, 4]
devices = ["default.qubit", "default.mixed"]


class TestIntegrationSingleReturn:
    """Test that single measurements return behavior does not change."""

    @pytest.mark.parametrize("wires", wires)
    def test_state_default(self, wires):
        """Return state with default.qubit."""
        qml.enable_return()

        dev = qml.device("default.qubit", wires=wires)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.state()

        qnode = qml.QNode(circuit, dev)
        res = qnode(0.5)

        qml.disable_return()

        assert res.shape == (2**wires,)
        assert isinstance(res, np.ndarray)

    @pytest.mark.parametrize("wires", wires)
    def test_state_mixed(self, wires):
        """Return state with default.mixed."""
        qml.enable_return()

        dev = qml.device("default.mixed", wires=wires)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.state()

        qnode = qml.QNode(circuit, dev)
        res = qnode(0.5)

        qml.disable_return()

        assert res.shape == (2**wires, 2**wires)
        assert isinstance(res, np.ndarray)

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("d_wires", wires)
    def test_density_matrix(self, d_wires, device):
        """Return density matrix with default.qubit."""
        qml.enable_return()

        dev = qml.device(device, wires=4)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.density_matrix(wires=range(0, d_wires))

        qnode = qml.QNode(circuit, dev)
        res = qnode(0.5)

        qml.disable_return()

        assert res.shape == (2**d_wires, 2**d_wires)
        assert isinstance(res, np.ndarray)

    @pytest.mark.parametrize("device", devices)
    def test_expval(self, device):
        """Return a single expval."""
        qml.enable_return()

        dev = qml.device(device, wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(wires=1))

        qnode = qml.QNode(circuit, dev)
        res = qnode(0.5)

        qml.disable_return()

        assert res.shape == ()
        assert isinstance(res, np.ndarray)

    @pytest.mark.parametrize("device", devices)
    def test_var(self, device):
        """Return a single var."""
        qml.enable_return()

        dev = qml.device(device, wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.var(qml.PauliZ(wires=1))

        qnode = qml.QNode(circuit, dev)
        res = qnode(0.5)

        qml.disable_return()

        assert res.shape == ()
        assert isinstance(res, np.ndarray)

    @pytest.mark.parametrize("device", devices)
    def test_vn_entropy(self, device):
        """Return a single vn entropy."""
        qml.enable_return()

        dev = qml.device(device, wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.vn_entropy(wires=0)

        qnode = qml.QNode(circuit, dev)
        res = qnode(0.5)

        qml.disable_return()

        assert res.shape == ()
        assert isinstance(res, np.ndarray)

    @pytest.mark.parametrize("device", devices)
    def test_mutual_info(self, device):
        """Return a single mutual information."""
        qml.enable_return()

        dev = qml.device(device, wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.mutual_info(wires0=[0], wires1=[1])

        qnode = qml.QNode(circuit, dev)
        res = qnode(0.5)

        qml.disable_return()

        assert res.shape == ()
        assert isinstance(res, np.ndarray)

    herm = np.diag([1, 2, 3, 4])
    probs_data = [
        (None, [0]),
        (None, [0, 1]),
        (qml.PauliZ(0), None),
        (qml.Hermitian(herm, wires=[1, 0]), None),
    ]

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("op,wires", probs_data)
    def test_probs(self, op, wires, device):
        """Return a single prob."""
        qml.enable_return()

        dev = qml.device(device, wires=3)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.probs(op=op, wires=wires)

        qnode = qml.QNode(circuit, dev)
        res = qnode(0.5)

        if wires is None:
            wires = op.wires

        qml.disable_return()

        assert res.shape == (2 ** len(wires),)
        assert isinstance(res, np.ndarray)

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("measurement", [qml.sample(qml.PauliZ(0)), qml.sample(wires=[0])])
    def test_sample(self, measurement, device, shots=100):
        """Test the sample measurement."""
        qml.enable_return()

        dev = qml.device(device, wires=2, shots=shots)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.apply(measurement)

        qnode = qml.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        qml.disable_return()

        assert isinstance(res, np.ndarray)
        assert res.shape == (shots,)

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("measurement", [qml.counts(qml.PauliZ(0)), qml.counts(wires=[0])])
    def test_counts(self, measurement, device, shots=100):
        """Test the counts measurement."""
        qml.enable_return()

        dev = qml.device(device, wires=2, shots=shots)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.apply(measurement)

        qnode = qml.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        qml.disable_return()

        assert isinstance(res, dict)
        assert sum(res.values()) == shots


wires = [([0], [1]), ([1], [0]), ([0], [0]), ([1], [1])]


class TestIntegrationMultipleReturns:
    """Test the new return types for multiple measurements, it should always return a tuple containing the single
    measurements.
    """

    @pytest.mark.parametrize("device", devices)
    def test_multiple_expval(self, device):
        """Return multiple expvals."""
        qml.enable_return()
        dev = qml.device(device, wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(wires=0)), qml.expval(qml.PauliZ(wires=1))

        qnode = qml.QNode(circuit, dev)
        res = qnode(0.5)

        qml.disable_return()

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], np.ndarray)
        assert res[0].shape == ()

        assert isinstance(res[1], np.ndarray)
        assert res[1].shape == ()

    @pytest.mark.parametrize("device", devices)
    def test_multiple_var(self, device):
        """Return multiple vars."""
        qml.enable_return()

        dev = qml.device(device, wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.var(qml.PauliZ(wires=0)), qml.var(qml.PauliZ(wires=1))

        qnode = qml.QNode(circuit, dev)
        res = qnode(0.5)

        qml.disable_return()

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], np.ndarray)
        assert res[0].shape == ()

        assert isinstance(res[1], np.ndarray)
        assert res[1].shape == ()

    # op1, wires1, op2, wires2
    multi_probs_data = [
        (None, [0], None, [0]),
        (None, [0], None, [0, 1]),
        (None, [0, 1], None, [0]),
        (None, [0, 1], None, [0, 1]),
        (qml.PauliZ(0), None, qml.PauliZ(1), None),
        (None, [0], qml.PauliZ(1), None),
        (qml.PauliZ(0), None, None, [0]),
        (qml.PauliZ(1), None, qml.PauliZ(0), None),
    ]

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("op1,wires1,op2,wires2", multi_probs_data)
    def test_multiple_prob(self, op1, op2, wires1, wires2, device):
        """Return multiple probs."""
        qml.enable_return()

        dev = qml.device("default.qubit", wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.probs(op=op1, wires=wires1), qml.probs(op=op2, wires=wires2)

        qnode = qml.QNode(circuit, dev)
        res = qnode(0.5)

        qml.disable_return()

        assert isinstance(res, tuple)
        assert len(res) == 2

        if wires1 is None:
            wires1 = op1.wires

        if wires2 is None:
            wires2 = op2.wires

        assert isinstance(res[0], np.ndarray)
        assert res[0].shape == (2 ** len(wires1),)

        assert isinstance(res[1], np.ndarray)
        assert res[1].shape == (2 ** len(wires2),)

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("op1,wires1,op2,wires2", multi_probs_data)
    @pytest.mark.parametrize("wires3, wires4", wires)
    def test_mix_meas(self, op1, wires1, op2, wires2, wires3, wires4, device):
        """Return multiple different measurements."""
        qml.enable_return()

        dev = qml.device(device, wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return (
                qml.probs(op=op1, wires=wires1),
                qml.vn_entropy(wires=wires3),
                qml.probs(op=op2, wires=wires2),
                qml.expval(qml.PauliZ(wires=wires4)),
            )

        qnode = qml.QNode(circuit, dev)
        res = qnode(0.5)

        if wires1 is None:
            wires1 = op1.wires

        if wires2 is None:
            wires2 = op2.wires

        qml.disable_return()
        assert isinstance(res, tuple)
        assert len(res) == 4

        assert isinstance(res[0], np.ndarray)
        assert res[0].shape == (2 ** len(wires1),)

        assert isinstance(res[1], np.ndarray)
        assert res[1].shape == ()

        assert isinstance(res[2], np.ndarray)
        assert res[2].shape == (2 ** len(wires2),)

        assert isinstance(res[3], np.ndarray)
        assert res[3].shape == ()

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("measurement", [qml.sample(qml.PauliZ(0)), qml.sample(wires=[0])])
    def test_expval_sample(self, measurement, device, shots=100):
        """Test the expval and sample measurements together."""
        qml.enable_return()

        dev = qml.device(device, wires=2, shots=shots)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.expval(qml.PauliX(1)), qml.apply(measurement)

        qnode = qml.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        qml.disable_return()

        # Expval
        assert isinstance(res[0], np.ndarray)
        assert res[0].shape == ()

        # Sample
        assert isinstance(res[1], np.ndarray)
        assert res[1].shape == (shots,)

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("measurement", [qml.counts(qml.PauliZ(0)), qml.counts(wires=[0])])
    def test_expval_counts(self, measurement, device, shots=100):
        """Test the expval and counts measurements together."""
        qml.enable_return()

        dev = qml.device(device, wires=2, shots=shots)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.expval(qml.PauliX(1)), qml.apply(measurement)

        qnode = qml.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        qml.disable_return()

        # Expval
        assert isinstance(res[0], np.ndarray)
        assert res[0].shape == ()

        # Counts
        assert isinstance(res[1], dict)
        assert sum(res[1].values()) == shots

    wires = [2, 3, 4, 5]

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("wires", wires)
    def test_list_one_expval(self, wires, device):
        """Return a comprehension list of one expvals."""
        qml.enable_return()

        dev = qml.device(device, wires=wires)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return [qml.expval(qml.PauliZ(wires=0))]

        qnode = qml.QNode(circuit, dev)
        res = qnode(0.5)

        qml.disable_return()

        assert isinstance(res, list)
        assert len(res) == 1
        assert isinstance(res[0], np.ndarray)
        assert res[0].shape == ()

    @pytest.mark.parametrize("device", devices)
    @pytest.mark.parametrize("wires", wires)
    def test_list_multiple_expval(self, wires, device):
        """Return a comprehension list of multiple expvals."""
        qml.enable_return()

        dev = qml.device(device, wires=wires)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(0, wires)]

        qnode = qml.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        qml.disable_return()
        assert isinstance(res, list)
        assert len(res) == wires

        for i in range(0, wires):
            assert isinstance(res[i], np.ndarray)
            assert res[i].shape == ()

    @pytest.mark.parametrize("device", devices)
    def test_array_multiple(self, device):
        """Return PennyLane array of multiple measurements"""
        qml.enable_return()

        dev = qml.device(device, wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.numpy.array([qml.expval(qml.PauliZ(wires=0)), qml.probs(wires=[0, 1])])

        qnode = qml.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        qml.disable_return()

        assert isinstance(res, qml.numpy.ndarray)
        assert res[0].shape == ()
        assert res[1].shape == (4,)


pauliz = qml.PauliZ(wires=1)
proj = qml.Projector([1], wires=1)
hermitian = qml.Hermitian(np.diag([1, 2]), wires=0)

# Note: mutual info and vn_entropy do not support some shot vectors
# qml.mutual_info(wires0=[0], wires1=[1]), qml.vn_entropy(wires=[0])]
single_scalar_output_measurements = [
    qml.expval(pauliz),
    qml.var(pauliz),
    qml.expval(proj),
    qml.var(proj),
    qml.expval(hermitian),
    qml.var(hermitian),
]

herm = np.diag([1, 2, 3, 4])
probs_data = [
    (None, [0]),
    (None, [0, 1]),
    (qml.PauliZ(0), None),
    (qml.Hermitian(herm, wires=[1, 0]), None),
]

shot_vectors = [[10, 1000], [1, 10, 10, 1000], [1, (10, 2), 1000]]
devices = ["default.qubit", "default.mixed"]


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("shot_vector", shot_vectors)
class TestIntegrationShotVectors:
    """Test the support for QNodes with single measurements using a device with shot vectors."""

    @pytest.mark.parametrize("measurement", single_scalar_output_measurements)
    def test_scalar(self, shot_vector, measurement, device):
        """Test a single scalar-valued measurement."""
        qml.enable_return()

        dev = qml.device(device, wires=2, shots=shot_vector)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.apply(measurement)

        qnode = qml.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        all_shots = sum([shot_tuple.copies for shot_tuple in dev.shot_vector])

        qml.disable_return()

        assert isinstance(res, tuple)
        assert len(res) == all_shots
        assert all(r.shape == () for r in res)

    @pytest.mark.parametrize("op,wires", probs_data)
    def test_probs(self, shot_vector, op, wires, device):
        """Test a single probability measurement."""
        qml.enable_return()

        dev = qml.device("default.qubit", wires=2, shots=shot_vector)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.probs(op=op, wires=wires)

        # Diff method is to be set to None otherwise use Interface execute
        qnode = qml.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        all_shots = sum([shot_tuple.copies for shot_tuple in dev.shot_vector])

        qml.disable_return()

        assert isinstance(res, tuple)
        assert len(res) == all_shots
        wires_to_use = wires if wires else op.wires
        assert all(r.shape == (2 ** len(wires_to_use),) for r in res)

    @pytest.mark.parametrize("wires", [[0], [2, 0], [1, 0], [2, 0, 1]])
    @pytest.mark.xfail
    def test_density_matrix(self, shot_vector, wires, device):
        """Test a density matrix measurement."""
        qml.enable_return()

        dev = qml.device(device, wires=3, shots=shot_vector)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.density_matrix(wires=wires)

        # Diff method is to be set to None otherwise use Interface execute
        qnode = qml.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        all_shots = sum([shot_tuple.copies for shot_tuple in dev.shot_vector])

        qml.disable_return()

        assert isinstance(res, tuple)
        assert len(res) == all_shots
        dim = 2 ** len(wires)
        assert all(r.shape == (dim, dim) for r in res)

    @pytest.mark.parametrize("measurement", [qml.sample(qml.PauliZ(0)), qml.sample(wires=[0])])
    def test_samples(self, shot_vector, measurement, device):
        """Test the sample measurement."""
        qml.enable_return()

        dev = qml.device(device, wires=2, shots=shot_vector)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.apply(measurement)

        # Diff method is to be set to None otherwise use Interface execute
        qnode = qml.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        all_shot_copies = [
            shot_tuple.shots for shot_tuple in dev.shot_vector for _ in range(shot_tuple.copies)
        ]

        qml.disable_return()

        assert len(res) == len(all_shot_copies)
        for r, shots in zip(res, all_shot_copies):

            if shots == 1:
                # Scalar tensors
                assert r.shape == ()
            else:
                assert r.shape == (shots,)

    @pytest.mark.parametrize("measurement", [qml.counts(qml.PauliZ(0)), qml.counts(wires=[0])])
    def test_counts(self, shot_vector, measurement, device):
        """Test the counts measurement."""

        qml.enable_return()

        dev = qml.device(device, wires=2, shots=shot_vector)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.apply(measurement)

        # Diff method is to be set to None otherwise use Interface execute
        qnode = qml.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        all_shots = sum([shot_tuple.copies for shot_tuple in dev.shot_vector])

        qml.disable_return()

        assert isinstance(res, tuple)
        assert len(res) == all_shots
        assert all(isinstance(r, dict) for r in res)


@pytest.mark.parametrize("shot_vector", shot_vectors)
@pytest.mark.parametrize("device", devices)
class TestIntegrationSameMeasurementShotVector:
    """Test the support for executing QNodes with the same type of measurement multiple times using a device with
    shot vectors"""

    def test_scalar(self, shot_vector, device):
        """Test multiple scalar-valued measurements."""
        qml.enable_return()

        dev = qml.device("default.qubit", wires=2, shots=shot_vector)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.expval(qml.PauliX(0)), qml.var(qml.PauliZ(1))

        qnode = qml.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        qml.disable_return()

        all_shots = sum([shot_tuple.copies for shot_tuple in dev.shot_vector])

        assert isinstance(res, tuple)
        assert len(res) == all_shots
        for r in res:
            assert len(r) == 2
            assert all(r.shape == () for r in r)

    probs_data2 = [
        (None, [2]),
        (None, [2, 3]),
        (qml.PauliZ(2), None),
        (qml.Hermitian(herm, wires=[3, 2]), None),
    ]

    @pytest.mark.parametrize("op1,wires1", probs_data)
    @pytest.mark.parametrize("op2,wires2", reversed(probs_data2))
    def test_probs(self, shot_vector, op1, wires1, op2, wires2, device):
        """Test multiple probability measurements."""
        dev = qml.device("default.qubit", wires=4, shots=shot_vector)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.probs(op=op1, wires=wires1), qml.probs(op=op2, wires=wires2)

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        all_shots = sum([shot_tuple.copies for shot_tuple in dev.shot_vector])

        assert isinstance(res[0], tuple)
        assert len(res[0]) == all_shots

        wires1 = wires1 if wires1 else op1.wires
        wires2 = wires2 if wires2 else op2.wires
        for r in res[0]:
            assert len(r) == 2
            assert r[0].shape == (2 ** len(wires1),)
            assert r[1].shape == (2 ** len(wires2),)

    @pytest.mark.parametrize("measurement1", [qml.sample(qml.PauliZ(0)), qml.sample(wires=[0])])
    @pytest.mark.parametrize("measurement2", [qml.sample(qml.PauliX(1)), qml.sample(wires=[1])])
    def test_samples(self, shot_vector, measurement1, measurement2, device):
        """Test multiple sample measurements."""
        dev = qml.device("default.qubit", wires=2, shots=shot_vector)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.apply(measurement1), qml.apply(measurement2)

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        all_shot_copies = [
            shot_tuple.shots for shot_tuple in dev.shot_vector for _ in range(shot_tuple.copies)
        ]

        assert len(res[0]) == len(all_shot_copies)
        for r, shots in zip(res[0], all_shot_copies):

            shape = () if shots == 1 else (shots,)
            assert all(res_item.shape == shape for res_item in r)

    @pytest.mark.parametrize("measurement1", [qml.counts(qml.PauliZ(0)), qml.counts(wires=[0])])
    @pytest.mark.parametrize("measurement2", [qml.counts(qml.PauliZ(0)), qml.counts(wires=[0])])
    def test_counts(self, shot_vector, measurement1, measurement2, device):
        """Test multiple counts measurements."""
        dev = qml.device("default.qubit", wires=2, shots=shot_vector)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.apply(measurement1), qml.apply(measurement2)

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        all_shots = sum([shot_tuple.copies for shot_tuple in dev.shot_vector])

        assert isinstance(res[0], tuple)
        assert len(res[0]) == all_shots
        for r in res[0]:
            assert isinstance(r, tuple)
            assert all(isinstance(res_item, dict) for res_item in r)


# -------------------------------------------------
# Shot vector multi measurement tests - test data
# -------------------------------------------------

pauliz_w2 = qml.PauliZ(wires=2)
proj_w2 = qml.Projector([1], wires=2)
hermitian = qml.Hermitian(np.diag([1, 2]), wires=0)
tensor_product = qml.PauliY(wires=2) @ qml.PauliX(wires=1)

# Expval/Var with Probs

scalar_probs_multi = [
    # Expval
    (qml.expval(pauliz_w2), qml.probs(wires=[2, 0])),
    (qml.expval(proj_w2), qml.probs(wires=[2, 0])),
    (qml.expval(tensor_product), qml.probs(wires=[2, 0])),
    # Var
    (qml.var(qml.PauliZ(wires=1)), qml.probs(wires=[0, 1])),
    (qml.var(proj_w2), qml.probs(wires=[2, 0])),
    (qml.var(tensor_product), qml.probs(wires=[2, 0])),
]

# Expval/Var with Sample

scalar_sample_multi = [
    # Expval
    (qml.expval(pauliz_w2), qml.sample(op=qml.PauliZ(1) @ qml.PauliZ(0))),
    (qml.expval(proj_w2), qml.sample(op=qml.PauliZ(1) @ qml.PauliZ(0))),
    (qml.expval(tensor_product), qml.sample(op=qml.PauliZ(0))),
    # Var
    (qml.var(proj_w2), qml.sample(op=qml.PauliZ(1) @ qml.PauliZ(0))),
    (qml.var(pauliz_w2), qml.sample(op=qml.PauliZ(1) @ qml.PauliZ(0))),
    (qml.var(tensor_product), qml.sample(op=qml.PauliZ(0))),
]

scalar_sample_no_obs_multi = [
    # TODO: for copy=1, the wires syntax has a bug
    # -----
    (qml.expval(qml.PauliZ(wires=1)), qml.sample(wires=[0, 1])),
    (qml.var(qml.PauliZ(wires=1)), qml.sample(wires=[0, 1])),
]

# Expval/Var with Counts

scalar_counts_multi = [
    # Expval
    (qml.expval(pauliz_w2), qml.counts(op=qml.PauliZ(1) @ qml.PauliZ(0))),
    (qml.expval(proj_w2), qml.counts(op=qml.PauliZ(1) @ qml.PauliZ(0))),
    (qml.expval(tensor_product), qml.counts(op=qml.PauliZ(0))),
    # Var
    (qml.var(proj_w2), qml.counts(op=qml.PauliZ(1) @ qml.PauliZ(0))),
    (qml.var(pauliz_w2), qml.counts(op=qml.PauliZ(1) @ qml.PauliZ(0))),
    (qml.var(tensor_product), qml.counts(op=qml.PauliZ(0))),
]

scalar_counts_no_obs_multi = [
    # TODO: for copy=1, the wires syntax has a bug
    # -----
    (qml.expval(qml.PauliZ(wires=1)), qml.counts(wires=[0, 1])),
    (qml.var(qml.PauliZ(wires=1)), qml.counts(wires=[0, 1])),
]


@pytest.mark.parametrize("shot_vector", shot_vectors)
@pytest.mark.parametrize("device", devices)
class TestIntegrationMultipleMeasurementsShotVector:
    """Test the support for executing QNodes with multiple different measurements using a device with shot vectors"""

    @pytest.mark.parametrize("meas1,meas2", scalar_probs_multi)
    def test_scalar_probs(self, shot_vector, meas1, meas2, device):
        """Test scalar-valued and probability measurements"""
        qml.enable_return()

        dev = qml.device(device, wires=3, shots=shot_vector)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.apply(meas1), qml.apply(meas2)

        qnode = qml.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        all_shots = sum([shot_tuple.copies for shot_tuple in dev.shot_vector])

        qml.disable_return()

        assert isinstance(res, tuple)
        assert len(res) == all_shots
        assert all(isinstance(r, tuple) for r in res)
        assert all(isinstance(m, np.ndarray) for measurement_res in res for m in measurement_res)
        for meas_res in res:
            for i, r in enumerate(meas_res):
                if i % 2 == 0:

                    # Scalar-val meas
                    assert r.shape == ()
                else:
                    assert r.shape == (2**2,)

                    # Probs add up to 1
                    assert np.allclose(sum(r), 1)

    @pytest.mark.parametrize("meas1,meas2", scalar_sample_multi)
    def test_scalar_sample_with_obs(self, shot_vector, meas1, meas2, device):
        """Test scalar-valued and sample measurements where sample takes an
        observable."""
        qml.enable_return()

        dev = qml.device(device, wires=3, shots=shot_vector)
        raw_shot_vector = [
            shot_tuple.shots for shot_tuple in dev.shot_vector for _ in range(shot_tuple.copies)
        ]

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.apply(meas1), qml.apply(meas2)

        qnode = qml.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        all_shots = sum([shot_tuple.copies for shot_tuple in dev.shot_vector])

        qml.disable_return()

        assert isinstance(res, tuple)
        assert len(res) == all_shots
        assert all(isinstance(r, tuple) for r in res)
        assert all(isinstance(m, np.ndarray) for measurement_res in res for m in measurement_res)

        for idx, shots in enumerate(raw_shot_vector):
            for i, r in enumerate(res[idx]):
                if i % 2 == 0 or shots == 1:
                    obs_provided = meas2.obs is not None
                    expected_shape = ()
                    assert r.shape == expected_shape
                else:
                    assert r.shape == (shots,)

    @pytest.mark.parametrize("meas1,meas2", scalar_sample_no_obs_multi)
    @pytest.mark.xfail
    def test_scalar_sample_no_obs(self, shot_vector, meas1, meas2, device):
        """Test scalar-valued and computational basis sample measurements."""
        qml.enable_return()

        dev = qml.device(device, wires=3, shots=shot_vector)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.apply(meas1), qml.apply(meas2)

        qnode = qml.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        all_shots = sum([shot_tuple.copies for shot_tuple in dev.shot_vector])

        qml.disable_return()

        assert isinstance(res, tuple)
        assert len(res) == all_shots
        assert all(isinstance(r, tuple) for r in res)
        assert all(isinstance(m, np.ndarray) for measurement_res in res for m in measurement_res)

        for shot_tuple in dev.shot_vector:
            for idx in range(shot_tuple.copies):
                for i, r in enumerate(res[idx]):
                    expected_sample_shape_item = len(meas2.wires)
                    if i % 2 == 0 or shot_tuple.shots == 1:
                        obs_provided = meas2.obs is not None
                        expected_shape = ()
                        assert r.shape == expected_shape
                    else:
                        assert r.shape == (shot_tuple.shots,)

    @pytest.mark.parametrize("meas1,meas2", scalar_counts_multi)
    def test_scalar_counts_with_obs(self, shot_vector, meas1, meas2, device):
        """Test scalar-valued and counts measurements where counts takes an
        observable."""
        qml.enable_return()

        dev = qml.device(device, wires=3, shots=shot_vector)
        raw_shot_vector = [
            shot_tuple.shots for shot_tuple in dev.shot_vector for _ in range(shot_tuple.copies)
        ]

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.apply(meas1), qml.apply(meas2)

        qnode = qml.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        all_shots = sum([shot_tuple.copies for shot_tuple in dev.shot_vector])

        qml.disable_return()

        assert isinstance(res, tuple)
        assert len(res) == all_shots
        assert all(isinstance(r, tuple) for r in res)

        for r in res:
            assert isinstance(r[0], np.ndarray)
            assert isinstance(r[1], dict)

        expected_outcomes = {-1, 1}

        for idx, shots in enumerate(raw_shot_vector):
            for i, r in enumerate(res[idx]):
                if i % 2 == 0:
                    obs_provided = meas2.obs is not None
                    expected_shape = ()
                    assert r.shape == expected_shape
                else:
                    # Samples are either -1 or 1
                    assert set(r.keys()).issubset(expected_outcomes)
                    assert sum(r.values()) == shots

    @pytest.mark.parametrize("meas1,meas2", scalar_counts_no_obs_multi)
    @pytest.mark.xfail
    def test_scalar_counts_no_obs(self, shot_vector, meas1, meas2, device):
        """Test scalar-valued and computational basis counts measurements."""
        dev = qml.device(device, wires=3, shots=shot_vector)
        raw_shot_vector = [
            shot_tuple.shots for shot_tuple in dev.shot_vector for _ in range(shot_tuple.copies)
        ]

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.apply(meas1), qml.apply(meas2)

        qnode = qml.QNode(circuit, dev)
        res = qnode(0.5)
        print(res)

        all_shots = sum([shot_tuple.copies for shot_tuple in dev.shot_vector])

        assert isinstance(res, tuple)
        assert len(res) == all_shots
        assert all(isinstance(r, tuple) for r in res)
        assert all(isinstance(m, np.ndarray) for measurement_res in res for m in measurement_res)

        for idx, shots in enumerate(raw_shot_vector):
            for i, r in enumerate(res[idx]):
                expected_sample_shape_item = len(meas2.wires)
                if i % 2 == 0 or shots == 1:
                    obs_provided = meas2.obs is not None
                    expected_shape = ()
                    assert r.shape == expected_shape
                else:
                    assert r.shape == (shots,)

    @pytest.mark.parametrize("sample_obs", [qml.PauliZ, None])
    def test_probs_sample(self, shot_vector, sample_obs, device):
        """Test probs and sample measurements."""
        qml.enable_return()

        dev = qml.device(device, wires=3, shots=shot_vector)
        raw_shot_vector = [
            shot_tuple.shots for shot_tuple in dev.shot_vector for _ in range(shot_tuple.copies)
        ]

        meas1_wires = [0, 1]
        meas2_wires = [2]

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            if sample_obs is not None:
                # Observable provided to sample
                return qml.probs(wires=meas1_wires), qml.sample(sample_obs(meas2_wires))

            # Only wires provided to sample
            return qml.probs(wires=meas1_wires), qml.sample(wires=meas2_wires)

        qnode = qml.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        all_shots = sum([shot_tuple.copies for shot_tuple in dev.shot_vector])

        qml.disable_return()

        assert isinstance(res, tuple)
        assert len(res) == all_shots
        assert all(isinstance(r, tuple) for r in res)
        assert all(isinstance(m, np.ndarray) for measurement_res in res for m in measurement_res)

        for idx, shots in enumerate(raw_shot_vector):
            for i, r in enumerate(res[idx]):
                expected_sample_shape_item = len(meas2_wires)
                if i % 2 == 0:
                    expected_shape = (len(meas1_wires) ** 2,)
                    assert r.shape == expected_shape

                    # Probs add up to 1
                    assert np.allclose(sum(r), 1)
                else:
                    if shots == 1:
                        assert r.shape == ()
                    else:
                        expected = (shots,)
                        assert r.shape == expected

    @pytest.mark.parametrize("sample_obs", [qml.PauliZ, None])
    def test_probs_counts(self, shot_vector, sample_obs, device):
        """Test probs and counts measurements."""
        qml.enable_return()

        dev = qml.device(device, wires=3, shots=shot_vector)
        raw_shot_vector = [
            shot_tuple.shots for shot_tuple in dev.shot_vector for _ in range(shot_tuple.copies)
        ]

        meas1_wires = [0, 1]
        meas2_wires = [2]

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            if sample_obs is not None:
                # Observable provided to sample
                return qml.probs(wires=meas1_wires), qml.counts(sample_obs(meas2_wires))

            # Only wires provided to sample
            return qml.probs(wires=meas1_wires), qml.counts(wires=meas2_wires)

        qnode = qml.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        all_shots = sum([shot_tuple.copies for shot_tuple in dev.shot_vector])

        qml.disable_return()

        assert isinstance(res, tuple)
        assert len(res) == all_shots
        assert all(isinstance(r, tuple) for r in res)
        assert all(isinstance(measurement_res[0], np.ndarray) for measurement_res in res)
        assert all(isinstance(measurement_res[1], dict) for measurement_res in res)

        expected_outcomes = {-1, 1} if sample_obs is not None else {"0", "1"}
        for idx, shots in enumerate(raw_shot_vector):
            for i, r in enumerate(res[idx]):
                if i % 2 == 0:
                    expected_shape = (len(meas1_wires) ** 2,)
                    assert r.shape == expected_shape

                    # Probs add up to 1
                    assert np.allclose(sum(r), 1)
                else:
                    # Samples are -1 or 1
                    assert set(r.keys()).issubset(expected_outcomes)
                    assert sum(r.values()) == shots

    @pytest.mark.parametrize("sample_wires", [[1], [0, 2]])
    @pytest.mark.parametrize("counts_wires", [[4], [3, 5]])
    def test_sample_counts(self, shot_vector, sample_wires, counts_wires, device):
        """Test sample and counts measurements, each measurement with custom
        samples or computational basis state samples."""
        qml.enable_return()

        dev = qml.device(device, wires=6, shots=shot_vector)
        raw_shot_vector = [
            shot_tuple.shots for shot_tuple in dev.shot_vector for _ in range(shot_tuple.copies)
        ]

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])

            # 1. Sample obs and Counts obs
            if len(sample_wires) == 1 and len(counts_wires) == 1:
                return qml.sample(qml.PauliY(sample_wires)), qml.counts(qml.PauliX(counts_wires))

            # 2. Sample no obs and Counts obs
            if len(sample_wires) > 1 and len(counts_wires) == 1:
                return qml.sample(wires=sample_wires), qml.counts(qml.PauliX(counts_wires))

            # 3. Sample obs and Counts no obs
            if len(sample_wires) == 1 and len(counts_wires) > 1:
                return qml.sample(qml.PauliY(sample_wires)), qml.counts(wires=counts_wires)

            # 4. Sample no obs and Counts no obs
            return qml.sample(wires=sample_wires), qml.counts(wires=counts_wires)

        qnode = qml.QNode(circuit, dev, diff_method=None)
        res = qnode(0.5)

        qml.disable_return()

        all_shots = sum([shot_tuple.copies for shot_tuple in dev.shot_vector])

        assert isinstance(res, tuple)
        assert len(res) == all_shots
        assert all(isinstance(r, tuple) for r in res)
        assert all(isinstance(measurement_res[0], np.ndarray) for measurement_res in res)
        assert all(isinstance(measurement_res[1], dict) for measurement_res in res)

        for idx, shots in enumerate(raw_shot_vector):
            for i, r in enumerate(res[idx]):
                num_wires = len(sample_wires)
                if shots == 1 and i % 2 == 0:
                    expected_shape = () if num_wires == 1 else (num_wires,)
                    assert r.shape == expected_shape
                elif i % 2 == 0:
                    expected_shape = (shots,) if num_wires == 1 else (shots, num_wires)
                    assert r.shape == expected_shape
                else:
                    assert isinstance(r, dict)

    @pytest.mark.parametrize("meas1,meas2", scalar_probs_multi)
    def test_scalar_probs_sample_counts(self, shot_vector, meas1, meas2, device):
        """Test scalar-valued, probability, sample and counts measurements all
        in a single qfunc."""
        qml.enable_return()

        dev = qml.device(device, wires=5, shots=shot_vector)
        raw_shot_vector = [
            shot_tuple.shots for shot_tuple in dev.shot_vector for _ in range(shot_tuple.copies)
        ]

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return (
                qml.apply(meas1),
                qml.apply(meas2),
                qml.sample(qml.PauliX(4)),
                qml.counts(qml.PauliX(3)),
            )

        qnode = qml.QNode(circuit, dev, diff_method=None)

        res = qnode(0.5)

        qml.disable_return()

        all_shots = sum([shot_tuple.copies for shot_tuple in dev.shot_vector])

        assert isinstance(res, tuple)
        assert len(res) == all_shots
        assert all(isinstance(r, tuple) for r in res)

        for res_idx, meas_res in enumerate(res):
            for i, r in enumerate(meas_res):
                num_meas = i % 4
                expval_or_var = num_meas == 0
                probs = num_meas == 1
                sample = num_meas == 2

                if expval_or_var:
                    assert r.shape == ()
                elif probs:
                    assert r.shape == (2**2,)

                    # Probs add up to 1
                    assert np.allclose(sum(r), 1)
                elif sample:
                    shots = raw_shot_vector[res_idx]
                    if shots == 1:
                        assert r.shape == ()
                    else:
                        expected = (shots,)
                        assert r.shape == expected
                else:
                    # Return is Counts
                    assert isinstance(r, dict)


class TestIntegrationJacobianBackpropMultipleReturns:
    """Test the new return types for the Jacobian of multiple measurements, with backprop."""

    @pytest.mark.autograd
    def test_multiple_expval_autograd(self):
        """Return Jacobian of multiple expvals."""
        dev = qml.device("default.qubit", wires=2)
        qml.enable_return()

        @qml.qnode(dev)
        def circuit(a):
            qml.RX(a[0], wires=0)
            qml.CNOT(wires=(0, 1))
            qml.RY(a[1], wires=1)
            qml.RZ(a[2], wires=1)
            return qml.expval(qml.PauliZ(wires=0)), qml.expval(qml.PauliZ(wires=1))

        x = qml.numpy.array([0.1, 0.2, 0.3], requires_grad=True)
        res = qml.jacobian(circuit)(x)

        qml.disable_return()

        assert isinstance(res, np.ndarray)
        assert res.shape == (2, 3)

    @pytest.mark.torch
    def test_multiple_expval_torch(self):
        """Return Jacobian of multiple expvals."""
        import torch

        dev = qml.device("default.qubit", wires=2)
        qml.enable_return()

        @qml.qnode(dev, interface="torch")
        def circuit(a):
            qml.RX(a[0], wires=0)
            qml.CNOT(wires=(0, 1))
            qml.RY(a[1], wires=1)
            qml.RZ(a[2], wires=1)
            return qml.expval(qml.PauliZ(wires=0)), qml.expval(qml.PauliZ(wires=1))

        x = torch.tensor([0.1, 0.2, 0.3])

        res = torch.autograd.functional.jacobian(circuit, x)

        qml.disable_return()

        assert isinstance(res, tuple)
        assert len(res) == 2
        for elem in res:
            assert isinstance(elem, torch.Tensor)
            assert elem.shape == (3,)

    @pytest.mark.tf
    def test_multiple_expval_tf(self):
        """Return Jacobian of multiple expvals."""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)
        qml.enable_return()

        @qml.qnode(dev, interface="tf")
        def circuit(a):
            qml.RX(a[0], wires=0)
            qml.CNOT(wires=(0, 1))
            qml.RY(a[1], wires=1)
            qml.RZ(a[2], wires=1)
            return qml.expval(qml.PauliZ(wires=0)), qml.expval(qml.PauliZ(wires=1))

        x = tf.Variable([0.1, 0.2, 0.3])

        with tf.GradientTape() as tape:
            out = circuit(x)

        res = tape.jacobian(out, x)

        qml.disable_return()

        assert isinstance(res, tf.Tensor)
        assert res.shape == (2, 3)

    @pytest.mark.tf
    def test_multiple_meas_tf_autograph(self):
        """Return Jacobian of multiple measurements with Tf Autograph."""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)
        qml.enable_return()

        @tf.function
        @qml.qnode(dev, interface="tf")
        def circuit(a):
            qml.RX(a[0], wires=0)
            qml.CNOT(wires=(0, 1))
            qml.RY(a[1], wires=1)
            qml.RZ(a[2], wires=1)
            return qml.expval(qml.PauliZ(wires=0)), qml.expval(qml.PauliZ(wires=1))

        # Autograph does not support multiple measurments with different shape.

        x = tf.Variable([0.1, 0.2, 0.3])

        with tf.GradientTape() as tape:
            out = circuit(x)

        res = tape.jacobian(out, x)
        qml.disable_return()

        assert isinstance(res, tf.Tensor)
        assert res.shape == (2, 3)

    @pytest.mark.jax
    def test_multiple_expval_jax(self):
        """Return Jacobian of multiple expvals."""
        import jax

        dev = qml.device("default.qubit", wires=2)
        qml.enable_return()

        @qml.qnode(dev, interface="jax")
        def circuit(a):
            qml.RX(a[0], wires=0)
            qml.CNOT(wires=(0, 1))
            qml.RY(a[1], wires=1)
            qml.RZ(a[2], wires=1)
            return qml.expval(qml.PauliZ(wires=0)), qml.expval(qml.PauliZ(wires=1))

        x = jax.numpy.array([0.1, 0.2, 0.3])
        res = jax.jacobian(circuit)(x)

        qml.disable_return()

        assert isinstance(res, tuple)
        assert len(res) == 2
        for elem in res:
            assert isinstance(elem, jax.numpy.ndarray)
            assert elem.shape == (3,)

    @pytest.mark.autograd
    def test_multiple_probs_autograd(self):
        """Return Jacobian of multiple probs."""
        dev = qml.device("default.qubit", wires=2)
        qml.enable_return()

        @qml.qnode(dev)
        def circuit(a):
            qml.RX(a[0], wires=0)
            qml.CNOT(wires=(0, 1))
            qml.RY(a[1], wires=1)
            qml.RZ(a[2], wires=1)
            return qml.probs(op=qml.PauliZ(wires=0)), qml.probs(wires=1)

        x = qml.numpy.array([0.1, 0.2, 0.3], requires_grad=True)
        res = qml.jacobian(circuit)(x)

        qml.disable_return()

        assert isinstance(res, np.ndarray)
        assert res.shape == (2, 2, 3)

    @pytest.mark.torch
    def test_multiple_probs_torch(self):
        """Return Jacobian of multiple probs."""
        import torch

        dev = qml.device("default.qubit", wires=2)
        qml.enable_return()

        @qml.qnode(dev, interface="torch")
        def circuit(a):
            qml.RX(a[0], wires=0)
            qml.CNOT(wires=(0, 1))
            qml.RY(a[1], wires=1)
            qml.RZ(a[2], wires=1)
            return qml.probs(op=qml.PauliZ(wires=0)), qml.probs(wires=1)

        x = torch.tensor([0.1, 0.2, 0.3])

        res = torch.autograd.functional.jacobian(circuit, x)

        qml.disable_return()

        assert isinstance(res, tuple)
        assert len(res) == 2
        for elem in res:
            assert isinstance(elem, torch.Tensor)
            assert elem.shape == (2, 3)

    @pytest.mark.tf
    def test_multiple_probs_tf(self):
        """Return Jacobian of multiple probs."""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)
        qml.enable_return()

        @qml.qnode(dev, interface="tf")
        def circuit(a):
            qml.RX(a[0], wires=0)
            qml.CNOT(wires=(0, 1))
            qml.RY(a[1], wires=1)
            qml.RZ(a[2], wires=1)
            return qml.probs(op=qml.PauliZ(wires=0)), qml.probs(wires=1)

        x = tf.Variable([0.1, 0.2, 0.3])

        with tf.GradientTape() as tape:
            out = circuit(x)

        res = tape.jacobian(out, x)

        qml.disable_return()

        assert isinstance(res, tf.Tensor)
        assert res.shape == (2, 2, 3)

    @pytest.mark.jax
    def test_multiple_probs_jax(self):
        """Return Jacobian of multiple probs."""
        import jax

        dev = qml.device("default.qubit", wires=2)
        qml.enable_return()

        @qml.qnode(dev, interface="jax")
        def circuit(a):
            qml.RX(a[0], wires=0)
            qml.CNOT(wires=(0, 1))
            qml.RY(a[1], wires=1)
            qml.RZ(a[2], wires=1)
            return qml.probs(op=qml.PauliZ(wires=0)), qml.probs(wires=1)

        x = jax.numpy.array([0.1, 0.2, 0.3])

        res = jax.jacobian(circuit)(x)

        qml.disable_return()

        assert isinstance(res, tuple)
        assert len(res) == 2
        for elem in res:
            assert isinstance(elem, jax.numpy.ndarray)
            assert elem.shape == (2, 3)

    @pytest.mark.autograd
    def test_multiple_meas_autograd(self):
        """Return Jacobian of multiple measurements."""
        dev = qml.device("default.qubit", wires=2)
        qml.enable_return()

        @qml.qnode(dev)
        def circuit(a):
            qml.RX(a[0], wires=0)
            qml.CNOT(wires=(0, 1))
            qml.RY(a[1], wires=1)
            qml.RZ(a[2], wires=1)
            return qml.expval(qml.PauliZ(wires=0)), qml.probs(wires=[0, 1]), qml.vn_entropy(wires=1)

        x = qml.numpy.array([0.1, 0.2, 0.3], requires_grad=True)
        res = qml.jacobian(circuit)(x)

        qml.disable_return()

        assert isinstance(res, np.ndarray)
        assert res.shape == (6, 3)

    @pytest.mark.torch
    def test_multiple_meas_torch(self):
        """Return Jacobian of multiple measurements."""
        import torch

        dev = qml.device("default.qubit", wires=2)
        qml.enable_return()

        @qml.qnode(dev, interface="torch")
        def circuit(a):
            qml.RX(a[0], wires=0)
            qml.CNOT(wires=(0, 1))
            qml.RY(a[1], wires=1)
            qml.RZ(a[2], wires=1)
            return qml.expval(qml.PauliZ(wires=0)), qml.probs(wires=[0, 1]), qml.vn_entropy(wires=1)

        x = torch.tensor([0.1, 0.2, 0.3])

        res = torch.autograd.functional.jacobian(circuit, x)

        qml.disable_return()

        assert isinstance(res, tuple)
        assert len(res) == 3
        for i, elem in enumerate(res):
            assert isinstance(elem, torch.Tensor)
            if i == 0:
                assert elem.shape == (3,)
            elif i == 1:
                assert elem.shape == (4, 3)
            elif i == 2:
                assert elem.shape == (3,)

    @pytest.mark.tf
    def test_multiple_meas_tf(self):
        """Return Jacobian of multiple measurements."""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)
        qml.enable_return()

        @qml.qnode(dev, interface="tf")
        def circuit(a):
            qml.RX(a[0], wires=0)
            qml.CNOT(wires=(0, 1))
            qml.RY(a[1], wires=1)
            qml.RZ(a[2], wires=1)
            return qml.expval(qml.PauliZ(wires=0)), qml.probs(wires=[0, 1]), qml.vn_entropy(wires=1)

        x = tf.Variable([0.1, 0.2, 0.3])

        with tf.GradientTape() as tape:
            out = circuit(x)

        res = tape.jacobian(out, x)

        qml.disable_return()

        assert isinstance(res, tf.Tensor)
        assert res.shape == (6, 3)

    @pytest.mark.jax
    def test_multiple_meas_jax(self):
        """Return Jacobian of multiple measurements."""
        import jax

        dev = qml.device("default.qubit", wires=2)
        qml.enable_return()

        @qml.qnode(dev, interface="jax")
        def circuit(a):
            qml.RX(a[0], wires=0)
            qml.CNOT(wires=(0, 1))
            qml.RY(a[1], wires=1)
            qml.RZ(a[2], wires=1)
            return qml.expval(qml.PauliZ(wires=0)), qml.probs(wires=[0, 1]), qml.vn_entropy(wires=1)

        x = jax.numpy.array([0.1, 0.2, 0.3])

        res = jax.jacobian(circuit)(x)

        qml.disable_return()

        assert isinstance(res, tuple)
        assert len(res) == 3
        for i, elem in enumerate(res):
            assert isinstance(elem, jax.numpy.ndarray)
            if i == 0:
                assert elem.shape == (3,)
            elif i == 1:
                assert elem.shape == (4, 3)
            elif i == 2:
                assert elem.shape == (3,)
