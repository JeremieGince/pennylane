# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for default qubit preprocessing."""
from typing import Sequence

from flaky import flaky
import numpy as np
import pytest

import pennylane as qml
from pennylane.devices.qubit.apply_operation import apply_mid_measure, MidMeasureMP
from pennylane.devices.qubit.simulate import accumulate_native_mcm, gather_mcm


def validate_counts(shots, results1, results2):
    if isinstance(results1, Sequence):
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            validate_counts(shots, r1, r2)
        return
    for r1, r2 in zip(sorted(results1.items()), sorted(results2.items())):
        key1, val1 = r1
        key2, val2 = r2
        assert key1 == key2
        assert abs(val1 - val2) < (abs(val1 + val2) // 10)


def validate_samples(shots, results1, results2):
    for res in [results1, results2]:
        if isinstance(shots, Sequence):
            for s, r1, r2 in zip(shots, results1, results2):
                validate_samples(s, r1, r2)
        else:
            sh1, sh2 = results1.shape[0], results2.shape[0]
            assert abs(sh1 - sh2) < (abs(sh1 + sh2) // 10)
            assert results1.shape[1] == results2.shape[1]
            assert abs(np.sum(results1) - np.sum(results2)) < (results2.size // 10)


def validate_expval(shots, results1, results2):
    if shots is None:
        assert np.allclose(results1, results2)
    assert np.allclose(results1, results2, atol=0.001, rtol=0.3)


def validate_measurements(func, shots, results1, results2):
    if func is qml.counts:
        validate_counts(shots, results1, results2)
        return

    if func is qml.sample:
        validate_samples(shots, results1, results2)
        return

    validate_expval(shots, results1, results2)


def test_apply_mid_measure():
    with pytest.raises(ValueError, match="MidMeasureMP cannot be applied to batched states."):
        _ = apply_mid_measure(MidMeasureMP(0), np.zeros((2, 2)), is_state_batched=True)
    state, sample = apply_mid_measure(MidMeasureMP(0), np.zeros(2))
    assert sample == 0
    assert np.allclose(state, 0.0)


def tests_accumulate_native_mcm():
    with pytest.raises(
        TypeError, match="Measurement should be of class SampleMP but is of class CountsMP"
    ):
        accumulate_native_mcm(
            qml.tape.QuantumScript([], [qml.counts(qml.PauliZ(0))]), [None], [None]
        )


@pytest.mark.parametrize(
    "measurement",
    [
        qml.state(),
        qml.density_matrix(0),
        qml.vn_entropy(0),
        qml.mutual_info(0, 1),
        qml.purity(0),
        qml.classical_shadow(0),
        qml.shadow_expval(0),
    ],
)
def test_gather_mcm(measurement):
    with pytest.raises(ValueError, match="Native mid-circuit measurement mode does not support"):
        gather_mcm(measurement, None, None)


def test_unsupported_measurement():
    dev = qml.device("default.qubit", shots=1000)
    params = np.pi / 4 * np.ones(2)

    @qml.qnode(dev)
    def func(x, y):
        qml.RX(x, wires=0)
        m0 = qml.measure(0)
        qml.cond(m0, qml.RY)(y, wires=1)
        return qml.classical_shadow(wires=0)

    with pytest.raises(
        ValueError,
        match="Native mid-circuit measurement mode does not support ClassicalShadowMP measurements.",
    ):
        func(*params)


@flaky(max_runs=5)
@pytest.mark.parametrize("shots", [None, 1000, [1000, 1001]])
@pytest.mark.parametrize("postselect", [None, 0, 1])
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("measure_f", [qml.counts, qml.expval, qml.probs, qml.sample, qml.var])
def test_single_mcm_single_measure_mcm(shots, postselect, reset, measure_f):
    """Tests that DefaultQubit handles a circuit with a single mid-circuit measurement and a
    conditional gate. A single measurement of the mid-circuit measurement value is performed at
    the end."""

    dev = qml.device("default.qubit", shots=shots)
    params = np.pi / 4 * np.ones(2)

    @qml.qnode(dev)
    def func1(x, y):
        qml.RX(x, wires=0)
        m0 = qml.measure(0, reset=reset, postselect=postselect)
        qml.cond(m0, qml.RY)(y, wires=1)
        return measure_f(op=m0)

    @qml.qnode(dev)
    @qml.defer_measurements
    def func2(x, y):
        qml.RX(x, wires=0)
        m0 = qml.measure(0, reset=reset, postselect=postselect)
        qml.cond(m0, qml.RY)(y, wires=1)
        return measure_f(op=m0)

    if shots is None and measure_f in (qml.counts, qml.sample):
        return

    results1 = func1(*params)
    results2 = func2(*params)

    if postselect is None or measure_f in (qml.expval, qml.probs, qml.var):
        validate_measurements(measure_f, shots, results1, results2)


@flaky(max_runs=5)
@pytest.mark.parametrize("shots", [None, 1000, [1000, 1001]])
@pytest.mark.parametrize("postselect", [None, 0, 1])
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("measure_f", [qml.counts, qml.expval, qml.probs, qml.sample, qml.var])
def test_single_mcm_single_measure_obs(shots, postselect, reset, measure_f):
    """Tests that DefaultQubit handles a circuit with a single mid-circuit measurement and a
    conditional gate. A single measurement of a common observable is performed at the end."""

    dev = qml.device("default.qubit", shots=shots)
    params = np.pi / 4 * np.ones(2)
    obs = qml.PauliZ(0)

    @qml.qnode(dev)
    def func1(x, y):
        qml.RX(x, wires=0)
        m0 = qml.measure(0, reset=reset, postselect=postselect)
        qml.cond(m0, qml.RY)(y, wires=1)
        return measure_f(op=obs)

    @qml.qnode(dev)
    @qml.defer_measurements
    def func2(x, y):
        qml.RX(x, wires=0)
        m0 = qml.measure(0, reset=reset, postselect=postselect)
        qml.cond(m0, qml.RY)(y, wires=1)
        return measure_f(op=obs)

    if shots is None and measure_f in (qml.counts, qml.sample):
        return

    results1 = func1(*params)
    results2 = func2(*params)

    if postselect is None or measure_f in (qml.expval, qml.probs, qml.var):
        validate_measurements(measure_f, shots, results1, results2)


@flaky(max_runs=5)
@pytest.mark.parametrize("shots", [1000])
@pytest.mark.parametrize("postselect", [None, 0, 1])
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("measure_f", [qml.counts, qml.expval, qml.probs, qml.sample, qml.var])
def test_single_mcm_multiple_measurements(shots, postselect, reset, measure_f):
    """Tests that DefaultQubit handles a circuit with a single mid-circuit measurement with reset
    and a conditional gate. Multiple measurements of the mid-circuit measurement value are
    performed."""

    dev = qml.device("default.qubit", shots=shots)
    params = np.pi / 4 * np.ones(2)
    obs = qml.PauliZ(0)

    @qml.qnode(dev)
    def func1(x, y):
        qml.RX(x, wires=0)
        m0 = qml.measure(0, reset=reset, postselect=postselect)
        qml.cond(m0, qml.RY)(y, wires=1)
        return measure_f(op=obs), measure_f(op=m0)

    @qml.qnode(dev)
    @qml.defer_measurements
    def func2(x, y):
        qml.RX(x, wires=0)
        m0 = qml.measure(0, reset=reset, postselect=postselect)
        qml.cond(m0, qml.RY)(y, wires=1)
        return measure_f(op=obs), measure_f(op=m0)

    results1 = func1(*params)
    results2 = func2(*params)

    if postselect is None or measure_f in (qml.expval, qml.probs, qml.var):
        for r1, r2 in zip(results1, results2):
            validate_measurements(measure_f, shots, r1, r2)


@flaky(max_runs=5)
@pytest.mark.parametrize("shots", [None, 10000, [10000, 10001]])
@pytest.mark.parametrize("postselect", [None, 0, 1])
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("measure_f", [qml.counts, qml.expval, qml.sample, qml.var])
def test_composite_mcm_measure_composite_mcm(shots, postselect, reset, measure_f):
    """Tests that DefaultQubit handles a circuit with a composite mid-circuit measurement and a
    conditional gate. A single measurement of a composite mid-circuit measurement is performed
    at the end."""

    dev = qml.device("default.qubit", shots=shots)
    param = np.pi / 3

    @qml.qnode(dev)
    def func1(x):
        qml.RX(x, 0)
        m0 = qml.measure(0)
        qml.RX(0.5 * x, 1)
        m1 = qml.measure(1, reset=reset, postselect=postselect)
        qml.cond((m0 + m1) == 2, qml.RY)(2.0 * x, 0)
        m2 = qml.measure(0)
        return measure_f(op=(m0 - 2 * m1) * m2 + 7)

    @qml.qnode(dev)
    @qml.defer_measurements
    def func2(x):
        qml.RX(x, 0)
        m0 = qml.measure(0)
        qml.RX(0.5 * x, 1)
        m1 = qml.measure(1, reset=reset, postselect=postselect)
        qml.cond((m0 + m1) == 2, qml.RY)(2.0 * x, 0)
        m2 = qml.measure(0)
        return measure_f(op=(m0 - 2 * m1) * m2 + 7)

    if shots is None and measure_f in (qml.counts, qml.sample):
        return

    results1 = func1(param)
    results2 = func2(param)

    if postselect is None or measure_f in (qml.expval, qml.probs, qml.var):
        validate_measurements(measure_f, shots, results1, results2)


@flaky(max_runs=5)
@pytest.mark.parametrize("shots", [None, 10000, [10000, 10001]])
@pytest.mark.parametrize("postselect", [None, 0, 1])
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("measure_f", [qml.counts, qml.expval, qml.probs, qml.sample, qml.var])
def test_composite_mcm_single_measure_obs(shots, postselect, reset, measure_f):
    """Tests that DefaultQubit handles a circuit with a composite mid-circuit measurement and a
    conditional gate. A single measurement of a common observable is performed at the end."""

    dev = qml.device("default.qubit", shots=shots)
    param = np.pi / 3
    obs = qml.PauliZ(0)

    @qml.qnode(dev)
    def func1(x):
        qml.RX(x, 0)
        m0 = qml.measure(0)
        qml.RX(0.5 * x, 1)
        m1 = qml.measure(1, reset=reset, postselect=postselect)
        qml.cond((m0 + m1) == 2, qml.RY)(2.0 * x, 0)
        return measure_f(op=obs)

    @qml.qnode(dev)
    @qml.defer_measurements
    def func2(x):
        qml.RX(x, 0)
        m0 = qml.measure(0)
        qml.RX(0.5 * x, 1)
        m1 = qml.measure(1, reset=reset, postselect=postselect)
        qml.cond((m0 + m1) == 2, qml.RY)(2.0 * x, 0)
        return measure_f(op=obs)

    if shots is None and measure_f in (qml.counts, qml.sample):
        return

    results1 = func1(param)
    results2 = func2(param)

    if postselect is None or measure_f in (qml.expval, qml.probs, qml.var):
        validate_measurements(measure_f, shots, results1, results2)


@flaky(max_runs=5)
@pytest.mark.parametrize("shots", [10000, [10000, 10001]])
@pytest.mark.parametrize("postselect", [None, 0, 1])
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("measure_f", [qml.counts, qml.probs, qml.sample])
def test_composite_mcm_measure_value_list(shots, postselect, reset, measure_f):
    """Tests that DefaultQubit handles a circuit with a composite mid-circuit measurement and a
    conditional gate. A single measurement of a composite mid-circuit measurement is performed
    at the end."""

    dev = qml.device("default.qubit", shots=shots)
    param = np.pi / 3

    @qml.qnode(dev)
    def func1(x):
        qml.RX(x, 0)
        m0 = qml.measure(0)
        qml.RX(0.5 * x, 1)
        m1 = qml.measure(1, reset=reset, postselect=postselect)
        qml.cond((m0 + m1) == 2, qml.RY)(2.0 * x, 0)
        m2 = qml.measure(0)
        return measure_f(op=[m0, m1, m2])

    @qml.qnode(dev)
    @qml.defer_measurements
    def func2(x):
        qml.RX(x, 0)
        m0 = qml.measure(0)
        qml.RX(0.5 * x, 1)
        m1 = qml.measure(1, reset=reset, postselect=postselect)
        qml.cond((m0 + m1) == 2, qml.RY)(2.0 * x, 0)
        m2 = qml.measure(0)
        return measure_f(op=[m0, m1, m2])

    results1 = func1(param)
    results2 = func2(param)

    validate_measurements(measure_f, shots, results1, results2)


@flaky(max_runs=5)
@pytest.mark.parametrize("shots", [10000])
@pytest.mark.parametrize("postselect", [None, 0, 1])
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("measure_f", [qml.expval])
def test_composite_mcm_gradient_measure_obs(shots, postselect, reset, measure_f):
    """Tests that DefaultQubit can differentiate a circuit with a composite mid-circuit
    measurement and a conditional gate. A single measurement of a common observable is
    performed at the end."""

    dev = qml.device("default.qubit", shots=shots)
    param = qml.numpy.array([np.pi / 3, np.pi / 6])
    obs = qml.PauliZ(0) @ qml.PauliZ(1)

    @qml.qnode(dev, diff_method="parameter-shift")
    def func1(x, y):
        qml.RX(x, 0)
        m0 = qml.measure(0)
        qml.RX(y, 1)
        m1 = qml.measure(1, reset=reset, postselect=postselect)
        qml.cond((m0 + m1) == 2, qml.RY)(2 * np.pi / 3, 0)
        qml.cond((m0 + m1) > 0, qml.RY)(2 * np.pi / 3, 1)
        return measure_f(op=obs)

    @qml.qnode(dev, diff_method="parameter-shift")
    @qml.defer_measurements
    def func2(x, y):
        qml.RX(x, 0)
        m0 = qml.measure(0)
        qml.RX(y, 1)
        m1 = qml.measure(1, reset=reset, postselect=postselect)
        qml.cond((m0 + m1) == 2, qml.RY)(2 * np.pi / 3, 0)
        qml.cond((m0 + m1) > 0, qml.RY)(2 * np.pi / 3, 1)
        return measure_f(op=obs)

    results1 = func1(*param)
    results2 = func2(*param)

    validate_measurements(measure_f, shots, results1, results2)

    grad1 = qml.grad(func1)(*param)
    grad2 = qml.grad(func2)(*param)

    assert np.allclose(grad1, grad2, atol=0.01, rtol=0.3)
