# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
Unit tests for functions needed for resource estimation with the double factorization method.
"""
import pytest

import pennylane as qml


@pytest.mark.parametrize(
    ("norm", "error", "cost_ref"),
    [  # cost_ref is computed manually
        (72.49779513025341, 0.001, 113880),
    ],
)
def test_estimation_cost(norm, error, cost_ref):
    r"""Test that estimation_cost returns the correct values."""
    cost = qml.resources.estimation_cost(norm, error)

    assert cost == cost_ref


@pytest.mark.parametrize(
    ("norm", "error"),
    [
        (5.28, 0.0),
        (5.28, -1.0),
        (-5.28, 0.01),
        (0.0, 0.01),
    ],
)
def test_estimation_cost_error(norm, error):
    r"""Test that estimation_cost raises an error with incorrect inputs."""
    with pytest.raises(ValueError, match="must be greater than zero"):
        qml.resources.estimation_cost(norm, error)


@pytest.mark.parametrize(
    ("constants", "cost_ref", "k_ref"),
    [  # The reference costs and k values are obtained manually, by computing the cost for a range
        # of k values and selecting the k that gives the minimum cost.
        (
            (26, 1, 0, 15.0, -1),
            27,
            1,
        ),
        (
            (26, 1, 0, 1, 0),
            11,
            4,
        ),
        (
            (151.0, 7.0, 151.0, 280, 0),
            589,
            1,
        ),
        (
            (151.0, 7.0, 151.0, 2, 0),
            52,
            16,
        ),
        (
            (151.0, 7.0, 151.0, 30.0, -1),
            168,
            4,
        ),
    ],
)
def test_qrom_cost(constants, cost_ref, k_ref):
    r"""Test that _qrom_cost returns the correct values."""
    cost, k = qml.resources._qrom_cost(constants)

    assert cost == cost_ref
    assert k == k_ref


@pytest.mark.parametrize(
    ("n", "rank_r", "rank_m", "br", "alpha", "beta", "cost_ref"),
    [
        (14, 26, 5.5, 7, 10, 20, 2007),
    ],
)
def test_unitary_cost(n, rank_r, rank_m, br, alpha, beta, cost_ref):
    r"""Test that unitary_cost returns the correct value."""
    cost = qml.resources.unitary_cost(n, rank_r, rank_m, br, alpha, beta)

    assert cost == cost_ref


@pytest.mark.parametrize(
    ("n", "rank_r", "rank_m", "br", "alpha", "beta"),
    [
        (14.5, 26, 5.5, 7, 10, 20),
        (-14, 26, 5.5, 7, 10, 20),
        (11, 26, 5.5, 7, 10, 20),
        (14, -26, 5.5, 7, 10, 20),
        (14, 26.1, 5.5, 7, 10, 20),
        (14, 26, -5.5, 7, 10, 20),
        (14, 26, 5.5, -7, 10, 20),
        (14, 26, 5.5, 7.5, 10, 20),
        (14, 26, 5.5, 7, -10, 20),
        (14, 26, 5.5, 7, 10.2, 20),
        (14, 26, 5.5, 7, 10, -20),
        (14, 26, 5.5, 7, 10, 20.9),
    ],
)
def test_unitary_cost_error(n, rank_r, rank_m, br, alpha, beta):
    r"""Test that unitary_cost raises an error with incorrect inputs."""
    with pytest.raises(ValueError, match="must be a positive"):
        qml.resources.unitary_cost(n, rank_r, rank_m, br, alpha, beta)


@pytest.mark.parametrize(
    ("n", "norm", "error", "rank_r", "rank_m", "br", "alpha", "beta", "cost_ref"),
    [
        (14, 52.98761457453095, 0.001, 26, 5.5, 7, 10, 20, 167048631),
    ],
)
def test_gate_cost(n, norm, error, rank_r, rank_m, br, alpha, beta, cost_ref):
    r"""Test that gate_cost returns the correct value."""
    cost = qml.resources.gate_cost(n, norm, error, rank_r, rank_m, br, alpha, beta)

    assert cost == cost_ref


@pytest.mark.parametrize(
    ("n", "norm", "error", "rank_r", "rank_m", "br", "alpha", "beta"),
    [
        (14.5, 5.5, 0.01, 26, 5.5, 7, 10, 20),
        (-14, 5.5, 0.01, 26, 5.5, 7, 10, 20),
        (11, 5.5, 0.01, 26, 5.5, 7, 10, 20),
        (14, 5.28, 0.0, 26, 5.5, 7, 10, 20),
        (14, 5.28, -1.0, 26, 5.5, 7, 10, 20),
        (14, -5.28, 0.01, 26, 5.5, 7, 10, 20),
        (14, 0.0, 0.01, 26, 5.5, 7, 10, 20),
        (14, 5.5, 0.01, -26, 5.5, 7, 10, 20),
        (14, 5.5, 0.01, 26.1, 5.5, 7, 10, 20),
        (14, 5.5, 0.01, 26, -5.5, 7, 10, 20),
        (14, 5.5, 0.01, 26, 5.5, -7, 10, 20),
        (14, 5.5, 0.01, 26, 5.5, 7.5, 10, 20),
        (14, 5.5, 0.01, 26, 5.5, 7, -10, 20),
        (14, 5.5, 0.01, 26, 5.5, 7, 10.2, 20),
        (14, 5.5, 0.01, 26, 5.5, 7, 10, -20),
        (14, 5.5, 0.01, 26, 5.5, 7, 10, 20.9),
    ],
)
def test_gate_cost_error(n, norm, error, rank_r, rank_m, br, alpha, beta):
    r"""Test that gate_cost raises an error with incorrect inputs."""
    with pytest.raises(ValueError, match="must be"):
        qml.resources.gate_cost(n, norm, error, rank_r, rank_m, br, alpha, beta)


@pytest.mark.parametrize(
    ("n", "norm", "error", "rank_r", "rank_m", "br", "alpha", "beta", "cost_ref"),
    [
        (14, 52.98761457453095, 0.001, 26, 5.5, 7, 10, 20, 292),
    ],
)
def test_qubit_cost(n, norm, error, rank_r, rank_m, br, alpha, beta, cost_ref):
    r"""Test that qubit_cost returns the correct value."""
    cost = qml.resources.qubit_cost(n, norm, error, rank_r, rank_m, br, alpha, beta)

    assert cost == cost_ref


@pytest.mark.parametrize(
    ("n", "norm", "error", "rank_r", "rank_m", "br", "alpha", "beta"),
    [
        (14.5, 5.5, 0.01, 26, 5.5, 7, 10, 20),
        (-14, 5.5, 0.01, 26, 5.5, 7, 10, 20),
        (11, 5.5, 0.01, 26, 5.5, 7, 10, 20),
        (14, 5.28, 0.0, 26, 5.5, 7, 10, 20),
        (14, 5.28, -1.0, 26, 5.5, 7, 10, 20),
        (14, -5.28, 0.01, 26, 5.5, 7, 10, 20),
        (14, 0.0, 0.01, 26, 5.5, 7, 10, 20),
        (14, 5.5, 0.01, -26, 5.5, 7, 10, 20),
        (14, 5.5, 0.01, 26.1, 5.5, 7, 10, 20),
        (14, 5.5, 0.01, 26, -5.5, 7, 10, 20),
        (14, 5.5, 0.01, 26, 5.5, -7, 10, 20),
        (14, 5.5, 0.01, 26, 5.5, 7.5, 10, 20),
        (14, 5.5, 0.01, 26, 5.5, 7, -10, 20),
        (14, 5.5, 0.01, 26, 5.5, 7, 10.2, 20),
        (14, 5.5, 0.01, 26, 5.5, 7, 10, -20),
        (14, 5.5, 0.01, 26, 5.5, 7, 10, 20.9),
    ],
)
def test_qubit_cost_error(n, norm, error, rank_r, rank_m, br, alpha, beta):
    r"""Test that qubit_cost raises an error with incorrect inputs."""
    with pytest.raises(ValueError, match="must be"):
        qml.resources.qubit_cost(n, norm, error, rank_r, rank_m, br, alpha, beta)
