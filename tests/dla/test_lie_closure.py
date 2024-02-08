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
"""Tests for pennylane/dla/lie_closure.py functionality"""
import pytest
import numpy as np

import pennylane as qml
from pennylane.dla import lie_closure
from pennylane.dla.lie_closure import VSpace, _is_any_col_propto_last
from pennylane.pauli import PauliWord, PauliSentence

ops1 = [
    PauliSentence({PauliWord({0: "X", 1: "X"}): 1.0, PauliWord({0: "Y", 1: "Y"}): 1.0}),
    PauliSentence(
        {
            PauliWord({0: "X", 1: "X"}): 1.0,
        }
    ),
    PauliSentence(
        {
            PauliWord({0: "Y", 1: "Y"}): 2.0,
        }
    ),
]

ops2 = [
    PauliSentence({PauliWord({0: "X", 1: "X"}): 1.0, PauliWord({0: "Y", 1: "Y"}): 1.0}),
    PauliSentence(
        {
            PauliWord({0: "X", 1: "X"}): 1.0,
        }
    ),
]

ops2plusY10 = ops2 + [PauliSentence({PauliWord({10: "Y"}): 1.0})]


class TestVSpace:
    """Unit and integration tests for VSpace class"""

    def test_init(self):
        """Unit tests for initialization"""
        vspace = VSpace(ops1)

        assert all(isinstance(op, PauliSentence) for op in vspace.basis)
        assert np.allclose(vspace.M, [[1.0, 1.0], [1.0, 0.0]]) or np.allclose(
            vspace.M, [[1.0, 0.0], [1.0, 1.0]]
        )
        assert vspace.basis == ops1[:-1]
        assert vspace.rank == 2
        assert vspace.num_pw == 2
        assert len(vspace.pw_to_idx) == 2

    ADD_LINEAR_INDEPENDENT = (
        (ops2, PauliWord({10: "Y"}), ops2plusY10),
        (ops2, PauliSentence({PauliWord({10: "Y"}): 1.0}), ops2plusY10),
        (ops2, qml.PauliY(10), ops2plusY10),
    )

    @pytest.mark.parametrize("ops, op, true_new_basis", ADD_LINEAR_INDEPENDENT)
    def test_add_lin_independent(self, ops, op, true_new_basis):
        """Test that adding new (linearly independent) operators works as expected"""
        vspace = VSpace(ops)
        new_basis = vspace.add(op)
        assert new_basis == true_new_basis

    ADD_LINEAR_DEPENDENT = (
        (ops2, PauliWord({0: "Y", 1: "Y"}), ops2),
        (ops2, PauliSentence({PauliWord({0: "Y", 1: "Y"}): 1.0}), ops2),
        (ops2, qml.PauliY(0) @ qml.PauliY(1), ops2),
        (ops2, 0.5 * ops2[0], ops2),
        (ops2, 0.5 * ops2[1], ops2),
    )

    @pytest.mark.parametrize("ops, op, true_new_basis", ADD_LINEAR_DEPENDENT)
    def test_add_lin_dependent(self, ops, op, true_new_basis):
        """Test that adding linearly dependent operators works as expected"""
        vspace = VSpace(ops)
        new_basis = vspace.add(op)
        assert new_basis == true_new_basis

    @pytest.mark.parametrize("ops, op, true_new_basis", ADD_LINEAR_INDEPENDENT)
    def test_len(self, ops, op, true_new_basis):
        """Test the length of the VSpace instance with inplace and non-inplace addition to the basis"""
        vspace = VSpace(ops)
        len_before_adding = len(vspace)
        len_basis_before_adding = len(vspace.basis)

        _ = vspace.add(op)
        len_after_adding = len(vspace)
        assert len_after_adding != len_before_adding
        assert len_after_adding == len(true_new_basis)
        assert len_before_adding == len_basis_before_adding


class TestLieClosure:
    """Tests for qml.dla.lie_closure()"""

    M1 = np.array(
        [
            [1.0, 0.0, 0.5, 0.5, 1.0],  # non-matching zeros 2nd to last and last
            [1.0, 0.5, 0.0, 1.0, 0.0],
            [1.0, 0.5, 1.0, 2.0, 4.0],
        ]
    )
    M2 = np.array(
        [
            [1.0, 0.0, 0.5, 0.5, 1.0],  # second-to-last col proportional to last -> True
            [1.0, 0.5, 0.0, 0.0, 0.0],
            [1.0, 0.5, 1.0, 2.0, 4.0],
        ]
    )
    M3 = np.array(
        [
            [1.0, 0.0, 0.5, 0.5, -1.0],  # second-to-last col proportional to last -> True
            [1.0, 0.5, 0.0, 0.0, 0.0],  # additional feature: minus signs reversed
            [1.0, 0.5, 1.0, 2.0, -4.0],
        ]
    )
    M4 = np.array(
        [
            [1.0, 0.0, 0.5, -0.5, 1.0],  # second-to-last col proportional to last -> True
            [1.0, 0.5, 0.0, 0.0, 0.0],  # additional feature: minus signs reversed
            [1.0, 0.5, 1.0, -2.0, 4.0],
        ]
    )
    M5 = np.array(
        [
            [1.0, 0.0, 0.5, 0.5, -1.0],  # second-to-last col proportional to last -> True
            [1.0, 0.5, 0.0, 0.0, 0.0],  # additional feature: minus signs opposites
            [1.0, 0.5, 1.0, -2.0, 4.0],
        ]
    )

    IS_ANY_COL_PROPTO_LAST = ((M1, False), (M2, True), (M3, True), (M4, True), (M5, True))

    @pytest.mark.parametrize("M, res", IS_ANY_COL_PROPTO_LAST)
    def test_is_any_col_propto_last(self, M, res):
        """Test utility function _is_any_col_propto_last that checks whether any column of the input is proportional to the last column"""
        assert _is_any_col_propto_last(M) == res

    def test_simple_lie_closure(self):
        """Test simple lie_closure example"""
        dla11 = [
            PauliSentence({PauliWord({0: "X", 1: "X"}): 1.0, PauliWord({0: "Y", 1: "Y"}): 1.0}),
            PauliSentence(
                {
                    PauliWord({0: "Z"}): 1.0,
                }
            ),
            PauliSentence(
                {
                    PauliWord({1: "Z"}): 1.0,
                }
            ),
            PauliSentence({PauliWord({0: "Y", 1: "X"}): -1.0, PauliWord({0: "X", 1: "Y"}): 1.0}),
        ]
        gen11 = dla11[:-1]
        res11 = lie_closure(gen11)
        assert res11 == dla11

        dla12 = [
            PauliSentence({PauliWord({0: "X", 1: "X"}): 1.0, PauliWord({0: "Y", 1: "Y"}): 1.0}),
            PauliSentence(
                {
                    PauliWord({0: "Z"}): 1.0,
                }
            ),
            PauliSentence({PauliWord({0: "Y", 1: "X"}): -1.0, PauliWord({0: "X", 1: "Y"}): 1.0}),
            PauliSentence({PauliWord({0: "Z"}): -2.0, PauliWord({1: "Z"}): 2.0}),
        ]
        gen12 = dla12[:-1]
        res12 = lie_closure(gen12)
        assert res12 == dla12

    @pytest.mark.parametrize("n", range(2, 5))
    def test_lie_closure_transverse_field_ising_1D_open(self, n):
        """Test the lie closure works correctly for the transverse Field Ising model with open boundary conditions, a8 in theorem IV.1 in https://arxiv.org/pdf/2309.05690.pdf"""
        generators = [
            PauliSentence({PauliWord({i: "X", (i + 1) % n: "X"}): 1.0}) for i in range(n - 1)
        ]
        generators += [
            PauliSentence({PauliWord({i: "X", (i + 1) % n: "Z"}): 1.0}) for i in range(n - 1)
        ]

        res = qml.dla.lie_closure(generators)
        assert len(res) == (2 * n - 1) * (2 * n - 2) // 2

    @pytest.mark.parametrize("n", range(3, 5))
    def test_lie_closure_transverse_field_ising_1D_cyclic(self, n):
        """Test the lie closure works correctly for the transverse Field Ising model with cyclic boundary conditions, a8 in theorem IV.2 in https://arxiv.org/pdf/2309.05690.pdf"""
        generators = [PauliSentence({PauliWord({i: "X", (i + 1) % n: "X"}): 1.0}) for i in range(n)]
        generators += [
            PauliSentence({PauliWord({i: "X", (i + 1) % n: "Z"}): 1.0}) for i in range(n)
        ]

        res = qml.dla.lie_closure(generators)
        assert len(res) == 2 * n * (2 * n - 1)

    def test_lie_closure_heisenberg_generators_odd(self):
        """Test the resulting DLA from Heisenberg generators with odd n, a7 in theorem IV.1 in https://arxiv.org/pdf/2309.05690.pdf"""
        n = 3
        # dim of su(N) is N ** 2 - 1
        # Heisenberg generates su(2**(n-1)) for n odd            => dim should be (2**(n-1))**2 - 1
        generators = [
            PauliSentence({PauliWord({i: "X", (i + 1) % n: "X"}): 1.0}) for i in range(n - 1)
        ]
        generators += [
            PauliSentence({PauliWord({i: "Y", (i + 1) % n: "Y"}): 1.0}) for i in range(n - 1)
        ]
        generators += [
            PauliSentence({PauliWord({i: "Z", (i + 1) % n: "Z"}): 1.0}) for i in range(n - 1)
        ]

        res = qml.dla.lie_closure(generators)
        len(res) == (2 ** (n - 1)) ** 2 - 1

    def test_lie_closure_heisenberg_generators_odd(self):
        """Test the resulting DLA from Heisenberg generators with odd n, a7 in theorem IV.1 in https://arxiv.org/pdf/2309.05690.pdf"""
        n = 4
        # dim of su(N) is N ** 2 - 1
        # Heisenberg generates (su(2**(n-2)))**4 for n>=4 even   => dim should be 4*((2**(n-2))**2 - 1)
        generators = [
            PauliSentence({PauliWord({i: "X", (i + 1) % n: "X"}): 1.0}) for i in range(n - 1)
        ]
        generators += [
            PauliSentence({PauliWord({i: "Y", (i + 1) % n: "Y"}): 1.0}) for i in range(n - 1)
        ]
        generators += [
            PauliSentence({PauliWord({i: "Z", (i + 1) % n: "Z"}): 1.0}) for i in range(n - 1)
        ]

        res = qml.dla.lie_closure(generators)
        len(res) == 4 * ((2 ** (n - 2)) ** 2 - 1)
