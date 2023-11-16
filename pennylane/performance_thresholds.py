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
"""A module for storing any tunable performance thresholds.

"""

DEFAULT_QUBIT_QFT_THRESHOLD = 6
"""
default qubit will decompose any QFT template with more than this number of wires.
"""

DEFAULT_QUBIT_GROVER_THRESHOLD = 13
"""
default qubit will decompose any Grover Operator template with more than this number of wires.
"""

EINSUM_OP_WIRECOUNT_PERF_THRESHOLD = 3
"""
apply_operation will use a tensordot based method when the operation has more wires than this
threshold.

"""

EINSUM_STATE_WIRECOUNT_PERF_THRESHOLD = 13
"""
apply_operation will use a tensordot based method when the state has more wires than this threshold.
"""

PAULIZ_TENSORDOT_THRESHOLD = 9
"""
apply_operation will apply PauliZ operations with tensordot is the state has
more than this number of wires.
"""

MEASURE_SUM_SPARSE_DOT = 7
"""
measure will use csr_dot_products if the number of observable wires is greater than this number.
Below this number, the observable will be diagonalized and the diagonalizing gates will be
applied to the state.
"""
