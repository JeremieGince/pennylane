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
This module contains the qml.bind_new_parameters function.
"""
# pylint: disable=missing-docstring

from typing import Sequence

has_optree = True
try:
    import optree
except ImportError:
    has_optree = False

from pennylane.typing import TensorLike
from pennylane.operation import Operator

from pennylane.pytrees import flatten, unflatten


def bind_new_parameters(op: Operator, params: Sequence[TensorLike]) -> Operator:
    """Create a new operator with updated parameters

    This function takes an :class:`~.Operator` and new parameters as input and
    returns a new operator of the same type with the new parameters. This function
    does not mutate the original operator.

    Args:
        op (.Operator): Operator to update
        params (Sequence[TensorLike]): New parameters to create operator with. This
            must have the same shape as `op.data`.

    Returns:
        .Operator: New operator with updated parameters
    """
    if has_optree:
        _, structure = optree.tree_flatten(op, namespace="qml")
        return optree.tree_unflatten(structure, params)
    _, structure = flatten(op)
    return unflatten(params, structure)
