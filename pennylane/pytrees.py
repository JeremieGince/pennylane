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
An internal module for working with pytrees.
"""

from typing import Callable, Tuple, Any, NamedTuple

has_jax = True
try:
    import jax.tree_util as jax_tree_util
except ImportError:
    has_jax = False

has_optree = True
try:
    import optree
except ImportError:
    has_optree = False

Leaves = Any
Metadata = Any

FlattenFn = Callable[[Any], Tuple[Leaves, Metadata]]
UnflattenFn = Callable[[Leaves, Metadata], Any]

flatten_registrations = {
    list: (lambda obj: (obj, None)),
    tuple: (lambda obj: (obj, None)),
    dict: (lambda obj: (obj.values(), obj.keys())),
}
unflatten_registrations = {
    list: (lambda data, _: list(data)),
    tuple: (lambda data, _: tuple(data)),
    dict: (lambda data, metadata: dict(zip(metadata, data))),
}


def _register_pytree_with_pennylane(
    pytree_type: type, flatten_fn: FlattenFn, unflatten_fn: UnflattenFn
):
    flatten_registrations[pytree_type] = flatten_fn
    unflatten_registrations[pytree_type] = unflatten_fn


def _register_pytree_with_jax(pytree_type: type, flatten_fn: FlattenFn, unflatten_fn: UnflattenFn):
    def jax_unflatten(aux, parameters):
        return unflatten_fn(parameters, aux)

    jax_tree_util.register_pytree_node(pytree_type, flatten_fn, jax_unflatten)


def _register_pytree_with_optree(
    pytree_type: type, flatten_fn: FlattenFn, unflatten_fn: UnflattenFn
):
    def optree_flatten(obj):
        data, metadata = flatten_fn(obj)
        return data, metadata, None

    def optree_unflatten(metadata, data):
        return unflatten_fn(data, metadata)

    optree.register_pytree_node(pytree_type, optree_flatten, optree_unflatten, namespace="qml")


def register_pytree(pytree_type: type, flatten_fn: FlattenFn, unflatten_fn: UnflattenFn):
    """Register a type with all available pytree backends.

    Current backends is jax.
    Args:
        pytree_type (type): the type to register, such as ``qml.RX``
        flatten_fn (Callable): a function that splits an object into trainable leaves and hashable metadata.
        unflatten_fn (Callable): a function that reconstructs an object from its leaves and metadata.

    Returns:
        None

    Side Effects:
        ``pytree`` type becomes registered with available backends.

    """

    _register_pytree_with_pennylane(pytree_type, flatten_fn, unflatten_fn)

    if has_jax:
        _register_pytree_with_jax(pytree_type, flatten_fn, unflatten_fn)

    if has_optree:
        _register_pytree_with_optree(pytree_type, flatten_fn, unflatten_fn)


class Structure(NamedTuple):
    node_type: type
    node_metadata: Tuple
    child_structures: Tuple["Structure"]
    unflatten_fn: UnflattenFn

    def __repr__(self):
        rep = f"Tree({self.node_type.__name__}, {self.node_metadata})\n\t"
        rep += "\n\t".join(repr(s) for s in self.child_structures)
        return rep


class Leaf:
    def __repr__(self):
        return "Leaf"

    def __eq__(self, other):
        return type(other) is Leaf

    def __hash__(self):
        return hash(Leaf)


leaf = Leaf()


def flatten(op):
    leaves, metadata = flatten_registrations[type(op)](op)

    flattened_leaves = []
    child_structures = []
    for l in leaves:
        if type(l) in flatten_registrations:
            child_leaves, child_structure = flatten(l)
            flattened_leaves += child_leaves
            child_structures.append(child_structure)
        else:
            flattened_leaves.append(l)
            child_structures.append(leaf)

    unflatten_fn = unflatten_registrations[type(op)]

    structure = Structure(type(op), metadata, tuple(child_structures), unflatten_fn)
    return flattened_leaves, structure


def unflatten(new_data, structure):
    return _unflatten(iter(new_data), structure)


def _unflatten(new_data, structure):
    if isinstance(structure, Leaf):
        return next(new_data)
    children = tuple(_unflatten(new_data, s) for s in structure.child_structures)
    return structure.unflatten_fn(children, structure.node_metadata)
