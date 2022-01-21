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
This module contains functions for validating gradient methods against a target device.
"""
import warnings

import pennylane as qml


# pylint: disable=too-many-return-statements
def get_gradient_fn(device, interface, diff_method="best"):
    """Determine the best differentiation method, interface, and device
    for a requested device, interface, and diff method.

    Args:
        device (.Device): PennyLane device
        interface (str): name of the requested interface
        diff_method (str or .gradient_transform): The requested method of differentiation.
            If a string, allowed options are ``"best"``, ``"backprop"``, ``"adjoint"``,
            ``"device"``, ``"parameter-shift"``, or ``"finite-diff"``. A gradient transform may
            also be passed here.

    Returns:
        tuple[str or .gradient_transform, dict, .Device: Tuple containing the ``gradient_fn``,
        ``gradient_kwargs``, and the device to use when calling the execute function.
    """

    if diff_method == "best":
        return get_best_method(device, interface)

    if diff_method == "backprop":
        return _validate_backprop_method(device, interface)

    if diff_method == "adjoint":
        return _validate_adjoint_method(device)

    if diff_method == "device":
        return _validate_device_method(device)

    if diff_method == "parameter-shift":
        return _validate_parameter_shift(device)

    if diff_method == "finite-diff":
        return qml.gradients.finite_diff, {}, device

    if isinstance(diff_method, str):
        raise qml.QuantumFunctionError(
            f"Differentiation method {diff_method} not recognized. Allowed "
            "options are ('best', 'parameter-shift', 'backprop', 'finite-diff', 'device', 'adjoint')."
        )

    if isinstance(diff_method, qml.gradients.gradient_transform):
        return diff_method, {}, device

    raise qml.QuantumFunctionError(
        f"Differentiation method {diff_method} must be a gradient transform or a string."
    )


def get_best_method(device, interface):
    """Returns the 'best' differentiation method
    for a particular device and interface combination.

    This method attempts to determine support for differentiation
    methods using the following order:

    * ``"device"``
    * ``"backprop"``
    * ``"parameter-shift"``
    * ``"finite-diff"``

    The first differentiation method that is supported (going from
    top to bottom) will be returned.

    Args:
        device (.Device): PennyLane device
        interface (str): name of the requested interface

    Returns:
        tuple[str or .gradient_transform, dict, .Device: Tuple containing the ``gradient_fn``,
        ``gradient_kwargs``, and the device to use when calling the execute function.
    """
    try:
        return _validate_device_method(device)
    except qml.QuantumFunctionError:
        try:
            return _validate_backprop_method(device, interface)
        except qml.QuantumFunctionError:
            try:
                return _validate_parameter_shift(device)
            except qml.QuantumFunctionError:
                return qml.gradients.finite_diff, {}, device


def _validate_backprop_method(device, interface):
    # determine if the device supports backpropagation
    backprop_interface = device.capabilities().get("passthru_interface", None)

    # determine if the device has any child devices that support backpropagation
    backprop_devices = device.capabilities().get("passthru_devices", None)

    if getattr(device, "cache", 0):
        # TODO: deprecate device caching, and replacing with QNode caching.
        raise qml.QuantumFunctionError(
            "Device caching is incompatible with the backprop diff_method"
        )

    if backprop_interface is not None:
        # device supports backpropagation natively

        if interface == backprop_interface:
            return "backprop", {}, device

        raise qml.QuantumFunctionError(
            f"Device {device.short_name} only supports diff_method='backprop' when using the "
            f"{backprop_interface} interface."
        )

    if backprop_devices is not None:
        if device.shots is None:
            # device is analytic and has child devices that support backpropagation natively

            if interface in backprop_devices:
                # TODO: need a better way of passing existing device init options
                # to a new device?
                expand_fn = device.expand_fn
                batch_transform = device.batch_transform

                device = qml.device(
                    backprop_devices[interface],
                    wires=device.wires,
                    shots=device.shots,
                )
                device.expand_fn = expand_fn
                device.batch_transform = batch_transform
                return "backprop", {}, device

            raise qml.QuantumFunctionError(
                f"Device {device.short_name} only supports diff_method='backprop' when using the "
                f"{list(backprop_devices.keys())} interfaces."
            )

        raise qml.QuantumFunctionError("Backpropagation is only supported when shots=None.")

    raise qml.QuantumFunctionError(
        f"The {device.short_name} device does not support native computations with "
        "autodifferentiation frameworks."
    )


def _validate_adjoint_method(device):
    # The conditions below provide a minimal set of requirements that we can likely improve upon in
    # future, or alternatively summarize within a single device capability. Moreover, we also
    # need to inspect the circuit measurements to ensure only expectation values are taken. This
    # cannot be done here since we don't yet know the composition of the circuit.

    required_attrs = ["_apply_operation", "_apply_unitary", "adjoint_jacobian"]
    supported_device = all(hasattr(device, attr) for attr in required_attrs)
    supported_device = supported_device and device.capabilities().get("returns_state")

    if not supported_device:
        raise ValueError(
            f"The {device.short_name} device does not support adjoint differentiation."
        )

    if device.shots is not None:
        warnings.warn(
            "Requested adjoint differentiation to be computed with finite shots."
            " Adjoint differentiation always calculated exactly.",
            UserWarning,
        )

    return "device", {"use_device_state": True, "method": "adjoint_jacobian"}, device


def _validate_device_method(device):
    # determine if the device provides its own jacobian method
    provides_jacobian = device.capabilities().get("provides_jacobian", False)

    if not provides_jacobian:
        raise qml.QuantumFunctionError(
            f"The {device.short_name} device does not provide a native "
            "method for computing the jacobian."
        )

    return "device", {}, device


def _validate_parameter_shift(device):
    model = device.capabilities().get("model", None)

    if model == "qubit":
        return qml.gradients.param_shift, {}, device

    if model == "cv":
        return qml.gradients.param_shift_cv, {"dev": device}, device

    raise qml.QuantumFunctionError(
        f"Device {device.short_name} uses an unknown model ('{model}') "
        "that does not support the parameter-shift rule."
    )
