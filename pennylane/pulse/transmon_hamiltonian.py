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
"""This module contains the classes/functions needed to simulate the evolution of ensembles of
individual (trapped) transmon atoms under the excitation of several laser fields."""
import warnings
from dataclasses import dataclass
from typing import Callable, List, Union

import numpy as np

import pennylane as qml
from pennylane.wires import Wires
from pennylane.operation import Operator
from pennylane.ops.qubit.hamiltonian import Hamiltonian
from pennylane.typing import TensorLike


from .parametrized_hamiltonian import ParametrizedHamiltonian

def a(wire, d=2):
    return qml.s_prod(0.5, qml.PauliX(wire)) + qml.s_prod(0.5j, qml.PauliY(wire))

def ad(wire, d=2):
    return qml.s_prod(0.5, qml.PauliX(wire)) + qml.s_prod(-0.5j, qml.PauliY(wire))



def transmon_interaction(
    connections: list, omega: Union[float, list], g: Union[float, list], delta=None, wires=None, d=2
):
    r"""Returns a :class:`ParametrizedHamiltonian` representing the interaction of an ensemble of
    Transmon atoms due to the Transmon blockade

    .. math::

        \sum_{i<j} V_{ij} n_i n_j

    where :math:`n_i` corresponds to the projector on the Transmon state of the atom :math:`i`, and
    :math:`V_{ij}` is the van der Waals potential:

    .. math::

        V_{ij} = \frac{C_6}{R_{ij}^6}

    where :math:`R_{ij}` is the distance between the atoms :math:`i` and :math:`j`, and :math:`C_6`
    is the Transmon interaction constant, which defaults to :math:`862690 \text{MHz} \times \mu \text{m}^6`.
    The unit of time for the evolution of this Transmon interaction term is in :math:`\mu \text{s}`.
    This interaction term can be combined with laser drive terms (:func:`~.transmon_drive`) to create
    a Hamiltonian describing a driven Transmon atom system.

    .. seealso::

        :func:`~.transmon_drive`

    Args:
        connections (list): list of coordinates of the Transmon atoms (in micrometers)
        wires (list): List of wires containing the wire values for all the atoms. This list should
            have the same length as ``connections``. If ``None``, each atom's wire value will
            correspond to its index in the ``connections`` list.
        interaction_coeff (float): Transmon interaction constant in units: :math:`\text{MHz} \times \mu \text{m}^6`.
            Defaults to :math:`862690 \text{ MHz} \times \mu \text{m}^6`. This value is based on an assumption that
            frequencies and energies in the Hamiltonian are provided in units of MHz.
        max_distance (float): Threshold for distance in :math:`\mu \text{m}` between two Transmon atoms beyond which their
            contribution to the interaction term is removed from the Hamiltonian.

    Returns:
        TransmonHamiltonian: a :class:`~.ParametrizedHamiltonian` representing the atom interaction

    **Example**

    We create a Hamiltonian describing the van der Waals interaction among 9 Transmon atoms in a square lattice:

    .. code-block:: python

        atom_coordinates = [[0, 0], [0, 5], [0, 10], [5, 0], [5, 5], [5, 10], [10, 0], [10, 5], [10, 10]]
        wires = [1, 5, 0, 2, 4, 3, 8, 6, 7]
        H_i = qml.pulse.transmon_interaction(atom_coordinates, wires=wires)

    >>> H_i
    ParametrizedHamiltonian: terms=36

    As expected, we have :math:`\frac{N(N-1)}{2} = 36` terms for N=6 atoms.

    The interaction term is dependent only on the number and positions of the Transmon atoms. We can execute this
    pulse program, which corresponds to all driving laser fields being turned off and therefore has no trainable
    parameters. To add a driving laser field, see :func:`~.transmon_drive`.

    .. code-block:: python

        dev = qml.device("default.qubit.jax", wires=9)

        @qml.qnode(dev, interface="jax")
        def circuit():
            qml.evolve(H_i)([], t=[0, 10])
            return qml.expval(qml.PauliZ(0))

    >>> circuit()
    Array(1., dtype=float32)
    """
    if wires is not None and not all(i in wires for i in qml.math.unique(connections)):
        raise ValueError(f"There are wires in connections {connections} that are not in the provided wires {wires}")

    wires = wires or qml.math.unique(connections)
    n_wires = len(wires)

    # Prepare coefficients
    # TODO: make coefficients callable / trainable. Currently not supported
    if qml.math.ndim(omega)==0:
        omega = [omega] * n_wires
    if len(omega) != n_wires:
        raise ValueError(f"Number of qubit frequencies omega = {omega} does not match the provided wires = {wires}")
    
    if qml.math.ndim(g)==0:
        g = [g] * n_wires
    if len(g) != len(connections):
        raise ValueError(f"Number of coupling terms {g} does not match the provided connections = {connections}")

    # qubit term
    coeffs = list(omega)
    observables = [ad(i, d) @ a(i, d) for i in wires]

    # coupling term term
    coeffs += list(g)
    observables += [ad(i, d) @ a(j, d) + ad(j, d) @ a(i, d) for (i, j) in connections]

    if d>2:
        if delta is None:
            delta = [0.] * n_wires
        if qml.math.ndim(delta)==0:
            delta = [delta] * n_wires
        if len(delta) != n_wires:
            raise ValueError(f"Number of qubit anharmonicities delta = {delta} does not match the provided wires = {wires}")
        # anharmonicity term
        coeffs += list(delta)
        observables += [ad(i, d) @ ad(i, d) @ a(i, d) @ a(i, d) for i in wires]
    

    return TransmonHamiltonian(
        coeffs, observables, omega, delta, g, connections=connections
    )


def transmon_drive(amplitude, phase, detuning, wires):
    r"""Returns a :class:`ParametrizedHamiltonian` representing the action of a driving laser
    field with the given rabi frequency, detuning and phase acting on the given wires

    .. math::
        \frac{1}{2} \Omega(t) \sum_{i \in \text{wires}} (\cos(\phi)\sigma_i^x - \sin(\phi)\sigma_i^y) -
        \frac{1}{2} \delta(t) \sum_{i \in \text{wires}} \sigma_i^z

    where :math:`\Omega`, :math:`\delta` and :math:`\phi` correspond to the rabi frequency, detuning
    and phase of the laser, :math:`i` correspond to the wire index, and :math:`\sigma^\alpha` for
    :math:`\alpha = x,y,z` are the Pauli matrices. The unit of time for the  evolution of this Transmon
    drive term is :math:`\mu \text{s}`. This driving term can be combined with an interaction term to
    create a Hamiltonian describing a driven Transmon atom system. Multiple driving terms can be combined
    by summing them (see example).

    Args:
        amplitude (Union[float, Callable]): float or callable returning the amplitude (in MHz) of a
            laser field
        phase (Union[float, Callable]): float or callable returning the phase (in radians) of the laser field
        detuning (Union[float, Callable]): float or callable returning the detuning (in MHz) of a
            laser field
        wires (Union[int, List[int]]): integer or list containing wire values for the Transmon atoms that
            the laser field acts on

    Returns:
        TransmonHamiltonian: a :class:`~.ParametrizedHamiltonian` representing the action of the laser field
        on the Transmon atoms.

    .. seealso::

        :func:`~.transmon_interaction`, :class:`~.ParametrizedHamiltonian`, :class:`~.ParametrizedEvolution`
        and :func:`~.evolve`

    **Example**

    We create a Hamiltonian describing a laser acting on 4 wires (Transmon atoms) with a fixed detuning and
    phase, and a parametrized, time-dependent amplitude. The Hamiltonian includes an interaction term for
    inter-atom interactions due to van der Waals forces, as well as the driving term for the laser driving
    the atoms:

    .. code-block:: python

        atom_coordinates = [[0, 0], [0, 4], [4, 0], [4, 4]]
        H_i = qml.pulse.transmon_interaction(atom_coordinates, wires)

        amplitude = lambda p, t: p * jnp.sin(jnp.pi * t)
        phase = jnp.pi / 2
        detuning = 3 * jnp.pi / 4
        wires = [0, 1, 2, 3]
        H_d = qml.pulse.transmon_drive(amplitude, phase, detuning, wires)

    >>> H_i
    ParametrizedHamiltonian: terms=6
    >>> H_d
    ParametrizedHamiltonian: terms=2

    The two terms of the drive field correspond to the first and second sum, corresponding to the drive and the shift term.
    This drive term corresponds to a global drive that acts on all wires of the device.

    .. code-block:: python

        dev = qml.device("default.qubit.jax", wires=wires)

        @qml.qnode(dev, interface="jax")
        def circuit(params):
            qml.evolve(H_i + H_d)(params, t=[0, 10])
            return qml.expval(qml.PauliZ(0))

    >>> params = [2.4]
    >>> circuit(params)
    Array(0.97137696, dtype=float32)
    >>> jax.grad(circuit)(params)
    [Array(0.10493923, dtype=float32)]

    We can also create a Hamiltonian with multiple local drives. The following circuit corresponds to the
    evolution where an additional local drive acting on wires ``[0, 1]`` is added to the Hamiltonian:

    .. code-block:: python

        amplitude_local = lambda p, t: p[0] * jnp.sin(2 * jnp.pi * t) + p[1]
        phase_local = jnp.pi / 4
        detuning_local = lambda p, t: p * jnp.exp(-0.25 * t)
        H_local = qml.pulse.transmon_drive(amplitude_local, phase_local, detuning_local, [0, 1])

        H = H_i + H_d + H_local

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def circuit_local(params):
            qml.evolve(H)(params, t=[0, 10])
            return qml.expval(qml.PauliZ(0))

    >>> params = [2.4, [1.3, -2.0]]
    >>> circuit_local(params)
    Array(0.45782223, dtype=float64)
    >>> jax.grad(circuit_local_drives)(params)
    [Array(-0.33522988, dtype=float64),
     [Array(0.40320718, dtype=float64, weak_type=True),
      Array(-0.12003976, dtype=float64, weak_type=True)]]
    """
    if isinstance(wires, int):
        wires = [wires]

    # We compute the `coeffs` and `observables` of the laser field
    coeffs = [
        amplitude_and_phase(qml.math.cos, amplitude, phase),
        amplitude_and_phase(qml.math.sin, amplitude, phase),
        detuning,
    ]

    drive_terms_1 = sum(qml.PauliX(wire) for wire in wires)
    drive_terms_2 = sum(-qml.PauliY(wire) for wire in wires)
    drive_terms_3 = sum(qml.PauliZ(wire) for wire in wires)

    observables = [drive_terms_1, drive_terms_2, drive_terms_3]

    # We convert the pulse data into a list of ``TransmonPulse`` objects
    pulses = [TransmonPulse(amplitude, phase, detuning, wires)]

    return TransmonHamiltonian(coeffs, observables, pulses=pulses)


class TransmonHamiltonian(ParametrizedHamiltonian):
    r"""Internal class used to keep track of the required information to translate a ``ParametrizedHamiltonian``
    into hardware.

    This class contains the ``coeffs`` and the ``observables`` that represent one or more
    terms of the Hamiltonian of an ensemble of Transmon atoms under the action of local and global
    laser fields:

    .. math::

        H = \frac{1}{2} \sum_i  \Omega_i(t) (\cos(\phi_i)\sigma_i^x - \sin(\phi_i)\sigma_i^y) -
        \frac{1}{2} \sum_i \delta_i(t) \sigma_i^z + \sum_{i<j} V_{ij} n_i n_j

    Additionally, it also contains two more attributes (``connections`` and ``pulses``) that contain
    the information that the hardware needs to execute this Hamiltonian.

    .. warning::

        This class should NEVER be initialized directly! Please use the functions
        :func:`transmon_interaction` and :func:`transmon_drive` instead.

    .. seealso:: :func:`transmon_interaction`, :func:`transmon_drive`, :class:`ParametrizedHamiltonian`

    Args:
        coeffs (Union[float, callable]): coefficients of the Hamiltonian expression, which may be
            constants or parametrized functions. All functions passed as ``coeffs`` must have two
            arguments, the first one being the trainable parameters and the second one being time.
        observables (Iterable[Observable]): observables in the Hamiltonian expression, of same
            length as ``coeffs``

    Keyword Args:
        connections (list): list of coordinates (in micrometers) of each atom in the ensemble
        pulses (list): list of ``TransmonPulse`` classes containing the information about the
            amplitude, phase, detuning and wires of each pulse
        interaction_coeff (float): Transmon interaction constant in units: :math:`\text{MHz} \times \mu m^6`.
            Defaults to :math:`862690 \text{MHz} \times \mu m^6`.

    Returns:
        TransmonHamiltonian: class representing the Hamiltonian of an ensemble of Transmon atoms
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        coeffs,
        observables,
        omega = None,
        delta = None,
        g = None,
        connections: list = None,
        pulses: List["TransmonPulse"] = None,
    ):
        self.omega = omega
        self.delta = delta,
        self.g = g,
        self.connections = connections
        self.pulses = [] if pulses is None else pulses
        super().__init__(coeffs, observables)

    def __call__(self, params, t):
        params = _transmon_reorder_parameters(params, self.coeffs_parametrized)
        return super().__call__(params, t)

    def __add__(self, other):
        if isinstance(other, TransmonHamiltonian):
            # Update coeffs, obs and hardware attributes
            if self.connections is not None:
                if other.connections is not None:
                    raise ValueError("We cannot add two Hamiltonians with an interaction term!")
                if not self.wires.contains_wires(other.wires):
                    warnings.warn(
                        "The wires of the laser fields are not present in the Transmon ensemble."
                    )
            elif other.connections is not None and not other.wires.contains_wires(self.wires):
                warnings.warn(
                    "The wires of the laser fields are not present in the Transmon ensemble."
                )

            new_connections = self.connections or other.connections
            new_pulses = self.pulses + other.pulses
            new_ops = self.ops + other.ops
            new_coeffs = self.coeffs + other.coeffs
            return TransmonHamiltonian(new_coeffs, new_ops, connections=new_connections, pulses=new_pulses)

        ops = self.ops.copy()
        coeffs = self.coeffs.copy()
        connections = self.connections
        pulses = self.pulses

        if isinstance(other, (Hamiltonian, ParametrizedHamiltonian)):
            new_coeffs = coeffs + other.coeffs.copy()
            new_ops = ops + other.ops.copy()
            return TransmonHamiltonian(new_coeffs, new_ops, connections=connections, pulses=pulses)

        if isinstance(other, qml.ops.SProd):  # pylint: disable=no-member
            new_coeffs = coeffs + [other.scalar]
            new_ops = ops + [other.base]
            return TransmonHamiltonian(new_coeffs, new_ops, connections=connections, pulses=pulses)

        if isinstance(other, Operator):
            new_coeffs = coeffs + [1]
            new_ops = ops + [other]
            return TransmonHamiltonian(new_coeffs, new_ops, connections=connections, pulses=pulses)

        return NotImplemented

    def __radd__(self, other):
        """Deals with the special case where a TransmonHamiltonian is added to a
        ParametrizedHamiltonian. Ensures that this returs a TransmonHamiltonian where
        the order of the parametrized coefficients and operators matches the order of
        the hamiltonians, i.e. that

        ParametrizedHamiltonian + TransmonHamiltonian

        returns a TransmonHamiltonian where the call expects params = [params_PH] + [params_RH]
        """
        if isinstance(other, ParametrizedHamiltonian):
            ops = self.ops.copy()
            coeffs = self.coeffs.copy()

            new_coeffs = other.coeffs.copy() + coeffs
            new_ops = other.ops.copy() + ops

            return TransmonHamiltonian(
                new_coeffs, new_ops, connections=self.connections, pulses=self.pulses
            )

        return self.__add__(other)


@dataclass
class TransmonPulse:
    """Dataclass that contains the information of a single Transmon pulse. This class is used
    internally in PL to group into a single object all the data related to a single laser field.

    Args:
        amplitude (Union[float, Callable]): float or callable returning the amplitude (in MHz) of a laser
            field
        phase (Union[float, Callable]): float containing the phase (in radians) of the laser field
        detuning (Union[float, Callable]): float or callable returning the detuning (in MHz) of a
            laser field
        wires (Union[int, List[int]]): integer or list containing wire values that the laser field
            acts on
    """

    amplitude: Union[float, Callable]
    phase: Union[float, Callable]
    detuning: Union[float, Callable]
    wires: List[Wires]

    def __post_init__(self):
        self.wires = Wires(self.wires)

    def __eq__(self, other):
        return (
            self.amplitude == other.amplitude
            and self.phase == other.phase
            and self.detuning == other.detuning
            and self.wires == other.wires
        )


def amplitude_and_phase(trig_fn, amp, phase):
    """Wrapper function for combining amplitude and phase into a single callable
    (or constant if neither amplitude nor phase are callable)."""
    if not callable(amp) and not callable(phase):
        return amp * trig_fn(phase)
    return AmplitudeAndPhase(trig_fn, amp, phase)


# pylint:disable = too-few-public-methods
class AmplitudeAndPhase:
    """Class storing combined amplitude and phase callable if either or both
    of amplitude nor phase are callable."""

    def __init__(self, trig_fn, amp, phase):
        self.amp_is_callable = callable(amp)
        self.phase_is_callable = callable(phase)

        def callable_amp_and_phase(params, t):
            return amp(params[0], t) * trig_fn(phase(params[1], t))

        def callable_amp(params, t):
            return amp(params, t) * trig_fn(phase)

        def callable_phase(params, t):
            return amp * trig_fn(phase(params, t))

        if self.amp_is_callable and self.phase_is_callable:
            self.func = callable_amp_and_phase

        elif self.amp_is_callable:
            self.func = callable_amp

        elif self.phase_is_callable:
            self.func = callable_phase

    def __call__(self, params, t):
        return self.func(params, t)


def _transmon_reorder_parameters(params, coeffs_parametrized):
    """Takes `params`, and reorganizes it based on whether the Hamiltonian has
    callable phase and/or callable amplitude.

    Consolidates phase and amplitude parameters in the case that both are callable,
    and duplicates phase and/or amplitude parameters if either are callables, since
    they will be passed to two operators in the Hamiltonian"""

    reordered_params = []

    coeff_idx = 0
    params_idx = 0

    for i, coeff in enumerate(coeffs_parametrized):
        if i == coeff_idx:
            if isinstance(coeff, AmplitudeAndPhase):
                if coeff.phase_is_callable and coeff.amp_is_callable:
                    # add the joined parameters twice, and skip an index
                    reordered_params.append([params[params_idx], params[params_idx + 1]])
                    reordered_params.append([params[params_idx], params[params_idx + 1]])
                    coeff_idx += 2
                    params_idx += 2
                elif coeff.phase_is_callable or coeff.amp_is_callable:
                    reordered_params.append(params[params_idx])
                    reordered_params.append(params[params_idx])
                    coeff_idx += 2
                    params_idx += 1
            else:
                reordered_params.append(params[params_idx])
                coeff_idx += 1
                params_idx += 1

    return reordered_params
