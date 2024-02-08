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

"""
This module contains the ABC triples for states and transformations in the ``Bargmann`` representation.

The ABC triples in this module follow the same standard order definition as the ``Wires`` class:
Pure states: (out_ket_1, out_ket_2, ...)
Mixed states: (out_bra_1, out_bra_2, ...; out_ket_1, out_ket_2, ...)
Unitaries: (out_ket_1, out_ket_2, ...; in_ket_1, in_ket_2, ...))
Channels: (out_bra_1, out_bra_2, ...; in_bra_1, in_bra_2, ...; out_ket_1, out_ket_2, ...; in_ket_1, in_ket_2, ...)
"""
import numpy as np
from typing import Union
from mrmustard import math
from mrmustard.utils.typing import Matrix, Vector, Scalar


#  ~~~~~~~~~~~~
#  Utilities
#  ~~~~~~~~~~~~


def _X_matrix() -> Matrix:
    r"""Returns the X matrix."""
    return math.array([[0, 1], [1, 0]])


def _X_matrix_for_unitary(num_modes: int) -> Matrix:
    r"""Returns the X matrix for the order of unitaries."""
    return math.cast(np.kron(math.array([[0, 1], [1, 0]]), math.eye(num_modes)), math.complex128)


def _vacuum_A_matrix(shape: int) -> Matrix:
    r"""Returns the A matrix with all zeros."""
    return math.zeros((shape, shape))


def _vacuum_B_vector(shape: int) -> Vector:
    r"""Returns the B vector with all zeros."""
    return math.zeros(shape)


#  ~~~~~~~~~~~~
#  Pure States
#  ~~~~~~~~~~~~


def vacuum_state_Abc_triples(num_modes: int) -> Union[Matrix, Vector, Scalar]:
    r"""Returns the Abc triples of the pure vacuum state.

    Args:
        num_modes: number of modes

    Returns:
        A matrix, b vector and c scalar of the pure vacuum state.
    """
    return _vacuum_A_matrix(num_modes), _vacuum_B_vector(num_modes), 1.0


def coherent_state_Abc_triples(
    x: Union[Scalar, list], y: Union[Scalar, list]
) -> Union[Matrix, Vector, Scalar]:
    r"""Returns the Abc triples of the pure coherent state.

    The dimension depends on the dimensions of ``x`` and ``y``. If one of them has dimension one, it is repeated
    to have the same dimension as the other one. For example, ``x = [1,2,3]`` and ``y = [1]`` become
    like ``x = [1,2,3], y = [1,1,1]``.

    Args:
        x: real part of displacement (in units of :math:`\sqrt{\hbar}`)
        y: imaginary part of displacement (in units of :math:`\sqrt{\hbar}`)

    Returns:
        A matrix, b vector and c scalar of the pure coherent state.
    """
    x = math.atleast_1d(x, math.float64)
    y = math.atleast_1d(y, math.float64)
    if x.shape[-1] == 1:
        x = math.tile(x, y.shape)
    if y.shape[-1] == 1:
        y = math.tile(y, x.shape)
    num_modes = x.shape
    return _vacuum_A_matrix(num_modes), x + 1j * y, math.exp(-0.5 * math.sum(x**2 + y**2))


def squeezed_vacuum_state_Abc_triples(
    r: Union[Scalar, list], phi: Union[Scalar, list]
) -> Union[Matrix, Vector, Scalar]:
    r"""Returns the Abc triples of a squeezed vacuum state.

    The dimension depends on the dimensions of ``r`` and ``phi``. If one of them has dimension one, it is repeated
    to have the same dimension as the other one. For example, ``r = [1,2,3]`` and ``phi = [1]`` become
    like ``r = [1,2,3], phi = [1,1,1]``.

    Args:
        r: squeezing magnitude
        phi: squeezing angle

    Returns:
        A matrix, b vector and c scalar of the squeezed vacuum state.
    """
    r = math.atleast_1d(r, math.float64)
    phi = math.atleast_1d(phi, math.float64)
    if r.shape[-1] == 1:
        r = math.tile(r, phi.shape)
    if phi.shape[-1] == 1:
        phi = math.tile(phi, r.shape)
    num_modes = phi.shape[-1]
    A = math.diag(-math.sinh(r) / math.cosh(r) * math.exp(1j * phi))
    return A, _vacuum_B_vector(num_modes), math.prod(1 / math.sqrt(math.cosh(r)))


def displaced_squeezed_vacuum_state_Abc_triples(
    x: Union[Scalar, list],
    y: Union[Scalar, list],
    r: Union[Scalar, list],
    phi: Union[Scalar, list],
) -> Union[Matrix, Vector, Scalar]:
    r"""Returns the Abc triples of a displaced squeezed vacuum state.

    Raises:
        ValueError: If the dimensions of ``x``, ``y``, ``r`` and ``phi`` are inconsistent.

    Args:
        r: squeezing magnitude
        phi: squeezing angle
        x: real part of displacement (in units of :math:`\sqrt{\hbar}`)
        y: imaginary part of displacement (in units of :math:`\sqrt{\hbar}`)

    Returns:
        A matrix, b vector and c scalar of the squeezed vacuum state.
    """
    r = math.atleast_1d(r, math.float64)
    phi = math.atleast_1d(phi, math.float64)
    x = math.atleast_1d(x, math.float64)
    y = math.atleast_1d(y, math.float64)
    if not (
        (r.shape[-1] == phi.shape[-1]) & (x.shape[-1] == y.shape[-1]) & (r.shape[-1] == x.shape[-1])
    ):
        raise ValueError("Found parameters of inconsistent shapes.")
    A = math.diag(-math.sinh(r) / math.cosh(r) * math.exp(1j * phi))
    b = (x + 1j * y) + (x - 1j * y) * math.sinh(r) / math.cosh(r) * math.exp(1j * phi)
    c = math.exp(
        -0.5 * (x**2 + y**2)
        - 0.5 * (x - 1j * y) ** 2 * math.sinh(r) / math.cosh(r) * math.exp(1j * phi)
    ) / math.sqrt(math.cosh(r))
    return A, b, c


def two_mode_squeezed_vacuum_state_Abc_triples(
    r: Union[Scalar, list],
    phi: Union[Scalar, list],
) -> Union[Matrix, Vector, Scalar]:
    r"""Returns the Abc triples of a two mode squeezed vacuum state.

    The dimension depends on the dimensions of ``r`` and ``phi``. If one of them has dimension one, it is repeated
    to have the same dimension as the other one. For example, ``r = [1,2,3]`` and `` phi = [1]`` become
    like ``r = [1,2,3], phi = [1,1,1]``.

    Args:
        r: squeezing magnitude
        phi: squeezing angle

    Returns:
        A matrix, b vector and c scalar of the two mode squeezed vacuum state.
    """
    r = math.atleast_1d(r, math.float64)
    phi = math.atleast_1d(phi, math.float64)
    if r.shape[-1] == 1:
        r = math.tile(r, phi.shape)
    if phi.shape[-1] == 1:
        phi = math.tile(phi, r.shape)
    num_modes = phi.shape[-1] * 2
    O_n = math.zeros((num_modes, num_modes))
    tanhr = math.diag(math.sinh(r) / math.cosh(r))
    A = math.block(
        [
            [O_n, -math.exp(1j * phi) * tanhr],
            [-math.exp(1j * phi) * tanhr, O_n],
        ]
    )
    return (
        A,
        _vacuum_B_vector(num_modes),
        math.prod(1 / math.cosh(r)),
    )


#  ~~~~~~~~~~~~
#  Mixed States
#  ~~~~~~~~~~~~


def thermal_state_Abc_triples(nbar: Vector) -> Union[Matrix, Vector, Scalar]:
    r"""Returns the Abc triples of a thermal state.

    Args:
        nbar: average number of photons per mode

    Returns:
        A matrix, b vector and c scalar of the thermal state.
    """
    nbar = math.atleast_1d(nbar, math.float64)
    num_modes = nbar.shape[-1]
    return nbar / (nbar + 1) * _X_matrix(), _vacuum_B_vector(num_modes), 1 / (nbar + 1)


# ~~~~~~~~~~~~~~~~~~~~~~~~
#  Unitary transformations
# ~~~~~~~~~~~~~~~~~~~~~~~~


def rotation_gate_Abc_triples(theta: Union[Scalar, list]):
    r"""Returns the Abc triples of a rotation gate.

    The gate is defined by
        :math:`R(\theta) = \exp(i\theta\hat{a}^\dagger\hat{a})`.

    The dimension depends on the dimensions of ``theta``.

    Args:
        theta: rotation angle

    Returns:
        A matrix, b vector and c scalar of the rotation gate.
    """
    theta = math.atleast_1d(theta, math.float64)
    num_modes = theta.shape[-1]
    A = math.cast(
        np.kron(math.array([[0, 1], [1, 0]]), math.exp(1j * theta) * math.eye(num_modes)),
        math.complex128,
    )
    return A, _vacuum_B_vector(num_modes), 1.0


def displacement_gate_Abc_triples(x: Union[Scalar, list], y: Union[Scalar, list]):
    r"""Returns the Abc triples of a displacement gate.

    The gate is defined by
        :math:`D(\gamma) = \exp(\gamma\hat{a}^\dagger-\gamma^*\hat{a})`,
    where ``\gamma = x + 1j*y``.

    The dimension depends on the dimensions of ``x`` and ``y``. If one of them has dimension one, it is repeated
    to have the same dimension as the other one. For example, ``x = [1,2,3]`` and ``y = [1]`` become
    like ``x = [1,2,3], y = [1,1,1]``.

    Args:
        x: real part of displacement (in units of :math:`\sqrt{\hbar}`)
        y: imaginary part of displacement (in units of :math:`\sqrt{\hbar}`)

    Returns:
        A matrix, b vector and c scalar of the displacement gate.
    """
    x = math.atleast_1d(x, math.float64)
    y = math.atleast_1d(y, math.float64)
    if x.shape[-1] == 1:
        x = math.tile(x, y.shape)
    if y.shape[-1] == 1:
        y = math.tile(y, x.shape)
    num_modes = x.shape
    b = math.concat([x + 1j * y, -x + 1j * y], axis=0)
    c = math.exp(-math.sum(x**2 + y**2) / 2)
    return _X_matrix_for_unitary(num_modes), b, c


def squeezing_gate_Abc_triples(r: Union[Scalar, list], delta: Union[Scalar, list]):
    r"""Returns the Abc triples of a squeezing gate.

    The gate is defined by
        :math:`S(\zeta) = \exp(\zeta^*\hat{a}^2 - \zeta\hat{a}^{\dagger 2})`,
    where ``\zeta = r\exp(i\delta)``.

    The dimension depends on the dimensions of ``r`` and ``\delta``. If one of them has dimension one, it is repeated
    to have the same dimension as the other one. For example, ``r = [1,2,3]`` and ``\delta = [1]`` become
    like ``r = [1,2,3], \delta = [1,1,1]``.

    Args:
        r: squeezing magnitude
        delta: squeezing angle

    Returns:
        A matrix, b vector and c scalar of the squeezing gate.
    """
    r = math.atleast_1d(r, math.float64)
    delta = math.atleast_1d(delta, math.float64)
    if r.shape[-1] == 1:
        r = math.tile(r, delta.shape)
    if delta.shape[-1] == 1:
        delta = math.tile(delta, r.shape)
    num_modes = delta.shape[-1]
    tanhr = math.diag(math.sinh(r) / math.cosh(r))
    sechr = math.diag(1 / math.cosh(r))
    A = math.block([[math.exp(1j * delta) * tanhr, sechr], [sechr, -math.exp(-1j * delta) * tanhr]])
    return A, _vacuum_B_vector(num_modes * 2), math.prod(1 / math.sqrt(math.cosh(r)))


def beamsplitter_gate_Abc_triples(theta: Union[Scalar, list], phi: Union[Scalar, list]):
    r"""Returns the Abc triples of a beamsplitter gate on two modes.

    The gate is defined by
        :math:`BS(\theta, \phi) = \exp()`.

    The dimension depends on the dimensions of ``theta`` and ``phi``. If one of them has dimension one, it is repeated
    to have the same dimension as the other one. For example, ``theta = [1,2,3]`` and ``phi = [1]`` become
    like ``theta = [1,2,3], phi = [1,1,1]``.

    Args:
        theta: transmissivity parameter
        phi: phase parameter

    Returns:
        A matrix, b vector and c scalar of the beamsplitter gate.
    """
    theta = math.atleast_1d(theta, math.float64)
    phi = math.atleast_1d(phi, math.float64)
    if theta.shape[-1] == 1:
        theta = math.tile(theta, phi.shape)
    if phi.shape[-1] == 1:
        phi = math.tile(phi, theta.shape)
    num_modes = phi.shape[-1]
    O_n = math.zeros((num_modes, num_modes))
    costheta = math.diag(math.cos(theta))
    sintheta = math.diag(math.sin(theta))
    V = math.block(
        [[costheta, -math.exp(-1j * phi) * sintheta], [math.exp(1j * phi) * sintheta, costheta]]
    )
    A = math.block([[O_n, V], [math.transpose(V), O_n]])
    return A, _vacuum_B_vector(num_modes * 2), 1.0


# ~~~~~~~~~~
#  Channels
# ~~~~~~~~~~


def attenuator_Abc_triples(eta: Union[Scalar, list]):
    r"""Returns the Abc triples of an atternuator.

    The dimension depends on the dimensions of ``eta``.

    Args:
        eta: value of the transmissivity, must be between 0 and 1

    Returns:
        A matrix, b vector and c scalar of the attenuator channel.
    """
    eta = math.atleast_1d(eta, math.float64)
    num_modes = eta.shape[-1]
    O_n = math.zeros((num_modes, num_modes))
    A = math.block(
        [
            [O_n, math.diag(math.sqrt(eta)), O_n, O_n],
            [math.diag(math.sqrt(eta)), O_n, O_n, math.eye(num_modes) - math.diag(eta)],
            [O_n, O_n, O_n, math.diag(math.sqrt(eta))],
            [O_n, math.eye(num_modes) - math.diag(eta), math.diag(math.sqrt(eta)), O_n],
        ]
    )
    return A, _vacuum_B_vector(num_modes * 2), np.prod(eta)


def amplifier_Abc_triples(g: Union[Scalar, list]):
    r"""Returns the Abc triples of an amplifier.

    The dimension depends on the dimensions of ``g``.

    Args:
        g: value of the ``gain > 1``

    Returns:
        A matrix, b vector and c scalar of the amplifier channel.
    """
    g = math.atleast_1d(g, math.float64)
    num_modes = g.shape[-1]
    O_n = math.zeros((num_modes, num_modes))
    A = math.block(
        [
            [O_n, 1 / math.sqrt(g), 1 - 1 / g, O_n],
            [1 / math.sqrt(g), O_n, O_n, 1 - g],
            [1 - 1 / g, O_n, O_n, 1 / math.sqrt(g)],
            [O_n, O_n, 1 / math.sqrt(g), O_n],
        ]
    )
    return A, _vacuum_B_vector(num_modes * 2), np.prod(1 / g)


def fock_damping_Abc_triples(num_modes: int):
    r"""Returns the Abc triples of a Fock damper.

    Args:
         num_modes: number of modes

    Returns:
        A matrix, b vector and c scalar of the Fock damping channel.
    """
    return _X_matrix_for_unitary(num_modes * 2), _vacuum_B_vector(num_modes * 4), 1.0
