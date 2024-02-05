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
This module contains the ABC triples for states, transformations in Bargmann representation.

Note that the ABC triples in this file follow the same standard order definition as Wires class:
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
#  Pure States
#  ~~~~~~~~~~~~


def vacuum_state_ABC_triples(num_modes: int) -> Union[Matrix, Vector, Scalar]:
    r"""Returns the ABC triples of the pure vacuum state.

    Args:
        num_modes (int): number of modes

    Returns:
        (Matrix, Vector, Scalar): A matrix, B vector and C scalar of the pure vacuum state.
    """
    return vacuum_A_matrix(num_modes), vacuum_B_vector(num_modes), 1.0


def coherent_state_ABC_triples(
    x: Union[Scalar, Vector], y: Union[Scalar, Vector]
) -> Union[Matrix, Vector, Scalar]:
    r"""Returns the ABC triples of the pure coherent state.

    The dimension depends on the dimensions of ``x`` and ``y``. If one of them has dimension one, we will repete it
    to have the same dimension as the other one. For example, if ``x = [1,2,3], y = [1]``, we will fill it automatically
    like ``x = [1,2,3], y = [1,1,1]``.

    Args:
        x (scalar or vector): real part of displacement (in units of :math:`\sqrt{\hbar}`)
        y (scalar or vector): imaginary part of displacement (in units of :math:`\sqrt{\hbar}`)

    Returns:
        (Matrix, Vector, Scalar): A matrix, B vector and C scalar of the pure coherent state.
    """
    x = math.atleast_1d(x, math.float64)
    y = math.atleast_1d(y, math.float64)
    if x.shape[-1] == 1:
        x = math.tile(x, y.shape)
    if y.shape[-1] == 1:
        y = math.tile(y, x.shape)
    num_modes = x.shape
    return vacuum_A_matrix(num_modes), x + 1j * y, math.exp(-0.5 * (x**2 + y**2))


def squeezed_vacuum_state_ABC_triples(
    r: Union[Scalar, Vector], phi: Union[Scalar, Vector]
) -> Union[Matrix, Vector, Scalar]:
    r"""Returns the ABC triples of a squeezed vacuum state.

    The dimension depends on the dimensions of ``r`` and ``phi``. If one of them has dimension one, we will repete it
    to have the same dimension as the other one. For example, if ``r = [1,2,3], phi = [1]``, we will fill it automatically
    like ``r = [1,2,3], phi = [1,1,1]``.

    Args:
        r (scalar or vector): squeezing magnitude
        phi (scalar or vector): squeezing angle

    Returns:
        (Matrix, Vector, Scalar): A matrix, B vector and C scalar of the squeezed vacuum state.
    """
    r = math.atleast_1d(r, math.float64)
    phi = math.atleast_1d(phi, math.float64)
    if r.shape[-1] == 1:
        r = math.tile(r, phi.shape)
    if phi.shape[-1] == 1:
        phi = math.tile(phi, r.shape)
    num_modes = phi.shape[-1]
    return squeezed_vacuum_A_matrix(r, phi), vacuum_B_vector(num_modes), 1 / math.sqrt(math.cosh(r))


def displaced_squeezed_vacuum_state_ABC_triples(
    x: Union[Scalar, Vector],
    y: Union[Scalar, Vector],
    r: Union[Scalar, Vector],
    phi: Union[Scalar, Vector],
) -> Union[Matrix, Vector, Scalar]:
    r"""Returns the ABC triples of a displaced squeezed vacuum state.

    ValueError will be raise if the dimensions of ``x``, ``y``, ``r`` and ``phi`` are different.

    Args:
        r (scalar or vector): squeezing magnitude
        phi (scalar or vector): squeezing angle
        x (scalar or vector): real part of displacement (in units of :math:`\sqrt{\hbar}`)
        y (scalar or vector): imaginary part of displacement (in units of :math:`\sqrt{\hbar}`)

    Returns:
        (Matrix, Vector, Scalar): A matrix, B vector and C scalar of the squeezed vacuum state.
    """
    r = math.atleast_1d(r, math.float64)
    phi = math.atleast_1d(phi, math.float64)
    x = math.atleast_1d(x, math.float64)
    y = math.atleast_1d(y, math.float64)
    if not (
        (r.shape[-1] == phi.shape[-1]) & (x.shape[-1] == y.shape[-1]) & (r.shape[-1] == x.shape[-1])
    ):
        raise ValueError("The shape of them must be the same.")
    return (
        squeezed_vacuum_A_matrix(r, phi),
        displaced_squeezed_vacuum_B_vector(x, y, r, phi),
        math.exp(
            -0.5 * (x**2 + y**2)
            - 0.5 * (x - 1j * y) ** 2 * math.sinh(r) / math.cosh(r) * math.exp(1j * phi)
        )
        / math.sqrt(math.cosh(r)),
    )


def two_mode_squeezed_vacuum_state_ABC_triples(
    r: Union[Scalar, Vector],
    phi: Union[Scalar, Vector],
) -> Union[Matrix, Vector, Scalar]:
    r"""Returns the ABC triples of a two mode squeezed vacuum state.

    The dimension depends on the dimensions of ``r`` and ``phi``. If one of them has dimension one, we will repete it
    to have the same dimension as the other one. For example, if ``r = [1,2,3], phi = [1]``, we will fill it automatically
    like ``r = [1,2,3], phi = [1,1,1]``.

    Args:
        r (scalar or vector): squeezing magnitude
        phi (scalar or vector): squeezing angle

    Returns:
        (Matrix, Vector, Scalar): A matrix, B vector and C scalar of the two mode squeezed vacuum state.
    """
    r = math.atleast_1d(r, math.float64)
    phi = math.atleast_1d(phi, math.float64)
    if r.shape[-1] == 1:
        r = math.tile(r, phi.shape)
    if phi.shape[-1] == 1:
        phi = math.tile(phi, r.shape)
    num_modes = phi.shape[-1] * 2
    return (
        two_mode_squeezed_vacuum_A_matrix(r, phi, num_modes),
        vacuum_B_vector(num_modes),
        1 / math.cosh(r),
    )


#  ~~~~~~~~~~~~
#  Mixed States
#  ~~~~~~~~~~~~


def thermal_state_ABC_triples(nbar: Vector) -> Union[Matrix, Vector, Scalar]:
    r"""Returns the ABC triples of a thermal state.

    Args:
        nbar (vector): average number of photons per mode

    Returns:
        (Matrix, Vector, Scalar): A matrix, B vector and C scalar of the thermal state.
    """
    nbar = math.atleast_1d(nbar, math.float64)
    num_modes = nbar.shape[-1]
    return nbar / (nbar + 1) * X_matrix(), vacuum_B_vector(num_modes), 1 / (nbar + 1)


# ~~~~~~~~~~~~~~~~~~~~~~~~
#  Unitary transformations
# ~~~~~~~~~~~~~~~~~~~~~~~~


def rotation_gate_ABC_triples(theta: Union[Scalar, Vector]):
    r"""Returns the ABC triples of a rotation gate.

    The gate is defined by
        :math:`R(\theta) = \exp(i\theta\hat{a}^\dagger\hat{a})`.

    The dimension depends on the dimensions of ``theta``.

    Args:
        theta (scalar or vector): rotation angle

    Returns:
        (Matrix, Vector, Scalar): A matrix, B vector and C scalar of the rotation gate.
    """
    theta = math.atleast_1d(theta, math.float64)
    num_modes = theta.shape[-1]
    return (
        math.cast(
            np.kron(math.array([[0, 1], [1, 0]]), math.exp(1j * theta) * math.eye(num_modes)),
            math.complex128,
        ),
        vacuum_B_vector(num_modes),
        1.0,
    )


def displacement_gate_ABC_triples(x: Union[Scalar, Vector], y: Union[Scalar, Vector]):
    r"""Returns the ABC triples of a displacement gate.

    The gate is defined by
        :math:`D(\gamma) = \exp(\gamma\hat{a}^\dagger-\gamma^*\hat{a})`,
    where ``\gamma = x + 1j*y``.

    The dimension depends on the dimensions of ``x`` and ``y``. If one of them has dimension one, we will repete it
    to have the same dimension as the other one. For example, if ``x = [1,2,3], y = [1]``, we will fill it automatically
    like ``x = [1,2,3], y = [1,1,1]``.

    Args:
        x (scalar or vector): real part of displacement (in units of :math:`\sqrt{\hbar}`)
        y (scalar or vector): imaginary part of displacement (in units of :math:`\sqrt{\hbar}`)

    Returns:
        (Matrix, Vector, Scalar): A matrix, B vector and C scalar of the displacement gate.
    """
    x = math.atleast_1d(x, math.float64)
    y = math.atleast_1d(y, math.float64)
    if x.shape[-1] == 1:
        x = math.tile(x, y.shape)
    if y.shape[-1] == 1:
        y = math.tile(y, x.shape)
    num_modes = x.shape
    return (
        X_matrix_for_unitary(num_modes),
        math.concat([x + 1j * y, -x + 1j * y], axis=0),
        math.exp(-math.sum(x**2 + y**2) / 2),
    )


def squeezing_gate_ABC_triples(r: Union[Scalar, Vector], delta: Union[Scalar, Vector]):
    r"""Returns the ABC triples of a squeezing gate.

    The gate is defined by
        :math:`S(\zeta) = \exp(\zeta^*\hat{a}^2 - \zeta\hat{a}^{\dagger 2})`,
    where ``\zeta = r\exp(i\delta)``.

    The dimension depends on the dimensions of ``r`` and ``\delta``. If one of them has dimension one, we will repete it
    to have the same dimension as the other one. For example, if ``r = [1,2,3], \delta = [1]``, we will fill it automatically
    like ``r = [1,2,3], \delta = [1,1,1]``.

    Args:
        r (scalar or vector): squeezing magnitude
        delta (scalar or vector): squeezing angle

    Returns:
        (Matrix, Vector, Scalar): A matrix, B vector and C scalar of the squeezing gate.
    """
    r = math.atleast_1d(r, math.float64)
    delta = math.atleast_1d(delta, math.float64)
    if r.shape[-1] == 1:
        r = math.tile(r, delta.shape)
    if delta.shape[-1] == 1:
        delta = math.tile(delta, r.shape)
    num_modes = delta.shape[-1]
    return (
        squeezing_gate_A_matrix(r, delta),
        vacuum_B_vector(num_modes * 2),
        1 / math.sqrt(math.cosh(r)),
    )


def beamsplitter_gate_ABC_triples(theta: Union[Scalar, Vector], phi: Union[Scalar, Vector]):
    r"""Returns the ABC triples of a beamsplitter gate on two modes.

    The gate is defined by
        :math:`BS(\theta, \phi) = \exp()`.

    The dimension depends on the dimensions of ``theta`` and ``phi``. If one of them has dimension one, we will repete it
    to have the same dimension as the other one. For example, if ``theta = [1,2,3], phi = [1]``, we will fill it automatically
    like ``theta = [1,2,3], phi = [1,1,1]``.

    Args:
        theta (scalar or vector): transmissivity parameter
        phi (scalar or vector): phase parameter

    Returns:
        (Matrix, Vector, Scalar): A matrix, B vector and C scalar of the beamsplitter gate.
    """
    theta = math.atleast_1d(theta, math.float64)
    phi = math.atleast_1d(phi, math.float64)
    if theta.shape[-1] == 1:
        theta = math.tile(theta, phi.shape)
    if phi.shape[-1] == 1:
        phi = math.tile(phi, theta.shape)
    num_modes = phi.shape[-1]
    O_n = math.zeros((num_modes, num_modes))
    V = beamsplitter_gate_V_matrix(theta, phi)
    return math.array([[O_n, V], [math.transpose(V), O_n]]), vacuum_B_vector(num_modes * 2), 1.0


#  ~~~~~~~~~~~~
#  Utilities
#  ~~~~~~~~~~~~


def X_matrix() -> Matrix:
    r"""Returns the X matrix."""
    return math.array([[0, 1], [1, 0]])


def X_matrix_for_unitary(num_modes: int) -> Matrix:
    r"""Returns the X matrix for the order of unitaries."""
    return math.cast(np.kron(math.array([[0, 1], [1, 0]]), math.eye(num_modes)), math.complex128)


def vacuum_A_matrix(num_modes: int) -> Matrix:
    r"""Returns the A matrix of a vacuum state."""
    return math.zeros((num_modes, num_modes))


def vacuum_B_vector(num_modes: int) -> Vector:
    r"""Returns the B vector of a vacuum state."""
    return math.zeros(num_modes)


def squeezed_vacuum_A_matrix(r: Union[Scalar, Vector], phi: Union[Scalar, Vector]) -> Matrix:
    r"""Returns the A matrix of a squeezed vacuum state."""
    return math.diag(-math.sinh(r) / math.cosh(r) * math.exp(1j * phi))


def displaced_squeezed_vacuum_B_vector(
    x: Union[Scalar, Vector],
    y: Union[Scalar, Vector],
    r: Union[Scalar, Vector],
    phi: Union[Scalar, Vector],
) -> Vector:
    r"""Returns the B vector of a displaced squeezed vacuum state."""
    return (x + 1j * y) + (x - 1j * y) * math.sinh(r) / math.cosh(r) * math.exp(1j * phi)


def two_mode_squeezed_vacuum_A_matrix(
    r: Union[Scalar, Vector], phi: Union[Scalar, Vector], num_modes: int
):
    r"""Returns the A matrix of a two-mode squeezed vacuum state."""
    O_n = math.zeros((num_modes, num_modes))
    return math.array(
        [
            [O_n, -math.exp(1j * phi) * math.sinh(r) / math.cosh(r)],
            [-math.exp(1j * phi) * math.sinh(r) / math.cosh(r), O_n],
        ]
    )


def squeezing_gate_A_matrix(r: Union[Scalar, Vector], delta: Union[Scalar, Vector]):
    r"""Returns the A matrix of a squeezing gate."""
    tanhr = math.sinh(r) / math.cosh(r)
    sechr = 1 / math.cosh(r)
    return math.array(
        [[math.exp(1j * delta) * tanhr, sechr], [sechr, -math.exp(-1j * delta) * tanhr]]
    )


def beamsplitter_gate_V_matrix(theta: Union[Scalar, Vector], phi: Union[Scalar, Vector]):
    r"""Returns the V transformation matrix of a beamsplitter."""
    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    return math.array(
        [[costheta, -math.exp(-1j * phi) * sintheta], [math.exp(1j * phi) * sintheta, costheta]]
    )
