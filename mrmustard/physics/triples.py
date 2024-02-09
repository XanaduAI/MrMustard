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
    return math.astensor([[0, 1], [1, 0]], dtype=math.float64)


def _X_matrix_for_unitary(n_modes: int) -> Matrix:
    r"""Returns the X matrix for the order of unitaries."""
    return math.cast(np.kron(math.astensor([[0, 1], [1, 0]]), math.eye(n_modes)), math.complex128)


def _vacuum_A_matrix(n_modes: int) -> Matrix:
    r"""Returns the A matrix with all zeros."""
    return math.zeros((n_modes, n_modes))


def _vacuum_B_vector(n_modes: int) -> Vector:
    r"""Returns the B vector with all zeros."""
    return math.zeros((n_modes,))


def _reshape(**kwargs):
    r"""
    A utility function to reshape parameters.
    """
    names = list(kwargs.keys())
    vars = list(kwargs.values())

    vars = [math.atleast_1d(var, math.complex128) for var in vars]
    n_modes = max([len(var) for var in vars])

    for i, var in enumerate(vars):
        if len(var) == 1:
            var = math.tile(var, (n_modes,))
        else:
            if len(var) != n_modes:
                msg = f"Parameter {names[i]} has an incompatible shape."
                raise ValueError(msg)
        yield var


#  ~~~~~~~~~~~~
#  Pure States
#  ~~~~~~~~~~~~


def vacuum_state_Abc(n_modes: int) -> Union[Matrix, Vector, Scalar]:
    r"""Returns the Abc triples of the pure vacuum state.

    Args:
        n_modes: number of modes

    Returns:
        A matrix, b vector and c scalar of the pure vacuum state.
    """
    A = _vacuum_A_matrix(n_modes)
    b = _vacuum_B_vector(n_modes)
    c = 1.0

    return A, b, c


def coherent_state_Abc(
    x: Union[Scalar, list], y: Union[Scalar, list]
) -> Union[Matrix, Vector, Scalar]:
    r"""Returns the Abc triples of the pure coherent state.

    The dimension depends on the dimensions of ``x`` and ``y``. If one of them has dimension one,
    it is repeated to have the same dimension as the other one. For example, ``x = [1,2,3]`` and
    ``y = [1]`` become ``x = [1,2,3], y = [1,1,1]``.

    Args:
        x: real part of displacement (in units of :math:`\sqrt{\hbar}`)
        y: imaginary part of displacement (in units of :math:`\sqrt{\hbar}`)

    Returns:
        A matrix, b vector and c scalar of the pure coherent state.
    """
    x, y = list(_reshape(x=x, y=y))
    n_modes = len(x)

    A = _vacuum_A_matrix(n_modes)
    b = x + 1j * y
    c = math.prod(math.exp(-0.5 * (x**2 + y**2)))

    return A, b, c


def squeezed_vacuum_state_Abc(
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
    r, phi = list(_reshape(r=r, phi=phi))
    n_modes = len(r)

    A = math.diag(-math.sinh(r) / math.cosh(r) * math.exp(1j * phi))
    b = _vacuum_B_vector(n_modes)
    c = math.prod(1 / math.sqrt(math.cosh(r)))

    return A, b, c


def displaced_squeezed_vacuum_state_Abc(
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
    x, y, r, phi = list(_reshape(x=x, y=y, r=r, phi=phi))

    A = math.diag(-math.sinh(r) / math.cosh(r) * math.exp(1j * phi))
    b = (x + 1j * y) + (x - 1j * y) * math.sinh(r) / math.cosh(r) * math.exp(1j * phi)
    c = math.exp(
        -0.5 * (x**2 + y**2)
        - 0.5 * (x - 1j * y) ** 2 * math.sinh(r) / math.cosh(r) * math.exp(1j * phi)
    )
    c = math.prod(c / math.sqrt(math.cosh(r)))

    return A, b, c


def two_mode_squeezed_vacuum_state_Abc(
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
    n_modes = phi.shape[-1] * 2
    O_n = math.zeros((n_modes, n_modes))
    tanhr = math.diag(math.sinh(r) / math.cosh(r))
    A = math.block(
        [
            [O_n, -math.exp(1j * phi) * tanhr],
            [-math.exp(1j * phi) * tanhr, O_n],
        ]
    )
    return (
        A,
        _vacuum_B_vector(n_modes),
        math.prod(1 / math.cosh(r)),
    )


#  ~~~~~~~~~~~~
#  Mixed States
#  ~~~~~~~~~~~~


def thermal_state_Abc(nbar: Vector) -> Union[Matrix, Vector, Scalar]:
    r"""Returns the Abc triples of a thermal state.

    Args:
        nbar: average number of photons per mode

    Returns:
        A matrix, b vector and c scalar of the thermal state.
    """
    nbar = math.atleast_1d(nbar, math.float64)
    n_modes = len(nbar)

    A = nbar / (nbar + 1) * _X_matrix()
    b = _vacuum_B_vector(n_modes)
    c = 1 / (nbar + 1)

    return A, b, c


# ~~~~~~~~~~~~~~~~~~~~~~~~
#  Unitary transformations
# ~~~~~~~~~~~~~~~~~~~~~~~~


def rotation_gate_Abc(theta: Union[Scalar, list]):
    r"""Returns the Abc triples of a rotation gate.

    The gate is defined by
        :math:`R(\theta) = \exp(i\theta\hat{a}^\dagger\hat{a})`.

    The dimension depends on the dimensions of ``theta``.

    Args:
        theta: rotation angle

    Returns:
        A matrix, b vector and c scalar of the rotation gate.
    """
    theta = math.atleast_1d(theta, math.complex128)
    n_modes = len(theta)

    A = math.astensor([[0, 1], [1, 0]], math.complex128)
    A = np.kron(A, math.exp(1j * theta) * math.eye(n_modes, math.complex128))
    b = _vacuum_B_vector(n_modes)
    c = 1.0

    return A, b, c


def displacement_gate_Abc(x: Union[Scalar, list], y: Union[Scalar, list]):
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
    x, y = _reshape(x=x, y=y)
    n_modes = len(x)

    A = _X_matrix_for_unitary(n_modes)
    b = math.concat([x + 1j * y, -x + 1j * y], axis=0)
    c = math.exp(-math.sum(x**2 + y**2) / 2)

    return A, b, c


def squeezing_gate_Abc(r: Union[Scalar, list], delta: Union[Scalar, list]):
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
    r, delta = _reshape(r=r, delta=delta)
    n_modes = len(delta)

    tanhr = math.diag(math.sinh(r) / math.cosh(r))
    sechr = math.diag(1 / math.cosh(r))

    A = math.block([[math.exp(1j * delta) * tanhr, sechr], [sechr, -math.exp(-1j * delta) * tanhr]])
    b = _vacuum_B_vector(n_modes * 2)
    c = math.prod(1 / math.sqrt(math.cosh(r)))

    return A, b, c


def beamsplitter_gate_Abc(theta: Union[Scalar, list], phi: Union[Scalar, list]):
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
    theta, phi = _reshape(theta=theta, phi=phi)
    n_modes = 2 * len(theta)

    O_n = math.zeros((n_modes, n_modes), math.complex128)
    costheta = math.diag(math.cos(theta))
    sintheta = math.diag(math.sin(theta))
    V = math.block(
        [[costheta, -math.exp(-1j * phi) * sintheta], [math.exp(1j * phi) * sintheta, costheta]]
    )

    A = math.block([[O_n, V], [math.transpose(V), O_n]])
    b = _vacuum_B_vector(n_modes * 2)
    c = 1

    return A, b, c


# ~~~~~~~~~~
#  Channels
# ~~~~~~~~~~


def attenuator_Abc(eta: Union[Scalar, list]):
    r"""Returns the Abc triples of an atternuator.

    The dimension depends on the dimensions of ``eta``.

    Args:
        eta: The value of the transmissivity.

    Returns:
        A matrix, b vector and c scalar of the attenuator channel.

    Raises:
        ValueError: If ``eta`` is larger than `1` or smaller than `0`.
    """
    eta = math.atleast_1d(eta, math.float64)
    n_modes = len(eta)

    for e in eta:
        if e > 1 or e < 0:
            msg = "Transmissivity must be a float in the interval ``[0, 1]``"
            raise ValueError(msg)

    O_n = math.zeros((n_modes, n_modes))
    A = math.block(
        [
            [O_n, math.diag(math.sqrt(eta)), O_n, O_n],
            [math.diag(math.sqrt(eta)), O_n, O_n, math.eye(n_modes) - math.diag(eta)],
            [O_n, O_n, O_n, math.diag(math.sqrt(eta))],
            [O_n, math.eye(n_modes) - math.diag(eta), math.diag(math.sqrt(eta)), O_n],
        ]
    )
    b = _vacuum_B_vector(n_modes * 2)
    c = np.prod(eta)

    return A, b, c


def amplifier_Abc(g: Union[Scalar, list]):
    r"""Returns the Abc triples of an amplifier.

    The dimension depends on the dimensions of ``g``.

    Args:
        g: value of the ``gain > 1``

    Returns:
        A matrix, b vector and c scalar of the amplifier channel.
    """
    g = math.atleast_1d(g, math.float64)
    n_modes = len(g)

    O_n = math.zeros((n_modes, n_modes))
    g0 = math.diag(math.astensor([1 - g]))
    g1 = math.diag(math.astensor([1 / math.sqrt(g)]))
    g2 = math.diag(math.astensor([1 - 1 / g]))

    A = math.block(
        [
            [O_n, g1, g2, O_n],
            [g1, O_n, O_n, g0],
            [g2, O_n, O_n, g1],
            [O_n, O_n, g1, O_n],
        ]
    )
    b = _vacuum_B_vector(n_modes * 2)
    c = np.prod(1 / g)

    return A, b, c


def fock_damping_Abc(n_modes: int):
    r"""Returns the Abc triples of a Fock damper.

    Args:
         n_modes: number of modes

    Returns:
        A matrix, b vector and c scalar of the Fock damping channel.
    """
    A = _X_matrix_for_unitary(n_modes * 2)
    b = _vacuum_B_vector(n_modes * 4)
    c = 1.0

    return A, b, c
