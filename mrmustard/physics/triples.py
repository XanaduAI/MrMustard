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
This module contains the ``(A, b, c)`` triples for the Fock-Bargmann representation of
various states and transformations.
"""

from typing import Generator, Iterable, Union
from mrmustard import math
from mrmustard.utils.typing import Matrix, Vector, Scalar

import numpy as np


#  ~~~~~~~~~
#  Utilities
#  ~~~~~~~~~


def _X_matrix_for_unitary(n_modes: int) -> Matrix:
    r"""
    The X matrix for the order of unitaries.
    """
    return math.cast(math.kron(math.astensor([[0, 1], [1, 0]]), math.eye(n_modes)), math.complex128)


def _vacuum_A_matrix(n_modes: int) -> Matrix:
    r"""
    The A matrix of the vacuum state.
    """
    return math.zeros((n_modes, n_modes), math.complex128)


def _vacuum_B_vector(n_modes: int) -> Vector:
    r"""
    The B vector of the vacuum state.
    """
    return math.zeros((n_modes,), math.complex128)


def _reshape(**kwargs) -> Generator:
    r"""
    A utility function to reshape parameters.
    """
    names = list(kwargs.keys())
    vars = list(kwargs.values())

    vars = [math.atleast_1d(var, math.complex128) for var in vars]
    n_modes = max(len(var) for var in vars)

    for i, var in enumerate(vars):
        if len(var) == 1:
            var = math.tile(var, (n_modes,))
        else:
            if len(var) != n_modes:
                msg = f"Parameter {names[i]} has an incompatible shape."
                raise ValueError(msg)
        yield var


def _get_reshape_list_for_channel(num_modes: int):
    r"""
    A utility function to find out the reorder list for the channel.
    """
    new_order = [[i, i + num_modes, i + num_modes * 2, i + num_modes * 3] for i in range(num_modes)]
    return np.array(new_order).reshape(num_modes * 4)


#  ~~~~~~~~~~~
#  Pure States
#  ~~~~~~~~~~~


def vacuum_state_Abc(n_modes: int) -> Union[Matrix, Vector, Scalar]:
    r"""
    The ``(A, b, c)`` triple of a tensor product of vacuum states on ``n_modes``.

    Args:
        n_modes: The number of modes.

    Returns:
        The ``(A, b, c)`` triple of the vacuum states.
    """
    A = _vacuum_A_matrix(n_modes)
    b = _vacuum_B_vector(n_modes)
    c = 1.0 + 0j

    return A, b, c


def coherent_state_Abc(
    x: Union[float, Iterable[float]], y: Union[float, Iterable[float]] = 0
) -> Union[Matrix, Vector, Scalar]:
    r"""
    The ``(A, b, c)`` triple of a tensor product of pure coherent states.

    The number of modes depends on the length of the input parameters.

    If one of the input parameters has length ``1``, it is tiled so that its length matches
    that of the other one. For example, passing ``x=[1,2,3]`` and ``y=1`` is equivalent to passing
    ``x=[1,2,3]`` and ``y=[1,1,1]``.

    Args:
        x: The real parts of the displacements, in units of :math:`\sqrt{\hbar}`.
        y: The imaginary parts of the displacements, in units of :math:`\sqrt{\hbar}`.

    Returns:
        The ``(A, b, c)`` triple of the pure coherent states.
    """
    x, y = list(_reshape(x=x, y=y))
    n_modes = len(x)

    A = _vacuum_A_matrix(n_modes)
    b = x + 1j * y
    c = math.prod(math.exp(-0.5 * (x**2 + y**2)))

    return A, b, c


def squeezed_vacuum_state_Abc(
    r: Union[float, Iterable[float]], phi: Union[float, Iterable[float]] = 0
) -> Union[Matrix, Vector, Scalar]:
    r"""
    The ``(A, b, c)`` triple of a tensor product of squeezed vacuum states.

    The number of modes depends on the length of the input parameters.

    If one of the input parameters has length ``1``, it is tiled so that its length matches
    that of the other one. For example, passing ``r=[1,2,3]`` and ``phi=1`` is equivalent to
    passing ``r=[1,2,3]`` and ``phi=[1,1,1]``.

    Args:
        r: The squeezing magnitudes.
        phi: The squeezing angles.

    Returns:
        The ``(A, b, c)`` triple of the squeezed vacuum states.
    """
    r, phi = list(_reshape(r=r, phi=phi))
    n_modes = len(r)

    A = math.diag(-math.sinh(r) / math.cosh(r) * math.exp(1j * phi))
    b = _vacuum_B_vector(n_modes)
    c = math.prod(1 / math.sqrt(math.cosh(r)))

    return A, b, c


def displaced_squeezed_vacuum_state_Abc(
    x: Union[float, Iterable[float]],
    y: Union[float, Iterable[float]] = 0,
    r: Union[float, Iterable[float]] = 0,
    phi: Union[float, Iterable[float]] = 0,
) -> Union[Matrix, Vector, Scalar]:
    r"""
    The ``(A, b, c)`` triple of a tensor product of displazed squeezed vacuum states.

    The number of modes depends on the length of the input parameters.

    If some of the input parameters have length ``1``, they are tiled so that their length
    matches that of the other ones. For example, passing ``r=[1,2,3]`` and ``phi=1`` is equivalent
    to passing ``r=[1,2,3]`` and ``phi=[1,1,1]``.

    Args:
        r: The squeezing magnitudes.
        phi: The squeezing angles.
        x: The real parts of the displacements, in units of :math:`\sqrt{\hbar}`.
        y: The imaginary parts of the displacements, in units of :math:`\sqrt{\hbar}`.

    Returns:
        The ``(A, b, c)`` triple of the squeezed vacuum states.
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


#  ~~~~~~~~~~~~
#  Mixed States
#  ~~~~~~~~~~~~


def thermal_state_Abc(nbar: Union[int, Iterable[int]]) -> Union[Matrix, Vector, Scalar]:
    r"""
    The ``(A, b, c)`` triple of a tensor product of thermal states.

    The number of modes depends on the length of the input parameters.

    Args:
        nbar: The average numbers of photons per mode.

    Returns:
        The ``(A, b, c)`` triple of the thermal states.
    """
    nbar = math.atleast_1d(nbar, math.complex128)
    n_modes = len(nbar)

    A = math.astensor([[0, 1], [1, 0]], math.complex128)
    A = np.kron((nbar / (nbar + 1)) * math.eye(n_modes, math.complex128), A)
    c = math.prod([1 / (_nbar + 1) for _nbar in nbar])
    b = _vacuum_B_vector(n_modes * 2)

    return A, b, c


# ~~~~~~~~~~~~~~~~~~~~~~~~
#  Unitary transformations
# ~~~~~~~~~~~~~~~~~~~~~~~~


def rotation_gate_Abc(theta: Union[float, Iterable[float]]) -> Union[Matrix, Vector, Scalar]:
    r"""
    The ``(A, b, c)`` triple of of a tensor product of rotation gates.

    The number of modes depends on the length of the input parameters.

    Args:
        theta: The rotation angles.

    Returns:
        The ``(A, b, c)`` triple of the rotation gates.
    """
    theta = math.atleast_1d(theta, math.complex128)
    n_modes = len(theta)

    A = math.astensor([[0, 1], [1, 0]], math.complex128)
    A = np.kron(A, math.exp(1j * theta) * math.eye(n_modes, math.complex128))
    b = _vacuum_B_vector(n_modes * 2)
    c = 1.0 + 0j

    return A, b, c


def displacement_gate_Abc(
    x: Union[float, Iterable[float]], y: Union[float, Iterable[float]] = 0
) -> Union[Matrix, Vector, Scalar]:
    r"""
    The ``(A, b, c)`` triple of a tensor product of displacement gates.

    The number of modes depends on the length of the input parameters.

    If one of the input parameters has length ``1``, it is tiled so that its length matches
    that of the other one. For example, passing ``x=[1,2,3]`` and ``y=1`` is equivalent to
    passing ``x=[1,2,3]`` and ``y=[1,1,1]``.

    Args:
        x: The real parts of the displacements, in units of :math:`\sqrt{\hbar}`.
        y: The imaginary parts of the displacements, in units of :math:`\sqrt{\hbar}`.

    Returns:
        The ``(A, b, c)`` triple of the displacement gates.
    """
    x, y = _reshape(x=x, y=y)
    n_modes = len(x)

    A = _X_matrix_for_unitary(n_modes)
    b = math.concat([x + 1j * y, -x + 1j * y], axis=0)
    c = math.exp(-math.sum(x**2 + y**2) / 2)

    return A, b, c


def squeezing_gate_Abc(
    r: Union[float, Iterable[float]], delta: Union[float, Iterable[float]] = 0
) -> Union[Matrix, Vector, Scalar]:
    r"""
    The ``(A, b, c)`` triple of a tensor product of squeezing gates.

    The number of modes depends on the length of the input parameters.

    If one of the input parameters has length ``1``, it is tiled so that its length matches
    that of the other one. For example, passing ``r=[1,2,3]`` and ``delta=1`` is equivalent to
    passing ``r=[1,2,3]`` and ``delta=[1,1,1]``.

    Args:
        r: The squeezing magnitudes.
        delta: The squeezing angles.

    Returns:
        The ``(A, b, c)`` triple of the squeezing gates.
    """
    r, delta = _reshape(r=r, delta=delta)
    n_modes = len(delta)

    tanhr = math.diag(math.sinh(r) / math.cosh(r))
    sechr = math.diag(1 / math.cosh(r))

    A = math.block([[-math.exp(1j * delta) * tanhr, sechr], [sechr, math.exp(-1j * delta) * tanhr]])
    b = _vacuum_B_vector(n_modes * 2)
    c = math.prod(1 / math.sqrt(math.cosh(r)))

    return A, b, c


def beamsplitter_gate_Abc(
    theta: Union[float, Iterable[float]], phi: Union[float, Iterable[float]] = 0
) -> Union[Matrix, Vector, Scalar]:
    r"""
    The ``(A, b, c)`` triple of a tensor product of two-mode beamsplitter gates.

    The number of modes depends on the length of the input parameters.

    If one of the input parameters has length ``1``, it is tiled so that its length matches
    that of the other one. For example, passing ``theta=[1,2,3]`` and ``phi=1`` is equivalent to
    passing ``theta=[1,2,3]`` and ``phi=[1,1,1]``.

    Args:
        theta: The transmissivity parameters.
        phi: The phase parameters.

    Returns:
        The ``(A, b, c)`` triple of the beamsplitter gates.
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
    c = 1.0 + 0j

    return A, b, c


# ~~~~~~~~~~
#  Channels
# ~~~~~~~~~~


def attenuator_Abc(eta: Union[float, Iterable[float]]) -> Union[Matrix, Vector, Scalar]:
    r"""
    The ``(A, b, c)`` triple of of a tensor product of atternuators.

    The number of modes depends on the length of the input parameters.

    Args:
        eta: The values of the transmissivities.

    Returns:
        The ``(A, b, c)`` triple of the attenuator channels.

    Raises:
        ValueError: If ``eta`` is larger than `1` or smaller than `0`.
    """
    eta = math.atleast_1d(eta, math.complex128)
    n_modes = len(eta)

    for e in eta:
        if math.real(e) > 1 or math.real(e) < 0:
            msg = "Transmissivity must be a float in the interval ``[0, 1]``"
            raise ValueError(msg)

    O_n = math.zeros((n_modes, n_modes), math.complex128)
    eta1 = math.diag(math.sqrt(eta)).reshape((n_modes, n_modes)).reshape((n_modes, n_modes))
    eta2 = math.eye(n_modes) - math.diag(eta).reshape((n_modes, n_modes))

    A = math.block(
        [
            [O_n, eta1, O_n, O_n],
            [eta1, O_n, O_n, eta2],
            [O_n, O_n, O_n, eta1],
            [O_n, eta2, eta1, O_n],
        ]
    )
    reshape_list = _get_reshape_list_for_channel(n_modes)

    A = A[reshape_list, :][:, reshape_list]
    b = _vacuum_B_vector(n_modes * 4)
    c = 1.0 + 0j

    return A, b, c


def amplifier_Abc(g: Union[float, Iterable[float]]) -> Union[Matrix, Vector, Scalar]:
    r"""
    The ``(A, b, c)`` triple of a tensor product of amplifiers.

    The number of modes depends on the length of the input parameters.

    Args:
        g: The values of the gains.

    Returns:
        The ``(A, b, c)`` triple of the amplifier channels.

    Raises:
        ValueError: If ``g`` is smaller than `1`.
    """
    g = math.atleast_1d(g, math.complex128)
    n_modes = len(g)

    for g_val in g:
        if math.real(g_val) < 1:
            msg = "Found amplifier with gain ``g`` smaller than `1`."
            raise ValueError(msg)

    g = math.atleast_1d(g, math.complex128)
    n_modes = len(g)
    reshape_list = _get_reshape_list_for_channel(n_modes)

    O_n = math.zeros((n_modes, n_modes), math.complex128)
    g1 = math.diag(math.astensor([1 / math.sqrt(g)])).reshape((n_modes, n_modes))
    g2 = math.diag(math.astensor([1 - 1 / g])).reshape((n_modes, n_modes))

    A = math.block(
        [
            [O_n, g1, g2, O_n],
            [g1, O_n, O_n, O_n],
            [g2, O_n, O_n, g1],
            [O_n, O_n, g1, O_n],
        ]
    )
    A = A[reshape_list, :][:, reshape_list]
    b = _vacuum_B_vector(n_modes * 4)
    c = np.prod(1 / g)

    return A, b, c


def fock_damping_Abc(n_modes: int) -> Union[Matrix, Vector, Scalar]:
    r"""
    The ``(A, b, c)`` triple of a tensor product of Fock dampers.

    Args:
         n_modes: The number of modes.

    Returns:
        The ``(A, b, c)`` triple of the Fock damping channels.
    """
    A = _X_matrix_for_unitary(n_modes * 2)
    b = _vacuum_B_vector(n_modes * 4)
    c = 1.0 + 0j

    return A, b, c
