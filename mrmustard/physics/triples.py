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

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mrmustard import math, settings
from mrmustard.physics.gaussian_integrals import complex_gaussian_integral_2
from mrmustard.utils.typing import ComplexMatrix, ComplexTensor, ComplexVector, RealMatrix

from .bargmann_utils import symplectic2Au

#  ~~~~~~~~~
#  Utilities
#  ~~~~~~~~~


def _X_matrix_for_unitary(n_modes: int) -> ComplexMatrix:
    r"""
    The X matrix for the order of unitaries.
    """
    return math.cast(math.kron(math.astensor([[0, 1], [1, 0]]), math.eye(n_modes)), math.complex128)


def _vacuum_A_matrix(n_modes: int) -> ComplexMatrix:
    r"""
    The A matrix of the vacuum state.
    """
    return math.zeros((n_modes, n_modes), dtype=math.complex128)


def _vacuum_B_vector(n_modes: int) -> ComplexVector:
    r"""
    The B vector of the vacuum state.
    """
    return math.zeros((n_modes,), dtype=math.complex128)


#  ~~~~~~~~~~~
#  Pure States
#  ~~~~~~~~~~~


def vacuum_state_Abc(n_modes: int) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
    r"""
    The ``(A, b, c)`` triple of a tensor product of vacuum states on ``n_modes``.

    Args:
        n_modes: The number of modes.

    Returns:
        The ``(A, b, c)`` triple of the vacuum states.
    """

    A = _vacuum_A_matrix(n_modes)
    b = _vacuum_B_vector(n_modes)
    c = math.astensor(1.0 + 0.0j)

    return A, b, c


def bargmann_eigenstate_Abc(
    alpha: complex | Sequence[complex],
) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
    r"""
    The Abc triple of a Bargmann eigenstate.

    Args:
        alpha: The eigenvalue of the Bargmann eigenstate.

    Returns:
        The ``(A, b, c)`` triple of the Bargmann eigenstate.
    """
    alpha = math.astensor(alpha, dtype=math.complex128)
    batch_shape = alpha.shape

    A = math.broadcast_to(_vacuum_A_matrix(1), (*batch_shape, 1, 1))
    b = math.reshape(alpha, (*batch_shape, 1))
    c = math.ones(batch_shape, math.complex128)

    return A, b, c


def coherent_state_Abc(
    x: float | Sequence[float],
    y: float | Sequence[float] = 0,
) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
    r"""
    The ``(A, b, c)`` triple of a pure coherent state.

    Args:
        x: The real part of the displacement, in units of :math:`\sqrt{\hbar}`.
        y: The imaginary part of the displacement, in units of :math:`\sqrt{\hbar}`.

    Returns:
        The ``(A, b, c)`` triple of the pure coherent state.
    """
    x, y = math.broadcast_arrays(
        math.astensor(x, dtype=math.complex128),
        math.astensor(y, dtype=math.complex128),
    )
    batch_shape = x.shape

    A = math.broadcast_to(_vacuum_A_matrix(1), (*batch_shape, 1, 1))
    b = math.reshape(x + 1j * y, (*batch_shape, 1))
    c = math.cast(math.exp(-0.5 * (x**2 + y**2)), math.complex128)

    return A, b, c


def squeezed_vacuum_state_Abc(
    r: float | Sequence[float],
    phi: float | Sequence[float] = 0,
) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
    r"""
    The ``(A, b, c)`` triple of a squeezed vacuum state.

    Args:
        r: The squeezing magnitudes.
        phi: The squeezing angles.

    Returns:
        The ``(A, b, c)`` triple of a squeezed vacuum state.
    """
    r, phi = math.broadcast_arrays(
        math.astensor(r, dtype=math.complex128),
        math.astensor(phi, dtype=math.complex128),
    )
    batch_shape = r.shape

    A = math.reshape(-math.sinh(r) / math.cosh(r) * math.exp(1j * phi), (*batch_shape, 1, 1))
    b = math.broadcast_to(_vacuum_B_vector(1), (*batch_shape, 1))
    c = 1 / math.sqrt(math.cosh(r))

    return A, b, c


def displaced_squeezed_vacuum_state_Abc(
    x: float | Sequence[float],
    y: float | Sequence[float] = 0,
    r: float | Sequence[float] = 0,
    phi: float | Sequence[float] = 0,
) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
    r"""
    The ``(A, b, c)`` triple of a displaced squeezed vacuum state.

    Args:
        r: The squeezing magnitudes.
        phi: The squeezing angles.
        x: The real parts of the displacements, in units of :math:`\sqrt{\hbar}`.
        y: The imaginary parts of the displacements, in units of :math:`\sqrt{\hbar}`.

    Returns:
        The ``(A, b, c)`` triple of the squeezed vacuum state.
    """
    x, y, r, phi = math.broadcast_arrays(
        math.astensor(x, dtype=math.complex128),
        math.astensor(y, dtype=math.complex128),
        math.astensor(r, dtype=math.complex128),
        math.astensor(phi, dtype=math.complex128),
    )
    batch_shape = x.shape

    A = math.reshape(-math.sinh(r) / math.cosh(r) * math.exp(1j * phi), (*batch_shape, 1, 1))
    b = math.reshape(
        (x + 1j * y) + (x - 1j * y) * math.sinh(r) / math.cosh(r) * math.exp(1j * phi),
        (*batch_shape, 1),
    )
    c = math.exp(
        -0.5 * (x**2 + y**2)
        - 0.5 * (x - 1j * y) ** 2 * math.sinh(r) / math.cosh(r) * math.exp(1j * phi),
    ) / math.sqrt(math.cosh(r))

    return A, b, c


def two_mode_squeezed_vacuum_state_Abc(
    r: float | Sequence[float],
    phi: float | Sequence[float] = 0,
) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
    r"""
    The ``(A, b, c)`` triple of a two mode squeezed vacuum state.

    Args:
        r: The squeezing magnitudes.
        phi: The squeezing angles.

    Returns:
        The ``(A, b, c)`` triple of the squeezed vacuum state.
    """
    r, phi = math.broadcast_arrays(math.astensor(r), math.astensor(phi))
    batch_shape = r.shape
    batch_dim = len(batch_shape)

    O_matrix = math.zeros(batch_shape, math.complex128)
    tanhr = math.exp(1j * phi) * math.sinh(r) / math.cosh(r)

    A = math.stack(
        [math.stack([O_matrix, tanhr], batch_dim), math.stack([tanhr, O_matrix], batch_dim)],
        batch_dim,
    )
    b = math.broadcast_to(_vacuum_B_vector(2), (*batch_shape, 2))
    c = math.cast(1 / math.cosh(r), math.complex128)

    return A, b, c


def gket_state_Abc(symplectic: RealMatrix):
    r"""
    The A,b,c parameters of a Gaussian Ket (Gket) state. This is simply a Gaussian acted on the vacuum.

    Args:
        symplectic: the symplectic representation of the Gaussian

    Returns:
        The ``(A,b,c)`` triple of the Gket state.
    """
    batch_shape = symplectic.shape[:-2]
    m = symplectic.shape[-1] // 2  # num of modes

    Au = symplectic2Au(symplectic)

    A = Au[..., :m, :m]
    b = math.zeros((*batch_shape, m), dtype=A.dtype)
    c = (
        (-1) ** m
        * math.det(Au[..., m:, m:] @ math.conj(Au[..., m:, m:]) - math.eye_like(Au[..., m:, m:]))
    ) ** 0.25

    return A, b, c


def gdm_state_Abc(betas: ComplexVector, symplectic: RealMatrix):
    r"""
    The A,b,c parameters of a Gaussian mixed state that is defined by the action of a Guassian on a thermal state

    Args:
        betas: the list of betas corresponding to the temperatures of the initial thermal state
        symplectic: the symplectic matrix of the Gaussian

    Returns:
        The ``(A,b,c)`` triple of the resulting Gaussian DM state.
    """
    batch_shape = symplectic.shape[:-2]
    m = len(betas)
    betas = math.broadcast_to(betas, (*batch_shape, m), dtype=math.complex128)
    Au = symplectic2Au(symplectic)
    A_udagger_u = math.block(
        [
            [math.conj(Au), math.zeros((2 * m, 2 * m), dtype="complex128")],
            [math.zeros((2 * m, 2 * m), dtype="complex128"), Au],
        ],
    )

    D = math.diag(math.exp(-betas))
    A_fd = math.block(
        [
            [math.zeros((m, m), dtype=math.complex128), D],
            [D, math.zeros((m, m), dtype=math.complex128)],
        ],
    )
    c_fd = math.prod(1 - math.exp(-betas))
    t_fd = (A_fd, math.zeros((*batch_shape, 2 * m), dtype=A_fd.dtype), c_fd)
    c_u = (
        (-1) ** m
        * math.det(Au[..., m:, m:] @ math.conj(Au[..., m:, m:]) - math.eye_like(Au[..., m:, m:]))
    ) ** (0.5)
    t_u = (A_udagger_u, math.zeros((*batch_shape, 4 * m)), c_u)
    return complex_gaussian_integral_2(
        t_fd,
        t_u,
        list(range(2 * m)),
        list(range(m, 2 * m)) + list(range(3 * m, 4 * m)),
    )


def sauron_state_Abc(n: int, epsilon: float) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
    r"""
    The A,b,c parametrization of Sauron states. These are Fock states written as a linear superposition of a
    ring of coherent states.

    Args:
        n: The number of photons.
        epsilon: The size of the ring. The approximation is exact in the limit for epsilon that goes to zero.

    Returns:
        The ``(A, b, c)`` triple of the sauron state.
    """

    phases = np.linspace(0, 2 * np.pi * (1 - 1 / (n + 1)), n + 1)
    cs = np.exp(1j * phases)
    bs = (epsilon * cs)[..., None]
    As = np.zeros([n + 1, 1, 1], dtype="complex128")

    # normalization
    probs = complex_gaussian_integral_2(
        (np.conj(As), np.conj(bs), np.conj(cs)),
        (As, bs, cs),
        [0],
        [0],
    )[2]
    prob = np.sum(probs)
    cs /= np.sqrt(prob)

    return As, bs, cs


def quadrature_eigenstates_Abc(
    x: float | Sequence[float],
    phi: float | Sequence[float],
) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
    r"""
    The ``(A, b, c)`` triple of a quadrature eigenstate.

    Args:
        r: The squeezing magnitudes.
        phi: The squeezing angles.

    Returns:
        The ``(A, b, c)`` triple of the squeezed vacuum state.
    """
    x, phi = math.broadcast_arrays(
        math.astensor(x, dtype=math.complex128),
        math.astensor(phi, dtype=math.complex128),
    )
    batch_shape = x.shape

    hbar = settings.HBAR
    A = math.reshape(-math.exp(1j * 2 * phi), (*batch_shape, 1, 1))
    b = math.reshape(x * math.exp(1j * phi) * math.sqrt(2 / hbar), (*batch_shape, 1))
    c = math.cast(1 / (np.pi) ** (1 / 4) * math.exp(-(x**2) / (2 * hbar)), math.complex128)

    return A, b, c


#  ~~~~~~~~~~~~
#  Mixed States
#  ~~~~~~~~~~~~


def thermal_state_Abc(
    nbar: int | Sequence[int],
) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
    r"""
    The ``(A, b, c)`` triple of a thermal state.

    Args:
        nbar: The average number of photons.

    Returns:
        The ``(A, b, c)`` triple of the thermal state.
    """
    nbar = math.astensor(nbar, dtype=math.complex128)
    batch_shape = nbar.shape
    batch_dim = len(batch_shape)

    O_matrix = math.zeros(batch_shape, math.complex128)

    A = math.stack(
        [
            math.stack([O_matrix, (nbar / (nbar + 1))], batch_dim),
            math.stack([(nbar / (nbar + 1)), O_matrix], batch_dim),
        ],
        batch_dim,
    )
    b = math.broadcast_to(_vacuum_B_vector(2), (*batch_shape, 2))
    c = math.cast(1 / (nbar + 1), math.complex128)

    return A, b, c


# ~~~~~~~~~~~~~~~~~~~~~~~~
#  Unitary transformations
# ~~~~~~~~~~~~~~~~~~~~~~~~


def rotation_gate_Abc(
    theta: float | Sequence[float],
) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
    r"""
    The ``(A, b, c)`` triple of of a tensor product of a rotation gate.

    Args:
        theta: The rotation angles.

    Returns:
        The ``(A, b, c)`` triple of the rotation gate.
    """
    theta = math.astensor(theta, dtype=math.complex128)
    batch_shape = theta.shape
    batch_dim = len(batch_shape)

    O_matrix = math.zeros(batch_shape, math.complex128)

    A = math.stack(
        [
            math.stack([O_matrix, math.exp(1j * theta)], batch_dim),
            math.stack([math.exp(1j * theta), O_matrix], batch_dim),
        ],
        batch_dim,
    )
    b = math.broadcast_to(_vacuum_B_vector(2), (*batch_shape, 2))
    c = math.ones(batch_shape, math.complex128)

    return A, b, c


def displacement_gate_Abc(
    x: float | Sequence[float],
    y: float | Sequence[float] = 0,
) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
    r"""
    The ``(A, b, c)`` triple of a tensor product of a displacement gate.

    Args:
        x: The real part of the displacement, in units of :math:`\sqrt{\hbar}`.
        y: The imaginary part of the displacement, in units of :math:`\sqrt{\hbar}`.

    Returns:
        The ``(A, b, c)`` triple of the displacement gate.
    """
    x, y = math.broadcast_arrays(
        math.astensor(x, dtype=math.complex128),
        math.astensor(y, dtype=math.complex128),
    )
    batch_shape = x.shape
    batch_dim = len(batch_shape)

    A = math.broadcast_to(_X_matrix_for_unitary(1), (*batch_shape, 2, 2))
    b = math.stack([x + 1j * y, -x + 1j * y], batch_dim)
    c = math.cast(math.exp(-(x**2 + y**2) / 2), math.complex128)

    return A, b, c


def squeezing_gate_Abc(
    r: float | Sequence[float],
    phi: float | Sequence[float] = 0,
) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
    r"""
    The ``(A, b, c)`` triple of a squeezing gate.

    Args:
        r: The squeezing magnitudes.
        phi: The squeezing angles.

    Returns:
        The ``(A, b, c)`` triple of the squeezing gate.
    """
    r, phi = math.broadcast_arrays(
        math.astensor(r, dtype=math.complex128),
        math.astensor(phi, dtype=math.complex128),
    )
    batch_shape = r.shape
    batch_dim = len(batch_shape)

    tanhr = math.sinh(r) / math.cosh(r)
    sechr = 1 / math.cosh(r)

    A = math.stack(
        [
            math.stack([-math.exp(1j * phi) * tanhr, sechr], batch_dim),
            math.stack([sechr, math.exp(-1j * phi) * tanhr], batch_dim),
        ],
        batch_dim,
    )
    b = math.broadcast_to(_vacuum_B_vector(2), (*batch_shape, 2))
    c = math.cast(1 / math.sqrt(math.cosh(r)), math.complex128)

    return A, b, c


def beamsplitter_gate_Abc(
    theta: float | Sequence[float],
    phi: float | Sequence[float] = 0,
) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
    r"""
    The ``(A, b, c)`` triple of a tensor product of a two-mode beamsplitter gate.

    Args:
        theta: The transmissivity parameters.
        phi: The phase parameters.

    Returns:
        The ``(A, b, c)`` triple of the beamsplitter gate.
    """
    theta, phi = math.broadcast_arrays(
        math.astensor(theta, dtype=math.complex128),
        math.astensor(phi, dtype=math.complex128),
    )
    batch_shape = theta.shape
    batch_dim = len(batch_shape)

    O_matrix = math.zeros((*batch_shape, 2, 2), math.complex128)
    costheta = math.cos(theta)
    sintheta = math.sin(theta)

    V = math.stack(
        [
            math.stack([costheta, -math.exp(math.astensor(-1j) * phi) * sintheta], batch_dim),
            math.stack([math.exp(math.astensor(1j) * phi) * sintheta, costheta], batch_dim),
        ],
        batch_dim,
    )
    A = math.block([[O_matrix, V], [math.einsum("...ij->...ji", V), O_matrix]])
    b = math.broadcast_to(_vacuum_B_vector(4), (*batch_shape, 4))
    c = math.ones(batch_shape, math.complex128)

    return A, b, c


def twomode_squeezing_gate_Abc(
    r: float | Sequence[float],
    phi: float | Sequence[float] = 0,
) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
    r"""
    The ``(A, b, c)`` triple of a tensor product of a two-mode squeezing gate.

    Args:
        r: The squeezing magnitudes.
        phi: The squeezing phase.

    Returns:
        The ``(A, b, c)`` triple of the two mode squeezing gate.
    """
    r, phi = math.broadcast_arrays(
        math.astensor(r, dtype=math.complex128),
        math.astensor(phi, dtype=math.complex128),
    )
    batch_shape = r.shape
    batch_dim = len(batch_shape)

    O_matrix = math.zeros(batch_shape, math.complex128)
    tanhr = math.exp(1j * phi) * math.sinh(r) / math.cosh(r)
    sechr = 1 / math.cosh(r)

    A_block1 = math.stack(
        [math.stack([O_matrix, tanhr], batch_dim), math.stack([tanhr, O_matrix], batch_dim)],
        batch_dim,
    )
    A_block2 = math.stack(
        [
            math.stack([O_matrix, -math.conj(tanhr)], batch_dim),
            math.stack([-math.conj(tanhr), O_matrix], batch_dim),
        ],
        batch_dim,
    )
    A_block3 = math.stack(
        [math.stack([sechr, O_matrix], batch_dim), math.stack([O_matrix, sechr], batch_dim)],
        batch_dim,
    )
    A = math.block([[A_block1, A_block3], [A_block3, A_block2]])
    b = math.broadcast_to(_vacuum_B_vector(4), (*batch_shape, 4))
    c = 1 / math.cosh(r)

    return A, b, c


def identity_Abc(n_modes: int) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
    r"""
    The ``(A, b, c)`` triple of a tensor product of identity gates.

    Args:
        n_modes: The number of modes.

    Returns:
        The ``(A, b, c)`` triple of the identities.
    """
    O_n = math.zeros((n_modes, n_modes), dtype=math.complex128)
    I_n = math.eye(n_modes, dtype=math.complex128)

    A = math.block([[O_n, I_n], [I_n, O_n]])
    b = _vacuum_B_vector(n_modes * 2)
    c = 1.0 + 0j

    return A, b, c


# ~~~~~~~~~~
#  Channels
# ~~~~~~~~~~


def attenuator_Abc(
    eta: float | Sequence[float],
) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
    r"""
    The ``(A, b, c)`` triple of an attenuator.

    Args:
        eta: The values of the transmissivities.

    Returns:
        The ``(A, b, c)`` triple of the attenuator channel.

    Raises:
        ValueError: If ``eta`` is larger than `1` or smaller than `0`.
    """
    eta = math.astensor(eta, dtype=math.complex128)
    batch_shape = eta.shape
    batch_dim = len(batch_shape)

    math.error_if(eta, math.any(math.real(eta) > 1), "Found transmissivity greater than `1`.")
    math.error_if(eta, math.any(math.real(eta) < 0), "Found transmissivity less than `0`.")

    O_matrix = math.zeros(batch_shape, math.complex128)
    eta1 = math.sqrt(eta)
    eta2 = 1 - eta

    A = math.stack(
        [
            math.stack([O_matrix, eta1, O_matrix, O_matrix], batch_dim),
            math.stack([eta1, O_matrix, O_matrix, eta2], batch_dim),
            math.stack([O_matrix, O_matrix, O_matrix, eta1], batch_dim),
            math.stack([O_matrix, eta2, eta1, O_matrix], batch_dim),
        ],
        batch_dim,
    )
    b = math.broadcast_to(_vacuum_B_vector(4), (*batch_shape, 4))
    c = math.ones(batch_shape, math.complex128)

    return A, b, c


def amplifier_Abc(g: float | Sequence[float]) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
    r"""
    The ``(A, b, c)`` triple of an amplifier.

    Args:
        g: The values of the gains.

    Returns:
        The ``(A, b, c)`` triple of the amplifier channel.

    Raises:
        ValueError: If ``g`` is smaller than `1`.
    """
    g = math.astensor(g, dtype=math.complex128)
    batch_shape = g.shape
    batch_dim = len(batch_shape)

    math.error_if(
        g,
        math.any(math.real(g) < 1),
        "Found amplifier with gain ``g`` smaller than `1`.",
    )

    O_matrix = math.zeros(batch_shape, math.complex128)
    g1 = 1 / math.sqrt(g)
    g2 = 1 - 1 / g
    A = math.stack(
        [
            math.stack([O_matrix, g1, g2, O_matrix], batch_dim),
            math.stack([g1, O_matrix, O_matrix, O_matrix], batch_dim),
            math.stack([g2, O_matrix, O_matrix, g1], batch_dim),
            math.stack([O_matrix, O_matrix, g1, O_matrix], batch_dim),
        ],
        batch_dim,
    )
    b = math.broadcast_to(_vacuum_B_vector(4), (*batch_shape, 4))
    c = 1 / g

    return A, b, c


def fock_damping_Abc(
    beta: float | Sequence[float],
) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
    r"""
    The ``(A, b, c)`` triple of a Fock damper.

    Args:
        beta: The damping parameter.

    Returns:
        The ``(A, b, c)`` triple of the Fock damping operator.
    """
    beta = math.astensor(beta, dtype=math.complex128)
    batch_shape = beta.shape
    batch_dim = len(batch_shape)

    O_matrix = math.zeros(batch_shape, math.complex128)
    B_n = math.exp(-beta)

    A = math.stack(
        [math.stack([O_matrix, B_n], batch_dim), math.stack([B_n, O_matrix], batch_dim)],
        batch_dim,
    )
    b = math.broadcast_to(_vacuum_B_vector(2), (*batch_shape, 2))
    c = math.ones(batch_shape, math.complex128)

    return A, b, c


def gaussian_random_noise_Abc(Y: RealMatrix) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
    r"""
    The triple (A, b, c) for the gaussian random noise channel.

    Args:
        Y: the Y matrix of the Gaussian random noise channel.

    Returns:
        The ``(A, b, c)`` triple of the Gaussian random noise channel.
    """
    batch_shape = Y.shape[:-2]

    m = Y.shape[-1] // 2
    xi = math.eye(2 * m, dtype=math.complex128) + Y / settings.HBAR
    xi_inv = math.inv(xi)
    xi_inv_in_blocks = math.block(
        [[math.eye(2 * m) - xi_inv, xi_inv], [xi_inv, math.eye(2 * m) - xi_inv]],
    )
    R = (
        1
        / math.sqrt(complex(2))
        * math.block(
            [
                [
                    math.eye(m, dtype=math.complex128),
                    1j * math.eye(m, dtype=math.complex128),
                    math.zeros((m, 2 * m), dtype=math.complex128),
                ],
                [
                    math.zeros((m, 2 * m), dtype=math.complex128),
                    math.eye(m, dtype=math.complex128),
                    -1j * math.eye(m, dtype=math.complex128),
                ],
                [
                    math.eye(m, dtype=math.complex128),
                    -1j * math.eye(m, dtype=math.complex128),
                    math.zeros((m, 2 * m), dtype=math.complex128),
                ],
                [
                    math.zeros((m, 2 * m), dtype=math.complex128),
                    math.eye(m, dtype=math.complex128),
                    1j * math.eye(m, dtype=math.complex128),
                ],
            ],
        )
    )

    A = math.Xmat(2 * m) @ R @ xi_inv_in_blocks @ math.conj(R).T
    b = math.zeros((*batch_shape, 4 * m))
    c = 1 / math.sqrt(math.det(xi))

    return A, b, c


def bargmann_to_quadrature_Abc(
    n_modes: int,
    phi: float | Sequence[float],
) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
    r"""
    The ``(A, b, c)`` triple of the multi-mode kernel :math:`\langle \vec{p}|\vec{z} \rangle` between bargmann representation with ABC Ansatz form and quadrature representation with ABC Ansatz.
    The kernel can be considered as a Unitary-like component: the out_ket wires are related to the real variable :math:`\vec{p}` in quadrature representation and the in_ket wires are related to the complex variable :math:`\vec{z}`.

    If one wants to transform from quadrature representation to Bargmann representation, the kernel will be the `dual` of this component, but be careful that the inner product will then have to use the real integral.

    Args:
         n_modes: The number of modes.
         phi: The quadrature angle. 0 corresponds to the `x` quadrature, and :math:`\pi/2` to the `p` quadrature.

    Returns:
        The ``(A, b, c)`` triple of the map from bargmann representation with ABC Ansatz to quadrature representation with ABC Ansatz.
    """
    phi = math.astensor(phi, dtype=math.complex128)
    batch_shape = phi.shape
    batch_dim = len(batch_shape)

    hbar = settings.HBAR
    Id = math.eye(n_modes, dtype=math.complex128)
    e = math.exp(-1j * phi + 1j * np.pi / 2)
    A = math.kron(
        math.stack(
            [
                math.stack(
                    [
                        math.broadcast_to(-1 / hbar, batch_shape, dtype=math.complex128),
                        -1j * e * np.sqrt(2 / hbar),
                    ],
                    batch_dim,
                ),
                math.stack([-1j * e * np.sqrt(2 / hbar), e * e], batch_dim),
            ],
            batch_dim,
        ),
        Id,
    )
    b = math.broadcast_to(_vacuum_B_vector(2 * n_modes), (*batch_shape, 2 * n_modes))
    c = math.ones(batch_shape, math.complex128) / (np.pi * hbar) ** (0.25 * n_modes)

    return A, b, c


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Maps between representations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def displacement_map_s_parametrized_Abc(
    s: float | Sequence[float],
    n_modes: int,
) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
    r"""
    The ``(A, b, c)`` triple of a multi-mode ``s``\-parametrized displacement map.

    :math:
        D_s(\vec{\gamma}^*, \vec{\gamma}) = e^{\frac{s}{2}|\vec{\gamma}|^2} D(\vec{\gamma}^*, \vec{\gamma}) = e^{\frac{s}{2}|\vec{\gamma}|^2} e^{\frac{1}{2}|\vec{z}|^2} e^{\vec{z}^*\vec{\gamma} - \vec{z} \vec{\gamma}^*}.

    The indices of the final triple correspond to the variables :math:`(\gamma_1^*, \gamma_2^*, ..., z_1, z_2, ..., \gamma_1, \gamma_2, ..., z_1^*, z_2^*, ...)` of the Bargmann function of the s-parametrized displacement map, and correspond to ``out_bra, in_bra, out_ket, in_ket`` wires.

    Args:
        s: The phase space parameter
        n_modes: the number of modes for this map.

    Returns:
        The ``(A, b, c)`` triple of the multi-mode ``s``-parametrized dispalcement map :math:`D_s(\gamma)`.
    """
    s = math.astensor(s, dtype=math.complex128)
    batch_shape = s.shape

    Zmat = math.broadcast_to(
        -math.Zmat(num_modes=n_modes),
        (*batch_shape, 2 * n_modes, 2 * n_modes),
        dtype=math.complex128,
    )
    Xmat = math.broadcast_to(
        math.Xmat(num_modes=n_modes),
        (*batch_shape, 2 * n_modes, 2 * n_modes),
        dtype=math.complex128,
    )
    A = math.block(
        [[(s[..., None, None] - 1) / 2 * math.Xmat(num_modes=n_modes), Zmat], [Zmat, Xmat]],
    )
    order_list = math.arange(4 * n_modes)  # [0,3,1,2]
    order_list = list(
        math.cast(
            math.concat(
                (
                    math.concat((order_list[:n_modes], order_list[3 * n_modes :]), axis=0),
                    order_list[n_modes : 3 * n_modes],
                ),
                axis=0,
            ),
            math.int32,
        ),
    )

    A = A[..., order_list, :][..., :, order_list]
    b = math.broadcast_to(_vacuum_B_vector(4 * n_modes), (*batch_shape, 4 * n_modes))
    c = math.ones(batch_shape, math.complex128)

    return A, b, c


def bargmann_to_wigner_Abc(
    s: float,
    n_modes: int,
) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
    r"""
    The Abc triple of the Bargmann to Wigner/Husimi transformation.
    """

    On = math.zeros((n_modes, n_modes), dtype=math.complex128)
    In = math.eye(n_modes, dtype=math.complex128)

    A = (
        2
        / (s - 1)
        * math.block(
            [
                [On, -In, In, On],
                [-In, On, On, (s + 1) / 2 * In],
                [In, On, On, -In],
                [On, (s + 1) / 2 * In, -In, On],
            ],
        )
    )
    b = math.zeros(4 * n_modes, dtype=math.complex128)
    c = (2 / (math.abs(s - 1) * np.pi)) ** (n_modes)
    return A, b, c


# ~~~~~~~~~~~~~~~~
# Kraus operators
# ~~~~~~~~~~~~~~~~


def attenuator_kraus_Abc(
    eta: float | Sequence[float],
) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
    r"""
    The entire family of Kraus operators of the attenuator (loss) channel as a single ``(A, b, c)`` triple.
    The last index is the "bond" index which should be summed/integrated over.

    Args:
        eta: The value of the transmissivity.

    Returns:
        The ``(A, b, c)`` triple of the kraus operators of the attenuator (loss) channel.
    """
    eta = math.astensor(eta, dtype=math.complex128)
    batch_shape = eta.shape
    batch_dim = len(batch_shape)

    costheta = math.sqrt(eta)
    sintheta = math.sqrt(1 - eta)

    O_matrix = math.zeros(batch_shape, math.complex128)

    A = math.stack(
        [
            math.stack([O_matrix, costheta, O_matrix], batch_dim),
            math.stack([costheta, O_matrix, -sintheta], batch_dim),
            math.stack([O_matrix, -sintheta, O_matrix], batch_dim),
        ],
        batch_dim,
    )
    b = math.broadcast_to(_vacuum_B_vector(3), (*batch_shape, 3))
    c = math.ones(batch_shape, math.complex128)

    return A, b, c


def XY_to_channel_Abc(
    X: RealMatrix,
    Y: RealMatrix,
    d: ComplexVector | None = None,
) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
    r"""
    The method to compute the A matrix of a channel based on its X, Y, and d.
    Args:
        X: The X matrix of the channel
        Y: The Y matrix of the channel
        d: The d (displacement) vector of the channel -- if None, we consider it as 0
    """
    m = Y.shape[-1] // 2
    # considering no displacement if d is None
    d = d if d else math.zeros(2 * m)

    if X.shape != Y.shape:
        raise ValueError(
            "The dimension of X and Y matrices are not the same."
            f"X.shape = {X.shape}, Y.shape = {Y.shape}",
        )
    batch_shape = X.shape[:-2]
    Im = math.broadcast_to(math.eye(2 * m, dtype=math.complex128), (*batch_shape, 2 * m, 2 * m))
    im = math.broadcast_to(math.eye(m, dtype=math.complex128), (*batch_shape, m, m))
    Xm = math.broadcast_to(math.Xmat(2 * m), (*batch_shape, 4 * m, 4 * m))
    X_transpose = math.einsum("...ij->...ji", X)
    xi = 1 / 2 * Im + 1 / 2 * X @ X_transpose + Y / settings.HBAR
    xi_inv = math.inv(xi)
    xi_inv_in_blocks = math.block(
        [[Im - xi_inv, xi_inv @ X], [X_transpose @ xi_inv, Im - X_transpose @ xi_inv @ X]],
    )
    R = (
        1
        / math.sqrt(complex(2))
        * math.block(
            [
                [
                    im,
                    1j * im,
                    math.zeros((*batch_shape, m, 2 * m), dtype=math.complex128),
                ],
                [
                    math.zeros((*batch_shape, m, 2 * m), dtype=math.complex128),
                    im,
                    -1j * im,
                ],
                [
                    im,
                    -1j * im,
                    math.zeros((*batch_shape, m, 2 * m), dtype=math.complex128),
                ],
                [
                    math.zeros((*batch_shape, m, 2 * m), dtype=math.complex128),
                    im,
                    1j * im,
                ],
            ],
        )
    )
    R_transpose = math.einsum("...ij->...ji", R)
    A = Xm @ R @ xi_inv_in_blocks @ math.conj(R_transpose)
    xi_inv_d = math.einsum("...ij,...j->...i", xi_inv, d)
    x_xi_inv_d = math.einsum("...ij,...jk,...k->...i", X_transpose, xi_inv, d)
    temp = math.concat([xi_inv_d, x_xi_inv_d], -1)
    b = 1 / math.sqrt(complex(settings.HBAR)) * math.einsum("...ij,...j->...i", math.conj(R), temp)
    sandwiched_xi_inv = math.einsum("...i,...ij,...j->...", d, xi_inv, d)
    c = math.exp(-0.5 / settings.HBAR * sandwiched_xi_inv) / math.sqrt(math.det(xi))

    return A, b, c
