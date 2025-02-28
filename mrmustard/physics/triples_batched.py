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

from typing import Iterable

import numpy as np

from mrmustard import math, settings
from mrmustard.utils.typing import Matrix, Vector, Scalar, RealMatrix
from mrmustard.physics.gaussian_integrals import complex_gaussian_integral_2
from .bargmann_utils import symplectic2Au


#  ~~~~~~~~~
#  Utilities
#  ~~~~~~~~~


def _compute_batch_size(*args) -> tuple[int, ...] | None:
    r"""
    Compute the final batch size of the input arguments.

    Args:
        *args: The input arguments.

    Returns:
        The final batch size and batch dimension of the input arguments.
    """
    batch_size = None
    for arg in args:
        arg = math.astensor(arg)
        if arg.shape:
            if batch_size is None:
                batch_size = arg
            else:
                batch_size = batch_size * arg
    batch_size = batch_size.shape if batch_size is not None else None
    return batch_size, len(batch_size or (1,))


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


#  ~~~~~~~~~~~
#  Pure States
#  ~~~~~~~~~~~

# TODO: how to handle batching here?
# i.e. does it always output batch (1,)?
def vacuum_state_Abc(n_modes: int) -> tuple[Matrix, Vector, Scalar]:
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


def bargmann_eigenstate_Abc(alpha: complex | Iterable[complex]) -> tuple[Matrix, Vector, Scalar]:
    r"""
    The Abc triple of a Bargmann eigenstate.

    Args:
        alpha: The eigenvalue of the Bargmann eigenstate.

    Returns:
        The ``(A, b, c)`` triple of the Bargmann eigenstate.
    """
    batch_size, _ = _compute_batch_size(alpha)
    batch_shape = batch_size or (1,)

    A = np.broadcast_to(_vacuum_A_matrix(1), batch_shape + (1, 1))
    b = math.cast(math.reshape(alpha, batch_shape + (1,)), math.complex128)
    c = math.ones(batch_shape, math.complex128)
    return A if batch_size else A[0], b if batch_size else b[0], c if batch_size else c[0]


def coherent_state_Abc(
    x: float | Iterable[float], y: float | Iterable[float] = 0
) -> tuple[Matrix, Vector, Scalar]:
    r"""
    The ``(A, b, c)`` triple of a pure coherent state.

    Args:
        x: The real part of the displacement, in units of :math:`\sqrt{\hbar}`.
        y: The imaginary part of the displacement, in units of :math:`\sqrt{\hbar}`.

    Returns:
        The ``(A, b, c)`` triple of the pure coherent state.
    """
    batch_size, _ = _compute_batch_size(x, y)
    batch_shape = batch_size or (1,)

    x = np.broadcast_to(x, batch_shape)
    y = np.broadcast_to(y, batch_shape)

    A = np.broadcast_to(_vacuum_A_matrix(1), batch_shape + (1, 1))
    b = math.reshape(x + 1j * y, batch_shape + (1,))
    c = math.cast(math.exp(-0.5 * (x**2 + y**2)), math.complex128)

    return A if batch_size else A[0], b if batch_size else b[0], c if batch_size else c[0]


def squeezed_vacuum_state_Abc(
    r: float | Iterable[float], phi: float | Iterable[float] = 0
) -> tuple[Matrix, Vector, Scalar]:
    r"""
    The ``(A, b, c)`` triple of a squeezed vacuum state.

    Args:
        r: The squeezing magnitudes.
        phi: The squeezing angles.

    Returns:
        The ``(A, b, c)`` triple of a squeezed vacuum state.
    """
    batch_size, _ = _compute_batch_size(r, phi)
    batch_shape = batch_size or (1,)

    r = np.broadcast_to(r, batch_shape)
    phi = np.broadcast_to(phi, batch_shape)

    A = math.reshape(-math.sinh(r) / math.cosh(r) * math.exp(1j * phi), batch_shape + (1, 1))
    b = math.tile(_vacuum_B_vector(1), batch_shape + (1,))
    c = 1 / math.sqrt(math.cosh(r))

    return A if batch_size else A[0], b if batch_size else b[0], c if batch_size else c[0]


def displaced_squeezed_vacuum_state_Abc(
    x: float | Iterable[float],
    y: float | Iterable[float] = 0,
    r: float | Iterable[float] = 0,
    phi: float | Iterable[float] = 0,
) -> tuple[Matrix, Vector, Scalar]:
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
    batch_size, _ = _compute_batch_size(x, y, r, phi)
    batch_shape = batch_size or (1,)

    x = np.broadcast_to(x, batch_shape)
    y = np.broadcast_to(y, batch_shape)
    r = np.broadcast_to(r, batch_shape)
    phi = np.broadcast_to(phi, batch_shape)

    A = math.reshape(-math.sinh(r) / math.cosh(r) * math.exp(1j * phi), batch_shape + (1, 1))
    b = math.reshape(
        (x + 1j * y) + (x - 1j * y) * math.sinh(r) / math.cosh(r) * math.exp(1j * phi),
        batch_shape + (1,),
    )
    c = math.exp(
        -0.5 * (x**2 + y**2)
        - 0.5 * (x - 1j * y) ** 2 * math.sinh(r) / math.cosh(r) * math.exp(1j * phi)
    )
    c = c / math.sqrt(math.cosh(r))

    return A if batch_size else A[0], b if batch_size else b[0], c if batch_size else c[0]


def two_mode_squeezed_vacuum_state_Abc(
    r: float | Iterable[float], phi: float | Iterable[float] = 0
) -> tuple[Matrix, Vector, Scalar]:
    r"""
    The ``(A, b, c)`` triple of a two mode squeezed vacuum state.

    Args:
        r: The squeezing magnitudes.
        phi: The squeezing angles.

    Returns:
        The ``(A, b, c)`` triple of the squeezed vacuum state.
    """
    batch_size, batch_dim = _compute_batch_size(r, phi)
    batch_shape = batch_size or (1,)

    r = np.broadcast_to(r, batch_shape)
    phi = np.broadcast_to(phi, batch_shape)

    O_matrix = math.zeros(batch_shape, math.complex128)
    tanhr = -math.exp(1j * phi) * math.sinh(r) / math.cosh(r)

    A = np.stack(
        [np.stack([O_matrix, tanhr], batch_dim), np.stack([tanhr, O_matrix], batch_dim)], batch_dim
    )
    b = math.tile(_vacuum_B_vector(2), batch_shape + (2,))
    c = math.cast(1 / math.cosh(r), math.complex128)

    return A if batch_size else A[0], b if batch_size else b[0], c if batch_size else c[0]


def gket_state_Abc(symplectic: RealMatrix):
    r"""
    The A,b,c parameters of a Gaussian Ket (Gket) state. This is simply a Gaussian acted on the vacuum.

    Args:
        symplectic: the symplectic representation of the Gaussian

    Returns:
        The ``(A,b,c)`` triple of the Gket state.
    """
    batch_size = symplectic.shape[:-2]
    batch_shape = batch_size or (1,)
    batch_dim = len(batch_size)

    symplectic = np.broadcast_to(symplectic, batch_shape + symplectic.shape[-2:])
    m = symplectic.shape[-1] // 2  # num of modes

    batch_slice = (slice(None, None, None),) * batch_dim

    Au = symplectic2Au(symplectic)

    A = Au[*batch_slice, :m, :m]
    b = math.zeros(batch_shape + (m,), dtype=A.dtype)
    c = (
        (-1) ** m
        * math.det(
            Au[*batch_slice, m:, m:] @ math.conj(Au[*batch_slice, m:, m:])
            - math.eye_like(Au[*batch_slice, m:, m:])
        )
    ) ** 0.25

    return A if batch_size else A[0], b if batch_size else b[0], c if batch_size else c[0]


def gdm_state_Abc(betas: Vector, symplectic: RealMatrix):
    r"""
    The A,b,c parameters of a Gaussian mixed state that is defined by the action of a Guassian on a thermal state

    Args:
        betas: the list of betas corresponding to the temperatures of the initial thermal state
        symplectic: the symplectic matrix of the Gaussian

    Returns:
        The ``(A,b,c)`` triple of the resulting Gaussian DM state.
    """
    betas = math.atleast_1d(betas)  # makes it work
    m = len(betas)
    Au = symplectic2Au(symplectic)
    A_udagger_u = math.block(
        [
            [math.conj(Au), math.zeros((2 * m, 2 * m), dtype="complex128")],
            [math.zeros((2 * m, 2 * m), dtype="complex128"), Au],
        ]
    )

    D = math.diag(math.exp(-betas))
    A_fd = math.block([[math.zeros((m, m)), D], [D, math.zeros((m, m))]])
    c_fd = math.prod((1 - math.exp(-betas)))
    t_fd = (math.atleast_3d(A_fd), math.zeros((1, 2 * m), dtype=A_fd.dtype), math.atleast_1d(c_fd))
    c_u = (
        (-1) ** m * math.det(Au[m:, m:] @ math.conj(Au[m:, m:]) - math.eye_like(Au[m:, m:]))
    ) ** (0.5)
    t_u = (math.atleast_3d(A_udagger_u), math.zeros((1, 4 * m)), math.atleast_1d(c_u))
    return complex_gaussian_integral_2(
        t_fd, t_u, list(range(2 * m)), list(range(m, 2 * m)) + list(range(3 * m, 4 * m))
    )


def sauron_state_Abc(
    n: int | Iterable[int], epsilon: float | Iterable[float]
) -> tuple[Matrix, Vector, Scalar]:
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
        (np.conj(As), np.conj(bs), np.conj(cs)), (As, bs, cs), [0], [0]
    )[2]
    prob = np.sum(probs)
    cs /= np.sqrt(prob)

    return As, bs, cs


def quadrature_eigenstates_Abc(
    x: float | Iterable[float], phi: float | Iterable[float]
) -> tuple[Matrix, Vector, Scalar]:
    r"""
    The ``(A, b, c)`` triple of a quadrature eigenstate.

    Args:
        r: The squeezing magnitudes.
        phi: The squeezing angles.

    Returns:
        The ``(A, b, c)`` triple of the squeezed vacuum state.
    """
    hbar = settings.HBAR

    batch_size, _ = _compute_batch_size(x, phi)
    batch_shape = batch_size or (1,)

    x = np.broadcast_to(x, batch_shape)
    phi = np.broadcast_to(phi, batch_shape)

    A = math.reshape(-math.exp(1j * 2 * phi), batch_shape + (1, 1))
    b = math.reshape(x * math.exp(1j * phi) * math.sqrt(2 / hbar), batch_shape + (1,))
    c = math.cast(1 / (np.pi) ** (1 / 4) * math.exp(-(x**2) / (2 * hbar)), math.complex128)
    return A if batch_size else A[0], b if batch_size else b[0], c if batch_size else c[0]


#  ~~~~~~~~~~~~
#  Mixed States
#  ~~~~~~~~~~~~


def thermal_state_Abc(nbar: int | Iterable[int]) -> tuple[Matrix, Vector, Scalar]:
    r"""
    The ``(A, b, c)`` triple of a thermal state.

    Args:
        nbar: The average number of photons.

    Returns:
        The ``(A, b, c)`` triple of the thermal state.
    """
    batch_size, batch_dim = _compute_batch_size(nbar)
    batch_shape = batch_size or (1,)

    nbar = np.broadcast_to(nbar, batch_shape)

    O_matrix = math.zeros(batch_shape, math.complex128)

    A = np.stack(
        [
            np.stack([O_matrix, (nbar / (nbar + 1))], batch_dim),
            np.stack([(nbar / (nbar + 1)), O_matrix], batch_dim),
        ],
        batch_dim,
    )
    b = math.tile(_vacuum_B_vector(2), batch_shape + (2,))
    c = math.cast(1 / (nbar + 1), math.complex128)
    return A if batch_size else A[0], b if batch_size else b[0], c if batch_size else c[0]


# ~~~~~~~~~~~~~~~~~~~~~~~~
#  Unitary transformations
# ~~~~~~~~~~~~~~~~~~~~~~~~


def rotation_gate_Abc(
    theta: float | Iterable[float],
) -> tuple[Matrix, Vector, Scalar]:
    r"""
    The ``(A, b, c)`` triple of of a tensor product of a rotation gate.

    Args:
        theta: The rotation angles.

    Returns:
        The ``(A, b, c)`` triple of the rotation gate.
    """
    batch_size, batch_dim = _compute_batch_size(theta)
    batch_shape = batch_size or (1,)

    theta = np.broadcast_to(theta, batch_shape)

    O_matrix = math.zeros(batch_shape, math.complex128)

    A = np.stack(
        [
            np.stack([O_matrix, math.exp(1j * theta)], batch_dim),
            np.stack([math.exp(1j * theta), O_matrix], batch_dim),
        ],
        batch_dim,
    )
    b = math.tile(_vacuum_B_vector(2), batch_shape + (2,))
    c = math.ones(batch_shape, math.complex128)

    return A if batch_size else A[0], b if batch_size else b[0], c if batch_size else c[0]


def displacement_gate_Abc(
    x: float | Iterable[float], y: float | Iterable[float] = 0
) -> tuple[Matrix, Vector, Scalar]:
    r"""
    The ``(A, b, c)`` triple of a tensor product of a displacement gate.

    Args:
        x: The real part of the displacement, in units of :math:`\sqrt{\hbar}`.
        y: The imaginary part of the displacement, in units of :math:`\sqrt{\hbar}`.

    Returns:
        The ``(A, b, c)`` triple of the displacement gate.
    """
    batch_size, batch_dim = _compute_batch_size(x, y)
    batch_shape = batch_size or (1,)

    x = np.broadcast_to(x, batch_shape)
    y = np.broadcast_to(y, batch_shape)

    A = np.broadcast_to(_X_matrix_for_unitary(1), batch_shape + (2, 2))
    b = np.stack([x + 1j * y, -x + 1j * y], batch_dim)
    c = math.cast(math.exp(-(x**2 + y**2) / 2), math.complex128)

    return A if batch_size else A[0], b if batch_size else b[0], c if batch_size else c[0]


def squeezing_gate_Abc(
    r: float | Iterable[float], delta: float | Iterable[float] = 0
) -> tuple[Matrix, Vector, Scalar]:
    r"""
    The ``(A, b, c)`` triple of a squeezing gate.

    Args:
        r: The squeezing magnitudes.
        delta: The squeezing angles.

    Returns:
        The ``(A, b, c)`` triple of the squeezing gate.
    """
    batch_size, batch_dim = _compute_batch_size(r, delta)
    batch_shape = batch_size or (1,)

    r = np.broadcast_to(r, batch_shape)
    delta = np.broadcast_to(delta, batch_shape)

    tanhr = math.sinh(r) / math.cosh(r)
    sechr = 1 / math.cosh(r)

    A = np.stack(
        [
            np.stack([-math.exp(1j * delta) * tanhr, sechr], batch_dim),
            np.stack([sechr, math.exp(-1j * delta) * tanhr], batch_dim),
        ],
        batch_dim,
    )
    b = math.tile(_vacuum_B_vector(2), batch_shape + (2,))
    c = math.cast(1 / math.sqrt(math.cosh(r)), math.complex128)

    return A if batch_size else A[0], b if batch_size else b[0], c if batch_size else c[0]


def beamsplitter_gate_Abc(
    theta: float | Iterable[float], phi: float | Iterable[float] = 0
) -> tuple[Matrix, Vector, Scalar]:
    r"""
    The ``(A, b, c)`` triple of a tensor product of a two-mode beamsplitter gate.

    Args:
        theta: The transmissivity parameters.
        phi: The phase parameters.

    Returns:
        The ``(A, b, c)`` triple of the beamsplitter gate.
    """
    batch_size, batch_dim = _compute_batch_size(theta, phi)
    batch_shape = batch_size or (1,)

    theta = np.broadcast_to(theta, batch_shape)
    phi = np.broadcast_to(phi, batch_shape)

    O_matrix = math.zeros(batch_shape + (2, 2), math.complex128)
    costheta = math.cos(theta)
    sintheta = math.sin(theta)

    V = np.stack(
        [
            np.stack([costheta, -math.exp(math.astensor(-1j) * phi) * sintheta], batch_dim),
            np.stack([math.exp(math.astensor(1j) * phi) * sintheta, costheta], batch_dim),
        ],
        batch_dim,
    )

    perm = tuple(range(len(V.shape)))
    perm = perm[:batch_dim] + perm[batch_dim:][::-1]

    A = math.concat(
        [math.concat([O_matrix, V], -1), math.concat([math.transpose(V, perm), O_matrix], -1)], -2
    )
    b = math.tile(_vacuum_B_vector(4), batch_shape + (4,))
    c = math.ones(batch_shape, math.complex128)
    return A if batch_size else A[0], b if batch_size else b[0], c if batch_size else c[0]


def twomode_squeezing_gate_Abc(
    r: float | Iterable[float], phi: float | Iterable[float] = 0
) -> tuple[Matrix, Vector, Scalar]:
    r"""
    The ``(A, b, c)`` triple of a tensor product of a two-mode squeezing gate.

    Args:
        r: The squeezing magnitudes.
        phi: The squeezing phase.

    Returns:
        The ``(A, b, c)`` triple of the two mode squeezing gate.
    """
    batch_size, batch_dim = _compute_batch_size(r, phi)
    batch_shape = batch_size or (1,)

    r = np.broadcast_to(r, batch_shape)
    phi = np.broadcast_to(phi, batch_shape)

    O_matrix = math.zeros(batch_shape, math.complex128)
    tanhr = math.exp(1j * phi) * math.sinh(r) / math.cosh(r)
    sechr = 1 / math.cosh(r)

    A_block1 = np.stack(
        [np.stack([O_matrix, tanhr], batch_dim), np.stack([tanhr, O_matrix], batch_dim)], batch_dim
    )
    A_block2 = np.stack(
        [
            np.stack([O_matrix, -math.conj(tanhr)], batch_dim),
            np.stack([-math.conj(tanhr), O_matrix], batch_dim),
        ],
        batch_dim,
    )
    A_block3 = np.stack(
        [np.stack([sechr, O_matrix], batch_dim), np.stack([O_matrix, sechr], batch_dim)], batch_dim
    )

    A = math.concat(
        [math.concat([A_block1, A_block3], -1), math.concat([A_block3, A_block2], -1)], -2
    )
    b = math.tile(_vacuum_B_vector(4), batch_shape + (4,))
    c = math.cast(1 / math.cosh(r), math.complex128)

    return A if batch_size else A[0], b if batch_size else b[0], c if batch_size else c[0]


# TODO: how to handle batching here?
# i.e. does it always output batch (1,)?
def identity_Abc(n_modes: int) -> tuple[Matrix, Vector, Scalar]:
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


def attenuator_Abc(eta: float | Iterable[float]) -> tuple[Matrix, Vector, Scalar]:
    r"""
    The ``(A, b, c)`` triple of an attenuator.

    Args:
        eta: The values of the transmissivities.

    Returns:
        The ``(A, b, c)`` triple of the attenuator channel.

    Raises:
        ValueError: If ``eta`` is larger than `1` or smaller than `0`.
    """
    batch_size, batch_dim = _compute_batch_size(eta)
    batch_shape = batch_size or (1,)

    eta = np.broadcast_to(eta, batch_shape)

    if math.any(math.real(eta) > 1) or math.any(math.real(eta) < 0):
        raise ValueError("Transmissivity must be a float in the interval ``[0, 1]``.")

    O_matrix = math.zeros(batch_shape, math.complex128)
    eta1 = math.sqrt(eta)
    eta2 = 1 - eta

    A = np.stack(
        [
            np.stack([O_matrix, eta1, O_matrix, O_matrix], batch_dim),
            np.stack([eta1, O_matrix, O_matrix, eta2], batch_dim),
            np.stack([O_matrix, O_matrix, O_matrix, eta1], batch_dim),
            np.stack([O_matrix, eta2, eta1, O_matrix], batch_dim),
        ],
        batch_dim,
    )
    b = math.tile(_vacuum_B_vector(4), batch_shape + (4,))
    c = math.ones(batch_shape, math.complex128)

    return A if batch_size else A[0], b if batch_size else b[0], c if batch_size else c[0]


def amplifier_Abc(g: float | Iterable[float]) -> tuple[Matrix, Vector, Scalar]:
    r"""
    The ``(A, b, c)`` triple of an amplifier.

    Args:
        g: The values of the gains.

    Returns:
        The ``(A, b, c)`` triple of the amplifier channel.

    Raises:
        ValueError: If ``g`` is smaller than `1`.
    """
    batch_size, batch_dim = _compute_batch_size(g)
    batch_shape = batch_size or (1,)

    g = np.broadcast_to(g, batch_shape)

    if math.any(math.real(g) < 1):
        raise ValueError("Found amplifier with gain ``g`` smaller than `1`.")

    O_matrix = math.zeros(batch_shape, math.complex128)
    g1 = 1 / math.sqrt(g)
    g2 = 1 - 1 / g
    A = np.stack(
        [
            np.stack([O_matrix, g1, g2, O_matrix], batch_dim),
            np.stack([g1, O_matrix, O_matrix, O_matrix], batch_dim),
            np.stack([g2, O_matrix, O_matrix, g1], batch_dim),
            np.stack([O_matrix, O_matrix, g1, O_matrix], batch_dim),
        ],
        batch_dim,
    )
    b = math.tile(_vacuum_B_vector(4), batch_shape + (4,))
    c = math.cast(1 / g, math.complex128)

    return A if batch_size else A[0], b if batch_size else b[0], c if batch_size else c[0]


def fock_damping_Abc(
    beta: float | Iterable[float],
) -> tuple[Matrix, Vector, Scalar]:
    r"""
    The ``(A, b, c)`` triple of a Fock damper.

    Args:
        beta: The damping parameter.

    Returns:
        The ``(A, b, c)`` triple of the Fock damping operator.
    """
    batch_size, batch_dim = _compute_batch_size(beta)
    batch_shape = batch_size or (1,)

    beta = np.broadcast_to(beta, batch_shape)

    O_matrix = math.zeros(batch_shape, math.complex128)
    B_n = math.exp(-beta)

    A = np.stack(
        [np.stack([O_matrix, B_n], batch_dim), np.stack([B_n, O_matrix], batch_dim)], batch_dim
    )
    b = math.tile(_vacuum_B_vector(2), batch_shape + (2,))
    c = math.ones(batch_shape, math.complex128)

    return A if batch_size else A[0], b if batch_size else b[0], c if batch_size else c[0]


def gaussian_random_noise_Abc(Y: RealMatrix) -> tuple[Matrix, Vector, Scalar]:
    r"""
    The triple (A, b, c) for the gaussian random noise channel.

    Args:
        Y: the Y matrix of the Gaussian random noise channel.

    Returns:
        The ``(A, b, c)`` triple of the Gaussian random noise channel.
    """
    m = Y.shape[-1] // 2
    xi = math.eye(2 * m, dtype=math.complex128) + Y / settings.HBAR
    xi_inv = math.inv(xi)
    xi_inv_in_blocks = math.block(
        [[math.eye(2 * m) - xi_inv, xi_inv], [xi_inv, math.eye(2 * m) - xi_inv]]
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
            ]
        )
    )

    A = math.Xmat(2 * m) @ R @ xi_inv_in_blocks @ math.conj(R).T
    b = math.zeros(4 * m)
    c = 1 / math.sqrt(math.det(xi))

    return A, b, c


def bargmann_to_quadrature_Abc(
    n_modes: int, phi: float | Iterable[float]
) -> tuple[Matrix, Vector, Scalar]:
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
    batch_size, batch_dim = _compute_batch_size(phi)
    batch_shape = batch_size or (1,)

    phi = np.broadcast_to(phi, batch_shape)

    hbar = settings.HBAR
    Id = math.eye(n_modes, dtype=math.complex128)
    e = math.exp(-1j * phi + 1j * np.pi / 2)
    A = math.kron(
        np.stack(
            [
                np.stack(
                    [np.broadcast_to(-1 / hbar, batch_shape), -1j * e * np.sqrt(2 / hbar)],
                    batch_dim,
                ),
                np.stack([-1j * e * np.sqrt(2 / hbar), e * e], batch_dim),
            ],
            batch_dim,
        ),
        Id,
    )
    b = math.tile(_vacuum_B_vector(2 * n_modes), batch_shape + (2 * n_modes,))
    c = math.ones(batch_shape, math.complex128) / (np.pi * hbar) ** (0.25 * n_modes)
    return A if batch_size else A[0], b if batch_size else b[0], c if batch_size else c[0]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Maps between representations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def displacement_map_s_parametrized_Abc(s: int, n_modes: int) -> tuple[Matrix, Vector, Scalar]:
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
    batch_size, batch_dim = _compute_batch_size(s)
    batch_shape = batch_size or (1,)
    batch_slice = (slice(None),) * batch_dim

    s = np.broadcast_to(s, batch_shape)

    A = math.concat(
        [
            np.concat(
                [
                    (s[..., None, None] - 1) / 2 * math.Xmat(num_modes=n_modes),
                    np.broadcast_to(
                        -math.Zmat(num_modes=n_modes), batch_shape + (2 * n_modes, 2 * n_modes)
                    ),
                ],
                -1,
            ),
            np.concat(
                [
                    np.broadcast_to(
                        -math.Zmat(num_modes=n_modes), batch_shape + (2 * n_modes, 2 * n_modes)
                    ),
                    np.broadcast_to(
                        math.Xmat(num_modes=n_modes), batch_shape + (2 * n_modes, 2 * n_modes)
                    ),
                ],
                -1,
            ),
        ],
        -2,
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
        )
    )

    A = math.astensor(math.asnumpy(A)[*batch_slice, order_list, :][*batch_slice, :, order_list])
    b = np.broadcast_to(_vacuum_B_vector(4 * n_modes), batch_shape + (4 * n_modes,))
    c = math.ones(batch_shape, math.complex128)
    return A, b, c


#  TODO: how to handle batching here?
def complex_fourier_transform_Abc(n_modes: int) -> tuple[Matrix, Vector, Scalar]:
    r"""
    The ``(A, b, c)`` triple of the complex Fourier transform between two pairs of complex variables.
    Given a function :math:`f(z^*, z)`, the complex Fourier transform is defined as
    :math:
        \hat{f} (y^*, y) = \int_{\mathbb{C}} \frac{d^2 z}{\pi} e^{yz^* - y^*z} f(z^*, z).
    The indices of this triple correspond to the variables :math:`(y^*, z^*, y, z)`.

    Args:
        n_modes: the number of modes for this map.

    Returns:
        The ``(A, b, c)`` triple of the complex fourier transform.
    """
    O2n = math.zeros((2 * n_modes, 2 * n_modes))
    Omega = math.J(n_modes)
    A = math.block([[O2n, -Omega], [Omega, O2n]])
    b = _vacuum_B_vector(4 * n_modes)
    c = 1.0 + 0j
    return A, b, c


# ~~~~~~~~~~~~~~~~
# Kraus operators
# ~~~~~~~~~~~~~~~~


def attenuator_kraus_Abc(eta: float | Iterable[float]) -> tuple[Matrix, Vector, Scalar]:
    r"""
    The entire family of Kraus operators of the attenuator (loss) channel as a single ``(A, b, c)`` triple.
    The last index is the "bond" index which should be summed/integrated over.

    Args:
        eta: The value of the transmissivity.

    Returns:
        The ``(A, b, c)`` triple of the kraus operators of the attenuator (loss) channel.
    """
    batch_size, batch_dim = _compute_batch_size(eta)
    batch_shape = batch_size or (1,)

    eta = np.broadcast_to(eta, batch_shape)

    costheta = math.sqrt(eta)
    sintheta = math.sqrt(1 - eta)

    O_matrix = math.zeros(batch_shape, math.complex128)

    A = np.stack(
        [
            np.stack([O_matrix, costheta, O_matrix], batch_dim),
            np.stack([costheta, O_matrix, -sintheta], batch_dim),
            np.stack([O_matrix, -sintheta, O_matrix], batch_dim),
        ],
        batch_dim,
    )
    b = math.tile(_vacuum_B_vector(3), batch_shape + (3,))
    c = math.ones(batch_shape, math.complex128)
    return A if batch_size else A[0], b if batch_size else b[0], c if batch_size else c[0]


def XY_to_channel_Abc(
    X: RealMatrix, Y: RealMatrix, d: Vector | None = None
) -> tuple[Matrix, Vector, Scalar]:
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
            f"X.shape = {X.shape}, Y.shape = {Y.shape}"
        )

    xi = 1 / 2 * math.eye(2 * m, dtype=math.complex128) + 1 / 2 * X @ X.T + Y / settings.HBAR
    xi_inv = math.inv(xi)
    xi_inv_in_blocks = math.block(
        [[math.eye(2 * m) - xi_inv, xi_inv @ X], [X.T @ xi_inv, math.eye(2 * m) - X.T @ xi_inv @ X]]
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
            ]
        )
    )

    A = math.Xmat(2 * m) @ R @ xi_inv_in_blocks @ math.conj(R).T
    temp = math.block([[(xi_inv @ d).reshape(2 * m, 1)], [(-X.T @ xi_inv @ d).reshape((2 * m, 1))]])
    b = 1 / math.sqrt(complex(settings.HBAR)) * math.conj(R) @ temp
    c = math.exp(-0.5 / settings.HBAR * d @ xi_inv @ d) / math.sqrt(math.det(xi))

    return A, b, c
