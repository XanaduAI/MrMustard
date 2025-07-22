# Copyright 2024 Xanadu Quantum Technologies Inc.

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
This module contains the symplectic matrices for the Fock-Bargmann representation of
various states and transformations.
"""

from __future__ import annotations

from collections.abc import Sequence

from mrmustard import math
from mrmustard.utils.typing import Matrix


def cxgate_symplectic(s: float | Sequence[float]) -> Matrix:
    r"""
    The symplectic matrix of a controlled X gate.

    Args:
        s: The control parameter.

    Returns:
        The symplectic matrix of a CX gate.
    """
    s = math.astensor(s, dtype=math.complex128)
    batch_shape = s.shape
    batch_dim = len(batch_shape)

    O_matrix = math.zeros(batch_shape, math.complex128)
    I_matrix = math.ones(batch_shape, math.complex128)

    return math.stack(
        [
            math.stack([I_matrix, O_matrix, O_matrix, O_matrix], batch_dim),
            math.stack([s, I_matrix, O_matrix, O_matrix], batch_dim),
            math.stack([O_matrix, O_matrix, I_matrix, -s], batch_dim),
            math.stack([O_matrix, O_matrix, O_matrix, I_matrix], batch_dim),
        ],
        batch_dim,
    )


def czgate_symplectic(s: float | Sequence[float]) -> Matrix:
    r"""
    The symplectic matrix of a controlled Z gate.

    Args:
        s: The control parameter.

    Returns:
        The symplectic matrix of a CZ gate.
    """
    s = math.astensor(s, dtype=math.complex128)
    batch_shape = s.shape
    batch_dim = len(batch_shape)

    O_matrix = math.zeros(batch_shape, math.complex128)
    I_matrix = math.ones(batch_shape, math.complex128)

    return math.stack(
        [
            math.stack([I_matrix, O_matrix, O_matrix, O_matrix], batch_dim),
            math.stack([O_matrix, I_matrix, O_matrix, O_matrix], batch_dim),
            math.stack([O_matrix, s, I_matrix, O_matrix], batch_dim),
            math.stack([s, O_matrix, O_matrix, I_matrix], batch_dim),
        ],
        batch_dim,
    )


def interferometer_symplectic(unitary: Matrix) -> Matrix:
    r"""
    The symplectic matrix of an N-mode interferometer parametrized by an NxN unitary matrix.

    Args:
        unitary : A unitary matrix. For N modes it must have shape `(N,N)`.

    Returns:
        The symplectic matrix of an N-mode interferometer.
    """
    return math.block(
        [[math.real(unitary), -math.imag(unitary)], [math.imag(unitary), math.real(unitary)]],
    )


def mzgate_symplectic(
    phi_a: float | Sequence[float],
    phi_b: float | Sequence[float],
    internal: bool,
) -> Matrix:
    r"""
    The symplectic matrix of a Mach-Zehnder gate.

    It supports two conventions:
        1. if ``internal=True``, both phases act inside the interferometer: ``phi_a`` on the upper arm, ``phi_b`` on the lower arm;
        2. if ``internal = False``, both phases act on the upper arm: ``phi_a`` before the first BS, ``phi_b`` after the first BS.

    Args:
        phi_a: The phase in the upper arm of the MZ interferometer
        phi_b: The phase in the lower arm or external of the MZ interferometer
        internal: Whether phases are both in the internal arms.

    Returns:
        The symplectic matrix of a Mach-Zehnder gate.
    """
    phi_a, phi_b = math.broadcast_arrays(
        math.astensor(phi_a, dtype=math.complex128),
        math.astensor(phi_b, dtype=math.complex128),
    )
    batch_shape = phi_a.shape
    batch_dim = len(batch_shape)

    ca = math.cos(phi_a)
    sa = math.sin(phi_a)
    cb = math.cos(phi_b)
    sb = math.sin(phi_b)
    cp = math.cos(phi_a + phi_b)
    sp = math.sin(phi_a + phi_b)
    if internal:
        symplectic = math.stack(
            [
                math.stack([ca - cb, -sa - sb, sb - sa, -ca - cb], batch_dim),
                math.stack([-sa - sb, cb - ca, -ca - cb, sa - sb], batch_dim),
                math.stack([sa - sb, ca + cb, ca - cb, -sa - sb], batch_dim),
                math.stack([ca + cb, sb - sa, -sa - sb, cb - ca], batch_dim),
            ],
            batch_dim,
        )
    else:
        symplectic = math.stack(
            [
                math.stack([cp - ca, -sb, sa - sp, -1 - cb], batch_dim),
                math.stack([-sa - sp, 1 - cb, -ca - cp, sb], batch_dim),
                math.stack([sp - sa, 1 + cb, cp - ca, -sb], batch_dim),
                math.stack([cp + ca, -sb, -sa - sp, 1 - cb], batch_dim),
            ],
            batch_dim,
        )
    return 0.5 * symplectic


def pgate_symplectic(n_modes: int, shearing: float | Sequence[float]) -> Matrix:
    r"""
    The symplectic matrix of a quadratic phase gate.

    Args:
        n_modes: The number of modes.
        shearing: The shearing parameter.

    Returns:
        The symplectic matrix of a phase gate.
    """
    shearing = math.astensor(shearing, dtype=math.complex128)
    batch_shape = shearing.shape

    I_matrix = math.broadcast_to(
        math.eye(n_modes, dtype=math.complex128),
        (*batch_shape, n_modes, n_modes),
    )
    O_matrix = math.zeros((*batch_shape, n_modes, n_modes), dtype=math.complex128)

    return math.block([[I_matrix, O_matrix], [I_matrix * shearing[..., None, None], I_matrix]])


def realinterferometer_symplectic(orthogonal: Matrix) -> Matrix:
    r"""
    The symplectic matrix of an N-mode interferometer parametrized by an NxN orthogonal matrix.

    Args:
        orthogonal : A real orthogonal matrix. For N modes it must have shape `(N,N)`.

    Returns:
        The symplectic matrix of an N-mode interferometer.
    """
    return math.block(
        [[orthogonal, -math.zeros_like(orthogonal)], [math.zeros_like(orthogonal), orthogonal]],
    )
