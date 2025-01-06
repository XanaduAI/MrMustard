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

from mrmustard import math
from mrmustard.utils.typing import Matrix


def cxgate_symplectic(s: float) -> Matrix:
    r"""
    The symplectic matrix of a controlled X gate.

    Args:
        s: The control parameter.

    Returns:
        The symplectic matrix of a CX gate.
    """
    return math.astensor(
        [[1, 0, 0, 0], [s, 1, 0, 0], [0, 0, 1, -s], [0, 0, 0, 1]], dtype="complex128"
    )


def czgate_symplectic(s: float) -> Matrix:
    r"""
    The symplectic matrix of a controlled Z gate.

    Args:
        s: The control parameter.

    Returns:
        The symplectic matrix of a CZ gate.
    """
    return math.astensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, s, 1, 0], [s, 0, 0, 1]])


def interferometer_symplectic(unitary: Matrix) -> Matrix:
    r"""
    The symplectic matrix of an N-mode interferometer parametrized by an NxN unitary matrix.

    Args:
        unitary : A unitary matrix. For N modes it must have shape `(N,N)`.

    Returns:
        The symplectic matrix of an N-mode interferometer.
    """
    return math.block(
        [[math.real(unitary), -math.imag(unitary)], [math.imag(unitary), math.real(unitary)]]
    )


def mzgate_symplectic(phi_a: float, phi_b: float, internal: bool) -> Matrix:
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
    ca = math.cos(complex(phi_a))
    sa = math.sin(complex(phi_a))
    cb = math.cos(complex(phi_b))
    sb = math.sin(complex(phi_b))
    cp = math.cos(complex(phi_a + phi_b))
    sp = math.sin(complex(phi_a + phi_b))
    if internal:
        return 0.5 * math.astensor(
            [
                [ca - cb, -sa - sb, sb - sa, -ca - cb],
                [-sa - sb, cb - ca, -ca - cb, sa - sb],
                [sa - sb, ca + cb, ca - cb, -sa - sb],
                [ca + cb, sb - sa, -sa - sb, cb - ca],
            ]
        )
    else:
        return 0.5 * math.astensor(
            [
                [cp - ca, -sb, sa - sp, -1 - cb],
                [-sa - sp, 1 - cb, -ca - cp, sb],
                [sp - sa, 1 + cb, cp - ca, -sb],
                [cp + ca, -sb, -sa - sp, 1 - cb],
            ]
        )


def pgate_symplectic(n_modes: int, shearing: float) -> Matrix:
    r"""
    The symplectic matrix of a quadratic phase gate.

    Args:
        shearing: The shearing parameter.

    Returns:
        The symplectic matrix of a phase gate.
    """
    return math.block(
        [
            [math.eye(n_modes), math.zeros((n_modes, n_modes))],
            [math.eye(n_modes) * shearing, math.eye(n_modes)],
        ]
    )


def realinterferometer_symplectic(orthogonal: Matrix) -> Matrix:
    r"""
    The symplectic matrix of an N-mode interferometer parametrized by an NxN orthogonal matrix.

    Args:
        orthogonal : A real orthogonal matrix. For N modes it must have shape `(N,N)`.

    Returns:
        The symplectic matrix of an N-mode interferometer.
    """
    return math.block(
        [[orthogonal, -math.zeros_like(orthogonal)], [math.zeros_like(orthogonal), orthogonal]]
    )
