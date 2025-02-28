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

from typing import Iterable

import numpy as np

from mrmustard import math
from mrmustard.utils.typing import Matrix


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


def cxgate_symplectic(s: float | Iterable[float]) -> Matrix:
    r"""
    The symplectic matrix of a controlled X gate.

    Args:
        s: The control parameter.

    Returns:
        The symplectic matrix of a CX gate.
    """
    batch_size, batch_dim = _compute_batch_size(s)
    batch_shape = batch_size or (1,)

    s = np.broadcast_to(s, batch_shape)

    O_matrix = math.zeros(batch_shape, math.complex128)
    I_matrix = math.ones(batch_shape, math.complex128)

    symplectic = np.stack(
        [
            np.stack([I_matrix, O_matrix, O_matrix, O_matrix], batch_dim),
            np.stack([s, I_matrix, O_matrix, O_matrix], batch_dim),
            np.stack([O_matrix, O_matrix, I_matrix, -s], batch_dim),
            np.stack([O_matrix, O_matrix, O_matrix, I_matrix], batch_dim),
        ],
        batch_dim,
    )
    return symplectic if batch_size else symplectic[0]


def czgate_symplectic(s: float | Iterable[float]) -> Matrix:
    r"""
    The symplectic matrix of a controlled Z gate.

    Args:
        s: The control parameter.

    Returns:
        The symplectic matrix of a CZ gate.
    """
    batch_size, batch_dim = _compute_batch_size(s)
    batch_shape = batch_size or (1,)

    s = np.broadcast_to(s, batch_shape)

    O_matrix = math.zeros(batch_shape, math.complex128)
    I_matrix = math.ones(batch_shape, math.complex128)

    symplectic = np.stack(
        [
            np.stack([I_matrix, O_matrix, O_matrix, O_matrix], batch_dim),
            np.stack([O_matrix, I_matrix, O_matrix, O_matrix], batch_dim),
            np.stack([O_matrix, s, I_matrix, O_matrix], batch_dim),
            np.stack([s, O_matrix, O_matrix, I_matrix], batch_dim),
        ],
        batch_dim,
    )
    return symplectic if batch_size else symplectic[0]


def interferometer_symplectic(unitary: Matrix) -> Matrix:
    r"""
    The symplectic matrix of an N-mode interferometer parametrized by an NxN unitary matrix.

    Args:
        unitary : A unitary matrix. For N modes it must have shape `(N,N)`.

    Returns:
        The symplectic matrix of an N-mode interferometer.
    """
    batch_size = unitary.shape[:-2]
    batch_shape = batch_size or (1,)
    unitary = np.broadcast_to(unitary, batch_shape + unitary.shape[-2:])
    symplectic = np.concat(
        [
            np.concat([math.real(unitary), -math.imag(unitary)], -1),
            np.concat([math.imag(unitary), math.real(unitary)], -1),
        ],
        -2,
    )
    return symplectic if batch_size else symplectic[0]


def mzgate_symplectic(
    phi_a: float | Iterable[float], phi_b: float | Iterable[float], internal: bool
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
    batch_size, batch_dim = _compute_batch_size(phi_a, phi_b)
    batch_shape = batch_size or (1,)

    phi_a = np.broadcast_to(phi_a, batch_shape)
    phi_b = np.broadcast_to(phi_b, batch_shape)

    ca = math.cos(phi_a)
    sa = math.sin(phi_a)
    cb = math.cos(phi_b)
    sb = math.sin(phi_b)
    cp = math.cos(phi_a + phi_b)
    sp = math.sin(phi_a + phi_b)
    if internal:
        symplectic = np.stack(
            [
                np.stack([ca - cb, -sa - sb, sb - sa, -ca - cb], batch_dim),
                np.stack([-sa - sb, cb - ca, -ca - cb, sa - sb], batch_dim),
                np.stack([sa - sb, ca + cb, ca - cb, -sa - sb], batch_dim),
                np.stack([ca + cb, sb - sa, -sa - sb, cb - ca], batch_dim),
            ],
            batch_dim,
        )
    else:
        symplectic = np.stack(
            [
                np.stack([cp - ca, -sb, sa - sp, -1 - cb], batch_dim),
                np.stack([-sa - sp, 1 - cb, -ca - cp, sb], batch_dim),
                np.stack([sp - sa, 1 + cb, cp - ca, -sb], batch_dim),
                np.stack([cp + ca, -sb, -sa - sp, 1 - cb], batch_dim),
            ],
            batch_dim,
        )
    symplectic = math.cast(0.5 * symplectic, math.complex128)
    return symplectic if batch_size else symplectic[0]


def pgate_symplectic(n_modes: int, shearing: float | Iterable[float]) -> Matrix:
    r"""
    The symplectic matrix of a quadratic phase gate.

    Args:
        n_modes: The number of modes.
        shearing: The shearing parameter.

    Returns:
        The symplectic matrix of a phase gate.
    """
    batch_size, _ = _compute_batch_size(shearing)
    batch_shape = batch_size or (1,)

    shearing = np.broadcast_to(shearing, batch_shape)

    I_matrix = np.broadcast_to(math.eye(n_modes), batch_shape + (n_modes, n_modes))
    O_matrix = math.zeros(batch_shape + (n_modes, n_modes))

    symplectic = np.concat(
        [
            np.concat([I_matrix, O_matrix], -1),
            np.concat([math.eye(n_modes) * shearing[..., None, None], I_matrix], -1),
        ],
        -2,
    )
    return symplectic if batch_size else symplectic[0]


def realinterferometer_symplectic(orthogonal: Matrix) -> Matrix:
    r"""
    The symplectic matrix of an N-mode interferometer parametrized by an NxN orthogonal matrix.

    Args:
        orthogonal : A real orthogonal matrix. For N modes it must have shape `(N,N)`.

    Returns:
        The symplectic matrix of an N-mode interferometer.
    """
    batch_size = orthogonal.shape[:-2]
    batch_shape = batch_size or (1,)
    orthogonal = np.broadcast_to(orthogonal, batch_shape + orthogonal.shape[-2:])
    symplectic = np.concat(
        [
            np.concat([orthogonal, -math.zeros_like(orthogonal)], -1),
            np.concat([math.zeros_like(orthogonal), orthogonal], -1),
        ],
        -2,
    )
    return symplectic if batch_size else symplectic[0]
