# Copyright 2026 Xanadu Quantum Technologies Inc.

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
This module contains methods for creating built-in ``State`` ``Ansatz``.
"""

from __future__ import annotations

from collections.abc import Sequence

from mrmustard import math
from mrmustard.physics import triples
from mrmustard.physics.ansatz import ArrayAnsatz, PolyExpAnsatz
from mrmustard.physics.bargmann_utils import wigner_to_bargmann_psi as wigner_to_bargmann_psi_Abc
from mrmustard.physics.bargmann_utils import wigner_to_bargmann_rho as wigner_to_bargmann_rho_Abc
from mrmustard.physics.fock_utils import fock_state
from mrmustard.utils.typing import RealMatrix, RealVector

__all__ = [
    "bargmann_eigenstate",
    "coherent_state",
    "displaced_squeezed_vacuum_state",
    "gdm_state",
    "gket_state",
    "number_state",
    "quadrature_eigenstate",
    "sauron_state",
    "squeezed_vacuum_state",
    "squeezed_vacuum_state_fock",
    "thermal_state",
    "two_mode_squeezed_vacuum_state",
    "vacuum_state",
    "wigner_to_bargmann_psi",
    "wigner_to_bargmann_rho",
]


def bargmann_eigenstate(
    alpha: complex | Sequence[complex],
    lin_sup: bool = False,
) -> PolyExpAnsatz:
    r"""
    The Bargmann eigenstate in the Bargmann representation.

    Args:
        alpha: The displacement of the state (i.e., the eigen-value).
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The Bargmann eigenstate in the Bargmann representation.
    """
    A, b, c = triples.bargmann_eigenstate_Abc(alpha)
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)


def coherent_state(
    alpha: complex | Sequence[complex],
    lin_sup: bool = False,
) -> PolyExpAnsatz:
    r"""
    The coherent state in the Bargmann representation.

    Args:
        alpha: The complex displacement.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The coherent state in the Bargmann representation.
    """
    A, b, c = triples.coherent_state_Abc(alpha)
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)


def displaced_squeezed_vacuum_state(
    alpha: complex | Sequence[complex] = 0.0j,
    r: float | Sequence[float] = 0.0,
    phi: float | Sequence[float] = 0.0,
    lin_sup: bool = False,
) -> PolyExpAnsatz:
    r"""
    The displaced squeezed vacuum state in the Bargmann representation.

    Args:
        alpha: The complex displacement.
        r: The squeezing magnitude.
        phi: The squeezing angle.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The displaced squeezed vacuum state in the Bargmann representation.
    """
    A, b, c = triples.displaced_squeezed_vacuum_state_Abc(alpha, r, phi)
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)


def gket_state(
    symplectic: RealMatrix,
    lin_sup: bool = False,
) -> PolyExpAnsatz:
    r"""
    The Gaussian ket state in the Bargmann representation.

    Args:
        symplectic: The symplectic matrix of the state.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The Gaussian ket state in the Bargmann representation.
    """
    A, b, c = triples.gket_state_Abc(symplectic)
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)


def gdm_state(
    beta: float | Sequence[float],
    symplectic: RealMatrix,
    lin_sup: bool = False,
) -> PolyExpAnsatz:
    r"""
    The Gaussian dm state in the Bargmann representation.

    Args:
        beta: The displacement of the state.
        symplectic: The symplectic matrix of the state.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The Gaussian dm state in the Bargmann representation.
    """
    A, b, c = triples.gdm_state_Abc(beta, symplectic)
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)


def number_state(
    n: int | Sequence[int],
    shape: tuple[int],
) -> ArrayAnsatz:
    r"""
    The number state in the Fock representation.

    Args:
        n: The number of photons.
        shape: The shape such that the first element is used as the cutoff.

    Returns:
        The number state in the Fock representation.
    """
    array = fock_state(n, shape[0] - 1)
    return ArrayAnsatz(array, batch_dims=len(math.shape(n)))


def quadrature_eigenstate(
    x: float | Sequence[float],
    phi: float | Sequence[float],
    lin_sup: bool = False,
) -> PolyExpAnsatz:
    r"""
    The quadrature eigenstate in the Bargmann representation.

    Args:
        x: The displacement of the state.
        phi: The angle of the state with `0` being a position eigenstate and `\pi/2` being the momentum eigenstate.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The quadrature eigenstate in the Bargmann representation.
    """
    A, b, c = triples.quadrature_eigenstates_Abc(x, phi)
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)


def sauron_state(
    n: int,
    epsilon: float,
    lin_sup: bool = True,
) -> PolyExpAnsatz:
    r"""
    The Sauron state in the Bargmann representation.

    Args:
        n: The number of photons.
        epsilon: The size of the ring. The approximation is exact in the limit for epsilon that goes to zero.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The Sauron state in the Bargmann representation.
    """
    A, b, c = triples.sauron_state_Abc(n, epsilon)
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)


def squeezed_thermal_state(
    nbar: float | Sequence[float],
    r: float | Sequence[float],
    phi: float | Sequence[float] = 0.0,
    lin_sup: bool = False,
) -> PolyExpAnsatz:
    r"""
    The squeezed thermal state in the Bargmann representation.

    Args:
        nbar: The expected number of photons.
        r: The squeezing magnitude.
        phi: The squeezing angle.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The squeezed thermal state in the Bargmann representation.
    """
    A, b, c = triples.squeezed_thermal_state_Abc(nbar, r, phi)
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)


def squeezed_vacuum_state(
    r: float | Sequence[float],
    phi: float | Sequence[float] = 0.0,
    lin_sup: bool = False,
) -> PolyExpAnsatz:
    r"""
    The squeezed vacuum state in the Bargmann representation.

    Args:
        r: The squeezing magnitude.
        phi: The squeezing angle.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The squeezed vacuum state in the Bargmann representation.
    """
    A, b, c = triples.squeezed_vacuum_state_Abc(r, phi)
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)


def squeezed_vacuum_state_fock(
    r: float | Sequence[float],
    phi: float | Sequence[float] = 0.0,
    shape: int | Sequence[int] | None = None,
    lin_sup: bool = False,
) -> ArrayAnsatz:
    r"""
    The squeezed vacuum state in the Fock representation.

    Args:
        r: The squeezing magnitude.
        phi: The squeezing angle.
        shape: The shape of the resulting Fock array.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The squeezed vacuum state in the Fock representation.
    """
    rs, phis = math.broadcast_arrays(
        r,
        phi,
    )
    batch_dims = rs.ndim
    ret = math.astensor(math.squeezed(rs, phis, shape=shape))
    if lin_sup:
        ret = math.sum(ret, axis=batch_dims - 1)
    return ArrayAnsatz(ret, batch_dims)


def thermal_state(
    nbar: float | Sequence[float],
    lin_sup: bool = False,
) -> PolyExpAnsatz:
    r"""
    The thermal state in the Bargmann representation.

    Args:
        nbar: The expected number of photons.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The thermal state in the Bargmann representation.
    """
    A, b, c = triples.thermal_state_Abc(nbar)
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)


def two_mode_squeezed_vacuum_state(
    r: float | Sequence[float],
    phi: float | Sequence[float] = 0.0,
    lin_sup: bool = False,
) -> PolyExpAnsatz:
    r"""
    The two-mode squeezed vacuum state in the Bargmann representation.

    Args:
        r: The squeezing magnitude.
        phi: The squeezing angle.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The two-mode squeezed vacuum state in the Bargmann representation.
    """
    A, b, c = triples.two_mode_squeezed_vacuum_state_Abc(r, phi)
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)


def vacuum_state(n_modes: int, lin_sup: bool = False) -> PolyExpAnsatz:
    r"""
    The `N`-mode vacuum state in the Bargmann representation.

    Args:
        n_modes: The number of modes.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The `N`-mode vacuum state in the Bargmann representation.
    """
    A, b, c = triples.vacuum_state_Abc(n_modes)
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)


def wigner_to_bargmann_psi(
    cov: RealMatrix,
    means: RealVector,
    lin_sup: bool = False,
) -> PolyExpAnsatz:
    r"""
    The Wigner ``Ket`` in the Bargmann representation.

    Args:
        cov: The covariance matrix of the state.
        means: The mean vector of the state.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The Wigner state in the Bargmann representation.
    """
    A, b, c = wigner_to_bargmann_psi_Abc(cov, means)
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)


def wigner_to_bargmann_rho(
    cov: RealMatrix,
    means: RealVector,
    lin_sup: bool = False,
) -> PolyExpAnsatz:
    r"""
    The Wigner ``DM`` in the Bargmann representation.

    Args:
        cov: The covariance matrix of the state.
        means: The mean vector of the state.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The Wigner ``DM`` in the Bargmann representation.
    """
    A, b, c = wigner_to_bargmann_rho_Abc(cov, means)
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)
