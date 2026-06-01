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

"""This module contains methods for creating built-in ``CircuitComponent`` utility ``Ansatz``."""

from __future__ import annotations

from collections.abc import Sequence

from mrmustard.physics import triples
from mrmustard.physics.ansatz import PolyExpAnsatz

__all__ = [
    "bargmann_to_quadrature",
    "bargmann_to_wigner",
    "displacement_map_s_parametrized",
]


def bargmann_to_quadrature(
    n_modes: int,
    phi: float | Sequence[float],
    lin_sup: bool = False,
) -> PolyExpAnsatz:
    r"""The Bargmann to quadrature transformation.

    Args:
        n_modes: The number of modes.
        phi: The quadrature angle.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The Bargmann to quadrature transformation.
    """
    A, b, c = triples.bargmann_to_quadrature_Abc(n_modes, phi)
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)


def bargmann_to_wigner(
    s: float | Sequence[float],
    n_modes: int,
    lin_sup: bool = False,
) -> PolyExpAnsatz:
    r"""The Bargmann to Wigner transformation.

    Args:
        s: The `s` parameter of this channel. The case `s=-1`  corresponds to Husimi, `s=0` to Wigner, and `s=1` to Glauber P function.
        n_modes: The number of modes.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The Bargmann to Wigner transformation.
    """
    A, b, c = triples.bargmann_to_wigner_Abc(s=s, n_modes=n_modes)
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)


def displacement_map_s_parametrized(
    s: float | Sequence[float],
    n_modes: int,
    lin_sup: bool = False,
) -> PolyExpAnsatz:
    r"""The `s`-parametrized displacement map.

    Args:
        s: The phase space parameter.
        n_modes: The number of modes.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The `s`-parametrized displacement map.
    """
    A, b, c = triples.displacement_map_s_parametrized_Abc(s=s, n_modes=n_modes)
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)
