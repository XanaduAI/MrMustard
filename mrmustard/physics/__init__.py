# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
The physics module contains high-level functions for performing calculations on quantum objects.
It allows for objects with different representations to be used seamlessly in the same calculation, e.g., ``fidelity(A, B)`` works whether A and B are in the Fock or Gaussian representation or a mix of both.

All the functions are automatically differentiated and can be used in conjunction with an
optimization routine.
"""

from mrmustard.physics import fock, gaussian
from mrmustard import settings


def fidelity(A, B) -> float:
    r"""Calculates the fidelity between two quantum states.

    Args:
        A (State) The first quantum state.
        B (State) The second quantum state.

    Returns:
        float: The fidelity between the two states.
    """
    if A.is_gaussian and B.is_gaussian:
        return gaussian.fidelity(A.means, A.cov, B.means, B.cov, settings.HBAR)
    return fock.fidelity(A.fock, B.fock, a_dm=A._dm is not None, b_dm=B._dm is not None)


def overlap(A, B) -> float:
    r"""Calculates the overlap between two quantum states.

    If the states are both pure it returns :math:`|<A|B>|^2`, if one is mixed it returns :math:`<A|B|A>`
    and if both are mixed it returns :math:`Tr[AB]`.

    Args:
        A (State) the first quantum state
        B (State) the second quantum state

    Returns:
        float: the overlap between the two states
    """
    raise NotImplementedError
    if A.is_gaussian and B.is_gaussian:
        return gaussian.overlap(A.means, A.cov, B.means, B.cov, settings.HBAR)
    return fock.overlap(A.fock, B.fock, a_dm=A.is_mixed, b_dm=B.is_mixed)


def von_neumann_entropy(A) -> float:
    r"""Calculates the Von Neumann entropy of a quantum state.

    Args:
        A (State) the quantum state

    Returns:
        float: the Von Neumann entropy of the state
    """
    if A.is_gaussian:
        return gaussian.von_neumann_entropy(A.cov, settings.HBAR)
    return fock.von_neumann_entropy(A.fock, a_dm=A.is_mixed)


def relative_entropy(A, B) -> float:
    r"""Calculates the relative entropy between two quantum states.

    Args:
        A (State) the first quantum state
        B (State) the second quantum state

    Returns:
        float: the relative entropy between the two states
    """
    raise NotImplementedError
    if A.is_gaussian and B.is_gaussian:
        return gaussian.relative_entropy(A.means, A.cov, B.means, B.cov, settings.HBAR)
    return fock.relative_entropy(A.fock, B.fock, a_dm=A.is_mixed, b_dm=B.is_mixed)


def trace_distance(A, B) -> float:
    r"""Calculates the trace distance between two quantum states.

    Args:
        A (State) the first quantum state
        B (State) the second quantum state

    Returns:
        float: the trace distance between the two states
    """
    raise NotImplementedError
    if A.is_gaussian and B.is_gaussian:
        return gaussian.trace_distance(A.means, A.cov, B.means, B.cov, settings.HBAR)
    return fock.trace_distance(A.fock, B.fock, a_dm=A.is_mixed, b_dm=B.is_mixed)
