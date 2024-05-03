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
The ``new`` physics module contains the representative theory of the quantum objects and all the physical propety calculations about them related to ``lab_dev``.

Different representations can work to represent a quantum object: such as the Fock basis (or number basis) represenatation, Bargmann representation (holomorphic quantum computing), phase space representation (the well-known Wigner function of a quantum state), characteristic phase space representation and etc.
The internal engine of MM is powered by the computation between Fock and Bargmann representations.
It is also capable to convert between different representations (by using the converters).

Under the level of the representation, we define some data structures to store the data related to the quantum object, which is called ``Ansatz``. 
``Ansatz`` is not only the data, but also includes the linear algebras of the data. We support three different ``Ansatz`` 

The bargmann triples of the objects are in triples.py.

The most important computation between representations is the inner product for Gaussian objects, and the logic functions are in gaussian_integrals.

Check out our guides to learn more about :mod:`~mrmustard.physics` and its core functionalities:

* The :mod:`~mrmustard.physics.ansatze` guide introduces the concept of an ansatz and three pre-defined ansatze.
* The :mod:`~mrmustard.physics.representations` guide how to initialize representations and two basic representations: :class:`~mrmustard.physics.representations.Fock` and :class:`~mrmustard.physics.representations.Bargmann`.
* The :mod:`~mrmustard.physics.converters` contains the conversion functions from one representation to :class:`~mrmustard.physics.representations.Fock` representation. The convert functions from wigner representation to :class:`~mrmustard.physics.representations.Bargmann` representation are stored in :mod:`~mrmustard.physics.bargmann`.
* The :mod:`~mrmustard.physics.triples` contains the triples in the :class:`~mrmustard.physics.representations.Bargmann` representation for all quantum objects.
* The :mod:`~mrmustard.physics.gaussian_integrals` contains the real and complex Gaussian integrals functions.
* Other modules are used in the ``lab``, which needs to be rearranged and well documented in the future.
* The functions in the init file of physics needs to be rearranged as well.
"""

from .ansatze import *
from .representations import *

from mrmustard.physics import fock, gaussian


# pylint: disable=protected-access
def fidelity(A, B) -> float:
    r"""Calculates the fidelity between two quantum states.

    Args:
        A (State) The first quantum state.
        B (State) The second quantum state.

    Returns:
        float: The fidelity between the two states.
    """
    if A.is_gaussian and B.is_gaussian:
        return gaussian.fidelity(A.means, A.cov, B.means, B.cov)
    return fock.fidelity(A.fock, B.fock, a_ket=A._ket is not None, b_ket=B._ket is not None)


def normalize(A):
    r"""Returns the normalized state.

    Args:
        A (State): the quantum state

    Returns:
        State: the normalized state
    """
    if A.is_gaussian:
        A._norm = 1.0
        return A

    if A.is_mixed:
        return A.__class__(dm=fock.normalize(A.dm(), is_dm=True))

    return A.__class__(ket=fock.normalize(A.ket(), is_dm=False))


def norm(A) -> float:
    r"""Calculates the norm of a quantum state.

    The norm is equal to the trace of the density matrix if the state
    is mixed and to the norm of the state vector if the state is pure.

    Args:
        A (State): the quantum state

    Returns:
        float: the norm of the state
    """
    if A.is_gaussian:
        return A._norm
    return fock.norm(A.fock, is_dm=A.is_mixed)


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


def von_neumann_entropy(A) -> float:
    r"""Calculates the Von Neumann entropy of a quantum state.

    Args:
        A (State) the quantum state

    Returns:
        float: the Von Neumann entropy of the state
    """
    if A.is_gaussian:
        return gaussian.von_neumann_entropy(A.cov)
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


def trace_distance(A, B) -> float:
    r"""Calculates the trace distance between two quantum states.

    Args:
        A (State) the first quantum state
        B (State) the second quantum state

    Returns:
        float: the trace distance between the two states
    """
    if A.is_gaussian and B.is_gaussian:
        return gaussian.trace_distance(A.means, A.cov, B.means, B.cov)
    return fock.trace_distance(A.fock, B.fock, a_dm=A.is_mixed, b_dm=B.is_mixed)
