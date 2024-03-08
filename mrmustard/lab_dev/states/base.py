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
This module contains the base classes for the available quantum states.

In the docstrings defining the available states we provide a definition in terms of
the covariance matrix :math:`V` and the vector of means :math:`r`. Additionally, we
provide the ``(A, b, c)`` triples that define the states in the Fock Bargmann
representation.
"""

from __future__ import annotations

from typing import Optional, Sequence

from mrmustard import math, settings
from mrmustard.utils.typing import ComplexMatrix, ComplexTensor, ComplexVector
from mrmustard.physics.gaussian import purity
from mrmustard.physics.bargmann import wigner_to_bargmann_psi
from mrmustard.physics.representations import Bargmann, Fock
from ..circuit_components import CircuitComponent
from ..transformations.transformations import Unitary, Channel

__all__ = ["State", "DM", "Ket"]


class State(CircuitComponent):
    r"""
    Base class for all states.
    """


class DM(State):
    r"""
    Base class for density matrices.

    Args:
        name: The name of this state.
        modes: The modes of this state.
    """

    def __init__(self, name: Optional[str] = None, modes: Optional[Sequence[int]] = None):
        modes = modes or []
        name = name or ""
        super().__init__(name, modes_out_bra=modes, modes_out_ket=modes)

    def __rshift__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Contracts ``self`` and ``other`` as it would in a circuit, adding the adjoints when
        they are missing.

        Returns a ``DM`` when ``other`` is a ``Unitary`` or a ``Channel``, and ``other`` acts on
        ``self``'s modes. Otherwise, it returns a ``CircuitComponent``.
        """
        component = super().__rshift__(other)

        if isinstance(other, (Unitary, Channel)) and set(other.modes).issubset(self.modes):
            dm = DM()
            dm._wires = component.wires
            dm._representation = component.representation
            return dm
        return component

    def __repr__(self) -> str:
        return super().__repr__().replace("CircuitComponent", "DM")


class Ket(State):
    r"""
    Base class for all pure states, potentially unnormalized.

    Arguments:
        name: The name of this state.
        modes: The modes of this states.
    """

    def __init__(self, name: Optional[str] = None, modes: Optional[Sequence[int]] = None):
        modes = modes or []
        name = name or ""
        super().__init__(name, modes_out_ket=modes)

    @classmethod
    def from_bargmann(
        cls,
        modes: Sequence[int],
        triple: tuple[ComplexMatrix, ComplexVector, complex],
        name: Optional[str] = None,
    ) -> Ket:
        r"""
        Returns a ``Ket`` from an ``(A, b, c)`` triple defining a Bargmann representation.

        .. code-block::

            >>> from mrmustard.physics.representations import Bargmann
            >>> from mrmustard.physics.triples import coherent_state_Abc
            >>> from mrmustard.lab_dev import Ket

            >>> modes = [0, 1]
            >>> triple = coherent_state_Abc(x=[0.1, 0.2])

            >>> coh = Ket.from_bargmann(modes, triple)
            >>> assert coh.modes == modes
            >>> assert coh.representation == Bargmann(*triple)

        Args:
            modes: The modes of this states.
            triple: The ``(A, b, c)`` triple.
            name: The name of this state.

        Returns:
            A ``Ket`` state.

        Raises:
            ValueError: If the ``A`` or ``b`` have a shape that is inconsistent with
                the number of modes.
        """
        A = math.astensor(triple[0])
        b = math.astensor(triple[1])
        c = math.astensor(triple[2])

        n_modes = len(modes)
        if A.shape != (n_modes, n_modes) or b.shape != (n_modes,):
            msg = f"Given triple is inconsistent with modes=``{modes}``."
            raise ValueError(msg)

        ret = Ket(name, modes)
        ret._representation = Bargmann(A, b, c)
        return ret

    @classmethod
    def from_fock(
        cls,
        modes: Sequence[int],
        array: ComplexTensor,
        name: Optional[str] = None,
    ) -> Ket:
        r"""
        Returns a ``Ket`` from an array describing the state in the Fock representation.

        .. code-block::

            >>> from mrmustard.physics.representations import Fock
            >>> from mrmustard.physics.triples import coherent_state_Abc
            >>> from mrmustard.lab_dev import Coherent, Ket

            >>> modes = [0]
            >>> array = Coherent(modes, x=0.1).to_fock().representation.array
            >>> coh = Ket.from_fock(modes, array)

            >>> assert coh.modes == modes
            >>> assert coh.representation == Fock(array)

        Args:
            modes: The modes of this states.
            triple: The ``(A, b, c)`` triple.
            name: The name of this state.

        Returns:
            A ``Ket`` state.

        Raises:
            ValueError: If the given array has a shape that is inconsistent with the number of
                modes.
        """
        array = math.astensor(array)

        n_modes = len(modes)
        if len(array.shape) != n_modes:
            msg = f"Given array is inconsistent with modes=``{modes}``."
            raise ValueError(msg)

        ret = Ket(name, modes)
        ret._representation = Fock(array)
        return ret

    def from_phasespace(
        cls,
        modes: Sequence[int],
        cov: ComplexMatrix,
        mean: ComplexMatrix,
        name: Optional[str] = None,
        check_purity: bool = True,
    ):
        r"""General constructor for kets in phase space representation.

        Args:
            cov: The covariance matrix
            mean (Batch[ComplexVector]): The vector of means.
            modes: The modes of this states.
            triple: The ``(A, b, c)`` triple.
            name: The name of this state.
            check_purity: Whether to check if the state is pure.

        Returns:
            A ``Ket`` state.

        Raises:
            ValueError
        """
        cov = math.astensor(cov)
        mean = math.astensor(mean)

        # n_modes = len(modes)
        # if len(mean.shape) != n_modes:
        #     msg = f"Given array is inconsistent with modes=``{modes}``."
        #     raise ValueError(msg)
        
        if check_purity:
            p = purity(cov)
            if p < 1.0 - settings.ATOL_PURITY:
                msg = f"Cannot initialize a ket: purity is {purity:.3f} (must be 1.0)."
                raise ValueError(msg)
        
        ret = Ket(name, modes)
        ret._representation = Bargmann(wigner_to_bargmann_psi(cov, mean))
        return ret

    def __rshift__(self, other: CircuitComponent) -> CircuitComponent:
        r"""
        Contracts ``self`` and ``other`` as it would in a circuit, adding the adjoints when
        they are missing.

        Returns a ``State`` (either ``Ket`` or ``DM``) when ``other`` is a ``Unitary`` or a
        ``Channel``, and ``other`` acts on ``self``'s modes. Otherwise, it returns a
        ``CircuitComponent``.
        """
        component = super().__rshift__(other)

        if isinstance(other, Unitary) and set(other.modes).issubset(set(self.modes)):
            ket = Ket()
            ket._wires = component.wires
            ket._representation = component.representation
            return ket
        elif isinstance(other, Channel) and set(other.modes).issubset(set(self.modes)):
            dm = DM()
            dm._wires = component.wires
            dm._representation = component.representation
            return dm
        return component

    def __repr__(self) -> str:
        return super().__repr__().replace("CircuitComponent", "Ket")
