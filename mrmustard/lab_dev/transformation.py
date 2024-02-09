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

"""This module contains the implementation of the :class:`State` class."""

from __future__ import annotations

from mrmustard import physics
from mrmustard import math
from mrmustard.physics.representations import Bargmann, Fock
from mrmustard.lab_dev.circuit_components import CircuitComponent
from mrmustard.utils.typing import Batch, ComplexTensor

class Transformation(CircuitComponent):
    r"""Note: Bare-bones implementation.
    Mixin class for quantum states. It supplies common functionalities and properties of Ket and DM.
    """

    def __getitem__(self, modes: int | tuple) -> CircuitComponent:
        r"""Re-initializes the state on an alternative set of modes"""
        modes = [modes] if isinstance(modes, int) else [i for i in modes]
        if len(modes) != self.num_modes:
            raise ValueError(f"modes must have length {self.num_modes}, got {len(modes)} instead")
        return self.__class__(self.representation, modes=modes)

    def bargmann(self):
        r"""Returns the bargmann parameters if available. Otherwise, the representation
        object raises an AttributeError.

        Returns:
            tuple[np.ndarray, np.ndarray, complex]: the bargmann parameters
        """
        if isinstance(self.representation, Bargmann):
            return self.representation.A, self.representation.b, self.representation.c
        raise AttributeError("Cannot convert to Bargmann representation.")



class Unitary(Transformation):
    """Density matrix class."""
    def __init__(self, representation: Bargmann | Fock, modes: list[int], name: str = "Unitary"):
        super().__init__(name=name, modes_out_ket=modes, modes_out_bra=modes, representation=representation)

    @classmethod
    def from_bargmann(cls, A, b, c, modes):
        return cls(Bargmann(A, b, c), modes)

    @classmethod
    def from_phasespace(cls, X, d, modes, name="Unitary"):
        r"""General constructor for density matrices from phase space representation.

        Args:
            X (Batch[ComplexMatrix]): symplectic / transformation matrix
            d (int): displacement
            modes (Sequence[int]): the modes of the state
            name (str): the name of the transformation
        """
        A, b, c = physics.bargmann.wigner_to_bargmann_U(X, d)
        return cls(Bargmann(A, b, c), modes, name=name)

    def phasespace(self):
        r"""Returns the phase space parameters (cov, means)

        Returns:
            X, d: the symplectic / transformation matrix, the noise matrix, and the displacement
        """
        raise NotImplementedError("use kernel that you calculated with Yuan")

    @classmethod
    def from_fock(cls, fock_array, modes, batched=False, name="DM"):
        return cls(Fock(fock_array, batched), modes, name=name)

    # will be cached if backend is numpy
    def fock(self, shape: list[int] = None, prob: float = None) -> Batch[ComplexTensor]:
        r"""Returns the fock array of the density matrix.

        Args:
            shape (optional List[int]): The shape of the desired Fock tensor. If None it is guessed automatically.
            prob (optional float): The maximum probability of the state. Defaults to settings.AUTOCUTOFF_PROBABILITY (99.99%).

        Returns:
            State: the converted state with the target Fock Representation
        """
        raise NotImplementedError

    def __rshift__(self, other: CircuitComponent) -> Unitary | Channel | CircuitComponent:
        r"""self is contracted with other on matching modes and ket/bra sides.
        It adds a bra if needed (e.g. unitary >> channel is treated as
        (unitary.adjoint & unitary) >> channel).
        """
        common = set(self.wires.output.modes).intersection(other.wires.input.modes)
        # it's important to check common to avoid adding unnecessary adjoints
        if not other.wires[common].input.bra or not other.wires[common].input.ket:
            return (self @ other) @ other.adjoint  # order doesn't matter as bra/ket wires don't overlap
        return self @ other



class Channel(Transformation):
    def __init__(self, representation: Bargmann | Fock, modes: list[int], name: str = "Channel"):
        super().__init__(name=name, modes_out_ket=modes, representation=representation)

    @classmethod
    def from_bargmann(cls, A, b, c, modes, name="Ch"):
        return cls(Bargmann(A, b, c), modes, name=name)

    @classmethod
    def from_phasespace(cls, X, Y, d, modes, name="Ch"):
        r"""General constructor for channels from their phase space representation.

        Args:
            X (Batch[ComplexMatrix]): the symplectic matrix
            Y (Batch[ComplexMatrix]): the noise matrix
            d (Batch[ComplexVector]): the displacement
            modes (Sequence[int]): the modes of the state
            name (str): the name of the state
        """
        A, b, c = physics.bargmann.wigner_to_bargmann_Choi(X, Y, d)
        return cls(Bargmann(A, b, c), modes, name=name)

    # will be cached if backend is numpy
    def phasespace(self):
        r"""Returns the s-parametrized phase space representation. Note that this actually is
        the s-parametrized phase space representation of the pure density matrix |self><self|.

        Returns:
            X, Y, d: the symplectic matrix, the noise matrix, and the displacement
        """
        raise NotImplementedError("use kernel that you calculated with Yuan")

    @classmethod
    def from_fock(cls, fock_array, modes, batched=False, name="Ch"):
        return cls(Fock(fock_array, batched), modes, name=name)

    # will be cached if backend is numpy
    def fock(self, shape: list[int]):
        r"""Converts the channel to fock representation.

        Args:
            shape (List[int]): The shape of the desired Fock tensor. If None it is guessed automatically.

        Returns:
            Array: the fock representation of the channel
        """
        if isinstance(self.representation, Fock):
            return  math.sum(self.representation.array, axis=0)
        arrays = [math.hermite_renormalized(A, b, c, shape) for A,b,c in zip(self.bargmann)]
        return math.sum(arrays, axis=0)
    
    def __rshift__(self, other: CircuitComponent) -> Ket | CircuitComponent:
        r"""self is contracted with other on matching modes and ket/bra sides.
        It adds a bra if needed (e.g. ket >> channel is treated as
        (ket.adjoint & ket) >> channel).
        """
        common = set(self.wires.output.modes).intersection(other.wires.input.modes)
        # it's important to check common to avoid adding unnecessary adjoints
        bras_incompatible = bool(other.wires[common].input.bra) != bool(self.wires[common].output.bra) 
        kets_incompatible = bool(other.wires[common].input.ket) != bool(self.wires[common].output.ket)
        if bras_incompatible or kets_incompatible:
            return (self @ other) @ other.adjoint  # order doesn't matter as bra/ket wires don't overlap
        return self @ other
