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
from multiprocessing import Value

import numpy as np

from mrmustard import physics
from mrmustard import settings
from mrmustard import math
from mrmustard.physics.representations import Bargmann, Fock
from mrmustard.lab_dev.circuit_components import CircuitComponent
from mrmustard.utils.typing import Batch, ComplexTensor

class State(CircuitComponent):
    r"""Note: Bare-bones implementation.
    Mixin class for quantum states. It supplies common functionalities and properties of Ket and DM.
    """
    @property
    def is_pure(self):
        r"""Returns whether the state is pure."""
        return np.isclose(self.purity, 1.0)  # children must provide purity

    @property # will be @lazy_if_numpy
    def L2_norm(self) -> float:
        r"""Returns the L2 norm of the Hilbert space vector or the Hilbert-Schmidt norm of a density matrix."""
        return float((self >> self.dual).representation)

    def __getitem__(self, modes: int | tuple) -> State:
        r"""Re-initializes the state on an alternative set of modes"""
        modes = [modes] if isinstance(modes, int) else [i for i in modes]
        if len(modes) != self.num_modes:
            raise ValueError(f"modes must have length {self.num_modes}, got {len(modes)} instead")
        return self.__class__(self.representation, modes=modes)

    def __eq__(self, other) -> bool:
        r"""Returns whether the states are equal. Modes and all."""
        return self.representation == other.representation and self.modes == other.modes

    def __lshift__(self, other: State) -> CircuitComponent | complex:
        r"""dual of __rshift__"""
        return (other.dual >> self.dual).dual

    def __add__(self, other: State):
        r"""Implements addition of states. The meaning will be superposition in Hilbert space
        if the states are Kets and mixture in the case of Density Matrices."""
        if self.modes != other.modes:
            raise ValueError(f"Can't add states on different modes (got {self.modes} and {other.modes})")
        return self.__class__(self.representation + other.representation, modes=self.modes)

    def __rmul__(self, other: Union[int, float, complex]):
        r"""Implements multiplication from the left if the object on the left
        does not implement __mul__ for the type of self.

        E.g., ``0.5 * psi``.
        """
        assert isinstance(other, (int, float, complex))
        return self.__class__(other * self.representation, modes=self.modes)

    def __mul__(self, other):
        r"""Implements multiplication of two objects."""
        if isinstance(other, (int, float, complex)):
            return other * self
        modes = [m for m in self.modes if m not in other.modes] + [
            m for m in other.modes if m not in self.modes
        ]
        return self.__class__(self.representation * other.representation, modes=modes)

    def __truediv__(self, other: Union[int, float, complex]):
        r"""Implements division by a scalar.

        E.g. ``psi / 0.5``
        """
        assert isinstance(other, (int, float, complex))
        return self.__class__(self.representation / other, modes=self.modes)

    def bargmann(self):
        r"""Returns the bargmann parameters if available. Otherwise, the representation
        object raises an AttributeError.

        Returns:
            tuple[np.ndarray, np.ndarray, complex]: the bargmann parameters
        """
        if isinstance(self.representation, Bargmann):
            return self.representation.A, self.representation.b, self.representation.c
        raise AttributeError("Cannot convert to Bargmann representation.")



class DM(State):
    """Density matrix class."""
    def __init__(self, representation: Bargmann | Fock, modes: list[int], name: str = "DM"):
        super().__init__(name=name, modes_out_ket=modes, modes_out_bra=modes, representation=representation)

    @property # will be @lazy_if_numpy
    def probability(self) -> float:
        traced = self.representation.trace(self.wires.output.ket.indices, self.wires.output.bra.indices)
        return float(traced)

    @property # will be @lazy_if_numpy
    def purity(self) -> float:
        normalized = self / self.probability
        return normalized.L2_norm

    @classmethod
    def from_bargmann(cls, A, b, c, modes):
        return cls(Bargmann(A, b, c), modes)

    @classmethod
    def from_phasespace(cls, cov, means, modes, name="DM"):
        r"""General constructor for density matrices from phase space representation.

        Args:
            cov (Batch[ComplexMatrix]): the covariance matrix
            means (Batch[ComplexVector]): the vector of means
            modes (Sequence[int]): the modes of the state
            name (str): the name of the state
        """
        A, b, c = physics.bargmann.wigner_to_bargmann_rho(cov, means) # (s, characteristic) not implemented yet
        return cls(Bargmann(A, b, c), modes, name=name)

    def phasespace(self):
        r"""Returns the phase space parameters (cov, means)

        Returns:
            cov, means: the covariance matrix and the vector of means
        """
        raise NotImplementedError("need to calculate the siegel-weil kernel")


    @classmethod
    def from_fock(cls, fock_array, modes, batched=False, name="DM"):
        return cls(Fock(fock_array, batched), modes, name=name)

    @classmethod
    def from_quadrature(self):
        raise NotImplementedError

    # will be cached if backend is numpy
    def fock(self, shape: list[int] = None, prob: float = None) -> Batch[ComplexTensor]:
        r"""Returns the fock array of the density matrix.

        Args:
            shape (optional List[int]): The shape of the desired Fock tensor. If None it is guessed automatically.
            prob (optional float): The maximum probability of the state. Defaults to settings.AUTOCUTOFF_PROBABILITY (99.99%).

        Returns:
            State: the converted state with the target Fock Representation
        """
        if isinstance(self.representation, Fock):
            return math.sum(self.representation.array)
        if shape is None:  # NOTE: we will support optional ints in the shape as well
            cutoffs = physics.fock.autocutoffs(*self.phasespace(), prob or settings.AUTOCUTOFF_PROBABILITY)
            shape = cutoffs + cutoffs  # NOTE: in the future we can use a shape for each batch element
        arrays = [math.hermite_renormalized(A, b, c, shape) for A,b,c in zip(self.bargmann)]
        return math.sum(arrays, axis=0)

    def __and__(self, other: State) -> State:
        r"""Tensor product of two states. Use sparingly: the combined DM can be a large object."""
        ids = (self.wires + other.wires).ids  # eventual exceptions are raised here
        representation = self.representation @ other.representation
        representation = representation.reorder([ids.index(id) for id in self.wires.ids + other.wires.ids])
        return self.__class__(representation, modes=sorted(set(self.modes).union(other.modes)))

    def __rshift__(self, other: CircuitComponent) -> DM | CircuitComponent:
        r"""self is contracted with other on matching modes and ket/bra sides.
        It adds a bra if needed (e.g. ket >> channel is treated as
        (ket.adjoint & ket) >> channel).
        """
        common = set(self.wires.output.modes).intersection(other.wires.input.modes)
        # it's important to check common to avoid adding unnecessary adjoints
        if not other.wires[common].input.bra or not other.wires[common].input.ket:
            return (self @ other) @ other.adjoint  # order doesn't matter as bra/ket wires don't overlap
        return self @ other




class Ket(State):
    def __init__(self, representation: Bargmann | Fock, modes: list[int], name: str = "Ket"):
        super().__init__(name=name, modes_out_ket=modes, representation=representation)

    @property # will be @lazy_if_numpy
    def probability(self) -> float:
        return self.L2_norm

    @property
    def purity(self) -> float:
        return 1.0

    @classmethod
    def from_bargmann(cls, A, b, c, modes, name="Ket"):
        return cls(Bargmann(A, b, c), modes, name=name)

    @classmethod
    def from_phasespace(cls, cov, mean, modes, name="Ket", pure_check=True):
        r"""General constructor for kets in phase space representation.

        Args:
            cov (Batch[ComplexMatrix]): the covariance matrix
            mean (Batch[ComplexVector]): the vector of means
            modes (Sequence[int]): the modes of the state
            name (str): the name of the state
            pure_check (bool): whether to check if the state is pure (default: True)
        """
        if pure_check and (purity := physics.gaussian.purity(cov)) < 1.0 - settings.ATOL_PURITY:
            raise ValueError(f"Cannot initialize a ket: purity is {purity:.3f} which is less than 1.0.")
        A, b, c = physics.bargmann.wigner_to_bargmann_psi(cov, mean)
        return cls(Bargmann(A, b, c), modes, name=name)

    # will be cached if backend is numpy
    def phasespace(self):
        r"""Returns the s-parametrized phase space representation. Note that this actually is
        the s-parametrized phase space representation of the pure density matrix |self><self|.

        Returns:
            cov, means: the covariance matrix and the vector of means
        """
        return (self.adjoint & self).phasespace()

    @classmethod
    def from_fock(cls, fock_array, modes, batched=False, name="Ket"):
        return cls(Fock(fock_array, batched), modes, name=name)

    # will be cached if backend is numpy
    def fock(
        self,
        shape: Optional[List[int]] = None,
        max_prob: Optional[float] = None,
    ):
        r"""Converts the representation of the Ket to Fock representation.

        Args:
            shape (optional List[int]): The shape of the desired Fock tensor. If None it is guessed automatically.
            max_prob (optional float): The maximum probability of the state. Defaults to settings.AUTOCUTOFF_PROBABILITY.
                (used to stop the calculation of the amplitudes early)

        Returns:
            State: the converted state with the target Fock Representation
        """
        if isinstance(self.representation, Fock):
            return  math.sum(self.representation.array, axis=0)
        if shape is None:
            shape = physics.fock.autocutoffs(*self.phasespace(), max_prob or settings.AUTOCUTOFF_PROBABILITY)
        arrays = [math.hermite_renormalized(A, b, c, shape) for A,b,c in zip(self.bargmann)]
        return math.sum(arrays, axis=0)

    def from_quadrature(self, cov, means, modes, name="Ket"):
        r"""General constructor for kets in quadrature representation."""
        A, b, c = physics.bargmann.ket_from_quadrature(cov, means)
        return self.__class__(Bargmann(A, b, c), modes, name=name)
    
    # will be cached if backend is numpy
    def quadrature(self, angle: float) -> tuple[Matrix, Vector]:
        r"""Returns the state converted to quadrature (wavefunction) representation with the given quadrature angle.
        Use angle=0 for the position quadrature and angle=pi/2 for the momentum quadrature.

        Args:
            angle (optional float): The quadrature angle.

        Returns:
            tuple[Matrix, Vector]: the quadrature representation of the state
        """
        raise NotImplementedError

    def __and__(self, other: Ket) -> Ket:  # TODO: needs to be in Ket and DM
        r"""Tensor product of two states."""
        if not isinstance(other, Ket):
            raise ValueError("Can only tensor with other Kets")
        if not set(self.wires.modes).isdisjoint(other.wires.modes):
            raise ValueError("Modes must be disjoint")
        representation = self.representation & other.representation
        ids = (self.wires >> other.wires).ids
        representation = representation.reorder([ids.index(id) for id in self.wires.ids+other.wires.ids])
        return self.__class__(representation, modes=sorted(self.modes + other.modes))
    
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