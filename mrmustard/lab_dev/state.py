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

from typing import (
    TYPE_CHECKING,
    Iterable,
    Optional,
    Union,
)

import numpy as np

from mrmustard import physics
from mrmustard import settings
from mrmustard.lab_dev.transformations import Transformation
from mrmustard.lab_dev.utils import lazy_if_numpy
from mrmustard import math
from mrmustard.physics.representations import Representation, Bargmann, Fock
from mrmustard.lab_dev.circuit_components import CircuitComponent


class State(CircuitComponent):
    r"""Mixin class for quantum states. It supplies common functionalities and properties of all states."""

    @property
    def is_mixed(self):
        r"""Returns whether the state is mixed."""
        return not self.is_pure

    @property
    def is_pure(self):
        r"""Returns whether the state is pure."""
        return np.isclose(self.purity, 1.0, atol=settings.PURITY_ATOL)

    @lazy_if_numpy
    def L2_norm(self) -> float:
        r"""Returns the L2 norm of the Hilbert space vector or the Hilbert-Schmidt norm of a density matrix."""
        return self >> self.dual

    def __getitem__(self, modes: int | Iterable) -> State:
        r"""Re-initializes the state on an alternative set of modes"""
        modes = [modes] if isinstance(modes, int) else [i for i in modes]
        if len(modes) != self.num_modes:
            raise ValueError(f"modes must have length {self.num_modes}, got {len(modes)} instead")
        return self.__class__(self.representation, modes=modes)

    def __eq__(self, other) -> bool:  # pylint: disable=too-many-return-statements
        r"""Returns whether the states are equal. Modes and all."""
        return self.representation == other.representation and self.modes == other.modes

    def __lshift__(self, other: State) -> State | complex:
        r"""dual of __rshift__"""
        return (other.dual >> self.dual).dual  # >> is implemented in DM and Ket

    def __add__(self, other: State):
        r"""Implements addition of states."""
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
        r"""Returns the bargmann parameters.

        Returns:
            tuple[np.ndarray, np.ndarray, complex]: the bargmann parameters
        """
        return self.representation.A, self.representation.b, self.representation.c

    @staticmethod  # TODO: move away from here?
    def _format_probability(prob: Optional[float]) -> str:
        if prob is None:
            return "None"
        if prob < 0.001:
            return f"{100*prob:.3e} %"
        else:
            return f"{prob:.3%}"

    def _repr_markdown_(self):
        r"""Prints the table to show the properties of the state."""
        purity = self.purity
        probability = self.state_probability
        num_modes = self.representation.num_modes
        bosonic_size = "1" if isinstance(self.representation, (WignerKet, WignerDM)) else "N/A"
        representation = self.representation.name
        table = (
            f"#### {self.__class__.__qualname__}\n\n"
            + "| Purity | Probability | Num modes | Bosonic size | Representation |\n"
            + "| :----: | :----: | :----: | :----: | :----: |\n"
            + f"| {purity :.2e} | "
            + self._format_probability(probability)
            + f" | {num_modes} | {bosonic_size} | {representation} |"
        )

        if self.num_modes == 1:
            graphics.mikkel_plot(math.asnumpy(self.to_Fock(0.999).representation))

        return table
    
    @lazy_if_numpy
    def cov(self):
        r"""Covariance matrix of the Gaussian distribution. Available only if the representation is continuous."""
        return Abc_to_sigmamu(*self.phasespace(1, False))[0]

    @lazy_if_numpy
    def mean(self):  # TODO check if this is correct
        r"""Mean of the Gaussian distribution. Available only if the representation is continuous."""
        return Abc_to_sigmamu(*self.phasespace(1, False))[1]



class DM(State):
    def __init__(self, representation: Bargmann | Fock, modes: list[int], name: str = "DM"):
        self._representation = representation
        super().__init__(name=name, modes_out_ket=modes, modes_out_bra=modes)

    @lazy_if_numpy
    def probability(self) -> float:  
        return self.representation.trace(self.wires.output.ket.indices, self.wires.output.bra.indices)

    @lazy_if_numpy
    def purity(self) -> float:
        normalized = self / self.probability
        return normalized.L2_norm
    
    @classmethod
    def from_bargmann(cls, A, b, c, modes):
        return cls(Bargmann(A, b, c), modes)

    @classmethod
    def from_phasespace(cls, cov, means, modes, s=1, characteristic=False, name="DM"):
        r"""General constructor for density matrices from phase space representation.

        Args:
            cov (Batch[ComplexMatrix]): the covariance matrix
            means (Batch[ComplexVector]): the vector of means
            modes (Sequence[int]): the modes of the state
            s (float): the parameter of the phase space representation
            characteristic (bool): whether from the characteristic function
            name (str): the name of the state
        """
        A, b, c = physics.bargmann.dm_from_phase_space(cov, means, s, characteristic)
        # TODO replace with more general Bargmann repr (e.g. poly x exp) when ready
        return cls(Bargmann(A, b, c), modes, name=name)

    @lazy_if_numpy
    def phasespace(self, s=1, characteristic=False):
        r"""Returns the density matrix converted to s-parametrized phase space representation.

        Args:
            s (optional float): The parameter s of the phase space representation. Defaults to 1 (Wigner).
                Use s=0 for the Husimi Q function and s=-1 for the Glauber-Sudarshan P function.
            characteristic (optional bool): Whether to compute to the characteristic function. Defaults to False.

        Returns:
            DM: the converted DM in the phase space representation
        """
        Abc = (self >> SWKernel(s, characteristic, modes=self.modes)).bargmann()
        return Abc_to_sigmamu(Abc)
    
    @classmethod
    def from_quadrature(cls, cov, means, modes, name="DM"):
        r"""General constructor for density matrices from quadrature representation."""
        A, b, c = physics.bargmann.dm_from_quadrature(cov, means)
        return cls(Bargmann(A, b, c), modes, name=name)
    
    @lazy_if_numpy
    def quadrature(self, angle: float) -> tuple[Matrix, Vector]:
        r"""Returns the state converted to quadrature (wavefunction) representation with the given quadrature angle.
        Use angle=0 for the position quadrature and angle=pi/2 for the momentum quadrature.

        Args:
            angle (optional float): The quadrature angle.

        Returns:
            tuple[Matrix, Vector]: the quadrature representation of the state
        """
        # transform Abc as in Yuan's notes
        # call physics.bargmann.real_gaussian_integral (get it back from previous commits)
        # Abc -> sigma,mu
        raise NotImplementedError

    @classmethod
    def from_fock(cls, fock_array, modes, name="DM"):
        return cls(Fock(fock_array), modes, name=name)
    
    @lazy_if_numpy
    def fock(self, shape: Optional[List[int]] = None):
        r"""Converts the representation of the DM to Fock representation.

        Args:
            shape (optional List[int]): The shape of the desired Fock tensor. If None it is guessed automatically.

        Returns:
            State: the converted state with the target Fock Representation
        """
        if isinstance(self.representation, Fock):
            return self.array
        if shape is None:
            cutoffs = physics.fock.autocutoffs(*self.phasespace(1,False), 0.9999)
            shape = cutoffs + cutoffs
        return math.hermite_renormalized(self.representation.A, self.representation.b, self.representation.c, shape)

    def substate(self, modes: list[int]) -> State:
        return DM((self & self.adjoint).representation.trace(keep=modes), modes=modes)

    def __and__(self, other: State) -> State:
        r"""Tensor product of two states."""
        if not set(self.wires.modes).isdisjoint(other.wires.modes):
            raise ValueError("Wires must be disjoint")
        representation = self.representation & other.representation
        ids = (self.wires + other.wires).ids
        representation = representation.reorder([ids.index(id) for id in self.wires.ids+other.wires.ids])
        return self.__class__(representation, modes=sorted(self.modes + other.modes))

    def __rshift__(self, other: CircuitComponent) -> DM | CircuitComponent:
        r"""If `other` is a transformation, it is applied to self, e.g. ``Thermal(0.5) >> Sgate(r=0.1)``.
        If other is a dual State (i.e. a povm element), self is projected onto it, e.g. ``Gaussian(2) >> Coherent(x=0.1).dual``.
        """
        # 1) check if rshift is possible
        common = set(self.wires.output.modes).intersection(other.wires.input.modes)
        if not (set(self.wires.output.modes) - common).isdisjoint(other.wires.output.modes):
            raise ValueError("Output modes must be disjoint")
        if not set(self.wires.input.modes).isdisjoint(set(other.wires.input.modes) - common):
            raise ValueError("Input modes must be disjoint")
        # 2) add bra/ket if needed
        if other.wires[common].input.bra:  # we want to check common to avoid unnecessary adjoints
            return (self & self.adjoint) >> other
        if other.wires[common].input.ket:  # rarer case, but possible
            return self >> (other.adjoint & other)
        self = self.light_copy()  # new wires
        other = other.light_copy()  # new wires
        self.wires[common].output.ids = other.wires[common].input.ids
        self_idx = self.wires[common].output.indices
        other_idx = other.wires[common].input.indices
        # 3) convert bargmann->fock if needed
        if isinstance(self.representation, Bargmann) and isinstance(other.representation, Fock):
            shape = [s if i in self_idx else None for i, s in enumerate(other.representation.shape)]
            self._representation = self.representation.to_fock(shape=shape)
        if isinstance(self.representation, Fock) and isinstance(other.representation, Bargmann):
            shape = [s if i in other_idx else None for i, s in enumerate(self.representation.shape)]
            other._representation = other.representation.to_fock(shape=shape)
        # 4) apply rshift
        new_repr = self.representation[self_idx] @ other.representation[other_idx]
        before_ids = [id for id in self.wires.ids+other.wires.ids if id not in self.wires[common].output.ids]
        after_ids = (self.wires >> other.wires).ids  # automatically yields the right order of ids
        order = [before_ids.index(id) for id in after_ids]
        new_repr = new_repr.reorder(order)
        return self.__class__(representation=new_repr, modes=self.modes)






class Ket(State):
    def __init__(self, representation: Bargmann | Fock, modes, name):
        self._representation = representation
        super().__init__(name=name, modes_out_ket=modes)

    @lazy_if_numpy
    def probability(self) -> float:
        return self.L2_norm()

    @property
    def purity(self) -> float:
        return 1.0

    @classmethod
    def from_bargmann(cls, A, b, c, modes, name="Ket"):
        B = Bargmann(A, b, c)
        n = B.A.shape[-1]
        if n // 2 != len(modes) or n % 2 == 1:
            raise ValueError(f"A matrix and/or modes are wrong")
        return cls(B, modes, name=name)

    @classmethod
    def from_phasespace(
        cls, cov, mean, modes, s=1, characteristic=True, name="Ket", pure_check=True
    ):
        r"""General constructor for kets in phase space representation.

        Args:
            cov (Batch[ComplexMatrix]): the covariance matrix
            mean (Batch[ComplexVector]): the vector of means
            modes (Sequence[int]): the modes of the state
            s (float): the parameter of the phase space representation
            characteristic (bool): whether to compute the characteristic function
            name (str): the name of the state
            pure_check (bool): whether to check if the state is pure (default: True)
        """
        if pure_check and physics.gaussian.purity(cov) < 1.0 - settings.PURITY_ATOL:
            raise ValueError("Initializing a Ket using a mixed state is not allowed")
        A, b, c = physics.bargmann.ket_from_phase_space(cov, mean, s, characteristic)
        return cls(Bargmann(A, b, c), modes, name=name)

    @lazy_if_numpy
    def phasespace(self, s=1, characteristic=False):
        r"""Returns the density matrix converted to s-parametrized phase space representation.
        Note that this actually is the s-parametrized phase space representation of the
        pure density matrix |self><self|.

        Args:
            s (optional float): The parameter s of the phase space representation. Defaults to 1 (Wigner).
                Use s=0 for the Husimi Q function and s=-1 for the Glauber-Sudarshan P function.
            characteristic (optional bool): Whether to compute to the characteristic function. Defaults to False.

        Returns:
            DM: the converted DM in the phase space representation
        """
        Abc = ((self & self.adjoint) >> SWKernel(s, characteristic, modes=self.modes)).bargmann()

    def from_fock(self, fock_array):  # TODO check if code already exists
        return self.__class__(FockKet(fock_array), self.modes, name=self.name)

    @lazy_if_numpy
    def fock(
        self,
        max_prob: Optional[float] = None,
        max_photons: Optional[int] = None,
        shape: Optional[List[int]] = None,
    ):
        r"""Converts the representation of the Ket to Fock representation.

        Args:
            max_prob (optional float): The maximum probability of the state. Defaults to settings.AUTOCUTOFF_PROBABILITY.
                (used to stop the calculation of the amplitudes early)
            max_photons (optional int): The maximum number of photons in the state, summing over all modes
                (used to stop the calculation of the amplitudes early)
            shape (optional List[int]): The shape of the desired Fock tensor. If None it is guessed automatically.

        Returns:
            State: the converted state with the target Fock Representation
        """
        if isinstance(self.representation, Fock):
            return self.array
        if shape is None:
            shape = physics.fock.autocutoffs(*self.phasespace(1,False), max_prob or settings.AUTOCUTOFF_PROBABILITY)
        return math.hermite_renormalized(self.representation.A, self.representation.b, self.representation.c, shape)

    def from_quadrature(self, cov, means, modes, name="Ket"):
        r"""General constructor for kets in quadrature representation."""
        A, b, c = physics.bargmann.ket_from_quadrature(cov, means)
        return self.__class__(Bargmann(A, b, c), modes, name=name)
    
    @lazy_if_numpy
    def quadrature(self, angle: float) -> tuple[Matrix, Vector]:
        r"""Returns the state converted to quadrature (wavefunction) representation with the given quadrature angle.
        Use angle=0 for the position quadrature and angle=pi/2 for the momentum quadrature.

        Args:
            angle (optional float): The quadrature angle.

        Returns:
            tuple[Matrix, Vector]: the quadrature representation of the state
        """
        # transform Abc as in Yuan's notes
        # call physics.bargmann.real_gaussian_integral (get it back from previous commits)
        # Abc -> sigma,mu
        raise NotImplementedError

    def substate(self, modes: int | Iterable) -> State:
        return self.__class__(self.representation.trace(keep=modes), modes=modes)

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

    def __rshift__(self, other: State | Transformation) -> Ket | DM | CircuitComponent:
        r"""If `other` is a transformation, it is applied to self, e.g. ``Thermal(0.5) >> Sgate(r=0.1)``.
        If other is a dual State (i.e. a povm element), self is projected onto it, e.g. ``Gaussian(2) >> Coherent(x=0.1).dual``.
        """
        # 1) check if rshift is possible
        common = set(self.wires.output.modes).intersection(other.wires.input.modes)
        if not (set(self.wires.output.modes) - common).isdisjoint(other.wires.output.modes):
            raise ValueError("Output modes must be disjoint")
        if not set(self.wires.input.modes).isdisjoint(set(other.wires.input.modes) - common):
            raise ValueError("Input modes must be disjoint")
        # 2) add bra/ket if needed
        if other.wires[common].input.ket and other.wires[common].input.bra:  # we want to check common to avoid unnecessary adjoints
            return (self.adjoint & self) >> other
        new_repr = self._safe_rshift(self, other)
        if not other.wires[common].input.bra:
            return Ket(representation=new_repr, modes=self.modes)
        return DM(representation=new_repr, modes=self.modes)
        
    
    def _safe_rshift(self, other: State | Transformation) -> Matrix:
        r"""like rshift but it has already checked that it is safe to apply rshift"""
        self = self.light_copy()  # new wires
        other = other.light_copy()  # new wires
        self.wires[common].output.ids = other.wires[common].input.ids
        self_idx = self.wires[common].output.indices
        other_idx = other.wires[common].input.indices
        # 3) convert bargmann->fock if needed
        if isinstance(self.representation, Bargmann) and isinstance(other.representation, Fock):
            shape = [s if i in self_idx else None for i, s in enumerate(other.representation.shape)]
            self._representation = self.representation.to_fock(shape=shape)
        if isinstance(self.representation, Fock) and isinstance(other.representation, Bargmann):
            shape = [s if i in other_idx else None for i, s in enumerate(self.representation.shape)]
            other._representation = other.representation.to_fock(shape=shape)
        # 4) apply rshift
        new_repr = self.representation[self_idx] @ other.representation[other_idx]
        before_ids = [id for id in self.wires.ids+other.wires.ids if id not in self.wires[common].output.ids]
        after_ids = (self.wires >> other.wires).ids  # automatically yields the right order of ids
        order = [before_ids.index(id) for id in after_ids]
        return new_repr.reorder(order)
        