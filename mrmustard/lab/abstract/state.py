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

from abc import abstractproperty
from typing import (
    TYPE_CHECKING,
    Iterable,
    List,
    Optional,
    Union,
)

import numpy as np

from mrmustard import math, physics, settings
from mrmustard.math.tensor_networks.wires import Wires

if TYPE_CHECKING:
    from .transformation import Transformation

from abc import ABC, abstractproperty


class State(ABC):
    r"""Mixin class for quantum states. It supplies common functionalities and properties of all states.
    Note that Ket and DM implement their own ``from_foo`` methods.

    The State class is a coordinator between the indices of the representation (in self.representation)
    and the wires/modes, which is handled by self as well, because State inherits from Tensor.

    When using a State object we can think of it as the mathematical symbol, e.g. a Hilbert space vector

    .. code-block:: python
        psi0 = Ket.from_abc(A0, b0, c0, modes=[0, 1, 2])
        psi1 = Ket.from_abc(A1, b1, c1, modes=[0, 1, 2])

        psi = a * psi0 + b * psi1

    A State object also supports the tensor network operations, e.g. contraction, tensor product, etc.

    .. code-block:: python
        TN = connect(rho.wires.output[0], channel.wires.input[0], {})
        new_rho = contract([rho, channel0], TN)

    The actual implementation of the algebraic functionality is beyond the representation interface.
    Representation objects come in two types: there is RepresentationCV and RepresentationDV.
    RepresentationCV is a parent of the Bargmann representation (which includes all the other continous ones that we know of)
    and RepresentationDV is a parent of the Fock representation, but also of the discretization of the continuous representations.
    """

    @property
    def is_mixed(self):
        r"""Returns whether the state is mixed."""
        return not self.is_pure

    @property
    def is_pure(self):
        r"""Returns whether the state is pure."""
        return np.isclose(self.purity, 1.0, atol=settings.PURITY_ATOL)
    
    @abstractproperty
    def purity(self) -> float:
        r"""Returns the purity of the state."""
        pass

    @property
    def wires(self) -> Wires:
        r"""Returns the wires of the state from the representation."""
        self._representation.wires

    @property
    def modes(self) -> List[int]:
        r"""Returns the modes of the state."""
        return self.wires.modes  # NOTE: implementation must have a wires attribute

    @property
    def num_modes(self) -> int:
        r"""Returns the number of modes of the state."""
        return len(self.modes)
    
    @property
    def dual(self) -> State:
        r"""Returns the dual of the state."""
        return self.__class__(self.representation.dual, modes=self.modes)
    
    @property
    def adjoint(self) -> State:
        r"""Returns the adjoint of the state."""
        return self.__class__(self.representation.adjoint, modes=self.modes)

    @trainable_property
    def L2_norm(self) -> float:
        r"""Returns the L2 norm of the Hilbert space vector or L2 norm (Hilbert-Schmidt) of a density matrix."""
        return self >> self.dual

    def get_modes(self, modes: int | Iterable) -> State:
        # TODO: write partial_trace in the representation
        self.__class__(self._representation.partial_trace(keep=modes), modes=modes)

    def __getitem__(self, modes: int | Iterable) -> State:
        r"""Returns a new state object with same internal data except modes
        psi[1] & psi[2] >> U[1,2]
        """
        modes = [modes] if isinstance(modes, int) else modes
        assert all(isinstance(m, int) for m in modes)
        if len(modes) != self.num_modes:
            raise ValueError(f"Got {len(modes)} ints, but this state has {self.num_modes} modes")
        return self.__class__(self._representation, modes=modes)

    def __and__(self, other: State) -> State:
        r"""Tensor product of two states."""
        if not isinstance(other, State):
            raise ValueError("Cannot tensor a State and a non-State object")
        if not set(self.modes).isdisjoint(other.modes):
            raise ValueError("Modes must be disjoint")
        return self.__class__(
            self._representation & other._representation, modes=self.modes + other.modes
        )

    def __eq__(self, other) -> bool:
        r"""Returns whether the states are equal. Modes are dealt with in the internal representation."""
        return self._representation == other._representation

    def __rshift__(self, other: DualView | Transformation | Measurement) -> State | complex:
        r"""If `other` is a transformation, it is applied to self, e.g. ``Coherent(x=0.1) >> Sgate(r=0.1)``.
        If other is a dual State (i.e. a povm element), self is projected onto it, e.g. ``Gaussian(5) >> Coherent(x=0.1).dual``.
        """
        common_modes = [m for m in self.modes_out if m in other.modes_in]
        TN = {}
        try:
            TN = connect(
                self.wires[common_modes].output.ket,
                other.wires[common_modes].input.ket
                or other.wires.adjoint[common_modes].input.ket,
                TN,
            )
        except AttributeError:  # if self is a bra with no ket
            pass
        try:
            TN = connect(
                self.wires[common_modes].output.bra,
                other.wires[common_modes].input.bra
                or other.wires.adjoint[common_modes].input.bra,
                TN,
            )
        except AttributeError:  # if self is a ket with no bra
            pass
        new = contract(self, other, TN)
        return self.__class__(representation=new._representation, modes=new.modes)

    def __lshift__(self, other: State) -> State | complex:
        r"""dual of __rshift__"""
        return (other.dual >> self.dual).dual

    def __add__(self, other: State):
        r"""Implements addition of states."""
        assert self.modes == other.modes
        return self.__class__(self._representation + other._representation, modes=self.modes)

    def __rmul__(self, other: Union[int, float, complex]):
        r"""Implements multiplication from the left if the object on the left
        does not implement __mul__ for the type of self.

        E.g., ``0.5 * psi``.
        """
        assert isinstance(other, (int, float, complex))
        return self.__class__(other * self._representation, modes=self.modes)

    def __mul__(self, other):
        r"""Implements multiplication of two objects."""
        if isinstance(other, (int, float, complex)):
            return other * self
        modes = list(set(self.modes).union(set(other.modes)))
        return self.__class__(self._representation * other._representation, modes=modes)

    def __truediv__(self, other: Union[int, float, complex]):
        r"""Implements division by a scalar.

        E.g. ``psi / 0.5``
        """
        assert isinstance(other, (int, float, complex))
        return self.__class__(self._representation / other, modes=self.modes)

    ####################
    # PARAMETRIZATIONS #
    ####################

    def bargmann(self) -> tuple:
        r"""Returns the Bargmann parametrization of the state.
        We fix the functional form of the Bargmann representation as:

        F(z) = sum_i poly_i(z) * exp(-0.5 * z^T A_i z + b_i^H z) * c_i

        Gaussian states have a single term in the sum and poly(z) = 1.
        E.g. a cat state has two terms in the sum and poly_i(z) = 1.
        E.g. a heralded GBS state has a single term in the sum and non-trivial poly(z).

        Returns:
            tuple: the parameters of the Bargmann representation:
                the vectors {A_i}, {b_i}, {c_i}, {poly_i}.
        """
        bargmann = self._representation
        return bargmann.A, bargmann.b, bargmann.c, bargmann.poly

    def fock(self, max_prob=None, max_photons=None, shape=None):
        r"""Converts the representation of the DM to Fock representation.
        For the final shape of the array the priority of options is shape > max_photons > max_prob.

        Args:
            max_prob (optional float): The maximum probability of the state. Defaults to settings.AUTOCUTOFF_PROBABILITY.
                (used to stop the calculation of the amplitudes early)
            max_photons (optional int): The maximum number of photons in the state, summing over all modes
                (used to stop the calculation of the amplitudes early)
            shape (optional List[int]): The shape of the desired Fock tensor. If None it is guessed automatically.

        Returns:
            State: the converted state with the target Fock Representation
        """
        assert (
            max_prob or max_photons or shape
        ), "At least one of max_prob, max_photons or shape must be specified"
        A, b, c, poly = self.bargmann()
        fock_array = math.hermite_renormalized(
            A,
            b,
            c,
            max_prob=max_prob or settings.AUTOCUTOFF_PROBABILITY,
            max_photons=max_photons,
            shape=shape,
        )
        return math.tensordot(poly, fock_array, [[0], [0]])

    ##############
    # REPR STUFF #
    ##############

    def __len__(self):
        return len(self._representation)

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
        probability = self.probability
        num_modes = self._representation.num_modes
        bosonic_size = len(self)
        representation = self._representation.name
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


#############################
#######  KET and DM  ########
#############################


class Ket(State):
    def __init__(self, representation, modes):
        self._representation = representation

    @property
    def wires(self):
        return self._representation.wires

    @property
    def purity(self) -> float:
        return 1.0

    @trainable_property
    def probability(self) -> float:
        return self.L2_norm

    ################
    # CONSTRUCTORS #
    ################

    @classmethod
    def from_bargmann(cls, A, b, c, modes, poly=None, name="Ket"):
        return cls(Bargmann(A, b, c, poly), modes, name=name)

    @classmethod
    def from_fock(cls, fock_array, modes, name="Ket"):
        return cls(Fock(fock_array), modes, name=name)

    def __rshift__(self, other: Transformation | Measurement) -> State | complex:
        r"""If `other` is a transformation, it is applied to self, e.g. ``Coherent(x=0.1) >> Sgate(r=0.1)``.
        If other is a Measurement (e.g. a povm element), self is projected onto it, e.g. ``Gaussian(5) >> Homodyne(x=0.1)``.
        """
        common_modes = [m for m in self.wires.modes_out if m in other.wires.modes_in]
        TN = connect(self.wires[common_modes].output.ket, other.wires[common_modes].input.ket, {})
        contracted = contract([self, other], TN)
        return self.__class__(representation=contracted._representation, modes=contracted.modes)


    @classmethod
    def from_phase_space(
        cls, cov, mean, coeff = 1.0, s=1, characteristic=True, pure_check=True, modes=None, name="Ket"
    ):
        r"""General constructor for kets from phase space representation.
        Warning: kets do not really exist in phase space, but given that we use bargmann under the hood, we can do this.

        Args:
            cov (Batch[ComplexMatrix]): the covariance matrix
            mean (Batch[ComplexVector]): the vector of means
            modes (Sequence[int]): the modes of the state
            s (float): the parameter of the phase space representation
            characteristic (bool): whether to compute the characteristic function
            name (str): the name of the state
            pure_check (bool): whether to check if the state is pure (default: True)
        """
        if pure_check and physics.phase_space.purity(cov, s, characteristic) < 1.0 - settings.PURITY_ATOL:
            raise ValueError("Initializing a Ket using a mixed cov matrix is not allowed")
        A, b, c = physics.bargmann.from_phase_space_ket(cov, mean, s, characteristic)
        return cls(Bargmann(A, b, c * coeff), modes, name=name)

    @classmethod
    def from_quadrature(cls, A, b, c, angle: float, modes, name="Ket"):
        r"""Returns the state converted from quadrature (wavefunction) representation with the given quadrature angle.
        Use angle=0 for the position quadrature and angle=pi/2 for the momentum quadrature.

        Args:
            angle (optional float): The quadrature angle.

        Returns:
            State: the converted state with the target quadrature representation
        """
        Abc = physics.bargmann.quadrature_kernel([angle] * A.shape[-1])
        kernel = Bargmann(*Abc)
        wavefunction = Bargmann(A, b, c)  # just wavefunction! do not use inner product
        new_repr = wavefunction.output.ket @ kernel.input.ket
        return cls(new_repr, modes=modes, name=name)

    ###########################
    # PARAMETRIZATIONS OF KET #
    ###########################

    def phase_space(self, s=0, characteristic=False):
        r"""Returns the DENSITY MATRIX (i.e. |psi><psi|) parametrization in s-parametrized phase space representation.

        Since the functional form is fixed to F(z) = sum_i poly_i(z) * exp(0.5 * z^T A_i z + z^T b_i) * c_i,
        it will still be one of these functions that parametrizes, e.g. the wigner function.
        If s=0 (wigner) we need to compute the (Sigma,mu) pair that parametrizes the more common form
        exp(-(x-mu)^T Sigma^-1 (x-mu)) / sqrt(det(Sigma)) (and in the p/q basis).

        Args:
            s (optional float): The parameter s of the phase space representation. Defaults to 1 (Wigner).
                Use s=0 for the Husimi Q function and s=-1 for the Glauber-Sudarshan P function.
            characteristic (optional bool): Whether to compute to the characteristic function. Defaults to False.

        Returns:
            DM: the converted DM in the phase space representation
        """
        assert s in [1, 0, -1]
        d = self._representation.dimension
        Abc = physics.bargmann.siegel_weil([s] * d) if not characteristic else physics.bargmann.s_displacement([s] * d)
        kernel = DM(Bargmann(*Abc).conj, modes=self.modes).dual
        new = self >> kernel
        return new._representation.A, new._representation.b, new._representation.c, new._representation.poly

    def quadrature(self, angle: float):
        r"""Returns the state converted to quadrature (wavefunction) representation with the given quadrature angle.
        Use angle=0 for the position quadrature and angle=pi/2 for the momentum quadrature.

        Args:
            angle (optional float): The quadrature angle.

        Returns:
            State: the converted state with the target quadrature representation
        """
        Abc = physics.bargmann.quadrature_kernel([angle] * self.representation.dimension)
        kernel = Bargmann(*Abc)
        kernel_adj = kernel.adjoint
        # TODO: finish


class DM(State):
    def __init__(self, representation, modes, name):
        self.representation = representation
        super().__init__(name=name, modes_out_ket=modes, modes_out_bra=modes)

    @trainable_property
    def probability(self) -> float:
        return math.real(self._representation.output.ket @ self._representation.output.bra)

    @trainable_property
    def purity(self) -> float:
        normalized = self / self.probability
        return normalized.L2_norm  # NOTE: for density matrices L2 norm is the purity

    ################
    # CONSTRUCTORS #
    ################

    @classmethod
    def from_bargmann(cls, A, b, c, modes, poly=None, name="DM"):
        return cls(Bargmann(A, b, c, poly), modes, name=name)

    @classmethod
    def from_fock(cls, fock_array, modes, name="DM"):
        return cls(Fock(fock_array), modes, name=name)

    @classmethod
    def from_phase_space(cls, cov, means, coeffs, s, characteristic, modes, name="DM"):
        assert s in [1, 0, -1]
        dim = cov.shape[-1] // 2
        assert len(modes) == dim
        Abc = (
            physics.bargmann.s_displacement([s] * dim)
            if characteristic
            else physics.bargmann.siegel_weil([s] * dim)
        )
        Delta = Bargmann(*Abc)
        connect(self._representation.output.ket, Delta.input.ket)
        connect(Delta.output.bra, Delta.input.bra)
        bargmann = contract([self._representation, Delta])
        if s == 0:  # if Wigner / characteristic
            pass
            # p/q basis
        else:
            return cls(bargmann, modes, name=name)

    @classmethod
    def from_quadrature(cls, A, b, c, angle: float):
        r"""Returns the state converted from quadrature (wavefunction) representation with the given quadrature angle.
        Use angle=0 for the position quadrature and angle=pi/2 for the momentum quadrature.

        Args:
            angle (optional float): The quadrature angle.

        Returns:
            State: the converted state with the target quadrature representation
        """
        Abc = physics.bargmann.quadrature_kernel([angle] * self.representation.dimension)
        kernel = Bargmann(*Abc)
        kernel_adj = kernel.adjoint
        # TODO

    ##########################
    # PARAMETRIZATIONS OF DM #
    ##########################

    def phase_space(self, s=1, characteristic=False):
        r"""Returns the density matrix converted to s-parametrized phase space representation.

        Args:
            s (optional float): The parameter s of the phase space representation. Defaults to 1 (Wigner).
                Use s=0 for the Husimi Q function and s=-1 for the Glauber-Sudarshan P function.
            characteristic (optional bool): Whether to compute to the characteristic function. Defaults to False.

        Returns:
            DM: the converted DM in the phase space representation
        """
        assert s in [1, 0, -1]
        if not characteristic:
            Abc = physics.bargmann.Kernel_siegel_weil([s] * self.representation.dimension)
        else:
            Abc = physics.bargmann.Kernel_s_displacement([s] * self.representation.dimension)
        Delta = Bargmann(*Abc)
        connect(self.representation.output.ket, Delta.input.ket)
        connect(self.representation.output.bra, Delta.input.bra)
        new_repr = contract([self.representation, Delta])
        if s == 0:  # if Wigner / characteristic
            R = math.Rmat(self.representation.dimension)

    def quadrature(self, angle: float):
        r"""Returns the state converted to quadrature (wavefunction) representation with the given quadrature angle.
        Use angle=0 for the position quadrature and angle=pi/2 for the momentum quadrature.

        Args:
            angle (optional float): The quadrature angle.

        Returns:
            State: the converted state with the target quadrature representation
        """
        Abc = physics.bargmann.quadrature_kernel([angle] * self.representation.dimension)
        kernel = Bargmann(*Abc)
        kernel_adj = kernel.adjoint
        connect(self.output.ket, kernel.input.ket)
        connect(self.output.bra, kernel_adj.input.bra)
        quadrature = contract([self.representation, kernel, kernel_adj])
        return quadrature.A, quadrature.b, quadrature.c, quadrature.poly
