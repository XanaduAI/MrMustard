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
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from mrmustard import physics
from mrmustard import settings
from mrmustard.lab.utils import trainable_property
from mrmustard.math.tensor_networks.networks import connect, contract
from mrmustard.physics.bargmann_repr import BargmannExp
from mrmustard.math.tensor_networks import Tensor
from mrmustard.math import Math
from mrmustard.utils import graphics

if TYPE_CHECKING:
    from .transformation import Transformation

math = Math()


class State(Tensor):
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
        connect(rho.output[0], channel0.input)
        contract([rho, channel0])

    A State object also supports the wire-wise matmul functionality:

    The actual implementation of the algebraic functionality is beyond the representation interface.
    Representation objects come in two types: there is RepresentationCV and RepresentationDV.
    RepresentationCV is a parent of the Bargmann representation (which includes all the other continous ones that we know of)
    and RepresentationDV is a parent of the Fock representation, but also of the discretization of the continuous representations.
    """

    def __getattr__(self, name):
        r"""Searches for the attribute in the representation if it is not found in self (Ket or DM)."""
        try:
            return getattr(self.representation, name)
        except AttributeError as e:
            raise AttributeError(
                f"Attribute {name} not found in {self.__class__.__qualname__} or its representation"
            ) from e

    @property
    def is_mixed(self):
        r"""Returns whether the state is mixed."""
        return not self.is_pure

    @property
    def is_pure(self):
        r"""Returns whether the state is pure."""
        return np.isclose(self.purity, 1.0, atol=settings.PURITY_ATOL)

    @trainable_property
    def L2_norm(self) -> float:
        r"""Returns the L2 norm of the Hilbert space vector or Hilbert-Schmidt norm of a density matrix."""
        return self >> self.dual

    def __and__(self, other: State) -> State:
        r"""Tensor product of two states."""
        if not set(self.modes).isdisjoint(other.modes):
            raise ValueError("Modes must be disjoint")
        return self.__class__(
            self.representation & other.representation, modes=self.modes + other.modes
        )

    def __getitem__(self, item: int | Iterable) -> State:
        r"""Returns a copy of the state on the given modes."""
        if isinstance(item, int):
            item = [item]
        elif isinstance(item, Iterable):
            item = list(item)
        else:
            raise TypeError("item must be int or iterable")
        if len(item) != self.num_modes:
            raise ValueError(f"item must have length {self.num_modes}, got {len(item)} instead")
        return self.__class__(self.representation, modes=item)

    def __matmul__(self, other: BargmannExp) -> BargmannExp:
        r"""Inner product with a generic Tensor object.
        It assumes that the _contract_idxs attribute has already been set.
        """
        return self.__class__(self.representation @ other, modes=self.modes)

    def get_modes(self, modes: int | Iterable) -> State:
        # TODO: write partial_trace in the representation
        self.__class__(self.representation.partial_trace(keep=modes), modes=modes)

    def __eq__(self, other) -> bool:  # pylint: disable=too-many-return-statements
        r"""Returns whether the states are equal. Modes and all."""
        return self.representation == other.representation and self.modes == other.modes

    def __rshift__(self, other: Transformation | State) -> State | complex:
        r"""If `other` is a transformation, it is applied to self, e.g. ``Coherent(x=0.1) >> Sgate(r=0.1)``.
        If other is a dual State (i.e. a povm element), self is projected onto it, e.g. ``Gaussian(2) >> Coherent(x=0.1).dual``.
        """
        common_modes = set(self.modes_out).intersection(other.modes_in)
        self_out = self.output[common_modes]
        other_in = other.input[common_modes]
        connect(self_out, other_in)
        return self.__class__(representation=contract([self, other]), modes=self.modes)

    def __lshift__(self, other: State) -> State | complex:
        r"""dual of __rshift__"""
        return (other.dual >> self.dual).dual

    def __add__(self, other: State):
        r"""Implements addition of states."""
        assert self.modes == other.modes
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
        modes = list(set(self.modes).union(set(other.modes)))
        return self.__class__(self.representation * other.representation, modes=modes)

    def __truediv__(self, other: Union[int, float, complex]):
        r"""Implements division by a scalar.

        E.g. ``psi / 0.5``
        """
        assert isinstance(other, (int, float, complex))
        return self.__class__(self.representation / other, modes=self.modes)

    def to_bargmann(self):
        r"""Converts the representation of the state to Bargmann Representation and returns self.

        Returns:
            State: the converted state with the target Bargmann Representation
        """
        if isinstance(self.representation, BargmannExp):
            return self

    def to_fock(
        self,
        max_prob: Optional[float] = None,
        max_photons: Optional[int] = None,
        shape: Optional[List[int]] = None,
    ):
        r"""Converts the representation of the state to Fock Representation.

        Args:
            max_prob (optional float): The maximum probability of the state. Defaults to settings.AUTOCUTOFF_PROBABILITY.
                (used to stop the calculation of the amplitudes early)
            max_photons (optional int): The maximum number of photons in the state, summing over all modes
                (used to stop the calculation of the amplitudes early)
            shape (optional List[int]): The shape of the desired Fock tensor. If None it is guessed automatically.

        Returns:
            State: the converted state with the target Fock Representation
        """
        fock_repr = self.representation.to_fock(
            max_prob=max_prob or settings.AUTOCUTOFF_PROBABILITY,
            max_photons=max_photons,
            shape=shape,
        )
        return self.__class__(representation=fock_repr, modes=self.modes)

    def to_quadrature(self, angle: float):
        r"""Returns the state converted to quadrature (wavefunction) representation with the given quadrature angle.
        Use angle=0 for the position quadrature and angle=pi/2 for the momentum quadrature.

        Args:
            angle (optional float): The quadrature angle.

        Returns:
            State: the converted state with the target quadrature representation
        """
        quadrature = physics.bargmann.quadrature_kernel(angle, dim=self.representation.dimension)
        connect(self.output.ket, quadrature.input.ket)
        connect(self.output.bra, quadrature.input.ket.adjoint)
        return self.__class__(contract([self, quadrature]), modes=self.modes)

    def to_phase_space(self, s=1, characteristic=False):
        r"""Returns the state converted to s-parametrized phase space representation.

        Args:
            s (optional float): The parameter s of the phase space representation. Defaults to 1 (Wigner).
                Use s=0 for the Husimi Q function and s=-1 for the Glauber-Sudarshan P function.
            characteristic (optional bool): Whether to compute to the characteristic function. Defaults to False.

        Returns:
            State: the converted state with the target Wigner Representation
        """
        assert s in [1, 0, -1]  # for now
        return self.__class__(
            self.representation.to_phase_space(s, characteristic), modes=self.modes
        )

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

    @trainable_property
    def cov(self):
        r"""Covariance matrix of the Gaussian distribution. Available only if the representation is continuous."""
        cov_complex = self.to_phase_space(s=1, characteristic=True).representation.A
        R = math.Rmat(self.representation.dimension)
        return math.matmul(math.dagger(R), cov_complex, R) * settings.HBAR

    @trainable_property
    def mean(self):  # TODO check if this is correct
        r"""Mean of the Gaussian distribution. Available only if the representation is continuous."""
        wigner = self.to_phase_space(s=1, characteristic=False).representation
        mean_complex = -math.matvec(wigner.A, wigner.b)
        R = math.Rmat(self.representation.dimension)
        return math.matmul(math.dagger(R), mean_complex) * math.sqrt(settings.HBAR)


class DM(State):
    def __init__(self, representation, modes, name):
        self.representation = representation
        super().__init__(name=name, modes_out_ket=modes, modes_out_bra=modes)

    @classmethod
    def from_abc(cls, A, b, c, modes, name="DM"):
        return cls(BargmannExp(A, b, c), modes, name=name)

    @classmethod
    def from_phase_space(cls, cov, mean, modes, s=1, characteristic=False, name="DM"):
        r"""General constructor for density matrices in phase space representation.

        Args:
            cov (Batch[ComplexMatrix]): the covariance matrix
            mean (Batch[ComplexVector]): the vector of means
            modes (Sequence[int]): the modes of the state
            s (float): the parameter of the phase space representation
            characteristic (bool): whether from the characteristic function
            name (str): the name of the state
        """
        A, b, c = physics.bargmann.from_phase_space_dm(cov, mean, s, characteristic)
        return cls(
            BargmannExp(A, b, c), modes, name=name
        )  # TODO replace with more general Bargmann repr (e.g. poly x exp) when ready

    @classmethod
    def from_fock_array(cls, fock_array, modes, name="DM"):
        return super().__init__(FockDM(fock_array), modes, name=name)

    @classmethod  # NOTE DO we really want this?
    def from_wf_array(cls, coords, wf_array):
        return cls(WaveFunctionDM(coords, wf_array))

    @trainable_property
    def probability(self) -> float:  # TODO: make lazy if backend is not TF?
        connect(self.output.ket, self.output.bra)  # TODO they remain connected though
        return math.real(contract([self]))

    @trainable_property
    def purity(self) -> float:
        normalized = self / self.probability
        return normalized.L2_norm  # NOTE: for density matrices it's the purity

    def to_phase_space(self, s=1, characteristic=False):
        r"""Returns the density matrix converted to s-parametrized phase space representation.

        Args:
            s (optional float): The parameter s of the phase space representation. Defaults to 1 (Wigner).
                Use s=0 for the Husimi Q function and s=-1 for the Glauber-Sudarshan P function.
            characteristic (optional bool): Whether to compute to the characteristic function. Defaults to False.

        Returns:
            DM: the converted DM in the phase space representation
        """
        assert s in [1, 0, -1]
        Delta = physics.bargmann.siegel_weil_kernel(s, dim=self.representation.dimension)
        connect(self.output.ket, Delta.input.ket)
        connect(self.output.bra, Delta.input.bra)
        return self.__class__(contract([self, Delta]), modes=self.modes)


class Ket(State):
    def __init__(self, representation, modes, name):
        self.representation = representation
        super().__init__(name=name, modes_out_ket=modes)  # Tensor init

    @classmethod
    def from_abc(cls, A, b, c, modes, name="Ket"):
        return cls(BargmannExp(A, b, c), modes, name=name)

    @classmethod
    def from_phase_space(
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
            raise ValueError("Initializing a Ket using a mixed state")
        A, b, c = physics.bargmann.from_phase_space_ket(cov, mean, s, characteristic)
        return cls(BargmannExp(A, b, c), modes, name=name)

    def to_phase_space(self, s=1, characteristic=False):
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
        assert s in [1, 0, -1]
        # TODO: write SW kernel
        Delta = physics.bargmann.siegel_weil_kernel(s, dim=self.representation.dimension)
        self_bra = self.output.ket.adjoint
        connect(self.output.ket, Delta.input.ket)
        connect(self_bra, Delta.input.bra)
        return self.__class__(contract([self, self_bra, Delta]), modes=self.modes)

    def from_fock_array(self, fock_array):
        return self.__class__(FockKet(fock_array), self.modes, name=self.name)

    def from_wf_q_array(self, qs, wf_q_array):
        return self.__class__(WaveFunctionKet(qs, wf_q_array), self.modes, name=self.name)

    @property
    def purity(self) -> float:
        return 1.0

    @trainable_property
    def probability(self) -> float:
        return self.L2_norm
