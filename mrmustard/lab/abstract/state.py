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

from mrmustard.math.tensor_networks.networks import connect, contract
from mrmustard.physics.bargmann_repr import BargmannExp
from mrmustard.physics import bargmann
from mrmustard import settings
from mrmustard.math.tensor_networks import Tensor
from mrmustard.math import Math
from mrmustard.physics import fock, gaussian
from mrmustard.utils import graphics

if TYPE_CHECKING:
    from .transformation import Transformation

math = Math()


# class SafeProperty:
#     r"""
#     A descriptor that catches AttributeError exceptions in a getter method
#     and raises a new AttributeError with a custom message.

#     Usage:

#     @SafeProperty
#     def some_property(self):
#         return self.some_attribute
#     """

#     def __init__(self, getter):
#         self.getter = getter

#     def __get__(self, instance, owner):
#         try:
#             return self.getter(instance)
#         except AttributeError:
#             raise AttributeError("Property unavailable in the current representation.") from None

####################################################################################################
####################################################################################################
# NOTE: primal is implemented via tensor contraction in the __lshift__ method. Also, all the
# contractions are moved to physics
####################################################################################################
####################################################################################################

# NOTE: we don't care about implementation of the operations and the attributes of the state.
# Those are usually in the representation, and we just delegate to it. So abstract away.


# pylint: disable=too-many-instance-attributes
class State(Tensor):  # pylint: disable=too-many-public-methods
    r"""Mixin class for quantum states. It supplies a simple initializer
    and common functionalities and properties of all states. Note that Ket and DM
    implement their own ``from_foo`` methods.

    The State class coordinates the indices of the representation (in the attribute self.representation)
    and the wires/modes business, which is handled by self because State is a Tensor.
    """

    def __init__(self, representation, name, **modes):
        self.representation = representation
        super().__init__(name, **modes)

    def __getattr__(self, name):
        try:
            return getattr(self, name)  # try with self (Ket or DM)
        except AttributeError:  # if not found
            try:
                return getattr(self.representation, name)  # try with representation
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
        return np.isclose(self.purity, 1.0, atol=1e-9)

    @property
    def norm(self) -> float:  # NOTE: risky: need to make sure user knows what kind of norm
        r"""Returns the norm of the state."""
        return self.representation.norm

    @property
    def probability(self) -> float:
        r"""Returns the probability of the state."""
        return self.representation.probability

    def __and__(self, other: State) -> State:
        r"""Tensor product of two states."""
        return self.__class__(
            self.representation & other.representation, modes=self.modes + other.modes
        )

    def __getitem__(self, item: int | Iterable) -> State:  # TODO: implement slices
        r"""Returns the marginal state on the given modes."""
        if isinstance(item, int):
            item = [item]
        elif isinstance(item, Iterable):
            item = list(item)
        else:
            raise TypeError("item must be int or iterable or slice")

        if not set(item).issubset(self.modes):
            raise ValueError(
                f"modes {item} are not a subset of the modes of the state {self.modes}"
            )

        # TODO: write partial_trace in the representation
        self.__class__(self.representation.partial_trace(keep=item), modes=item)

    def __eq__(self, other) -> bool:  # pylint: disable=too-many-return-statements
        r"""Returns whether the states are equal."""
        return self.representation == other.representation  # do we care about modes?

    def __rshift__(self, other: Transformation | State) -> State | complex:
        r"""If `other` is a transformation, it is applied to self, e.g. ``Coherent(x=0.1) >> Sgate(r=0.1)``.
        If other is a dual State (i.e. a povm element), self is projected onto it, e.g. ``Gaussian(2) >> Coherent(x=0.1).dual``.
        """
        common_modes = set(self.output.ket).intersection(other.input.ket)
        for wire1, wire2 in ((self.output.ket[m], other.input.ket[m]) for m in common_modes):
            connect(wire1, wire2)
        try:  # if objects have a bra side, connect it too
            for wire1, wire2 in ((self.output.bra[m], other.input.bra[m]) for m in common_modes):
                connect(wire1, wire2)
        except KeyError:
            pass
        # TODO: contract should return a representation?
        return self.__class__(representation=contract([self, other]), modes=self.modes)

    def __lshift__(self, other: State) -> State | complex:
        r"""Dual picture of ``__rshift__``: self << other is the same as other.dual >> self.dual."""
        common_modes = set(self.input.ket).intersection(other.output.ket)
        for wire1, wire2 in zip(self.input.ket[common_modes], other.output.ket[common_modes]):
            connect(wire1, wire2)
        try:
            for wire1, wire2 in zip(self.input.bra[common_modes], other.output.bra[common_modes]):
                connect(wire1, wire2)
        except AttributeError:
            pass
        return self.__class__(representation=contract([self, other]), modes=self.modes)

    def __add__(self, other: State):
        r"""Implements addition of states."""
        assert self.modes == other.modes
        return self.__class__(self.representation + other.representation, modes=self.modes)

    def __rmul__(self, other: Union[int, float, complex]):
        r"""Implements multiplication by a scalar from the left.

        E.g., ``0.5 * psi``.
        """
        assert isinstance(other, (int, float, complex))
        return self.__class__(other * self.representation, modes=self.modes)

    def __mul__(self, other: Union[int, float, complex]):
        r"""Implements multiplication of two objects."""
        return other * self

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
        return self.__class__(self.representation.to_bargmann(), modes=self.modes)

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
        return self.__class__(self.representation.to_quadrature(angle), modes=self.modes)

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

        # TODO:
        # if settings.DEBUG:
        #     detailed_info = f"\ncov={repr(self.cov)}\n" + f"means={repr(self.means)}\n"
        #     return f"{table}\n{detailed_info}"

        return table


class DM(State):
    def __init__(self, representation, modes, name):
        super().__init__(
            representation, name=name, modes_out_ket=modes, modes_out_bra=modes
        )  # Tensor init

    @classmethod
    def from_abc(cls, A, b, c, modes, name="DM"):
        return cls(BargmannExp(A, b, c), modes, name=name)

    @classmethod
    def from_phase_space(cls, mat, vec, s, characteristic, modes, name="DM"):
        r"""General constructor for density matrices in phase space representation."""
        A, b, c = bargmann.from_phase_space_dm(mat, vec, s, characteristic)
        return cls(BargmannExp(A, b, c), modes, name=name)

    @classmethod
    def from_wigner(cls, cov, means, modes, name="DM"):
        A, b, c = bargmann.from_phase_space_dm(cov, means, s=1, characteristic=False)
        return cls(BargmannExp(A, b, c), modes, name=name)

    @classmethod
    def from_wigner_char(cls, cov, means, modes, name="DM"):
        A, b, c = bargmann.from_phase_space_dm(cov, means, s=1, characteristic=True)
        return cls(BargmannExp(A, b, c), modes, name=name)

    @classmethod
    def from_fock_array(cls, fock_array, modes, name="DM"):
        return super().__init__(FockArray(fock_array), modes, name=name)

    @classmethod
    def from_wf_array(cls, coords, wf_array):
        return cls(WaveFunctionDM(coords, wf_array))

    def purity(self) -> float:
        # TODO mark all indices to be contracted
        return self.representation @ self.representation.dual


class Ket(State):
    def __init__(self, representation, modes, name):
        self.representation = representation
        super().__init__(name=name, modes_out_ket=modes)  # Tensor init

    @classmethod
    def from_abc(cls, A, b, c, modes, name="Ket"):
        return cls(BargmannExp(A, b, c), modes, name=name)

    @classmethod
    def from_cov_means(cls, cov, means, modes, name="Ket"):  # should become from_phase_space?
        A, b, c = bargmann.wigner_to_bargmann_psi(cov, means)
        return cls(BargmannExp(A, b, c), modes, name=name)

    def from_fock_array(self, fock_array):
        return self.__class__(FockKet(fock_array), self.modes, name=self.name)

    def from_wf_q_array(self, qs, wf_q_array):
        return self.__class__(WaveFunctionKet(qs, wf_q_array), self.modes, name=self.name)

    @property
    def purity(self) -> float:
        return 1.0
