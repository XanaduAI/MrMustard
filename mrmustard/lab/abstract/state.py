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

import warnings
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
import copy

from mrmustard import settings
from mrmustard.math import Math
from mrmustard.physics import fock, gaussian
from mrmustard.lab.representations.representation import Representation
from mrmustard.lab.representations.fock_dm import FockDM
from mrmustard.lab.representations.fock_ket import FockKet
from mrmustard.lab.representations.wigner_ket import WignerKet
from mrmustard.lab.representations.wigner_dm import WignerDM
from mrmustard.lab.representations.wavefunctionq_ket import WaveFunctionQKet
from mrmustard.lab.representations.wavefunctionq_dm import WaveFunctionQDM
from mrmustard.lab.representations.converter import Converter
from mrmustard.typing import (
    ComplexMatrix,
    ComplexTensor,
    ComplexVector,
    RealMatrix,
    RealTensor,
    RealVector,
)
from mrmustard.utils import graphics

if TYPE_CHECKING:
    from .transformation import Transformation

math = Math()
converter = Converter()


class SafeProperty:
    r"""
    A descriptor that catches AttributeError exceptions in a getter method
    and raises a new AttributeError with a custom message.

    Usage:

    @SafeProperty
    def some_property(self):
        return self.some_attribute
    """

    def __init__(self, getter):
        self.getter = getter

    def __get__(self, instance, owner):
        try:
            return self.getter(instance)
        except AttributeError:
            raise AttributeError("Property unavailable for the current representation.") from None


# pylint: disable=too-many-instance-attributes
class State:  # pylint: disable=too-many-public-methods
    r"""Base class for quantum states."""

    def __init__(
        self,
        A=None,
        b=None,
        c=None,
        cov=None,
        means=None,
        fock_array=None,
        qs=None,
        wf_q_array=None,
        representation=None,
        modes=None,
    ):
        if A is not None and b is not None and c is not None:
            self._from_Abc(A, b, c)  # init Fock-Bargmann representation
        elif cov is not None and means is not None:
            self._from_cov_means(cov, means)  # init Wigner representation
        elif fock_array is not None:
            self._from_fock_array(fock_array)  # init Fock representation
        elif qs is not None and wf_q_array is not None:
            self._from_wf_q_array(qs, wf_q_array)  # init q-wavefunction representation
        elif representation is not None:
            self.representation = representation
        else:
            raise ValueError(
                f"A state must be initialized with (A,b,c) or (cov, means) or `fock_array` or (qs, wf_q_array)."
            )

        # TODO: this attribute modes is linked to the Circuit. Need to be modified afterwards.
        self.num_modes = self.representation.num_modes

        if modes is not None:  # TODO: will be refactored with CircuitPart
            self._modes = modes
            assert (
                len(modes) == self.num_modes
            ), f"Number of modes supplied ({len(modes)}) must match the representation dimension {self.num_modes}"

    def _from_Abc(self, A, b, c):
        raise NotImplementedError("Initialize a state using Ket or DM class.")

    def _from_cov_means(self, cov, means):
        raise NotImplementedError("Initialize a state using Ket or DM class.")

    def _from_fock_array(self, fock_array):
        raise NotImplementedError("Initialize a state using Ket or DM class.")

    def _from_wf_q_array(self, wf_q_array):
        raise NotImplementedError("Initialize a state using Ket or DM class.")

    def _from_wf_p_array(self, wf_p_array):
        raise NotImplementedError("Initialize a state using Ket or DM class.")

    # def __init__(  # TODO: remove and leave only init in Ket and DM
    #     self,
    #     cov: RealMatrix = None,
    #     means: RealVector = None,
    #     symplectic: RealMatrix = None,
    #     displacement: RealVector = None,
    #     fock: ComplexTensor = None,
    #     qs: RealVector = None,
    #     wavefunctionq: RealVector = None,
    #     modes: Sequence[int] = None,
    #     flag_ket: bool = None,
    #     representation: Representation = None,
    # ):
    #     r"""Initializes the state.

    #     Supply either:
    #         * a covariance matrix and means vector
    #         * a fock representation (ket or dm)
    #         * a quadrature variable vector and corresponding wavefunction samples
    #     together with the flag_ket to indicate the nature of the state.
    #     Or supply simply with the Representation Class.

    #     Args:
    #         cov (Matrix): the covariance matrix
    #         means (Vector): the means vector
    #         symplectic (Matrix): the symplectic matrix
    #         displacement (Vector): the displacement vector
    #         fock (Tensor): the Fock representation
    #         qs (Vector): the point value for corresponding q-wavefunction
    #         wavefunctionq (Vector): the value of the point of the q-wavefunction
    #         flag_ket (boolean): whether this state is a ket (pure) or a density matrix (mixed)
    #         representation (Representation): the Representation Class contains all information about the state
    #     """
    #     # Case 0: All data of the state is wrapped inside one of the Representation classes
    #     if representation:
    #         self.representation = representation
    #     else:
    #         # Case 1: Wigner representation #TODO: add coeff?
    #         if cov is not None and means is not None and flag_ket is not None:
    #             self.representation = WignerDM(cov, means)
    #         elif symplectic is not None and displacement is not None and flag_ket:
    #             self.representation = WignerKet(symplectic, displacement)
    #         # Case 2: Fock representation
    #         elif fock is not None and flag_ket is not None:
    #             if flag_ket:
    #                 self.representation = FockKet(fock)
    #             else:
    #                 self.representation = FockDM(fock)
    #         # Case 3: q-Wavefunction representation
    #         elif qs is not None and wavefunctionq is not None and flag_ket is not None:
    #             if flag_ket:
    #                 self.representation = WaveFunctionQKet(qs, wavefunctionq)
    #             else:
    #                 self.representation = WaveFunctionQDM(qs, wavefunctionq)
    #         else:
    #             raise ValueError(
    #                 "State must be initialized with either a wrapped Representation class, a covariance matrix and means vector, a fock representation, a point-wise wavefunction with its point values and the flag_ket."
    #             )

    @property
    def modes(self):  # will be updated from TN project, also modes in init.
        r"""Returns the modes of the state."""
        if self._modes is None:
            return list(range(self.representation.num_modes))
        return self._modes

    def indices(self, modes) -> Union[Tuple[int], int]:  # TODO write like the other methods
        r"""Returns the indices of the given modes. Only works for Fock.

        Args:
            modes (Sequence[int] or int): the modes or mode

        Returns:
            Tuple[int] or int: a tuple of indices of the given modes or the single index of a single mode
        """
        if isinstance(self.representation, (FockKet, FockDM)):
            if isinstance(modes, int):
                return self.modes.index(modes)
            return tuple(self.modes.index(m) for m in modes)
        else:
            raise AttributeError(
                "The representation of your state do not have this attribute, transform it with the Converter please!"
            )

    @property
    def purity(self) -> float:  # can override in Ket
        r"""Returns the purity of the state."""
        return self.representation.purity

    @property
    def is_mixed(self):
        r"""Returns whether the state is mixed."""
        return not self.is_pure

    @property
    def is_pure(self):
        r"""Returns whether the state is pure."""
        return np.isclose(self.purity, 1.0, atol=1e-6)  # NOTE: is this atol okay?

    # @property
    # def is_wigner(self):
    #     r"""Returns if the state is in Wigner representation or not. (Notes: works as the previous is_gaussian function.)"""
    #     if isinstance(self.representation, (WignerKet, WignerDM)):
    #         return True
    #     else:
    #         return False
    @SafeProperty
    def means(self) -> Optional[RealVector]:
        r"""Returns the means vector of the state."""
        return self.representation.data.means

    @SafeProperty
    def cov(self) -> Optional[RealMatrix]:  # override in Ket
        r"""Returns the covariance matrix of the state."""
        return self.representation.data.cov

    @property
    def number_stdev(self) -> RealVector:
        r"""Returns the square root of the photon number variances (standard deviation) in each mode."""
        return self.representation.number_stdev()

    @property  # NOTE: this is going to become ambiguous when each wire can have a different representation
    def cutoffs(self) -> List[int]:
        r"""Returns the cutoff dimensions for each mode."""
        return self.representation.cutoffs

    # @property
    # #TODO: Depends on the representation. Shape means something else.
    # def shape(self) -> List[int]:
    #     r"""Returns the shape of the state.
    #     """
    #     return self.cutoffs if self.is_pure else self.cutoffs + self.cutoffs

    @SafeProperty
    def fock_array(self) -> ComplexTensor:
        r"""Returns the Fock representation of the state."""
        return self.representation.data.array  # if unavailable will raise AttributeError from data

    @property
    def number_means(self) -> RealVector:
        r"""Returns the mean photon number for each mode."""
        return self.representation.number_means()

    @property
    def number_cov(self) -> RealMatrix:
        r"""Returns the complete photon number covariance matrix."""
        return self.representation.number_cov()

    @property
    def norm(self) -> float:  # NOTE: risky: need to make sure user knows what kind of norm
        r"""Returns the norm of the state."""
        return self.representation.norm

    @property
    def state_probability(self) -> float:  # TODO: fix this
        r"""Returns the probability of the state."""
        norm = self.norm
        if isinstance(self.representation, FockKet):
            return norm**2
        return norm

    def fock_probabilities(self, cutoffs: Sequence[int]) -> RealTensor:
        r"""Returns the probabilities in Fock representation.

        If the state is pure, they are the absolute value squared of the ket amplitudes.
        If the state is mixed they are the multi-dimensional diagonals of the density matrix.

        Args:
            cutoffs List[int]: the cutoff dimensions for each mode

        Returns:
            Tensor: the probabilities
        """
        # TODO: deal with the cutoff issue here
        if isinstance(self.representation, (FockKet, FockDM)):
            return self.representation.probabilities(cutoffs)

    def primal(self, other: Union[State, Transformation]) -> State:
        r"""Returns the post-measurement state after ``other`` is projected onto ``self``.

        ``other << self`` is other projected onto ``self``.

        If ``other`` is a ``Transformation``, it returns the dual of the transformation applied to
        ``self``: ``other << self`` is like ``self >> other^dual``.

        Note that the returned state is not normalized. To normalize a state you can use
        ``mrmustard.physics.normalize``.
        """
        # TODO: touch this primal when refactor measurement

        if isinstance(other, State):
            return self._project_onto_state(other)
        try:
            return other.dual(self)
        except AttributeError as e:
            raise TypeError(
                f"Cannot apply {other.__class__.__qualname__} to {self.__class__.__qualname__}"
            ) from e

    def _project_onto_state(self, other: State) -> Union[State, float]:
        r"""If states are gaussian use generaldyne measurement, else use
        the states' Fock representation."""

        # if both states are gaussian
        if self.is_wigner and other.is_wigner:
            return self._project_onto_gaussian(other)

        # either self or other is not gaussian
        return self._project_onto_fock(other)

    def _project_onto_fock(self, other: State) -> Union[State, float]:
        r"""Returns the post-measurement state of the projection between two non-Gaussian
        states on the remaining modes or the probability of the result. When doing homodyne sampling,
        returns the post-measurement state or the measument outcome if no modes remain.

        Args:
            other (State): state being projected onto self

        Returns:
            State or float: returns the conditional state on the remaining modes
                or the probability.
        """
        remaining_modes = list(set(other.modes) - set(self.modes))

        out_fock = self._contract_with_other(other)
        if len(remaining_modes) > 0:
            return (
                DM(fock_array=out_fock, modes=remaining_modes)
                if other.is_mixed or self.is_mixed
                else Ket(fock_array=out_fock, modes=remaining_modes)
            )

        # return the probability (norm) of the state when there are no modes left
        return (
            fock.math.abs(out_fock) ** 2
            if other.is_pure and self.is_pure
            else fock.math.abs(out_fock)  # TODO check this
        )

    def _contract_with_other(self, other):
        other_cutoffs = [
            None if m not in self.modes else other.cutoffs[other.indices(m)] for m in other.modes
        ]
        if hasattr(self, "_preferred_projection"):
            out_fock = self._preferred_projection(other, other.indices(self.modes))
        else:
            # matching other's cutoffs
            self_cutoffs = [other.cutoffs[other.indices(m)] for m in self.modes]
            out_fock = fock.contract_states(
                stateA=other.ket(other_cutoffs) if other.is_pure else other.dm(other_cutoffs),
                stateB=self.ket(self_cutoffs) if self.is_pure else self.dm(self_cutoffs),
                a_is_dm=other.is_mixed,
                b_is_dm=self.is_mixed,
                modes=other.indices(self.modes),
                normalize=self._normalize if hasattr(self, "_normalize") else False,
            )

        return out_fock

    def _project_onto_gaussian(self, other: State) -> Union[State, float]:
        r"""Returns the result of a generaldyne measurement given that states ``self`` and
        ``other`` are gaussian.

        Args:
            other (State): gaussian state being projected onto self

        Returns:
            State or float: returns the output conditional state on the remaining modes
                or the probability.
        """
        # here `self` is the measurement device state and `other` is the incoming state
        # being projected onto the measurement state
        remaining_modes = list(set(other.modes) - set(self.modes))

        _, probability, new_cov, new_means = gaussian.general_dyne(
            other.cov,
            other.means,
            self.cov,
            self.means,
            self.modes,
        )

        if len(remaining_modes) > 0:
            return State(
                means=new_means,
                cov=new_cov,
                modes=remaining_modes,
                _norm=probability if not getattr(self, "_normalize", False) else 1.0,
            )

        return probability

    # def __iter__(self) -> Iterable[State]:
    #     """Iterates over the modes and their corresponding tensors."""
    #     return (self.get_modes(i) for i in range(self.representation.num_modes))

    def __and__(self, other: State) -> State:
        r"""Concatenates two states."""
        return self.representation.data.__and__(other)

    def __getitem__(self, item) -> State:  # NOTE should this be getting the modes instead?
        "setting the modes of a state (same API of `Transformation`)"
        if isinstance(item, int):
            item = [item]
        elif isinstance(item, Iterable):
            item = list(item)
        else:
            raise TypeError("item must be int or iterable")
        if len(item) != self.num_modes:
            raise ValueError(
                f"there are {self.num_modes} modes (item has {len(item)} elements, perhaps you're looking for .get_modes()?)"
            )
        self._modes = item  # TODO return a new state with the modes set instead of mutating
        return self

    def get_modes(self, item) -> State:
        r"""Returns the state on the given modes."""
        if isinstance(item, int):
            item = [item]
        elif isinstance(item, Iterable):
            item = list(item)
        else:
            raise TypeError("item must be int or iterable")

        if item == self._modes:
            return self

        if not set(item) & set(self._modes):
            raise ValueError(
                f"Failed to request modes {item} for state {self} on modes {self._modes}."
            )
        item_idx = [self._modes.index(m) for m in item]
        if self.is_wigner:
            cov, _, _ = gaussian.partition_cov(self.cov, item_idx)
            means, _ = gaussian.partition_means(self.means, item_idx)
            return State(cov=cov, means=means, modes=item)

        fock_partitioned = fock.trace(
            self.dm(self.cutoffs), keep=item_idx
        )  # TODO: this self.cutoffs is not correct now with the new structure
        return State(dm=fock_partitioned, modes=item)

    def __eq__(self, other) -> bool:  # pylint: disable=too-many-return-statements
        r"""Returns whether the states are equal."""
        return self.representation.data.__eq__(other)

    def __rshift__(self, other: Transformation) -> State:
        r"""Applies other (a Transformation) to self (a State), e.g., ``Coherent(x=0.1) >> Sgate(r=0.1)``."""
        if issubclass(other.__class__, State):
            raise TypeError(
                f"Cannot apply {other.__class__.__qualname__} to a state. Are you looking for the << operator?"
            )
        return other.primal(self)

    def __lshift__(self, other: State):
        r"""Implements projection onto a state or the dual transformation applied on a state.

        E.g., ``self << other`` where other is a ``State`` and ``self`` is either a ``State`` or a ``Transformation``.
        """
        return other.primal(self)

    def __add__(self, other: State):
        r"""Implements a mixture of states (only available in fock representation for the moment)."""
        return self.representation.data.__add__(other)

    def __rmul__(self, other):
        r"""Implements multiplication by a scalar from the left.

        E.g., ``0.5 * psi``.
        """
        return self.representation.data.__rmul__(other)

    def __mul__(self, other: State):
        r"""Implements multiplication of two objects."""
        return self.representation.data.__rmul__(other)

    def __truediv__(self, other):
        r"""Implements division by a scalar from the left.

        E.g. ``psi / 0.5``
        """
        return self.representation.data.__truediv__(other)

    @staticmethod
    def _format_probability(prob: float) -> str:
        if prob < 0.001:
            return f"{100*prob:.3e} %"
        else:
            return f"{prob:.3%}"

    def to_Bargmann(self):
        r"""Converts the representation of the state to Bargmann Representation and returns self.

        Returns:
            State: the converted state with the target Bargmann Representation
        """
        self.representation = converter.convert(self.representation, "Bargmann")
        return self

    def to_Fock(
        self,
        max_prob: Optional[float] = None,
        max_photon: Optional[int] = None,
        cutoffs: Optional[List[int]] = None,
    ):
        r"""Converts the representation of the state to Fock Representation and returns self.

        Args:
            max_prob (optional float): The maximum probability of the state. Defaults to settings.AUTOCUTOFF_PROBABILITY.
                (used to stop the calculation of the amplitudes early)
            max_photons (optional int): The maximum number of photons in the state, summing over all modes
                (used to stop the calculation of the amplitudes early)
            cutoffs (optional List[int]): The cutoffs of the desired Fock tensor. Defaults to autocutoffs.

        Returns:
            State: the converted state with the target Fock Representation
        """
        self.representation = converter.convert(
            self.representation,
            "Fock",
            max_prob=max_prob or settings.AUTOCUTOFF_PROBABILITY,
            max_photon=max_photon,
            cutoffs=cutoffs,
        )
        return self

    def to_WaveFunctionQ(self):
        r"""Converts the representation of the state to q-wavefunction Representation and returns self.

        Returns:
            State: the converted state with the target q-Wavefunction Representation
        """
        self.representation = converter.convert(self.representation, "WaveFunctionQ")
        return self

    def to_Wigner(self):
        r"""Converts the representation of the state to Wigner Representation and returns self.

        Returns:
            State: the converted state with the target Wigner Representation
        """
        self.representation = converter.convert(self.representation, "Wigner")
        return self

    def _repr_markdown_(self):
        r"""Prints the table to show the properties of the state."""
        table = (
            f"#### {self.__class__.__qualname__}\n\n"
            + "| Purity | Probability | Num modes | Bosonic size | Representation |\n"
            + "| :----: | :----: | :----: | :----: | :----: |\n"
            + f"| {self.purity :.2e} | "
            + self._format_probability(self.state_probability)
            if self.norm
            else "N/A"
            + f" | {self.representation.num_modes} | {'1' if isinstance(self.representation, (WignerKet, WignerDM)) else 'N/A'} | {'✅' if isinstance(self.representation, (WignerKet, WignerDM)) else '❌'} | {'✅' if isinstance(self.representation, (FockKet, FockDM)) else '❌'} |"
        )

        if self.num_modes == 1:
            graphics.mikkel_plot(math.asnumpy(self.dm(cutoffs=self.representation.data.cutoffs)))

        # TODO:
        # if settings.DEBUG:
        #     detailed_info = f"\ncov={repr(self.cov)}\n" + f"means={repr(self.means)}\n"
        #     return f"{table}\n{detailed_info}"

        return table


class Ket(State):
    def _from_Abc(self, A, b, c):
        self.representation = BargmannKet(A, b, c)

    def _from_cov_means(self, cov, means):
        # decomposes cov into SS^T and uses S internally
        symplectic = math.cholesky(cov)
        self.representation = WignerKet(symplectic, means)

    def _from_fock_array(self, fock_array):
        self.representation = FockKet(fock_array)

    def _from_wf_q_array(self, qs, wf_q_array):
        self.representation = WaveFunctionQKet(qs, wf_q_array)

    @property
    def purity(self) -> float:
        r"""Returns the purity of the state."""
        return 1.0

    @property
    def cov(self):  # override because Ket doesn't have a covariance matrix per se
        r"""Returns the covariance matrix of the state."""
        S = self.representation.symplectic
        return math.matmul(S, math.transpose(S))

    def ket(self, cutoffs: List[int] = None) -> Optional[ComplexTensor]:
        r"""Returns the ket of the state if it is in Fock representation.

        Args:
            cutoffs List[int or None]: The cutoff dimensions for each mode. If a mode cutoff is
                ``None``, it's guessed automatically.

        Returns:
            Tensor: the ket
        """
        if isinstance(self.representation, FockKet):
            if cutoffs is None:
                return self.representation.data.array
            else:  # TODO: cutoffs could be smaller too!
                return fock.pad_array_with_cutoffs(self.representation.data.array, cutoffs)
        else:
            raise AttributeError(
                "Use .to_Fock to transform the state to Fock representation first."
            )

    def dm(self, cutoffs: Optional[List[int]] = None) -> ComplexTensor:
        pass  # TODO implement from ket


class DM(State):
    def _from_Abc(self, A, b, c):
        self.representation = BargmannDM(A, b, c)

    def _from_cov_means(self, cov, means):
        self.representation = WignerDM(cov, means, check_purity=True)

    def _from_fock_array(self, fock_array):
        self.representation = FockDM(fock_array)

    def _from_wf_q_array(self, qs, wf_q_array):
        self.representation = WaveFunctionQDM(qs, wf_q_array)

    def ket(self, cutoffs: Optional[List[int]] = None) -> Optional[ComplexTensor]:
        pass  # compute only if state is pure

    def dm(self, cutoffs: Optional[List[int]] = None) -> ComplexTensor:
        r"""Returns the density matrix of the state in Fock representation.

        Args:
            cutoffs (optional List[int]): The cutoff dimensions for each mode. If a mode cutoff is ``None``,
                it's automatically computed.

        Returns:
            Tensor: the density matrix
        """

        if isinstance(self.representation, FockDM):
            if cutoffs is None:
                cutoffs = [settings.AUTOCUTOFF_MAX_CUTOFF for _ in range(2 * self.num_modes)]
            return fock.pad_array_with_cutoffs(
                self.representation.data.array, cutoffs
            )  # TODO: use shape (this is a dm)
        else:
            raise AttributeError(
                "Use .to_Fock to transform the state to Fock representation first."
            )
