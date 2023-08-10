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
from mrmustard.physics import fock, gaussian, bargmann
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


# pylint: disable=too-many-instance-attributes
class State:  # pylint: disable=too-many-public-methods
    r"""Base class for quantum states."""

    def __init__(
        self,
        cov: RealMatrix = None,
        means: RealVector = None,
        eigenvalues: RealVector = None,
        symplectic: RealMatrix = None,
        ket: ComplexTensor = None,
        dm: ComplexTensor = None,
        A: ComplexMatrix = None,
        B: ComplexVector = None,
        C: float = None,
        modes: Sequence[int] = None,
        cutoffs: Sequence[int] = None,
    ):
        r"""Initializes the state.

        Supply either:
            * a covariance matrix and means vector
            * a symplectic matrix and displacement vector
            * a fock representation (ket or dm)
            * a ABC triple related to the Bargmann representation

        Args:
            cov (Matrix): the covariance matrix
            means (Vector): the means vector
            symplectic (Matrix): the symplectic matrix
            eigenvalues (Vector): the eigenvalues of the state
            ket (Tensor): the ket of the Fock representation
            dm (Tensor): the density matrix of the Fock representation
            A (Matrix): the matrix of the Bargmann representation
            B (Vector): the vector of the Bargmann representation
            C (Scalar): the coefficient of the Bargmann representation
            modes (optional, Sequence[int]): the modes in which the state is defined
            cutoffs (Sequence[int], default=None): set to force the cutoff dimensions of the state

        """
        self._cutoffs = cutoffs
        #Case 1: Wigner representation
        if cov is not None and means is not None:
            self.is_gaussian = True
            self.num_modes = cov.shape[-1] // 2
            #check if the state is pure and init with different representation classes
            self.representation = WignerKet(cov, means) if np.allclose(gaussian.purity(self.cov, settings.HBAR), 1.0) else WignerDM(cov, means)
        # #TODO: what is this eigenvalues? a list?
        # elif eigenvalues is not None and symplectic is not None:
        #     self.is_gaussian = True
        #     self.num_modes = symplectic.shape[-1] // 2
        #     means = math.zeros_like(self.num_modes)
        #     cov = (settings.HBAR / 2) * math.matmul(symplectic, math.matmul(eigenvalues, symplectic))
        #     self.representation = WignerKet(cov, means) if np.allclose(eigenvalues, 2.0 / settings.HBAR) else WignerDM(cov, means)
        #Case 2: Fock representation
        elif ket is not None:
            self.is_gaussian = False
            self.num_modes = len(ket.shape)
            self._purity = 1.0
            self.representation = FockKet(ket)
        elif dm is not None:
            self.is_gaussian = False
            self.num_modes = len(dm.shape) // 2
            self._purity = None
            self.representation = FockDM(dm)
        #Case 3: Bargmann representation
        elif A is not None and B is not None and C is not None:
            self.is_gaussian = True or False 
            self.representation = BargmannKet(A,B,C)
            #TODO: pure or mixed bargmann you are talking about? is_gaussian?
        else:
            raise ValueError(
                "State must be initialized with either a wrapped Representation class, a covariance matrix and means vector, a symplectic matrix and displacement, or a ket or dm in fock representation."
            )
                

        self._modes = modes
        if modes is not None:
            assert (
                len(modes) == self.num_modes
            ), f"Number of modes supplied ({len(modes)}) must match the representation dimension {self.num_modes}"


    @property
    def modes(self):
        r"""Returns the modes of the state.
        """
        if self._modes is None:
            return list(range(self.num_modes))
        return self._modes
    
    #TODO: something here?
    def indices(self, modes) -> Union[Tuple[int], int]:
        r"""Returns the indices of the given modes.

        Args:
            modes (Sequence[int] or int): the modes or mode

        Returns:
            Tuple[int] or int: a tuple of indices of the given modes or the single index of a single mode
        """
        if isinstance(modes, int):
            return self.modes.index(modes)
        return tuple(self.modes.index(m) for m in modes)
    

    @property
    def purity(self) -> float:
        """Returns the purity of the state."""
        return self.representation.purity

    @property
    def is_mixed(self):
        r"""Returns whether the state is mixed."""
        return not self.is_pure

    @property
    def is_pure(self):
        r"""Returns ``True`` if the state is pure and ``False`` otherwise."""
        return np.isclose(self.purity, 1.0, atol=1e-6)
    

    @property
    def means(self) -> Optional[RealVector]:
        r"""Returns the means vector of the state."""
        if isinstance(self.representation, (WignerKet, WignerDM)):
            return self.representation.data.means
        else:
            raise AttributeError("The representation of your state do not have this attribute, transform it with the Converter please!")

    @property
    def cov(self) -> Optional[RealMatrix]:
        r"""Returns the covariance matrix of the state."""
        if isinstance(self.representation, (WignerKet, WignerDM)):
            return self.representation.data.cov
        else:
            raise AttributeError("The representation of your state do not have this attribute, transform it with the Converter please!") from e

    @property
    def number_stdev(self) -> RealVector:
        r"""Returns the square root of the photon number variances (standard deviation) in each mode."""
        if isinstance(self.representation, (FockKet, FockDM)):
            return self.representation.number_stdev
        else:
            raise AttributeError("The representation of your state do not have this attribute, transform it with the Converter please!")
    
    
    @property
    def cutoffs(self) -> List[int]:
        r"""Returns the cutoff dimensions for each mode."""
        if not isinstance(self.representation, (FockKet, FockDM, WignerKet, WignerDM)):
            raise AttributeError("The representation of your state do not have this attribute, transform it with the Converter please!")

        if isinstance(self.representation, (FockKet, FockDM)):
            self._cutoffs = self.representation.data.cutoffs
        elif isinstance(self.representation, (WignerKet, WignerDM)):
            self._cutoffs = fock.autocutoffs(
                    self.cov, self.means, settings.AUTOCUTOFF_PROBABILITY
                )
        return self._cutoffs
            
    # @property
    # #TODO: Depends on the representation. Shape means something else.
    # def shape(self) -> List[int]:
    #     r"""Returns the shape of the state.
    #     """
    #     return self.cutoffs if self.is_pure else self.cutoffs + self.cutoffs


    @property
    def number_means(self) -> RealVector:
        r"""Returns the mean photon number for each mode."""
        if not isinstance(self.representation, (FockKet, FockDM, WignerKet, WignerDM)):
            raise AttributeError("The representation of your state do not have this attribute, transform it with the Converter please!")
        return self.representation.number_means

    @property
    def number_cov(self) -> RealMatrix:
        r"""Returns the complete photon number covariance matrix."""
        if not isinstance(self.representation, (WignerKet, WignerDM)):
            raise AttributeError("The representation of your state do not have this attribute, transform it with the Converter please!")
        return self.representation.number_cov


    @property
    def norm(self) -> float:
        r"""Returns the norm of the state."""
        if not isinstance(self.representation, (FockKet, FockDM, WaveFunctionQKet, WaveFunctionQDM)):
            raise AttributeError("The representation of your state do not have this attribute, transform it with the Converter please!")
        return self.representation.norm
   

    @property
    def probability(self) -> float:
        r"""Returns the probability of the state."""
        norm = self.norm
        if isinstance(self.representation, FockKet):
            return norm**2
        return norm


    def ket(
        self,
        cutoffs: List[int] = None,
        max_prob: float = 1.0,
        max_photons: int = None,
    ) -> Optional[ComplexTensor]:
        r"""Returns the ket of the state in Fock representation or ``None`` if the state is mixed.

        Args:
            cutoffs List[int or None]: The cutoff dimensions for each mode. If a mode cutoff is
                ``None``, it's guessed automatically.
            max_prob (float): The maximum probability of the state. Defaults to 1.0.
                (used to stop the calculation of the amplitudes early)
            max_photons (int): The maximum number of photons in the state, summing over all modes
                (used to stop the calculation of the amplitudes early)

        Returns:
            Tensor: the ket
        """

        if self.is_mixed:
            return None
        #TODO: which one is better? two lines above and below.
        # if self.representation.__class__.__name__.endswith('DM'):
        #     raise AttributeError("We do not support to decompose the density matrix to ket now!")
        
        if isinstance(self.representation, FockKet):
            if cutoffs is None:
                return self.representation.data.array
            return fock.pad_array_with_cutoffs(self.representation.data.array, cutoffs)
    
        if isinstance(self.representation, WignerKet):
            cutoffs = cutoffs or self._cutoffs
            self_copy = copy.deepcopy(self)
            return self_copy.to_Fock(cutoffs=cutoffs, max_prob=max_prob, max_photon=max_photons).ket()
        
        raise AttributeError("The representation of your state do not have this attribute, transform it with the Converter please!")

            

    def dm(self, cutoffs: Optional[List[int]] = None) -> ComplexTensor:
        r"""Returns the density matrix of the state in Fock representation.

        Args:
            cutoffs List[int]: The cutoff dimensions for each mode. If a mode cutoff is ``None``,
                it's automatically computed.

        Returns:
            Tensor: the density matrix
        """
        if not isinstance(self.representation, (FockKet, FockDM, WignerKet, WignerDM)):
            raise AttributeError("The representation of your state do not have this attribute, transform it with the Converter please!")
        
        if isinstance(self.representation, FockDM):
            if cutoffs is None:
                return self.representation.data.array
            else:
                return fock.pad_array_with_cutoffs(self.representation.data.array, cutoffs)
        
        if isinstance(self.representation, FockKet):
            if cutoffs is None:
                return fock.ket_to_dm(self.representation.data.array)
            else:
                return fock.pad_array_with_cutoffs(fock.ket_to_dm(self.representation.data.array), cutoffs) 
        
        if isinstance(self.representation, WignerKet):
            if cutoffs is None:
                cutoffs = self.cutoffs
            self_copy = copy.deepcopy(self)
            self_copy.to_Fock()
            return fock.pad_array_with_cutoffs(fock.ket_to_dm(self_copy.ket()), cutoffs)
        
        if isinstance(self.representation, WignerDM):
            self_copy = copy.deepcopy(self)
            self_copy.to_Fock(cutoffs=cutoffs)
            return self_copy.representation.data.array


    @property
    def fock(self) -> ComplexTensor:
        r"""Returns the Fock representation of the state."""
        if not isinstance(self.representation, (FockKet, FockDM, WignerKet, WignerDM)):
            raise AttributeError("The representation of your state do not have this attribute, transform it with the Converter please!")
        
        return self.ket if isinstance(self.representation, (FockKet, WignerKet)) else self.dm


    @property
    def fock_probabilities(self, cutoffs: Sequence[int]) -> RealTensor:
        r"""Returns the probabilities in Fock representation.

        If the state is pure, they are the absolute value squared of the ket amplitudes.
        If the state is mixed they are the multi-dimensional diagonals of the density matrix.

        Args:
            cutoffs List[int]: the cutoff dimensions for each mode

        Returns:
            Tensor: the probabilities
        """
        if isinstance(self.representation, (FockKet, FockDM)):
            return self.representation.probability(cutoffs)
        else:
            raise AttributeError("The representation of your state do not have this attribute, transform it with the Converter please!")


    def primal(self, other: Union[State, Transformation]) -> State:
        r"""Returns the post-measurement state after ``other`` is projected onto ``self``.

        ``other << self`` is other projected onto ``self``.

        If ``other`` is a ``Transformation``, it returns the dual of the transformation applied to
        ``self``: ``other << self`` is like ``self >> other^dual``.

        Note that the returned state is not normalized. To normalize a state you can use
        ``mrmustard.physics.normalize``.
        """
 
        if isinstance(other, State):
            return self._project_onto_state(other)
        try:
            return other.dual(self)
        except AttributeError as e:
            raise TypeError(
                f"Cannot apply {other.__class__.__qualname__} to {self.__class__.__qualname__}"
            ) from e
        

    def _project_onto_state(self, other: State) -> Union[State, float]:
        """If states are gaussian use generaldyne measurement, else use
        the states' Fock representation."""

        # if both states are gaussian
        if self.is_gaussian and other.is_gaussian:
            return self._project_onto_gaussian(other)

        # either self or other is not gaussian
        return self._project_onto_fock(other)


    def _project_onto_fock(self, other: State) -> Union[State, float]:
        """Returns the post-measurement state of the projection between two non-Gaussian
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
                State(dm=out_fock, modes=remaining_modes)
                if other.is_mixed or self.is_mixed
                else State(ket=out_fock, modes=remaining_modes)
            )

        # return the probability (norm) of the state when there are no modes left
        return (
            math.abs(out_fock) ** 2
            if other.is_pure and self.is_pure
            else fock.math.abs(out_fock)
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
        """Returns the result of a generaldyne measurement given that states ``self`` and
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


    def __iter__(self) -> Iterable[State]:
        """Iterates over the modes and their corresponding tensors."""
        return (self.get_modes(i) for i in range(self.num_modes))


    def __and__(self, other: State) -> State:
        r"""Concatenates two states."""
        return self.representation.data.__and__(other)


    def __getitem__(self, item) -> State:
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
        self._modes = item
        return self
    

    def bargmann(self, numpy=False) -> Optional[tuple[ComplexMatrix, ComplexVector, complex]]:
        r"""Returns the Bargmann representation of the state.
        If numpy=True, returns the numpy arrays instead of the backend arrays.
        """
        if self.is_gaussian:
            if self.is_pure:
                A, B, C = bargmann.wigner_to_bargmann_psi(self.cov, self.means)
            else:
                A, B, C = bargmann.wigner_to_bargmann_rho(self.cov, self.means)
        else:
            return None
        if numpy:
            return math.asnumpy(A), math.asnumpy(B), math.asnumpy(C)
        return A, B, C


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
        if self.is_gaussian:
            cov, _, _ = gaussian.partition_cov(self.cov, item_idx)
            means, _ = gaussian.partition_means(self.means, item_idx)
            return State(cov=cov, means=means, modes=item)

        fock_partitioned = fock.trace(self.dm(self.cutoffs), keep=item_idx) #TODO: this self.cutoffs is not correct now with the new structure
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


    #TODO: ask for this implementation of rmul and mul!
    def __rmul__(self, other):
        r"""Implements multiplication by a scalar from the left.

        E.g., ``0.5 * psi``.
        """
        return self.representation.data.__rmul__(other)
    

    def __mul__(self, other: State):
        r"""Implements multiplication of two objects.
        """
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
        r'''Converts the representation of the state to Bargmann Representation and returns a new State.
        Args:
            source (State): the state itself
            target (Srting): the target Bargmann representation of the state

        Returns:
            State: the converted state with the target Bargmann Representation
        '''
        self.representation = converter.convert(self.representation, "Bargmann") 
        return self
        

    def to_Fock(self, max_prob: float = 1.0, max_photon: int = None, cutoffs: Union(List[int], int) = None):        
        r"""Converts the representation of the state to Fock Representation and returns a new State.

        Args:
            source (State): the state itself
            target (Srting): the target Fock representation of the state
            max_prob (float): The maximum probability of the state. Defaults to 1.0.
                (used to stop the calculation of the amplitudes early)
            max_photons (int): The maximum number of photons in the state, summing over all modes
                (used to stop the calculation of the amplitudes early)
            cutoffs (List or int): The cutoffs of the desired Fock tensor

        Returns:
            State: the converted state with the target Fock Representation
        """
        self.representation = converter.convert(self.representation, "Fock", max_prob=max_prob, max_photon=max_photon, cutoffs=cutoffs)
        return self
        

    def to_WaveFunctionQ(self, qs):
        r'''Converts the representation of the state to q-wavefunction Representation and returns a new State.

        Args:
            source (State): the state itself
            qs(array): list of values on q-axis of the wavefunction
            target (Srting): the target q-Wavefunction representation of the state

        Returns:
            State: the converted state with the target q-Wavefunction Representation
        '''
        self.representation = converter.convert(self.representation, qs, "WaveFunctionQ") 
        return self
        

    def to_Wigner(self):
        r'''Converts the representation of the state to Wigner Representation and returns a new State.

        Args:
            source (State): the state itself
            target (Srting): the target Wigner representation of the state

        Returns:
            State: the converted state with the target Wigner Representation
        '''
        self.representation = converter.convert(self.representation, "Wigner")
        return self


    def _repr_markdown_(self):
        table = (
            f"#### {self.__class__.__qualname__}\n\n"
            + "| Purity | Probability | Num modes | Bosonic size | Gaussian | Fock |\n"
            + "| :----: | :----: | :----: | :----: | :----: | :----: |\n"
            + f"| {self.purity :.2e} | "
            + self._format_probability(self.probability)
            + f" | {self.num_modes} | {'1' if self.is_gaussian else 'N/A'} | {'✅' if self.is_gaussian else '❌'} | {'✅' if self._ket is not None or self._dm is not None else '❌'} |"
        )

        if self.num_modes == 1:
            graphics.mikkel_plot(math.asnumpy(self.dm(cutoffs=self.cutoffs)))

        if settings.DEBUG:
            detailed_info = f"\ncov={repr(self.cov)}\n" + f"means={repr(self.means)}\n"
            return f"{table}\n{detailed_info}"

        return table
