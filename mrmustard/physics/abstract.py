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

from __future__ import annotations
from abc import ABC, abstractmethod
from mrmustard.physics import gaussian, fock
from mrmustard.utils.types import *
from mrmustard import settings

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~ State ~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class State:
    r"""Base class for quantum states"""
    def __init__(self, mixed: bool = None, cov=None, means=None, fock=None):
        r"""
        Initializes the state. Either supply the cov,means pair of the fock tensor.
        Arguments:
            mixed (bool): whether the state is mixed
            cov (Matrix): the covariance matrix
            means (Vector): the means vector
            fock (Array): the Fock representation
        """
        if (cov is None) != (means is None) or (cov is None) != (fock is None):
            raise ValueError("either cov and means or fock must be supplied")
        self._num_modes = None
        if mixed is not None:
            self.is_mixed: bool = mixed
        self._fock = fock
        self._means = means
        self._cov = cov

    @classmethod
    def from_gaussian(cls, cov: Matrix, means: Vector, mixed: bool) -> State:
        r"""
        Returns a state from a Gaussian distribution.
        Arguments:
            cov (Matrix): the covariance matrix
            means (Vector): the means vector
            mixed (bool): whether the state is mixed
        Returns:
            State: the state
        """
        return cls(mixed, cov, means)

    @classmethod
    def from_fock(cls, fock: Array, mixed: bool) -> State:
        r"""
        Returns a state from a Fock representation.
        Arguments:
            fock (Array): the Fock representation
            mixed (bool): whether the state is mixed
        Returns:
            State: the state
        """
        return cls(mixed=mixed, fock=fock)

    @property
    def num_modes(self) -> int:
        r"""
        Returns the number of modes in the state.
        """
        if self._num_modes is None:
            if self._fock is not None:
                num_indices = len(self._fock.shape)
                self._num_modes = num_indices if self.is_pure else num_indices // 2
            else:
                self._num_modes = self.means.shape[-1] // 2
        return self._num_modes

    @property
    def is_pure(self):
        return not self.is_mixed

    @property
    def means(self) -> Optional[Vector]:
        r"""
        Returns the means vector of the state.
        """
        return self._means

    @property
    def cov(self) -> Optional[Matrix]:
        r"""
        Returns the covariance matrix of the state.
        """
        return self._cov

    @property
    def number_means(self) -> Vector:
        r"""
        Returns the mean photon number for each mode
        """
        try:
            return gaussian.number_means(self.cov, self.means, settings.HBAR)
        except ValueError:
            return gaussian.number_means(self._fock)

    @property
    def number_cov(self) -> Matrix:
        r"""
        Returns the complete photon number covariance matrix
        """
        try:
            return gaussian.number_cov(self.cov, self.means, settings.HBAR)
        except ValueError:
            return gaussian.number_cov(self._fock)

    def ket(self, cutoffs: Sequence[int]) -> Optional[Tensor]:
        r"""
        Returns the ket of the state in Fock representation or `None` if the state is mixed.
        Arguments:
            cutoffs List[int]: the cutoff dimensions for each mode
        Returns:
            Tensor: the ket
        """
        if self.is_pure:
            if self.means is not None:  # TODO: this may trigger recomputation of means: find a better way
                self._fock = fock.fock_representation(self.cov, self.means, cutoffs=cutoffs, mixed=False)
            return self._fock
        else:
            return None

    def dm(self, cutoffs: List[int]) -> Tensor:
        r"""
        Returns the density matrix of the state in Fock representation.
        Arguments:
            cutoffs List[int]: the cutoff dimensions for each mode
        Returns:
            Tensor: the density matrix
        """
        if self.is_pure:
            ket = self.ket(cutoffs=cutoffs)
            return fock.ket_to_dm(ket)
        else:
            if self.means is not None:  # TODO: this may trigger recomputation of means: find a better way
                self._fock = fock.fock_representation(self.cov, self.means, cutoffs=cutoffs, mixed=True)
            return self._fock

    def fock_probabilities(self, cutoffs: Sequence[int]) -> Tensor:
        r"""
        Returns the probabilities in Fock representation. If the state is pure, they are
        the absolute value squared of the ket amplitudes. If the state is mixed they are
        the diagonals of the density matrix.
        Arguments:
            cutoffs List[int]: the cutoff dimensions for each mode
        Returns:
            Array: the probabilities
        """
        if self.is_mixed:
            dm = self.dm(cutoffs=cutoffs)
            return fock.dm_to_probs(dm)
        else:
            ket = self.ket(cutoffs=cutoffs)
            return fock.ket_to_probs(ket)

    def __and__(self, other: State) -> State:
        r"""
        Concatenates two states.
        """
        cov = gaussian.join_covs([self.cov, other.cov])
        means = gaussian.join_means([self.means, other.means])
        return State.from_gaussian(cov, means, self.is_mixed or other.is_mixed)

    def __getitem__(self, item):
        r"""
        Returns the state on the given modes.
        """
        if isinstance(item, int):
            item = [item]
        elif isinstance(item, Iterable):
            item = list(item)
        else:
            raise TypeError("item must be int or iterable")
        cov, _, _ = gaussian.partition_cov(self.cov, item)
        means, _ = gaussian.partition_means(self.means, item)
        return State.from_gaussian(cov, means, gaussian.is_mixed_cov(cov))

    def __eq__(self, other):
        r"""
        Returns whether the states are equal.
        """
        if not allclose(self.means, other.means):
            return False
        if not allclose(self.cov, other.cov):
            return False
        return True

    def __repr__(self):
        info = f"num_modes={self.num_modes} | pure={self.is_pure}\n"
        detailed_info = f"\ncov={repr(self.cov)}\n" + f"means={repr(self.means)}\n"
        if self.num_modes == 1:
            if self._fock is not None:
                cutoffs = self._fock.shape if self.is_pure else self._fock.shape[:1]
            else:
                cutoffs = [20]
            graphics.mikkel_plot(self.dm(cutoffs=cutoffs))
        return info + "-" * len(info) + detailed_info


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~ Transformation ~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Transformation(ABC):
    r"""
    Base class for all transformations.
    Note that measurements are CP but not TP, so they have their own abstract class.
    Transformations include:
        * unitary transformations
        * non-unitary CPTP channels
    """

    def __call__(self, state: State) -> State:
        d = self.d_vector()
        X = self.X_matrix()
        Y = self.Y_matrix()
        cov, means = gaussian.CPTP(state.cov, state.means, X, Y, d, self.modes)
        return State.from_gaussian(cov, means, mixed=state.is_mixed or Y is not None)

    def __repr__(self):
        with np.printoptions(precision=6, suppress=True):
            lst = [f"{name}={np.array(np.atleast_1d(self.__dict__[name]))}" for name in self.param_names]
            return f"{self.__class__.__qualname__}(modes={self.modes}, {', '.join(lst)})"

    @property
    def modes(self) -> Sequence[int]:
        if self._modes in (None, []):
            if (d := self.d_vector()) is not None:
                self._modes = list(range(d.shape[-1] // 2))
            elif (X := self.X_matrix()) is not None:
                self._modes = list(range(X.shape[-1] // 2))
            elif (Y := self.Y_matrix()) is not None:
                self._modes = list(range(Y.shape[-1] // 2))
        return self._modes

    @property
    def bell(self):
        "N pairs of two-mode squeezed vacuum where N is the number of modes of the circuit"
        pass

    def X_matrix(self) -> Optional[Matrix]:
        return None

    def Y_matrix(self) -> Optional[Matrix]:
        return None

    def d_vector(self) -> Optional[Vector]:
        return None

    def fock(self, cutoffs=Sequence[int]):  # only single-mode for now
        unnormalized = self(self.bell).ket(cutoffs=cutoffs)
        return fock.normalize_choi_trick(unnormalized, settings.TMSV_DEFAULT_R)

    def trainable_parameters(self) -> Dict[str, List[Trainable]]:
        return {"symplectic": [], "orthogonal": [], "euclidean": self._trainable_parameters}

    def __getitem__(self, items) -> Callable:
        r"""
        Allows transformations to be used as:
        output = op[0,1](input)  # e.g. acting on modes 0 and 1
        """
        if isinstance(items, int):
            modes = [items]
        elif isinstance(items, slice):
            modes = list(range(items.start, items.stop, items.step))
        elif isinstance(items, (Sequence, Iterable)):
            modes = list(items)
        else:
            raise ValueError(f"{items} is not a valid slice or list of modes.")
        self._modes = modes
        return self

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~ Measurement ~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GaussianMeasurement(ABC):
    r"""
    Base class for all Gaussian measurements.
    """

    def __call__(self, state: State, **kwargs) -> Tuple[Scalar, State]:
        r"""
        Applies a general-dyne Gaussian measurement to the state, i.e. it projects
        onto the state with given cov and outcome means vector.
        Args:
            state (State): the state to be measured.
            kwargs (optional): same arguments as in the init, but specify if they are different
            from the arguments supplied at init time (e.g. for training the measurement).
        Returns:
            (float, state) The measurement probabilities and the remaining post-measurement state.
            Note that the post-measurement state is trivial if all modes are measured.
        """
        if len(kwargs) > 0:
            self._project_onto = self.recompute_project_onto(**kwargs)
        prob, cov, means = gaussian.general_dyne(
            state.cov, state.means, self._project_onto.cov, self._project_onto.means, self._modes, settings.HBAR
        )
        remaining_modes = [m for m in range(state.num_modes) if m not in self._modes]

        if len(remaining_modes) > 0:
            remaining_state = State.from_gaussian(cov, means, gaussian.is_mixed_cov(cov))  # TODO: avoid using is_mixed_cov from TW
            return prob, remaining_state
        else:
            return prob

    def recompute_project_onto(self, **kwargs) -> State:
        ...


# TODO: push all math methods into the physics module
class FockMeasurement(ABC):
    r"""
    A Fock measurement projecting onto a Fock measurement pattern.
    It works by representing the state in the Fock basis and then applying
    a stochastic channel matrix P(meas|n) to the Fock probabilities (belief propagation).
    It outputs the measurement probabilities and the remaining post-measurement state (if any)
    in the Fock basis.
    """

    def project(self, state: State, cutoffs: Sequence[int], measurement: Sequence[Optional[int]]) -> Tuple[State, Tensor]:
        r"""
        Projects the state onto a Fock measurement in the form [a,b,c,...] where integers
        indicate the Fock measurement on that mode and None indicates no projection on that mode.

        Returns the measurement probability and the renormalized state (in the Fock basis) in the unmeasured modes.
        """
        if (len(cutoffs) != state.num_modes) or (len(measurement) != state.num_modes):
            raise ValueError("the length of cutoffs/measurements does not match the number of modes")
        dm = state.dm(cutoffs=cutoffs)
        measured = 0
        for mode, (stoch, meas) in enumerate(zip(self._stochastic_channel, measurement)):
            if meas is not None:
                # put both indices last and compute sum_m P(meas|m)rho_mm for every meas
                last = [mode - measured, mode + state.num_modes - 2 * measured]
                perm = list(set(range(dm.ndim)).difference(last)) + last
                dm = fock.math.transpose(dm, perm)
                dm = fock.math.diag_part(dm)
                dm = fock.math.tensordot(dm, stoch[meas, : dm.shape[-1]], [[-1], [0]])
                measured += 1
        prob = fock.math.sum(fock.math.all_diagonals(dm, real=False))
        return fock.math.abs(prob), dm / prob

    def apply_stochastic_channel(self, stochastic_channel, fock_probs: Tensor) -> Tensor:
        cutoffs = [fock_probs.shape[m] for m in self._modes]
        for i, mode in enumerate(self._modes):
            if cutoffs[mode] > stochastic_channel[i].shape[1]:
                raise IndexError(
                    f"This detector does not support so many input photons in mode {mode} (you could increase max_input_photons or reduce the cutoff)"
                )
        detector_probs = fock_probs
        for i, mode in enumerate(self._modes):
            detector_probs = fock.math.tensordot(
                detector_probs,
                stochastic_channel[i][: cutoffs[mode], : cutoffs[mode]],
                [[mode], [1]],
            )
            indices = list(range(fock_probs.ndim - 1))
            indices.insert(mode, fock_probs.ndim - 1)
            detector_probs = fock.math.transpose(detector_probs, indices)
        return detector_probs

    def __call__(self, state: State, cutoffs: List[int], outcomes: Optional[Sequence[Optional[int]]] = None) -> Tuple[Tensor, Tensor]:
        fock_probs = state.fock_probabilities(cutoffs)
        all_probs = self.apply_stochastic_channel(self._stochastic_channel, fock_probs)
        if outcomes is None:
            return all_probs
        else:
            probs, dm = self.project(state, cutoffs, outcomes)
            return dm, probs

    def recompute_stochastic_channel(self, **kwargs) -> State:
        ...