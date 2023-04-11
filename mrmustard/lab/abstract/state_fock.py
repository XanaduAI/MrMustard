from typing import Iterable, List, Optional, Sequence, Union

from mrmustard.math import Math
from mrmustard.physics import fock
from mrmustard.types import Matrix, Tensor, Vector

math = Math()


class FockState:
    r"""Base class for quantum states in Fock representation."""

    def __init__(
        self,
        ket: Optional[Tensor] = None,
        dm: Optional[Tensor] = None,
    ):
        r"""Initializes the state.

        Supply either a ket or dm

        Args:
            ket (Array): the Fock representation as a ket
            dm (Array): the Fock representation as a density matrix
        """
        if ket is None and dm is None:
            raise ValueError("Either ket or dm must be provided.")
        self.num_modes = len(ket.shape) if ket is not None else len(dm.shape) // 2
        self.cutoffs = (
            ket.shape if ket is not None else dm.shape[: self.num_modes]
        )  # cutoffs refers to the modes, not the tensor axes
        self._ket = ket
        self._dm = dm
        self._fock_probabilities = None
        self.LR = dm is not None
        self._purity = 1.0 if ket is not None else None

    @property
    def purity(self) -> float:
        """Returns the purity of the state."""
        if self._purity is None:  # lazy
            self._purity = fock.purity(self._dm)
        return self._purity

    @property
    def is_mixed(self):
        r"""Returns whether the state is mixed."""
        return not self.is_pure

    @property
    def is_pure(self):
        r"""Returns ``True`` if the state is pure and ``False`` otherwise."""
        return True if self._ket is not None else np.isclose(self.purity, 1.0, atol=1e-6)

    @property
    def number_stdev(self) -> Vector:
        r"""Returns the square root of the photon number variances (standard deviation) in each mode."""
        return math.sqrt(fock.number_variances(self.dm() if self.LR else self.ket(), is_dm=self.LR))

    @property
    def shape(self) -> List[int]:
        r"""Returns the shape of the state, accounting for ket/dm representation.

        If the state is in Gaussian representation, the shape is inferred from
        the first two moments of the number operator.
        """
        return self.cutoffs if self._ket is not None else self.cutoffs + self.cutoffs

    @property
    def fock(self) -> Tensor:
        r"""Returns the Fock representation of the state: ket if available, dm otherwise."""
        return self._ket if self._ket is not None else self._dm

    @property
    def number_means(self) -> Vector:
        r"""Returns the mean photon number for each mode."""
        return fock.number_means(tensor=self.fock, is_dm=self.is_mixed)

    @property
    def number_cov(self) -> Matrix:
        r"""Returns the complete photon number covariance matrix."""
        raise NotImplementedError("number_cov not yet implemented for non-gaussian states")

    @property
    def norm(self) -> float:
        r"""Returns the norm of the state."""
        return fock.norm(self.fock, self._dm is not None)

    @property
    def probability(self) -> float:
        r"""Returns the probability of the state."""
        norm = self.norm
        if self.is_pure and self._ket is not None:
            return norm**2
        return norm

    @classmethod
    def from_gaussian(state: GaussianState, cutoffs: Sequence[int]) -> FockState:
        r"""Returns the Fock representation of a Gaussian state."""
        if state.is_pure:
            return FockState(
                ket=fock.wigner_to_fock_state(
                    state.cov, state.means, shape=state.cutoffs, return_dm=False
                )
            )
        return FockState(
            dm=fock.wigner_to_fock_state(
                state.cov, state.means, shape=state.cutoffs, return_dm=True
            )
        )

    def ket(self, cutoffs: Sequence[int] = None) -> Optional[Tensor]:
        r"""Returns the ket of the state in Fock representation or ``None`` if the state is mixed.
        If the requested cutoffs are below the current cutoffs, the ket is sliced.
        If the requested cutoffs are above the current cutoffs, the ket is padded with zeros as needed.

        Args:
            cutoffs List[int or None]: The cutoff dimensions for each mode. If cutoff[m] is
                ``None`` for mode m, the current cutoff is used.

        Returns:
            Tensor: the ket

        Example:
            >>> state = FockState(ket=np.array([1.0, 2.0, 3.0]))
            >>> state.ket(cutoffs=[2])
            array([1., 2.])
            >>> state.ket(cutoffs=[4])
            array([1., 2., 3., 0.])

        """
        if self.is_mixed:
            return None

        if cutoffs is None:
            cutoffs = self.cutoffs
        else:
            cutoffs = [
                c if c is not None else self.cutoffs[i] for i, c in enumerate(cutoffs)
            ]  # TODO: improve autocutoff for None values

        if self._ket is None:
            # if state is pure but has a density matrix, calculate the ket
            if self.is_pure:
                self._ket = fock.dm_to_ket(self._dm)
        current_cutoffs = list(self._ket.shape[: self.num_modes])
        if cutoffs != current_cutoffs:
            paddings = [(0, max(0, new - old)) for new, old in zip(cutoffs, current_cutoffs)]
            if any(p != (0, 0) for p in paddings):
                padded = fock.math.pad(self._ket, paddings, mode="constant")
            else:
                padded = self._ket
            return padded[tuple(slice(s) for s in cutoffs)]
        self.LR = False
        return self._ket[tuple(slice(s) for s in cutoffs)]

    def dm(self, cutoffs: List[int] = None) -> Tensor:
        r"""Returns the density matrix of the state in Fock representation.
        If the requested cutoffs are below the current cutoffs, the dm is sliced.
        If the requested cutoffs are above the current cutoffs, the dm is padded with zeros as needed.

        Args:
            cutoffs List[int]: The cutoff dimensions for each mode. If a mode cutoff is ``None``,
                the current cutoff is used.

        Returns:
            Tensor: the density matrix
        """
        if cutoffs is None:
            cutoffs = self.cutoffs
        else:
            cutoffs = [c if c is not None else self.cutoffs[i] for i, c in enumerate(cutoffs)]
        if self.is_pure:
            ket = self.ket(cutoffs=cutoffs)
            if ket is not None:
                return fock.ket_to_dm(ket)
        else:
            if cutoffs != (current_cutoffs := list(self._dm.shape[: self.num_modes])):
                paddings = [(0, max(0, new - old)) for new, old in zip(cutoffs, current_cutoffs)]
                if any(p != (0, 0) for p in paddings):
                    padded = fock.math.pad(self._dm, paddings + paddings, mode="constant")
                else:
                    padded = self._dm
                return padded[tuple(slice(s) for s in cutoffs + cutoffs)]
        return self._dm[tuple(slice(s) for s in cutoffs + cutoffs)]

    def fock_probabilities(self, cutoffs: Sequence[int]) -> Tensor:
        r"""Returns the probabilities in Fock representation.

        If the state is pure, they are the absolute value squared of the ket amplitudes.
        If the state is mixed they are the multi-dimensional diagonals of the density matrix.

        Args:
            cutoffs List[int]: the cutoff dimensions for each mode

        Returns:
            Array: the probabilities
        """
        if self._fock_probabilities is None:
            if self.is_mixed:
                dm = self.dm(cutoffs=cutoffs)
                self._fock_probabilities = fock.dm_to_probs(dm)
            else:
                ket = self.ket(cutoffs=cutoffs)
                self._fock_probabilities = fock.ket_to_probs(ket)
        return self._fock_probabilities

    def primal(self, other: Union[State, Transformation]) -> State:
        r"""Returns the post-measurement state after ``other`` is projected onto ``self``.

        ``other << self`` is other projected onto ``self``, i.e. self acts as a measurement.

        If ``other`` is a ``Transformation``, it returns the dual of the transformation applied to
        ``self``, i.e. ``other << self`` is like ``self >> other^dual``.

        Note that the returned state is not normalized. To normalize a state you can use
        ``mrmustard.physics.normalize``.
        """
        if isinstance(other, Union[State, Operation]):
            remaining_modes = list(set(other.out_modes) - set(self.out_modes))

            out_fock = self._contract_with_other(other)
            if len(remaining_modes) > 0:
                return (
                    State(dm=out_fock, modes=remaining_modes)
                    if other.is_mixed or self.is_mixed
                    else State(ket=out_fock, modes=remaining_modes)
                )

            # return the probability (norm) of the state when there are no modes left
            return (
                fock.math.abs(out_fock) ** 2
                if other.is_pure and self.is_pure
                else fock.math.abs(out_fock)
            )

        try:
            return other.dual(self)
        except AttributeError as e:
            raise TypeError(
                f"Cannot apply {other.__class__.__qualname__} to {self.__class__.__qualname__}"
            ) from e

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

    # def _project_onto_gaussian(self, other: State) -> Union[State, float]:
    #     """Returns the result of a generaldyne measurement given that states ``self`` and
    #     ``other`` are gaussian.

    #     Args:
    #         other (State): gaussian state being projected onto self

    #     Returns:
    #         State or float: returns the output conditional state on the remaining modes
    #             or the probability.
    #     """
    #     # here `self` is the measurement device state and `other` is the incoming state
    #     # being projected onto the measurement state
    #     remaining_modes = list(set(other.modes) - set(self.modes))

    #     _, probability, new_cov, new_means = gaussian.general_dyne(
    #         other.cov,
    #         other.means,
    #         self.cov,
    #         self.means,
    #         self.modes,
    #     )

    #     if len(remaining_modes) > 0:
    #         return State(
    #             means=new_means,
    #             cov=new_cov,
    #             modes=remaining_modes,
    #             _norm=probability if not getattr(self, "_normalize", False) else 1.0,
    #         )

    #     return probability

    def __iter__(self) -> Iterable[State]:
        """Iterates over the modes and their corresponding tensors."""
        return (self.get_modes(i) for i in range(self.num_modes))

    def __and__(self, other: State) -> State:
        r"""Concatenates two states."""
        # TODO: would be more efficient if we could keep pure states as kets
        if self.is_mixed or other.is_mixed:
            self_fock = self.dm()
            other_fock = other.dm()
            dm = fock.math.tensordot(self_fock, other_fock, [[], []])
            # e.g. self has shape [1,3,1,3] and other has shape [2,2]
            # we want self & other to have shape [1,3,2,1,3,2]
            # before transposing shape is [1,3,1,3]+[2,2]
            self_idx = list(range(len(self_fock.shape)))
            other_idx = list(range(len(self_idx), len(self_idx) + len(other_fock.shape)))
            return State(
                dm=math.transpose(
                    dm,
                    self_idx[: len(self_idx) // 2]
                    + other_idx[: len(other_idx) // 2]
                    + self_idx[len(self_idx) // 2 :]
                    + other_idx[len(other_idx) // 2 :],
                ),
                modes=self.modes + [m + max(self.modes) + 1 for m in other.modes],
            )
        # else, all states are pure
        self_fock = self.ket()
        other_fock = other.ket()
        return State(
            ket=fock.math.tensordot(self_fock, other_fock, [[], []]),
            modes=self.modes + [m + max(self.modes) + 1 for m in other.modes],
        )

    # def __getitem__(self, item):
    #     "setting the modes of a state (same API as `Transformation`)"
    #     if isinstance(item, int):
    #         item = [item]
    #     elif isinstance(item, Iterable):
    #         item = list(item)
    #     else:
    #         raise TypeError("item must be int or iterable")
    #     if len(item) != self.num_modes:
    #         raise ValueError(
    #             f"there are {self.num_modes} modes (item has {len(item)} elements, perhaps you're looking for .get_modes()?)"
    #         )
    #     self._modes = item
    #     return Operation(self)

    def get_modes(self, item):
        r"""Returns the state on the given modes."""
        if isinstance(item, int):
            item = [item]
        elif isinstance(item, Iterable):
            item = list(item)
        else:
            raise TypeError("item must be int or iterable")

        if item == self.modes:
            return self

        if not set(item) & set(self.modes):
            raise ValueError(
                f"Failed to request modes {item} for state {self} on modes {self.modes}."
            )
        item_idx = [self.modes.index(m) for m in item]

        fock_partitioned = fock.trace(self.dm(self.cutoffs), keep=item_idx)
        return State(dm=fock_partitioned, modes=item)

    # TODO: refactor
    def __eq__(self, other):
        r"""Returns whether the states are equal."""
        if self.num_modes != other.num_modes:
            return False
        if not np.isclose(self.purity, other.purity, atol=1e-6):
            return False
        try:
            return np.allclose(
                self.ket(cutoffs=other.cutoffs), other.ket(cutoffs=other.cutoffs), atol=1e-6
            )
        except TypeError:
            return np.allclose(
                self.dm(cutoffs=other.cutoffs), other.dm(cutoffs=other.cutoffs), atol=1e-6
            )

    def __rshift__(self, other):
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
        if not isinstance(other, State):
            raise TypeError(f"Cannot add {other.__class__.__qualname__} to a state")
        warnings.warn("mixing states forces conversion to fock representation", UserWarning)
        return State(dm=self.dm(self.cutoffs) + other.dm(self.cutoffs))

    def __rmul__(self, other):
        r"""Implements multiplication by a scalar from the left.

        E.g., ``0.5 * psi``.
        """
        if self.is_gaussian:
            warnings.warn(
                "scalar multiplication forces conversion to fock representation", UserWarning
            )
            return self.fock  # trigger creation of fock representation
        if self._dm is not None:
            return State(dm=self.dm() * other)
        if self._ket is not None:
            return State(ket=self.ket() * other)
        raise ValueError("No fock representation available")

    def __truediv__(self, other):
        r"""Implements division by a scalar from the left.

        E.g. ``psi / 0.5``
        """
        if self.is_gaussian:
            warnings.warn("scalar division forces conversion to fock representation", UserWarning)
            return self.fock

        if self._dm is not None:
            return State(dm=self.dm() / other)
        if self._ket is not None:
            return State(ket=self.ket() / other)
        raise ValueError("No fock representation available")

    @staticmethod
    def _format_probability(prob: float) -> str:
        if prob < 0.001:
            return f"{100*prob:.3e} %"
        else:
            return f"{prob:.3%}"

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
