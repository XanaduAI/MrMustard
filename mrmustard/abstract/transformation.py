import numpy as np  # for repr
from abc import ABC, abstractproperty
from mrmustard.plugins import gaussian, fock
from mrmustard.abstract import State
from mrmustard._typing import *
from mrmustard import tmsv_r


class Transformation(ABC):
    r"""
    Base class for all transformations.
    Note that measurements are CP but not TP, so they have their own abstract class.
    Transformations include:
        * unitary transformations
        * non-unitary CPTP channels
    """

    def __call__(self, state: State) -> State:
        d = self.d_vector(state._hbar)
        X = self.X_matrix()
        Y = self.Y_matrix(state._hbar)
        cov, means = gaussian.CPTP(state.cov, state.means, X, Y, d, self.modes)
        return State.from_gaussian(cov, means, mixed=state.is_mixed or Y is not None, hbar=state._hbar)

    def __repr__(self):
        with np.printoptions(precision=6, suppress=True):
            lst = [f"{name}={np.array(np.atleast_1d(self.__dict__[name]))}" for name in self.param_names]
            return f"{self.__class__.__qualname__}(modes={self.modes}, {', '.join(lst)})"

    @property
    def modes(self) -> Sequence[int]:
        if self._modes in (None, []):
            if (d := self.d_vector(hbar=2.0)) is not None:
                self._modes = list(range(d.shape[-1] // 2))
            elif (X := self.X_matrix()) is not None:
                self._modes = list(range(X.shape[-1] // 2))
            elif (Y := self.Y_matrix(hbar=2.0)) is not None:
                self._modes = list(range(Y.shape[-1] // 2))
        return self._modes

    @property
    def bell(self):
        "N pairs of two-mode squeezed vacuum where N is the number of modes of the circuit"
        pass

    def X_matrix(self) -> Optional[Matrix]:
        return None

    def Y_matrix(self, hbar: float) -> Optional[Matrix]:
        return None

    def d_vector(self, hbar: float) -> Optional[Vector]:
        return None

    def fock(self, cutoffs=Sequence[int]):  # only single-mode for now
        unnormalized = self(self.bell).ket(cutoffs=cutoffs)
        return fock.normalize_choi_trick(unnormalized, tmsv_r)

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
