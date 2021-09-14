import numpy as np  # for repr
from abc import ABC
from mrmustard.functionality import gaussian
from mrmustard.abstract import State
from experimental import XPTensor
from mrmustard._typing import *


class Transformation(ABC):
    r"""
    Base class for all transformations.
    Note that measurements are CP but not TP, so they have their own abstract class.
    Transformations include:
        * unitary transformations
        * non-unitary CPTP channels
    """

    def __call__(self, state: State, modes: Sequence[int] = []) -> State:
        r"""
        Apply the transformation to the state. If modes are not specified, it is assumed that the transformation acts on
        modes 0, 1, 2, ..., N-1 where N is the number of modes in the state.
        Arguments:
            state: The state to transform.
            modes (optional): The modes of the state the transformation acts on, relative to the state.
        Returns:
            The transformed state.
        """
        if modes == []:
            modes = list(range(state.num_modes))
        d = XPTensor(self.d_vector(state.hbar), modes=(modes,))
        X = XPTensor(self.X_matrix(), modes=(modes,modes))  # TODO: confirm with nico which of (X,Y,d) depend on hbar
        Y = XPTensor(self.Y_matrix(state.hbar), modes=(modes,modes))
        cov = XPTensor(state.cov, modes=(list(range(state.num_modes)), list(range(state.num_modes))))
        means = XPTensor(state.means, modes=(list(range(state.num_modes)),))
        cov, means = gaussian.CPTP(cov, means, X, Y, d, modes)
        output = State(hbar=state.hbar, mixed=Y is not None, cov=cov, means=means)
        return output

    def __repr__(self):
        with np.printoptions(precision=6, suppress=True):
            lst = [f"{name}={np.array(np.atleast_1d(self.__dict__[name]))}" for name in self.param_names]
            return f"{self.__class__.__qualname__}(modes={self._modes}, {', '.join(lst)})"

    def X_matrix(self) -> Optional[Matrix]:
        return None

    def d_vector(self, hbar: float) -> Optional[Vector]:
        return None

    def Y_matrix(self, hbar: float) -> Optional[Matrix]:
        return None

    def trainable_parameters(self) -> Dict[str, List[Trainable]]:
        return {"symplectic": [], "orthogonal": [], "euclidean": self._trainable_parameters}

    def __getitem__(self, item) -> Callable:
        r"""
        Allows transformations to be used as:
        op[0,1](input)  # acting on modes 0 and 1
        """
        if isinstance(item, int):
            modes = [item]
        elif isinstance(item, slice):
            modes = list(range(item.start, item.stop, item.step))
        elif isinstance(item, Iterable):
            modes = list(item)
        else:
            raise ValueError(f"{item} is not a valid slice or list of modes.")
        return lambda *args, **kwargs: self(*args, modes=modes, **kwargs)
