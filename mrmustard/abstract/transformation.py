import numpy as np  # for repr
from abc import ABC
from mrmustard.plugins import GaussianPlugin
from mrmustard.abstract import State
from mrmustard.typing import *



class Transformation(ABC):
    r"""
    Base class for all transformations.
    Transformations include:
        * unitary transformations
        * non-unitary CPTP channels
    Given that measurements are CP but not TP, they have their own abstract class.
    """

    _gaussian GaussianPlugin

    # the following 3 lines are so that mypy doesn't complain,
    # but all subclasses of Op have these 3 attributes
    _modes: List[int]
    _trainable_parameters: List[Tensor]
    _constant_parameters: List[Tensor]

    def __call__(self, state: State) -> State:
        displacement = self.displacement_vector(state.hbar)
        symplectic = self.symplectic_matrix(state.hbar)
        noise = self.noise_matrix(state.hbar)

        output = State(state.num_modes, hbar=state.hbar, mixed=noise is not None)
        output.cov, output.means = self._gaussian.CPTP(state.cov, state.means, symplectic, noise, displacement, self._modes)
        return output

    def __repr__(self):
        with np.printoptions(precision=6, suppress=True):
            lst = [f"{name}={self.nparray(self.math.atleast_1d(self.__dict__[name]))}" for name in self.param_names]
            return f"{self.__class__.__qualname__}(modes={self._modes}, {', '.join(lst)})"

    def symplectic_matrix(self, hbar: float) -> Optional[Matrix]:
        return None

    def displacement_vector(self, hbar: float) -> Optional[Vector]:
        return None

    def noise_matrix(self, hbar: float) -> Optional[Matrix]:
        return None

    @property
    def euclidean_parameters(self) -> List[Tensor]:
        return [par for par in self._trainable_parameters if par.ndim == 1]

    @property
    def symplectic_parameters(self) -> List[Tensor]:
        return []  # NOTE overridden in Ggate

    @property
    def orthogonal_parameters(self) -> List[Tensor]:
        return []  # NOTE overridden in Interferometer
