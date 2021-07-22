import numpy as np  # for repr
from abc import ABC
from typing import List, Optional
from mrmustard.plugins import MathBackendInterface
from mrmustard.abstract.state import State


class Op(ABC):
    r"""
    Base class for all operations (ops).
    Ops include: 
        * Unitary transformations
        * CPTP Channels
        * Measurements
    """

    _math: MathBackendInterface

    # the following 3 lines are so that mypy doesn't complain,
    # but all subclasses of Op have these 3 attributes
    _modes: List[int]
    _trainable_parameters: List[Tensor]
    _constant_parameters: List[Tensor]

    def __call__(self, state: State) -> State:
        displacement = self._math.tile_vec(self.displacement_vector(state.hbar), len(self._modes))
        symplectic = self._math.tile_mat(self.symplectic_matrix(state.hbar), len(self._modes))
        noise = self._math.tile_mat(self.noise_matrix(state.hbar), len(self._modes))

        output = State(state.num_modes, hbar=state.hbar, mixed=noise is not None)
        output.cov = self._math.sandwich(bread=symplectic, filling=state.cov, modes=self._modes)
        output.cov = self._math.add(old=output.cov, new=noise, modes=self._modes)
        output.means = self._math.matvec(mat=symplectic, vec=state.means, modes=self._modes)
        output.means = self._math.add(old=output.means, new=displacement, modes=self._modes)
        return output

    def __repr__(self):
        with np.printoptions(precision=6, suppress=True):
            lst = [f"{name}={self.nparray(self.math.atleast_1d(self.__dict__[name]))}" for name in self.param_names]
            return f"{self.__class__.__qualname__}(modes={self._modes}, {', '.join(lst)})"

    # TODO: refactor/update for non-TP channels

    def symplectic_matrix(self, hbar: float) -> Optional[Tensor]:
        return None

    def displacement_vector(self, hbar: float) -> Optional[Tensor]:
        return None

    def noise_matrix(self, hbar: float) -> Optional[Tensor]:
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
