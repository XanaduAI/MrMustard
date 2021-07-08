from abc import ABC
from typing import List, Optional
import numpy as np  # NOTE: only needed for the repr
from mrmustard.core.backends import MathBackendInterface, SymplecticBackendInterface
from mrmustard.core.baseclasses.state import State


class Gate(ABC):
    _math_backend: MathBackendInterface
    _symplectic_backend: SymplecticBackendInterface

    def __call__(self, state: State) -> State:
        displacement = self._math_backend.tile_vec(self.displacement_vector(state.hbar), len(self._modes))
        symplectic = self._math_backend.tile_mat(self.symplectic_matrix(state.hbar), len(self._modes))
        noise = self._math_backend.tile_mat(self.noise_matrix(state.hbar), len(self._modes))

        output = State(state.num_modes, hbar=state.hbar, mixed=noise is not None)
        output.cov = self._math_backend.sandwich(bread=symplectic, filling=state.cov, modes=self._modes)
        output.cov = self._math_backend.add(old=output.cov, new=noise, modes=self._modes)
        output.means = self._math_backend.matvec(mat=symplectic, vec=state.means, modes=self._modes)
        output.means = self._math_backend.add(old=output.means, new=displacement, modes=self._modes)
        return output

    def __repr__(self):
        with np.printoptions(precision=6, suppress=True):
            lst = [f'{name}={np.asarray(np.atleast_1d(self.__dict__[name]))}' for name in self.param_names]
            return f"{self.__class__.__qualname__}(modes={self._modes}, {', '.join(lst)})"

    def symplectic_matrix(self, hbar: float) -> Optional:
        return None

    def displacement_vector(self, hbar: float) -> Optional:
        return None

    def noise_matrix(self, hbar: float) -> Optional:
        return None

    @property
    def variables(self) -> List:
        return self._trainable_parameters

    @property
    def constants(self) -> List:
        return self._constant_parameters

    @property
    def all_parameters(self) -> List:
        return self.variables + self.constants

    @property
    def euclidean_parameters(self) -> List:
        return self._trainable_parameters  # NOTE overridden in Ggate and Interferometer

    @property
    def symplectic_parameters(self) -> List:
        return []  # NOTE overridden in Ggate

    @property
    def orthogonal_parameters(self) -> List:
        return []  # NOTE overridden in Interferometer
