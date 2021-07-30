from mrmustard.core.baseclasses import State
from mrmustard.typing import *


class Vacuum(State):
    r"""
    The N-mode vacuum state.
    """
    def __init__(self, num_modes: int, hbar: float = 2.0):
        super().__init__(num_modes, hbar, mixed=False)
        self.cov, self.means = self._gaussian.vacuum_state(self.num_modes, hbar)


class SqueezedVacuum(State):
    r"""
    The N-mode squeezed vacuum state.
    """
    def __init__(self, r: Tensor, phi: Tensor, hbar: float = 2.0):
        num_modes = self.math.atleast_1d(r).shape[-1]
        super().__init__(num_modes, hbar, mixed=False)
        self.cov, self.means = self._gaussian.squeezed_vacuum_state(r, phi, hbar)


class Coherent(State):
    r"""
    The N-mode coherent state.
    """
    def __init__(self, x: Tensor, y: Tensor, hbar: float = 2.0):
        num_modes = self.math.atleast_1d(x).shape[-1]
        super().__init__(num_modes, hbar, mixed=False)
        self.cov, self.means = self._gaussian.coherent_state(x, y, hbar)


class Thermal(State):
    r"""
    The N-mode thermal state.
    """
    def __init__(self, nbar: Tensor, hbar: float = 2.0):
        num_modes = self.math.atleast_1d(nbar).shape[-1]
        super().__init__(num_modes, hbar, mixed=False)
        self.cov, self.means = self._gaussian.thermal_state(nbar, hbar)


class DisplacedSqueezed(State):
    r"""
    The N-mode displaced squeezed state.
    """
    def __init__(self, r: Tensor, phi: Tensor, x: Tensor, y: Tensor, hbar: float = 2.0):
        num_modes = self.math.atleast_1d(r).shape[-1]
        super().__init__(num_modes, hbar, mixed=False)
        self.cov, self.means = self._gaussian.displaced_squeezed_state(r, phi, x, y, hbar)
