from mrmustard.core.baseclasses import State


class Vacuum(State):
    r"""
    The N-mode vacuum state.
    """

    def __init__(self, num_modes: int, hbar: float = 2.0):
        super().__init__(num_modes, hbar, mixed=False)
        self.cov = hbar * self._math_backend.identity(2 * self.num_modes) / 2.0
        self.means = self._math_backend.zeros(2 * self.num_modes)


class SqueezedVacuum(State):
    pass


class Coherent(State):
    pass


class Thermal(State):
    pass
