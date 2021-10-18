from mrmustard._typing import *
from mrmustard.abstract import State, Parametrized
from mrmustard.plugins import gaussian

__all__ = ["Vacuum", "SqueezedVacuum", "Coherent", "Thermal", "DisplacedSqueezed", "TMSV"]


class Vacuum(State):
    r"""
    The N-mode vacuum state.
    """

    def __init__(self, num_modes: int = None, hbar: float = 2.0):
        cov, means = gaussian.vacuum_state(num_modes, hbar)
        super().__init__(False, hbar, cov, means)


class Coherent(Parametrized, State):
    r"""
    The N-mode coherent state.
    """

    def __init__(
        self,
        x: Union[Optional[float], Optional[List[float]]],
        y: Union[Optional[float], Optional[List[float]]],
        hbar: float = 2.0,
        x_trainable: bool = True,
        y_trainable: bool = True,
        x_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        y_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
    ):
        super().__init__(x=x, y=y, x_trainable=x_trainable, y_trainable=y_trainable, x_bounds=x_bounds, y_bounds=y_bounds, hbar=hbar)


    @property
    def cov(self):
        cov, means = gaussian.coherent_state(self.x, self.y, self._hbar)  # TODO: split into cov and means functions
        return cov

    @property
    def means(self):
        cov, means = gaussian.coherent_state(self.x, self.y, self._hbar)  # TODO: split into cov and means functions
        return means


class SqueezedVacuum(State):
    r"""
    The N-mode squeezed vacuum state.
    """

    def __init__(self, r: Union[Scalar, Vector], phi: Union[Scalar, Vector], hbar: float = 2.0):
        cov, means = gaussian.squeezed_vacuum_state(r, phi, hbar)
        super().__init__(False, hbar, cov=cov, means=means)


class TMSV(State):
    r"""
    The 2-mode squeezed vacuum state.
    """

    def __init__(self, r: Union[Scalar, Vector], phi: Union[Scalar, Vector], hbar: float = 2.0):
        cov, means = gaussian.two_mode_squeezed_vacuum_state(r, phi, hbar)
        super().__init__(False, hbar, cov=cov, means=means)


class Thermal(State):
    r"""
    The N-mode thermal state.
    """

    def __init__(self, nbar: Union[Scalar, Vector], hbar: float = 2.0):
        cov, means = gaussian.thermal_state(nbar, hbar)
        super().__init__(False, hbar, cov=cov, means=means)


class DisplacedSqueezed(State):
    r"""
    The N-mode displaced squeezed state.
    """

    def __init__(
        self, r: Union[Scalar, Vector], phi: Union[Scalar, Vector], x: Union[Scalar, Vector], y: Union[Scalar, Vector], hbar: float = 2.0
    ):
        cov, means = gaussian.displaced_squeezed_state(r, phi, x, y, hbar)
        super().__init__(False, hbar, cov=cov, means=means)
