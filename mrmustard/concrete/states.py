from mrmustard._typing import *
from mrmustard.abstract import State, Parametrized
from mrmustard.plugins import gaussian

__all__ = ["Vacuum", "SqueezedVacuum", "Coherent", "Thermal", "DisplacedSqueezed", "TMSV"]


class Vacuum(State):
    r"""
    The N-mode vacuum state.
    """

    def __init__(self, num_modes: int = None, hbar: float = 2.0):
        cov = gaussian.vacuum_cov(num_modes, hbar)
        means = gaussian.vacuum_means(num_modes, hbar)
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
        x_trainable: bool = False,
        y_trainable: bool = False,
        x_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        y_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
    ):
        Parametrized.__init__(self, x=x, y=y, x_trainable=x_trainable, y_trainable=y_trainable, x_bounds=x_bounds, y_bounds=y_bounds)
        means = gaussian.displacement(x, y, hbar)
        cov = gaussian.vacuum_cov(means.shape[-1] // 2, hbar)
        State.__init__(self, False, hbar, cov, means)

    @property
    def means(self):
        return gaussian.displacement(self.x, self.y, self._hbar)


class SqueezedVacuum(Parametrized, State):
    r"""
    The N-mode squeezed vacuum state.
    """

    def __init__(
        self,
        r: Union[Scalar, Vector],
        phi: Union[Scalar, Vector],
        hbar: float = 2.0,
        r_trainable: bool = False,
        phi_trainable: bool = False,
        r_bounds: Tuple[Optional[float], Optional[float]] = (0, None),
        phi_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
    ):
        Parametrized.__init__(
            self, r=r, phi=phi, r_trainable=r_trainable, phi_trainable=phi_trainable, r_bounds=r_bounds, phi_bounds=phi_bounds
        )
        cov = gaussian.squeezed_vacuum_cov(r, phi, hbar)
        means = gaussian.vacuum_means(cov.shape[-1] // 2, hbar)
        State.__init__(self, False, hbar, cov=cov, means=means)

    @property
    def cov(self):
        return gaussian.squeezed_vacuum_cov(self.r, self.phi, self._hbar)


class TMSV(Parametrized, State):
    r"""
    The 2-mode squeezed vacuum state.
    """

    def __init__(
        self,
        r: Union[Scalar, Vector],
        phi: Union[Scalar, Vector],
        hbar: float = 2.0,
        r_trainable: bool = False,
        phi_trainable: bool = False,
        r_bounds: Tuple[Optional[float], Optional[float]] = (0, None),
        phi_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
    ):
        Parametrized.__init__(
            self, r=r, phi=phi, r_trainable=r_trainable, phi_trainable=phi_trainable, r_bounds=r_bounds, phi_bounds=phi_bounds
        )
        cov = gaussian.two_mode_squeezed_vacuum_cov(r, phi, hbar)
        means = gaussian.vacuum_means(2, hbar)
        State.__init__(self, False, hbar, cov=cov, means=means)

    @property
    def cov(self):
        return gaussian.two_mode_squeezed_vacuum_cov(self.r, self.phi, self._hbar)


class Thermal(Parametrized, State):
    r"""
    The N-mode thermal state.
    """

    def __init__(
        self,
        nbar: Union[Scalar, Vector],
        hbar: float = 2.0,
        nbar_trainable: bool = False,
        nbar_bounds: Tuple[Optional[float], Optional[float]] = (0, None),
    ):
        Parametrized.__init__(self, nbar=nbar, nbar_trainable=nbar_trainable, nbar_bounds=nbar_bounds)
        cov = gaussian.thermal_cov(nbar, hbar)
        means = gaussian.vacuum_means(cov.shape[-1] // 2, hbar)
        State.__init__(self, True, hbar, cov=cov, means=means)

    @property
    def cov(self):
        return gaussian.thermal_cov(self.nbar, self._hbar)


class DisplacedSqueezed(Parametrized, State):
    r"""
    The N-mode displaced squeezed state.
    """

    def __init__(
        self,
        r: Union[Scalar, Vector],
        phi: Union[Scalar, Vector],
        x: Union[Scalar, Vector],
        y: Union[Scalar, Vector],
        hbar: float = 2.0,
        r_trainable: bool = False,
        phi_trainable: bool = False,
        x_trainable: bool = False,
        y_trainable: bool = False,
        r_bounds: Tuple[Optional[float], Optional[float]] = (0, None),
        phi_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        x_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        y_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
    ):
        Parametrized.__init__(
            self,
            r=r,
            phi=phi,
            x=x,
            y=y,
            r_trainable=r_trainable,
            phi_trainable=phi_trainable,
            x_trainable=x_trainable,
            y_trainable=y_trainable,
            r_bounds=r_bounds,
            phi_bounds=phi_bounds,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )
        cov = gaussian.squeezed_vacuum_cov(r, phi, hbar)
        means = gaussian.displacement(x, y, hbar)
        State.__init__(self, False, hbar, cov=cov, means=means)

    @property
    def cov(self):
        return gaussian.squeezed_vacuum_cov(self.r, self.phi, self._hbar)

    @property
    def means(self):
        return gaussian.displacement(self.x, self.y, self._hbar)
