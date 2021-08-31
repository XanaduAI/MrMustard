from mrmustard._typing import *
from mrmustard.abstract import State

__all__ = ["Vacuum", "SqueezedVacuum", "Coherent", "Thermal", "DisplacedSqueezed"]


class Vacuum(State):
    r"""
    The N-mode vacuum state.
    """

    def __init__(self, num_modes: int, hbar: float = 2.0):
        cov, means = self._gaussian.vacuum_state(num_modes, hbar)
        super().__init__(num_modes, hbar, mixed=False, cov=cov, means=means)


class Coherent(State):
    r"""
    The N-mode coherent state.
    """

    def __init__(self, x: Union[Scalar, Vector], y: Union[Scalar, Vector], hbar: float = 2.0):
        x = self._gaussian._backend.atleast_1d(x, dtype="float64" if not hasattr(x, "dtype") else x.dtype)
        y = self._gaussian._backend.atleast_1d(y, dtype="float64" if not hasattr(y, "dtype") else y.dtype)
        cov, means = self._gaussian.coherent_state(x, y, hbar)
        super().__init__(x.shape[-1], hbar, mixed=False, cov=cov, means=means)


class SqueezedVacuum(State):
    r"""
    The N-mode squeezed vacuum state.
    """

    def __init__(self, r: Union[Scalar, Vector], phi: Union[Scalar, Vector], hbar: float = 2.0):
        r = self._gaussian._backend.atleast_1d(r, dtype="float64" if not hasattr(r, "dtype") else r.dtype)
        phi = self._gaussian._backend.atleast_1d(phi, dtype="float64" if not hasattr(phi, "dtype") else phi.dtype)
        cov, means = self._gaussian.squeezed_vacuum_state(r, phi, hbar)
        super().__init__(r.shape[-1], hbar, mixed=False, cov=cov, means=means)


class Thermal(State):
    r"""
    The N-mode thermal state.
    """

    def __init__(self, nbar: Union[Scalar, Vector], hbar: float = 2.0):
        nbar = self._gaussian._backend.atleast_1d(nbar, dtype="float64" if not hasattr(nbar, "dtype") else nbar.dtype)
        cov, means = self._gaussian.thermal_state(nbar, hbar)
        super().__init__(nbar.shape[-1], hbar, mixed=False, cov=cov, means=means)


class DisplacedSqueezed(State):
    r"""
    The N-mode displaced squeezed state.
    """

    def __init__(
        self, r: Union[Scalar, Vector], phi: Union[Scalar, Vector], x: Union[Scalar, Vector], y: Union[Scalar, Vector], hbar: float = 2.0
    ):
        r = self._gaussian._backend.atleast_1d(r, dtype="float64" if not hasattr(r, "dtype") else r.dtype)
        phi = self._gaussian._backend.atleast_1d(phi, dtype="float64" if not hasattr(phi, "dtype") else phi.dtype)
        x = self._gaussian._backend.atleast_1d(x, dtype="float64" if not hasattr(x, "dtype") else x.dtype)
        y = self._gaussian._backend.atleast_1d(y, dtype="float64" if not hasattr(y, "dtype") else y.dtype)
        cov, means = self._gaussian.displaced_squeezed_state(r, phi, x, y, hbar)
        super().__init__(r.shape[-1], hbar, mixed=False, cov=cov, means=means)
