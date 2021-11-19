# this is the topmost __init__.py file of the mrmustard package

import importlib
from rich.pretty import install  # NOTE: just for the looks

install()

__version__ = "0.1.0"


class Settings:
    def __init__(self):
        self._backend = "tensorflow"
        self.HBAR = 2.0
        self.CHOI_R = 0.881373587019543  # np.arcsinh(1.0)
        self.DEBUG = False
        self.AUTOCUTOFF_FACTOR = 5  # 5x the sqrt of the photon number variance when auto-detecting the Fock cutoff for a mode

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, backend_name: str):
        if backend_name not in ["tensorflow", "pytorch"]:
            raise ValueError("Backend must be either 'tensorflow' or 'pytorch'")
        self._backend = backend_name


settings = Settings()
settings.backend = "tensorflow"
