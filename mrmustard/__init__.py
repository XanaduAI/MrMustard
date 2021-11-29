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
        # mean + 7*std when auto-detecting the Fock cutoff for a mode
        self.AUTOCUTOFF_STDEV_FACTOR = 7
        self.AUTOCUTOFF_MAX_CUTOFF = 100
        self.AUTOCUTOFF_MIN_CUTOFF = 1
        # using cutoff=5 for each mode when determining if two transformations in fock repr are equal
        self.EQ_TRANSFORMATION_CUTOFF = 5
        self.EQ_TRANSFORMATION_RTOL_FOCK = 1e-3
        self.EQ_TRANSFORMATION_RTOL_GAUSS = 1e-6

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
