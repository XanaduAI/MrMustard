# this is the topmost __init__.py file of the mrmustard package

import importlib
from rich.pretty import install

install()  # NOTE: just for the looks, not stricly required

__version__ = "0.1.0"


class Settings:
    def __init__(self):
        self._backend = "tensorflow"
        self.HBAR = 2.0
        self.TMSV_DEFAULT_R = 3.0
        self.DEBUG = False
        self.N_SIGMA_CUTOFF = 4  # 4 sigma when auto-detecting the cutoff

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
