# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import rich.table
from rich import print as rprint

# pylint: disable=too-many-instance-attributes
class Settings:
    """Settings class."""

    def __new__(cls):  # singleton
        if not hasattr(cls, "instance"):
            cls.instance = super(Settings, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self._backend: str = "tensorflow"
        self.DEBUG: bool = False

        self.HBAR = 2.0
        """reduced Planck constant unit value"""
        self.CHOI_R = 0.881373587019543  # np.arcsinh(1.0)

        # clip(mean + 5*std, min, max) when auto-detecting the Fock cutoff
        self.AUTOCUTOFF_STDEV_FACTOR = 5
        self.AUTOCUTOFF_MAX_CUTOFF = 100
        self.AUTOCUTOFF_MIN_CUTOFF = 1

        self.DATA_MAX_SAMPLES_1D = 1000

        # using cutoff=5 per mode for determining if two transformations in fock repr are equal
        self.EQ_TRANSFORMATION_CUTOFF = 5
        self.EQ_TRANSFORMATION_RTOL_FOCK = 1e-3
        self.EQ_TRANSFORMATION_RTOL_GAUSS = 1e-6

        # for the detectors
        self.PNR_INTERNAL_CUTOFF = 50
        self.HOMODYNE_SQUEEZING = 10.0

        # misc
        self.PROGRESSBAR: bool = True

        # random numbers
        self._seed = np.random.randint(0, 2**31 - 1)
        self.rng = np.random.default_rng(self._seed)

    @property
    def SEED(self):
        """Returns the seed value if set, otherwise returns a random seed."""
        return self._seed

    @SEED.setter
    def SEED(self, value: int):
        """Sets the seed value."""
        if not isinstance(value, int):
            raise ValueError("Value of `SEED` should be an integer.")

        self._seed = value
        self.rng = np.random.default_rng(self._seed)

    # @property
    # def BACKEND(self):
    #     """The backend which is used.

    #     Can be either ``'tensorflow'`` or ``'torch'``.
    #     """
    #     return self._backend

    # @BACKEND.setter
    # def BACKEND(self, backend_name: str):
    #     if backend_name != "tensorflow":  # pragma: no cover
    #         raise ValueError("Backend must be either 'tensorflow' or 'torch'")
    #     self._backend = backend_name

    # use rich.table to print the settings
    def __repr__(self):
        """Returns a string representation of the settings."""
        table = rich.table.Table(title="MrMustard Settings")
        table.add_column("Setting")
        table.add_column("Value")
        table.add_row("BACKEND", self.BACKEND)
        table.add_row("SEED", str(self.SEED))
        for key, value in self.__dict__.items():
            if key == key.upper():
                table.add_row(key, str(value))
        rprint(table)
        return ""

settings = Settings()
