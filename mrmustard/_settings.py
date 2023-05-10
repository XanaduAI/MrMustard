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

"""The :py:class:`.Settings` contains the global configurations parameters used when
executing MrMustard.

To set any of the parameters simply assign the new value for them:

.. code::

    from mrmustard import settings

    # set reduced Planck constant value
    settings.HBAR = 1

    # increase the minimum and maximum automatic photon-number cutoff value
    settings.AUTOCUTOFF_MIN_CUTOFF = 10
    settings.AUTOCUTOFF_MAX_CUTOFF = 50

"""

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
        self.HBAR = 2.0
        """reduced Planck constant unit value"""

        # States autocutoff
        self.AUTOCUTOFF_PROBABILITY = 0.999
        """sets the threshold probability when automatically computing photon-number cutoffs"""
        self.AUTOCUTOFF_MAX_CUTOFF: int = 100
        """maximum automatic photon-number cutoff value"""
        self.AUTOCUTOFF_MIN_CUTOFF: int = 1
        """minimum automatic photon-number cutoff value"""

        self.DATA_MAX_SAMPLES_1D: int = 1000

        # Cutoff when comparing transformations
        self.EQ_TRANSFORMATION_CUTOFF: int = 5
        """sets the cutoff value per mode when determining if the Choi representation of
        two transformations in the photon-number basis are equal"""
        self.EQ_TRANSFORMATION_RTOL_FOCK = 1e-3
        """tolerance for equality comparison between transformations in Fock representation"""
        self.EQ_TRANSFORMATION_RTOL_GAUSS = 1e-6
        """tolerance for equality comparison between transformations in Wigner representation"""

        # for the detectors
        self.PNR_INTERNAL_CUTOFF = 50
        """default internal cutoff of photon-number resolving detectors"""
        self.HOMODYNE_SQUEEZING = 10.0
        """squeezing value of the squeezed state used for homodyne detection"""

        self.PROGRESSBAR: bool = True
        """wheter to display progress bar when running the optimizer"""

        # random numbers
        self._seed = np.random.randint(0, 2**31 - 1)
        self.rng = np.random.default_rng(self._seed)
        """random number generator"""

        self.BACKEND: str = "tensorflow"

    @property
    def SEED(self):
        """random seed value"""
        return self._seed

    @SEED.setter
    def SEED(self, value: int):
        if not isinstance(value, int):
            raise ValueError("Value of `SEED` should be an integer.")

        self._seed = value
        self.rng = np.random.default_rng(self._seed)

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
