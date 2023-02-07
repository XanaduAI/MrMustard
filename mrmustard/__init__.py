# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This is the top-most `__init__.py` file of MrMustard package."""

import numpy as np
import rich.table
from rich import print

from ._version import __version__


# pylint: disable=too-many-instance-attributes
class Settings:
    """Settings class."""

    def __new__(cls):  # singleton
        if not hasattr(cls, "instance"):
            cls.instance = super(Settings, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self._backend = "tensorflow"
        self.HBAR = 2.0
        self.CHOI_R = 0.881373587019543  # np.arcsinh(1.0)
        self.DEBUG = False
        # clip(mean + 5*std, min, max) when auto-detecting the Fock cutoff
        self.AUTOCUTOFF_STDEV_FACTOR = 5
        self.AUTOCUTOFF_MAX_CUTOFF = 100
        self.AUTOCUTOFF_MIN_CUTOFF = 1
        self.DATA_MAX_SAMPLES_1D = 1000
        # using cutoff=5 for each mode when determining if two transformations in fock repr are equal
        self.EQ_TRANSFORMATION_CUTOFF = 5
        self.EQ_TRANSFORMATION_RTOL_FOCK = 1e-3
        self.EQ_TRANSFORMATION_RTOL_GAUSS = 1e-6
        # for the detectors
        self.PNR_INTERNAL_CUTOFF = 50
        self.HOMODYNE_SQUEEZING = 10.0
        # misc
        self.PROGRESSBAR = True
        self._seed = np.random.randint(0, 2**31 - 1)
        self.rng = np.random.default_rng(self._seed)

    @property
    def SEED(self):
        """Returns the seed value if set, otherwise returns a random seed."""
        if self._seed is None:
            self._seed = np.random.randint(0, 2**31 - 1)
            self.rng = np.random.default_rng(self._seed)
        return self._seed

    @SEED.setter
    def SEED(self, value):
        """Sets the seed value."""
        self._seed = value
        self.rng = np.random.default_rng(self._seed)

    @property
    def BACKEND(self):
        """The backend which is used.

        Can be either ``'tensorflow'`` or ``'torch'``.
        """
        return self._backend

    @BACKEND.setter
    def BACKEND(self, backend_name: str):
        if backend_name not in ["tensorflow", "torch"]:  # pragma: no cover
            raise ValueError("Backend must be either 'tensorflow' or 'torch'")
        self._backend = backend_name

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
        print(table)
        return ""


settings = Settings()
"""Settings object."""


def version():
    r"""Version number of Mr Mustard.

    Returns:
      str: package version number
    """
    return __version__


def about():
    """Mr Mustard information.

    Prints the installed version numbers for Mr Mustard and its dependencies,
    and some system info. Please include this information in bug reports.

    **Example:**

    .. code-block:: pycon

        >>> mm.about()
        Mr Mustard: a differentiable bridge between phase space and Fock space.
        Copyright 2018-2021 Xanadu Quantum Technologies Inc.

        Python version:            3.6.10
        Platform info:             Linux-5.8.18-1-MANJARO-x86_64-with-arch-Manjaro-Linux
        Installation path:         /home/mrmustard/
        Mr Mustard version:        0.1.0
        Numpy version:             1.21.4
        Numba version:             0.48.0
        Scipy version:             1.7.3
        The Walrus version:        0.17.0
        TensorFlow version:        2.7.0
        Torch version:             1.10.0+cu102
    """
    # pylint: disable=import-outside-toplevel
    import os
    import platform
    import sys

    import numba
    import numpy
    import scipy
    import tensorflow
    import thewalrus

    # a QuTiP-style infobox
    print("\nMr Mustard: a differentiable bridge between phase space and Fock space.")
    print("Copyright 2018-2021 Xanadu Quantum Technologies Inc.\n")

    print("Python version:            {}.{}.{}".format(*sys.version_info[0:3]))
    print("Platform info:             {}".format(platform.platform()))
    print("Installation path:         {}".format(os.path.dirname(__file__)))
    print("Mr Mustard version:        {}".format(__version__))
    print("Numpy version:             {}".format(numpy.__version__))
    print("Numba version:             {}".format(numba.__version__))
    print("Scipy version:             {}".format(scipy.__version__))
    print("The Walrus version:        {}".format(thewalrus.__version__))
    print("TensorFlow version:        {}".format(tensorflow.__version__))

    try:  # pragma: no cover
        import torch

        torch_version = torch.__version__
        print("Torch version:             {}".format(torch_version))
    except ImportError:
        torch_version = None
