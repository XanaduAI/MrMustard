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

# ruff: noqa: PLC0415
# ruff: noqa: ICN001
import os

from rich import print as rprint

from ._version import __version__
from .utils.filters import add_complex_warning_filter
from .utils.settings import *


def version():
    r"""Version number of Mr Mustard.

    Returns:
      str: package version number
    """
    return __version__


def about():
    r"""Mr Mustard information.

    Prints the installed version numbers for Mr Mustard and its dependencies,
    and some system info. Please include this information in bug reports.

    **Example:**

    .. code-block:: pycon

        >>> mm.about()
        Mr Mustard: a differentiable bridge between phase space and Fock space.
        Copyright 2021 Xanadu Quantum Technologies Inc.

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
    import platform
    import sys

    import numba
    import numpy
    import scipy
    import tensorflow
    import thewalrus

    # a QuTiP-style infobox
    rprint("\nMr Mustard: a differentiable bridge between phase space and Fock space.")
    rprint("Copyright 2021 Xanadu Quantum Technologies Inc.\n")

    rprint("Python version:            {}.{}.{}".format(*sys.version_info[0:3]))
    rprint(f"Platform info:             {platform.platform()}")
    rprint(f"Installation path:         {os.path.dirname(__file__)}")
    rprint(f"Mr Mustard version:        {__version__}")
    rprint(f"Numpy version:             {numpy.__version__}")
    rprint(f"Numba version:             {numba.__version__}")
    rprint(f"Scipy version:             {scipy.__version__}")
    rprint(f"The Walrus version:        {thewalrus.__version__}")
    rprint(f"TensorFlow version:        {tensorflow.__version__}")


# filter tensorflow cast warnings
add_complex_warning_filter()
