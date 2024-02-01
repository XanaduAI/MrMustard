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

from rich import print

from ._version import __version__
from .utils.settings import *
from .utils.filters import add_complex_warning_filter


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
    print("Copyright 2021 Xanadu Quantum Technologies Inc.\n")

    print("Python version:            {}.{}.{}".format(*sys.version_info[0:3]))
    print("Platform info:             {}".format(platform.platform()))
    print("Installation path:         {}".format(os.path.dirname(__file__)))
    print("Mr Mustard version:        {}".format(__version__))
    print("Numpy version:             {}".format(numpy.__version__))
    print("Numba version:             {}".format(numba.__version__))
    print("Scipy version:             {}".format(scipy.__version__))
    print("The Walrus version:        {}".format(thewalrus.__version__))
    print("TensorFlow version:        {}".format(tensorflow.__version__))


# filter tensorflow cast warnings
add_complex_warning_filter()
