# this is the topmost __init__.py file of the mrmustard package

# from rich.pretty import install  # NOTE: just for the looks
# install()
from ._version import __version__


class Settings:
    """Settings class."""

    def __init__(self):
        self._backend = "tensorflow"
        self.HBAR = 2.0
        self.CHOI_R = 0.881373587019543  # np.arcsinh(1.0)
        self.DEBUG = False
        # min + mean + 5*std when auto-detecting the Fock cutoff
        self.AUTOCUTOFF_STDEV_FACTOR = 5
        self.AUTOCUTOFF_MAX_CUTOFF = 100
        self.AUTOCUTOFF_MIN_CUTOFF = 1
        # using cutoff=5 for each mode when determining if two transformations in fock repr are equal
        self.EQ_TRANSFORMATION_CUTOFF = 5
        self.EQ_TRANSFORMATION_RTOL_FOCK = 1e-3
        self.EQ_TRANSFORMATION_RTOL_GAUSS = 1e-6
        # for the detectors
        self.PNR_INTERNAL_CUTOFF = 50
        self.HOMODYNE_SQUEEZING = 10.0

    @property
    def backend(self):
        """The backend which is used.

        Can be either ``'tensorflow'`` or ``'pytorch'``.
        """
        return self._backend

    @backend.setter
    def backend(self, backend_name: str):
        if backend_name not in ["tensorflow", "pytorch"]:
            raise ValueError("Backend must be either 'tensorflow' or 'pytorch'")
        self._backend = backend_name


settings = Settings()
"""Settings object."""

settings.backend = "tensorflow"


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
        Mr Mustard version:       0.1.0
        Numpy version:             1.21.4
        Scipy version:             1.7.3
        The Walrus version:        0.17.0
        TensorFlow version:        2.7.0
        Torch version:             1.10.0+cu102
    """
    # pylint: disable=import-outside-toplevel
    import sys
    import platform
    import os
    import numpy
    import scipy
    import thewalrus
    import tensorflow

    # a QuTiP-style infobox
    print("\nMr Mustard: a differentiable bridge between phase space and Fock space.")
    print("Copyright 2018-2021 Xanadu Quantum Technologies Inc.\n")

    print("Python version:            {}.{}.{}".format(*sys.version_info[0:3]))
    print("Platform info:             {}".format(platform.platform()))
    print("Installation path:         {}".format(os.path.dirname(__file__)))
    print("Mr Mustard version:       {}".format(__version__))
    print("Numpy version:             {}".format(numpy.__version__))
    print("Scipy version:             {}".format(scipy.__version__))
    print("The Walrus version:        {}".format(thewalrus.__version__))
    print("TensorFlow version:        {}".format(tensorflow.__version__))

    try:
        import torch

        torch_version = torch.__version__
    except ImportError:
        torch_version = None

    print("Torch version:             {}".format(torch_version))
