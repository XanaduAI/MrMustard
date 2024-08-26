# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A module containing global settings."""

from __future__ import annotations
from pathlib import Path
from rich import print
import rich.table
import numpy as np

from mrmustard.utils.filters import (
    add_complex_warning_filter,
    remove_complex_warning_filter,
)

__all__ = ["settings"]


# pylint: disable=too-many-instance-attributes
class Settings:
    r"""A class containing various settings that are used by Mr Mustard throughout a session.

    Some of these settings (such as those representing cutoff values) can be changed at any time,
    while others (such as the value of the Planck constant) can only be changed before being
    queried for the first time.

    .. code-block::

        from mrmustard import settings

        >>> settings.AUTOCUTOFF_MAX_CUTOFF  # check the default values
        100

        >>> settings.AUTOCUTOFF_MAX_CUTOFF = 150  # update to new values
        >>> settings.AUTOCUTOFF_MAX_CUTOFF
        150
    """

    def __new__(cls):  # singleton
        if not hasattr(cls, "instance"):
            cls.instance = super(Settings, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self._hbar: float = 1.0
        self._hbar_locked: bool = False
        self._seed: int = np.random.randint(0, 2**31 - 1)
        self._complex_warning: bool = False
        self.rng = np.random.default_rng(self._seed)
        self._precision_bits_hermite_poly: int = 128
        self._complex_warning: bool = False
        self._julia_initialized: bool = False
        self._precision_bits_hermite_poly: int = 128
        self._cache_dir = Path(__file__).parents[2].absolute() / ".serialize_cache"

        self.UNSAFE_ZIP_BATCH: bool = False
        "Whether to operate element-wise within a batch of Ansatze. If True, the length of the batch dimension of two circuit components must be the same. Default is False."

        self.USE_VANILLA_AVERAGE: bool = False
        "Whether to use the vanilla_average function when computing Fock amplitudes (more stable, but slower). Default is False."

        self.DEBUG: bool = False
        "Whether or not to print the vector of means and the covariance matrix alongside the html representation of a state. Default is False."

        self.AUTOSHAPE_PROBABILITY: float = 0.99999
        "The minimum l2_norm to reach before automatically stopping the Bargmann-to-Fock conversion. Default is 0.999."

        self.AUTOCUTOFF_MAX_CUTOFF: int = 100  # TODO: remove in MM 1.0
        r"""The maximum value for autocutoff. Default is ``100``."""

        self.AUTOCUTOFF_MIN_CUTOFF: int = 1  # TODO: remove in MM 1.0
        r"""The minimum value for autocutoff. Default is ``1``."""

        self.AUTOSHAPE_MAX: int = 50
        r"""The max shape for the autoshape. Default is ``50``."""

        self.DRAW_CIRCUIT_PARAMS: bool = True
        "Whether or not to draw the parameters of a circuit."

        self.CIRCUIT_DECIMALS: int = 3
        "The number of decimal places to display when drawing a circuit."

        self.DISCRETIZATION_METHOD: str = "clenshaw"
        "The method used to discretize the Wigner function. Can be ``clenshaw`` (better, default) or ``iterative`` (worse, faster)."

        self.EQ_TRANSFORMATION_CUTOFF: int = 3  # enough for a full step of rec rel
        "The cutoff used when comparing two transformations via the Choi–Jamiolkowski isomorphism. Default is 3."

        self.EQ_TRANSFORMATION_RTOL_FOCK: float = 1e-3
        "The relative tolerance used when comparing two transformations via the Choi–Jamiolkowski isomorphism. Default is 1e-3."

        self.EQ_TRANSFORMATION_RTOL_GAUSS: float = 1e-6
        "The relative tolerance used when comparing two transformations on Gaussian states. Default is 1e-6."

        self.PRN_INTERNAL_CUTOFF: int = 50
        "The cutoff used when computing the output of a PNR detection. Default is 50."

        self.HOMODYNE_SQUEEZING: float = 10.0
        "The value of squeezing for homodyne measurements. Default is 10.0."

        self.PROGRESSBAR: bool = True
        "Whether or not to display the progress bar when performing training. Default is True."

        self.PNR_INTERNAL_CUTOFF: int = 50
        "The cutoff used when computing the output of a PNR detection. Default is 50."

        self.BS_FOCK_METHOD: str = "vanilla"  # can be 'vanilla' or 'schwinger'
        "The method for computing a beam splitter in the Fock basis . Default is ``vanilla``."

        self.ATOL: float = 1e-8
        "The absolute tolerance when comparing two values or arrays. Default is 1e-8."

    @property
    def COMPLEX_WARNING(self):
        r"""Whether tensorflow's ``ComplexWarning``s should be raised when a complex is cast to a float. Default is ``False``."""
        return self._complex_warning

    @COMPLEX_WARNING.setter
    def COMPLEX_WARNING(self, value: bool):
        self._complex_warning = value
        if value:
            remove_complex_warning_filter()
        else:
            add_complex_warning_filter()

    @property
    def HBAR(self):
        r"""The value of the Planck constant. Default is ``2``.

        Cannot be changed after its value is queried for the first time.
        """
        self._hbar_locked = True
        return self._hbar

    @HBAR.setter
    def HBAR(self, value: float):
        if value != self._hbar and self._hbar_locked:
            raise ValueError("Cannot change the value of `settings.HBAR`.")
        self._hbar = value

    @property
    def SEED(self) -> int:
        r"""Returns the seed value if set, otherwise returns a random seed."""
        if self._seed is None:
            self._seed = np.random.randint(0, 2**31 - 1)
            self.rng = np.random.default_rng(self._seed)
        return self._seed

    @SEED.setter
    def SEED(self, value: int | None):
        self._seed = value
        self.rng = np.random.default_rng(self._seed)

    @property
    def PRECISION_BITS_HERMITE_POLY(self):
        r"""
        The number of bits used to represent a single Fock amplitude when calculating Hermite polynomials.
        Default is 128 (i.e. the Fock representation has dtype complex128).
        Currently allowed values: 128, 256, 384, 512
        """
        return self._precision_bits_hermite_poly

    @PRECISION_BITS_HERMITE_POLY.setter
    def PRECISION_BITS_HERMITE_POLY(self, value: int):
        allowed_values = [128, 256, 384, 512]
        if value not in allowed_values:
            raise ValueError(
                f"precision_bits_hermite_poly must be one of the following values: {allowed_values}"
            )
        self._precision_bits_hermite_poly = value
        if (
            value != 128 and not self._julia_initialized
        ):  # initialize Julia when precision > complex128 and if it wasn't initialized before
            from juliacall import Main as jl  # pylint: disable=import-outside-toplevel

            # import Julia functions
            julia_directory = (
                Path(__file__)
                .parent.parent.joinpath("math", "lattice", "strategies", "julia")
                .resolve()
                .absolute()
            )
            jl.cd(str(julia_directory))
            jl.include("getPrecision.jl")
            jl.include("vanilla.jl")
            jl.include("compactFock/helperFunctions.jl")
            jl.include("compactFock/diagonal_amps.jl")
            jl.include("compactFock/diagonal_grad.jl")
            jl.include("compactFock/singleLeftoverMode_amps.jl")
            jl.include("compactFock/singleLeftoverMode_grad.jl")

            self._julia_initialized = True

    @property
    def CACHE_DIR(self) -> Path:
        """The directory in which serialized MrMustard objects are saved."""
        return self._cache_dir

    @CACHE_DIR.setter
    def CACHE_DIR(self, path: str | Path):
        self._cache_dir = Path(path)
        self._cache_dir.mkdir(exist_ok=True, parents=True)

    # use rich.table to print the settings
    def __repr__(self) -> str:
        r"""Returns a string representation of the settings."""

        # attributes that should not be displayed in the table
        not_displayed = ["rng"]

        table = rich.table.Table(title="MrMustard Settings")
        table.add_column("Setting")
        table.add_column("Value")

        for key, val in self.__dict__.items():
            if key in not_displayed or key.startswith("_"):
                continue
            key = key.upper()
            value = str(val)
            table.add_row(key, value)

        print(table)
        return ""


settings = Settings()
"""Settings object."""
