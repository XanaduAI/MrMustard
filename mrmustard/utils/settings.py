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

import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import rich.table
from rich import print as rprint

from mrmustard.utils.filters import add_complex_warning_filter, remove_complex_warning_filter

__all__ = ["settings"]


class Settings:
    r"""
    A class containing various settings that are used by Mr Mustard throughout a session.

    .. code-block::

        from mrmustard import settings

        >>> settings.HBAR  # check the default values
        1.0

        >>> settings.HBAR=2.0  # update globally to new values
        >>> settings.HBAR
        2.0

        >>> with settings(HBAR=3.0): # update with context manager
        >>>      settings.HBAR
        3.0

        >>> settings.HBAR # previous value remains
        2.0
    """

    _frozen = False

    def __new__(cls):  # singleton
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._hbar: float = 1.0
        self._seed: int = np.random.randint(0, 2**31 - 1)  # noqa: NPY002
        self.rng = np.random.default_rng(self._seed)
        self._precision_bits_hermite_poly: int = 128
        self._complex_warning: bool = False
        self._cache_dir = Path(__file__).parents[2].absolute() / ".serialize_cache"

        self.AUTOSHAPE_PROBABILITY: float = 0.99999
        r"""The minimum l2_norm to reach before automatically stopping the Bargmann-to-Fock conversion. Default is ``0.99999``."""

        self.AUTOSHAPE_MAX: int = 50
        r"""The max shape for the autoshape. Default is ``50``."""

        self.AUTOSHAPE_MIN: int = 1
        r"""The min shape for the autoshape. Default is ``1``."""

        self.ATOL: float = 1e-8
        r"""The absolute tolerance when comparing two values or arrays. Default is ``1e-8``."""

        self.DEFAULT_FOCK_SIZE: int = 50
        r"""The default size for the Fock representation. Default is ``50``."""

        self.DEFAULT_REPRESENTATION: Literal["Bargmann", "Fock"] = "Fock"
        r"""The representation to use when contracting two circuit components in different representations. Can be ``Fock`` or ``Bargmann``. Default is ``Fock``."""

        self.DISCRETIZATION_METHOD: Literal["clenshaw", "iterative"] = "clenshaw"
        r"""The method used to discretize the Wigner function. Can be ``clenshaw`` (better, default) or ``iterative`` (worse, faster). Default is ``clenshaw``."""

        self.DRAW_CIRCUIT_PARAMS: bool = True
        r"""Whether or not to draw the parameters of a circuit. Default is ``True``."""

        self.EINSUM_OPTIMIZE: bool | Literal["greedy", "optimal", "auto"] = "greedy"
        r"""Whether to optimize the contraction order when using the Einstein summation convention.
        Allowed values are True, False, "greedy", "optimal" or "auto".
        Note the TF backend does not support False and converts it to "greedy".
        Default is ``"greedy"``.
        """

        self.PROGRESSBAR: bool = True
        r"""Whether or not to display the progress bar when performing training. Default is ``True``."""

        self.STABLE_FOCK_CONVERSION: bool = False
        r"""Whether to use the ``stable`` function when computing Fock amplitudes (more stable, but slower). Default is ``False``."""

        self._original_values = {}
        self._frozen = True

    @property
    def CACHE_DIR(self) -> Path:
        r"""
        The directory in which serialized MrMustard objects are saved.
        """
        return self._cache_dir

    @CACHE_DIR.setter
    def CACHE_DIR(self, path: str | Path):
        self._cache_dir = Path(path)
        self._cache_dir.mkdir(exist_ok=True, parents=True)

    @property
    def COMPLEX_WARNING(self) -> bool:
        r"""Whether tensorflow's ``ComplexWarning`` should be raised when a complex is cast to a float. Default is ``False``."""
        return self._complex_warning

    @COMPLEX_WARNING.setter
    def COMPLEX_WARNING(self, value: bool):
        self._complex_warning = value
        if value:
            remove_complex_warning_filter()
        else:
            add_complex_warning_filter()

    @property
    def HBAR(self) -> float:
        r"""
        The value of the Planck constant. Default is ``1``.
        """
        return self._hbar

    @HBAR.setter
    def HBAR(self, value: float):
        warnings.warn("Changing HBAR can conflict with prior computations.", stacklevel=1)
        self._hbar = value

    @property
    def SEED(self) -> int:
        r"""Returns the seed value if set, otherwise returns a random seed."""
        if self._seed is None:
            self._seed = np.random.randint(0, 2**31 - 1)  # noqa: NPY002
            self.rng = np.random.default_rng(self._seed)
        return self._seed

    @SEED.setter
    def SEED(self, value: int | None):
        self._seed = value
        self.rng = np.random.default_rng(self._seed)

    def __call__(self, **kwargs):
        r"""
        Allows for setting multiple settings at once and saving the original values.
        """
        self._original_values = {k: getattr(self, k) for k in kwargs}
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def __enter__(self):
        "context manager enter method"
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        "context manager exit method that resets the settings to their original values"
        for k, v in self._original_values.items():
            setattr(self, k, v)

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
            key = key.upper()  # noqa: PLW2901
            value = str(val)
            table.add_row(key, value)

        rprint(table)
        return ""

    def __setattr__(self, name, value):
        r"""
        Once the class is initialized, do not allow the addition of new settings.
        """
        if self._frozen and not hasattr(self, name):
            raise AttributeError(f"unknown MrMustard setting: '{name}'")
        return super().__setattr__(name, value)


settings = Settings()
"""Settings object."""
