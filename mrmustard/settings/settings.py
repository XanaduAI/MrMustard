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
from typing import Annotated, Literal

import numpy as np
import rich.table
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    ValidationError,
    field_validator,
)
from rich import print as rprint

__all__ = [
    "Settings",
]


class Settings(BaseModel):
    r"""
    A class containing various settings that are used by Mr Mustard throughout a session.

    >>> from mrmustard import settings
    >>> assert settings.HBAR == 1.0  # check the default values
    >>> settings.HBAR=2.0  # update globally to new values
    >>> assert settings.HBAR == 2.0
    >>> with settings(HBAR=3.0): # update with context manager
    >>>      assert settings.HBAR == 3.0
    >>> assert settings.HBAR == 2.0 # previous value remains
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    AUTOSHAPE_PROBABILITY: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.99999
    r"""The minimum l2_norm to reach before automatically stopping the Bargmann-to-Fock conversion."""

    AUTOSHAPE_MAX: PositiveInt = 50
    r"""The max shape for the autoshape."""

    AUTOSHAPE_MIN: PositiveInt = 1
    r"""The min shape for the autoshape."""

    ATOL: PositiveFloat = 1e-8
    r"""The absolute tolerance when comparing two values or arrays."""

    CACHE_DIR: Path = Path(__file__).parents[2].absolute() / ".serialize_cache"
    r"""The directory in which serialized MrMustard objects are saved."""

    @field_validator("CACHE_DIR")
    @classmethod
    def create_cache_dir(cls, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        return path

    DEFAULT_FOCK_SIZE: PositiveInt = 50
    r"""The default size for the Fock representation."""

    DEFAULT_REPRESENTATION: Literal["Bargmann", "Fock", None] = "Fock"
    r"""The representation to use when contracting two circuit components in
    different representations. Can be ``Fock``, ``Bargmann`` or ``None``. 
    If ``None``, a ``TypeError`` is raised  instead."""

    DISCRETIZATION_METHOD: Literal["clenshaw", "iterative"] = "clenshaw"
    r"""The method used to discretize the Wigner function. Can be ``clenshaw`` (better, default) or
    ``iterative`` (worse, faster)."""

    DRAW_CIRCUIT_PARAMS: bool = True
    r"""Whether or not to draw the parameters of a circuit."""

    EINSUM_OPTIMIZE: bool | Literal["greedy", "optimal", "auto"] = "greedy"
    r"""Whether to optimize the contraction order when using the Einstein summation convention.
    Allowed values are True, False, "greedy", "optimal" or "auto".
    """

    HBAR: float = 1.0
    r"""The value of the Planck constant."""

    @field_validator("HBAR")
    @classmethod
    def hbar_warning(cls, value: float) -> float:
        warnings.warn("Changing HBAR can conflict with prior computations.", stacklevel=1)
        return value

    PROGRESSBAR: bool = True
    r"""Whether or not to display the progress bar when performing training."""

    @property
    def SEED(self) -> int:
        r"""Returns the seed value if set, otherwise returns a random seed."""
        if self._seed is None:
            self._seed = np.random.randint(0, 2**31 - 1)  # noqa: NPY002
            self._rng = np.random.default_rng(self._seed)
        return self._seed

    @SEED.setter
    def SEED(self, value: int | None):
        if value is not None and (not isinstance(value, int) or value < 0):
            raise ValueError("Input should be a non-negative integer or None")
        self._seed = value
        self._rng = np.random.default_rng(self._seed)

    STABLE_FOCK_CONVERSION: bool = True
    r"""Whether to use the ``stable`` function when computing Fock amplitudes 
    (more stable, but slower)."""

    def __new__(cls):  # singleton
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        super().__init__()
        self._seed: int = np.random.randint(0, 2**31 - 1)  # noqa: NPY002
        self._rng = np.random.default_rng(self._seed)
        self._precision_bits_hermite_poly: int = 128

        self._original_values = {}

    def get_rng(self, seed: int | None = None) -> np.random.Generator:
        if seed is not None:
            return np.random.default_rng(seed)
        return self._rng

    def __call__(self, **kwargs):
        r"""
        Allows for setting multiple settings at once and saving the original values.
        """
        self._original_values = {k: getattr(self, k) for k in kwargs}
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def __enter__(self):
        "Context manager enter method"
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        "Context manager exit method that resets the settings to their original values"
        for k, v in self._original_values.items():
            setattr(self, k, v)

    # use rich.table to print the settings
    def __repr__(self) -> str:
        r"""Returns a string representation of the settings."""

        table = rich.table.Table(title="MrMustard Settings")
        table.add_column("Setting")
        table.add_column("Value")

        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            table.add_row(key, str(value))

        rprint(table)
        return ""

    def __setattr__(self, name, value):
        r"""
        The addition of new settings is not allowed. A custom error message is provided.
        """
        try:
            return super().__setattr__(name, value)
        except ValidationError as e:
            if "Object has no attribute" in str(e):
                raise AttributeError(f"unknown MrMustard setting: '{name}'") from None
            raise
