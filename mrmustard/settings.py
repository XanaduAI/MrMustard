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

"""A module containing the settings.
"""

from rich import print
import rich.table
import numpy as np

__all__ = ["Settings", "settings"]


class ImmutableSetting:
    r"""A setting that becomes immutable after the first time its value is queried.

    Args:
        value (any): the default value of this setting
        name (str): the name of this setting
    """

    def __init__(self, value: any, name: str) -> None:
        self._value = value
        self._name = name
        self._is_immutable = False

    @property
    def name(self):
        r"""The name of this setting."""
        return self._name

    @property
    def value(self):
        r"""The value of this setting."""
        self._is_immutable = True
        return self._value

    @value.setter
    def value(self, value):
        if self._is_immutable:
            raise ValueError(f"Cannot change the value of `settings.{self.name}`.")
        self._value = value


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
        self._backend = "tensorflow"
        self._hbar = ImmutableSetting(2.0, "HBAR")
        self._debug = False
        self._autocutoff_probability = 0.999  # capture at least 99.9% of the probability
        self._autocutoff_max_cutoff = 100
        self._autocutoff_min_cutoff = 1
        self._circuit_decimals = 3
        self._discretization_method = "iterative"
        # use cutoff=5 for each mode when determining if two transformations in fock repr are equal
        # 3 is enough to include a full step of the rec relations
        self._eq_transformation_cutoff = 3
        self._eq_transformation_rtol_fock = 1e-3
        self._eq_transformation_rtol_gauss = 1e-6
        # for the detectors
        self._pnr_internal_cutoff = 50
        self._homodyne_squeezing = 10.0
        # misc
        self._progressbar = True
        self._seed = np.random.randint(0, 2**31 - 1)
        self.rng = np.random.default_rng(self._seed)
        self._default_bs_method = "vanilla"  # can be 'vanilla' or 'schwinger'

    @property
    def AUTOCUTOFF_MAX_CUTOFF(self):
        r"""The maximum value for autocutoff. Default is ``100``."""
        return self._autocutoff_max_cutoff

    @AUTOCUTOFF_MAX_CUTOFF.setter
    def AUTOCUTOFF_MAX_CUTOFF(self, value: str):
        self._autocutoff_max_cutoff = value

    @property
    def AUTOCUTOFF_MIN_CUTOFF(self):
        r"""The minimum value for autocutoff. Default is ``1``."""
        return self._autocutoff_min_cutoff

    @AUTOCUTOFF_MIN_CUTOFF.setter
    def AUTOCUTOFF_MIN_CUTOFF(self, value: str):
        self._autocutoff_min_cutoff = value

    @property
    def AUTOCUTOFF_PROBABILITY(self):
        r"""The autocutoff probability. Default is ``0.999``."""
        return self._autocutoff_probability

    @AUTOCUTOFF_PROBABILITY.setter
    def AUTOCUTOFF_PROBABILITY(self, value: str):
        self._autocutoff_probability = value

    @property
    def BACKEND(self):
        r"""The backend which is used. Default is ``tensorflow``.

        Can be either ``'tensorflow'`` or ``'torch'``.
        """
        return self._backend

    @BACKEND.setter
    def BACKEND(self, value: str):
        if value not in ["tensorflow", "torch"]:  # pragma: no cover
            raise ValueError("Backend must be either 'tensorflow' or 'torch'")
        self._backend = value

    @property
    def CIRCUIT_DECIMALS(self):
        r"""The number of decimals displayed when drawing a circuit with parameters. Default is ``3``."""
        return self._circuit_decimals

    @CIRCUIT_DECIMALS.setter
    def CIRCUIT_DECIMALS(self, value: str):
        self._circuit_decimals = value

    @property
    def DEBUG(self):
        r"""Whether or not to print the vector of means and the covariance matrix alongside the
        html representation of a state. Default is ``False``.
        """
        return self._debug

    @DEBUG.setter
    def DEBUG(self, value: str):
        self._debug = value

    @property
    def DISCRETIZATION_METHOD(self):
        r"""The method used to discretize the Wigner function. Default is ``iterative``.

        Can be either ``'iterative'`` or ``'clenshaw'``.
        """
        return self._discretization_method

    @DISCRETIZATION_METHOD.setter
    def DISCRETIZATION_METHOD(self, value: str):
        self._discretization_method = value

    @property
    def DEFAULT_BS_METHOD(self):
        r"""The default method for computing the transformation operated by a beam splitter in
         the Fock basis . Default is ``vanilla``.

        Can be either ``'vanilla'`` or ``'schwinger'``.
        """
        return self._default_bs_method

    @DEFAULT_BS_METHOD.setter
    def DEFAULT_BS_METHOD(self, value: str):
        self._default_bs_method = value

    @property
    def EQ_TRANSFORMATION_CUTOFF(self):
        r"""The cutoff used when comparing two transformations via the Choi–Jamiolkowski
        isomorphism. Default is ``3``."""
        return self._eq_transformation_cutoff

    @EQ_TRANSFORMATION_CUTOFF.setter
    def EQ_TRANSFORMATION_CUTOFF(self, value: str):
        self._eq_transformation_cutoff = value

    @property
    def EQ_TRANSFORMATION_RTOL_FOCK(self):
        r"""The relative tolerance used when comparing two transformations via the Choi–Jamiolkowski
        isomorphism. Default is ``1e-3``."""
        return self._eq_transformation_rtol_fock

    @EQ_TRANSFORMATION_RTOL_FOCK.setter
    def EQ_TRANSFORMATION_RTOL_FOCK(self, value: str):
        self._eq_transformation_rtol_fock = value

    @property
    def EQ_TRANSFORMATION_RTOL_GAUSS(self):
        r"""The relative tolerance used when comparing two transformations on Gaussian states.
        Default is ``1e-6``."""
        return self._eq_transformation_rtol_gauss

    @EQ_TRANSFORMATION_RTOL_GAUSS.setter
    def EQ_TRANSFORMATION_RTOL_GAUSS(self, value: str):
        self._eq_transformation_rtol_gauss = value

    @property
    def HBAR(self):
        r"""The value of the Planck constant. Default is ``2``.

        Cannot be changed after its value is queried for the first time.
        """
        return self._hbar.value

    @HBAR.setter
    def HBAR(self, value: str):
        self._hbar.value = value

    @property
    def HOMODYNE_SQUEEZING(self):
        r"""The value of squeezing for homodyne measurements. Default is ``10``."""
        return self._homodyne_squeezing

    @HOMODYNE_SQUEEZING.setter
    def HOMODYNE_SQUEEZING(self, value: str):
        self._homodyne_squeezing = value

    @property
    def PNR_INTERNAL_CUTOFF(self):
        r"""The cutoff used when computing the output of a PNR detection. Default is ``50``."""
        return self._pnr_internal_cutoff

    @PNR_INTERNAL_CUTOFF.setter
    def PNR_INTERNAL_CUTOFF(self, value: str):
        self._pnr_internal_cutoff = value

    @property
    def PROGRESSBAR(self):
        r"""Whether or not to display the progress bar when performing training. Default is ``True``."""
        return self._progressbar

    @PROGRESSBAR.setter
    def PROGRESSBAR(self, value: str):
        self._progressbar = value

    @property
    def SEED(self):
        r"""Returns the seed value if set, otherwise returns a random seed."""
        if self._seed is None:
            self._seed = np.random.randint(0, 2**31 - 1)
            self.rng = np.random.default_rng(self._seed)
        return self._seed

    @SEED.setter
    def SEED(self, value):
        self._seed = value
        self.rng = np.random.default_rng(self._seed)

    # use rich.table to print the settings
    def __repr__(self) -> str:
        r"""Returns a string representation of the settings."""

        # attributes that should not be displayed in the table
        not_displayed = ["rng"]

        table = rich.table.Table(title="MrMustard Settings")
        table.add_column("Setting")
        table.add_column("Value")

        for key, val in self.__dict__.items():
            if key in not_displayed:
                continue
            key = key.upper()[1:]
            value = str(val._value) if isinstance(val, ImmutableSetting) else str(val)
            table.add_row(key, value)

        print(table)
        return ""


settings = Settings()
"""Settings object."""
