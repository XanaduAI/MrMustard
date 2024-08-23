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

"""
Tests for the Settings class.
"""

from mrmustard import math
from mrmustard.utils.settings import Settings
import pytest

from ..conftest import skip_np


class TestSettings:
    """Tests the Settings class"""

    def test_init(self):
        """Test the default values of the settings"""
        settings = Settings()

        assert settings.HBAR == 1.0
        assert settings.DEBUG is False
        assert (
            settings.AUTOSHAPE_PROBABILITY == 0.99999
        )  # capture at least 99.9% of the probability
        assert settings.AUTOCUTOFF_MAX_CUTOFF == 100
        assert settings.AUTOCUTOFF_MIN_CUTOFF == 1
        assert settings.CIRCUIT_DECIMALS == 3
        assert settings.DISCRETIZATION_METHOD == "clenshaw"
        assert settings.EQ_TRANSFORMATION_CUTOFF == 3
        assert settings.EQ_TRANSFORMATION_RTOL_FOCK == 1e-3
        assert settings.EQ_TRANSFORMATION_RTOL_GAUSS == 1e-6
        assert settings.PNR_INTERNAL_CUTOFF == 50
        assert settings.HOMODYNE_SQUEEZING == 10.0
        assert settings.PRECISION_BITS_HERMITE_POLY == 128
        assert settings.PROGRESSBAR is True
        assert settings.BS_FOCK_METHOD == "vanilla"  # can be 'vanilla' or 'schwinger'

    def test_setters(self):
        settings = Settings()

        ap0 = settings.AUTOSHAPE_PROBABILITY
        settings.AUTOSHAPE_PROBABILITY = 0.1
        assert settings.AUTOSHAPE_PROBABILITY == 0.1
        settings.AUTOSHAPE_PROBABILITY = ap0

        db0 = settings.DEBUG
        settings.DEBUG = True
        assert settings.DEBUG is True
        settings.DEBUG = db0

        dbsm0 = settings.BS_FOCK_METHOD
        settings.BS_FOCK_METHOD = "schwinger"
        assert settings.BS_FOCK_METHOD == "schwinger"
        settings.BS_FOCK_METHOD = dbsm0

        eqtc0 = settings.EQ_TRANSFORMATION_CUTOFF
        settings.EQ_TRANSFORMATION_CUTOFF = 2
        assert settings.EQ_TRANSFORMATION_CUTOFF == 2
        settings.EQ_TRANSFORMATION_CUTOFF = eqtc0

        pnr0 = settings.PNR_INTERNAL_CUTOFF
        settings.PNR_INTERNAL_CUTOFF = False
        assert settings.PNR_INTERNAL_CUTOFF is False
        settings.PNR_INTERNAL_CUTOFF = pnr0

        pb0 = settings.PROGRESSBAR
        settings.PROGRESSBAR = False
        assert settings.PROGRESSBAR is False
        settings.PROGRESSBAR = pb0

        s0 = settings.SEED
        settings.SEED = None
        assert settings.SEED is not None
        settings.SEED = s0

        hs0 = settings.HOMODYNE_SQUEEZING
        settings.HOMODYNE_SQUEEZING = 20.1
        assert settings.HOMODYNE_SQUEEZING == 20.1
        settings.HOMODYNE_SQUEEZING = hs0

        fock_rtol = settings.EQ_TRANSFORMATION_RTOL_FOCK
        settings.EQ_TRANSFORMATION_RTOL_FOCK = 0.02
        assert settings.EQ_TRANSFORMATION_RTOL_FOCK == 0.02
        settings.EQ_TRANSFORMATION_RTOL_FOCK = fock_rtol

        gauss_rtol = settings.EQ_TRANSFORMATION_RTOL_GAUSS
        settings.EQ_TRANSFORMATION_RTOL_GAUSS = 0.02
        assert settings.EQ_TRANSFORMATION_RTOL_GAUSS == 0.02
        settings.EQ_TRANSFORMATION_RTOL_GAUSS = gauss_rtol

        assert settings.HBAR == 1.0
        with pytest.raises(ValueError, match="Cannot change"):
            settings.HBAR = 3

        with pytest.raises(ValueError, match="precision_bits_hermite_poly"):
            settings.PRECISION_BITS_HERMITE_POLY = 9

    def test_settings_seed_randomness_at_init(self):
        """Test that the random seed is set randomly as MM is initialized."""
        settings = Settings()
        seed0 = settings.SEED
        del Settings.instance
        settings = Settings()
        seed1 = settings.SEED
        assert seed0 != seed1

    def test_reproducibility(self):
        """Test that the random state is reproducible."""
        settings = Settings()
        settings.SEED = 42
        seq0 = [settings.rng.integers(0, 2**31 - 1) for _ in range(10)]
        settings.SEED = 42
        seq1 = [settings.rng.integers(0, 2**31 - 1) for _ in range(10)]
        assert seq0 == seq1

    def test_complex_warnings(self, caplog):
        """Tests that complex warnings can be correctly activated and deactivated."""
        skip_np()

        settings = Settings()

        assert settings.COMPLEX_WARNING is False
        math.cast(1 + 1j, math.float64)
        assert len(caplog.records) == 0

        settings.COMPLEX_WARNING = True
        math.cast(1 + 1j, math.float64)
        assert len(caplog.records) == 1
        assert "You are casting an input of type complex128" in caplog.records[0].msg

        settings.COMPLEX_WARNING = False
        math.cast(1 + 1j, math.float64)
        assert len(caplog.records) == 1
