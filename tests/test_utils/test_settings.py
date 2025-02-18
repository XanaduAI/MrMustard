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

import pytest

from mrmustard import math
from mrmustard.utils.settings import Settings

from ..conftest import skip_np_and_jax


class TestSettings:
    """Tests the Settings class"""

    def test_init(self):
        """Test the default values of the settings"""
        settings = Settings()

        assert settings.HBAR == 1.0
        assert settings.DEBUG is False
        assert settings.AUTOSHAPE_PROBABILITY == 0.99999
        assert settings.AUTOCUTOFF_MAX_CUTOFF == 100
        assert settings.AUTOCUTOFF_MIN_CUTOFF == 1
        assert settings.CIRCUIT_DECIMALS == 3
        assert settings.DISCRETIZATION_METHOD == "clenshaw"
        assert settings.EQ_TRANSFORMATION_CUTOFF == 3
        assert settings.EQ_TRANSFORMATION_RTOL_FOCK == 1e-3
        assert settings.EQ_TRANSFORMATION_RTOL_GAUSS == 1e-6
        assert settings.PNR_INTERNAL_CUTOFF == 50
        assert settings.HOMODYNE_SQUEEZING == 10.0
        assert settings.PROGRESSBAR is True
        assert settings.BS_FOCK_METHOD == "vanilla"

    def test_setters(self):
        settings = Settings()

        cw = settings.COMPLEX_WARNING
        settings.COMPLEX_WARNING = not cw
        assert settings.COMPLEX_WARNING == (not cw)
        settings.COMPLEX_WARNING = cw

        s0 = settings.SEED
        settings.SEED = None
        assert settings.SEED is not None
        settings.SEED = s0

        assert settings.HBAR == 1.0
        with pytest.warns(UserWarning, match="Changing HBAR can conflict with prior computations"):
            settings.HBAR = 3

    def test_settings_seed_randomness_at_init(self):
        """Test that the random seed is set randomly as MM is initialized."""
        settings = Settings()
        seed0 = settings.SEED
        del Settings._instance
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
        skip_np_and_jax()

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

    def test_cannot_add_new_settings(self):
        """Test that new settings are rejected (eg. typos)."""
        settings = Settings()
        with pytest.raises(AttributeError, match="unknown MrMustard setting: 'HBARR'"):
            settings.HBARR = 1.0

    def test_context_manager(self):
        """Test that the context manager works correctly."""
        settings = Settings()

        with settings(AUTOSHAPE_PROBABILITY=0.1, HBAR=5.0):
            assert settings.AUTOSHAPE_PROBABILITY == 0.1
            assert settings.HBAR == 5.0
        assert settings.AUTOSHAPE_PROBABILITY == 0.99999
        assert settings.HBAR == 1.0
