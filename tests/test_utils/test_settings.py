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


class TestSettings:
    """Tests the Settings class"""

    def test_init(self):
        """Test the default values of the settings"""
        settings = Settings()

        assert settings.HBAR == 1.0
        assert settings.AUTOSHAPE_PROBABILITY == 0.99999
        assert settings.AUTOSHAPE_MAX == 50
        assert settings.AUTOSHAPE_MIN == 1
        assert settings.DISCRETIZATION_METHOD == "clenshaw"
        assert settings.DEFAULT_FOCK_SIZE == 50
        assert settings.DEFAULT_REPRESENTATION == "Fock"
        assert settings.PROGRESSBAR is True

    def test_setters(self):
        settings = Settings()

        cw = settings.COMPLEX_WARNING
        settings.COMPLEX_WARNING = not cw
        assert (not cw) == settings.COMPLEX_WARNING
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

    @pytest.mark.requires_backend("tensorflow")
    def test_complex_warnings(self, caplog):
        """Tests that complex warnings can be correctly activated and deactivated."""

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
