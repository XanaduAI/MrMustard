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

from mrmustard.settings import Settings, ImmutableSetting
import pytest


class TestImmutableSettings:
    """Tests the ImmutableSettings class"""

    def test_init(self):
        """Tests the default values of the immutable settings"""
        s = ImmutableSetting("foo", "bar")
        assert s.value == "foo"
        assert s.name == "bar"

    def test_setting_becomes_immutable(self):
        """Tests that immutable settings become immutable"""
        s = ImmutableSetting(1, "my_name")

        s.value = 2
        assert s.value == 2

        with pytest.raises(ValueError, match=f"value of `settings.{s.name}`"):
            s.value = 3


class TestSettings:
    """Tests the Settings class"""

    def test_init(self):
        """Test the default values of the settings"""
        settings = Settings()

        assert settings.BACKEND == "tensorflow"
        assert settings.HBAR == 2.0
        assert settings.CHOI_R == 0.881373587019543  # np.arcsinh(1.0)
        assert settings.DEBUG is False
        assert settings.AUTOCUTOFF_PROBABILITY == 0.999  # capture at least 99.9% of the probability
        assert settings.AUTOCUTOFF_MAX_CUTOFF == 100
        assert settings.AUTOCUTOFF_MIN_CUTOFF == 1
        assert settings.CIRCUIT_DECIMALS == 3
        assert settings.EQ_TRANSFORMATION_CUTOFF == 3
        assert settings.EQ_TRANSFORMATION_RTOL_FOCK == 1e-3
        assert settings.EQ_TRANSFORMATION_RTOL_GAUSS == 1e-6
        assert settings.PNR_INTERNAL_CUTOFF == 50
        assert settings.HOMODYNE_SQUEEZING == 10.0
        assert settings.PROGRESSBAR is True
        assert settings.DEFAULT_BS_METHOD == "vanilla"  # can be 'vanilla' or 'schwinger'

        with pytest.raises(ValueError, match="Cannot change"):
            settings.HBAR = 3

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
