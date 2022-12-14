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

from mrmustard import Settings


def test_settings_seed_randomness_at_init():
    """Test that the random seed is set randomly as MM is initialized."""
    settings = Settings()
    seed0 = settings.SEED
    del Settings.instance
    settings = Settings()
    seed1 = settings.SEED
    assert seed0 != seed1


def test_reproducibility():
    """Test that the random state is reproducible."""
    settings = Settings()
    settings.SEED = 42
    seq0 = [settings.random_state.randint(0, 2**32) for _ in range(10)]
    settings.SEED = 42
    seq1 = [settings.random_state.randint(0, 2**32) for _ in range(10)]
    assert seq0 == seq1
