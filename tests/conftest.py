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

import os
from pathlib import Path
import pytest

from mrmustard import math
from hypothesis import Verbosity, settings as hyp_settings

print("pytest.conf -----------------------")

# ~~~~~~~~~~
# Hypothesis
# ~~~~~~~~~~

hyp_settings.register_profile("ci", max_examples=10, deadline=None)
hyp_settings.register_profile("dev", max_examples=10, deadline=None)
hyp_settings.register_profile(
    "debug", max_examples=10, verbosity=Verbosity.verbose, deadline=None
)

hyp_settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))

# ~~~~~~
# Pytest
# ~~~~~~


def pytest_addoption(parser):
    r"""
    Adds the option to select the backend using the ``--backend`` flag. For example,
    ``pytest --backend=tensorflow`` runs all the tests with tensorflow backend. The command
    ``pytest`` defaults to ``pytest --backend=numpy``.
    """
    parser.addoption("--backend", default="numpy", help="``numpy`` or ``tensorflow``.")


@pytest.fixture
def backend(request):
    r"""
    Extracts ``backend`` from request.
    """
    return request.config.getoption("--backend")


def pytest_ignore_collect(path, config):
    """Skip test_training when using the numpy backend."""
    if config.getoption("--backend") == "numpy" and "test_training" in Path(path).parts:
        return True
    return False


@pytest.fixture(autouse=True)
def set_backend(backend):
    r"""
    Sets backend for all the tests.
    """
    math.change_backend(f"{backend}")


def skip_np():
    if math.backend_name == "numpy":
        pytest.skip("numpy")


def pytest_configure(config):
    pass  # your code goes here
