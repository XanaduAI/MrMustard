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
"""
Unit tests for the :mod:`thewalrus` configuration class :class:`Configuration`.
"""

import contextlib
import io
import re

import mrmustard as mm


def test_about():
    """Tests if the about string prints correctly."""
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        mm.about()
    out = f.getvalue().strip()

    assert "Python version:" in out
    pl_version_match = re.search(r"Mr Mustard version:\s+([\S]+)\n", out).group(1)
    assert mm.version() in pl_version_match
    assert "Numpy version" in out
    assert "Scipy version" in out
    assert "The Walrus version" in out
    assert "TensorFlow version" in out
