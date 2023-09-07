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

"""Version information.

Version number retrieved from pyproject.toml file
"""
from pathlib import Path
import tomli


def _get_project_root():
    """Compute and return root dir"""
    return Path(__file__).parent.parent


def _get_project_version():
    """Parse 'pyproject.toml' and return current version"""
    with open(f"{_get_project_root()}/pyproject.toml", mode="rb") as pyproject:
        return tomli.load(pyproject)["tool"]["poetry"]["version"]


__version__ = str(_get_project_version())
