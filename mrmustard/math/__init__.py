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

r"""
The point of entry for the backend.
"""

import sys

from .autocast import *
from .backend_base import *
from .backend_manager import BackendManager
from .backend_numpy import *
from .caching import *
from .jax_vjps import *
from .lattice import *
from .parameter_set import *
from .parameters import *

sys.modules[__name__] = BackendManager()
