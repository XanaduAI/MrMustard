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

import importlib
from rich.pretty import install
install()  # NOTE: just for the looks, not stricly required

__version__ = "0.1.0"

def activate_backend(backend_name: str):
    "Activates the backend in the core"
    Backend = importlib.import_module("mrmustard.backends." + backend_name).Backend
    from mrmustard.core import fock;
    fock.backend = Backend()
    
    from mrmustard.core import gaussian;
    gaussian.backend = Backend()

    from mrmustard.core import train;
    train.backend = Backend();
    train.euclidean_opt = train.backend.DefaultEuclideanOptimizer()

    from mrmustard.concrete import circuit
    circuit.backend = Backend()
    
    from mrmustard.experimental import xptensor
    xptensor.backend = Backend()
    
    return 

activate_backend("tensorflow")

from mrmustard.concrete import *
from mrmustard.abstract import State
