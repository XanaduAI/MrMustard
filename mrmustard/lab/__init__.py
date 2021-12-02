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
The lab module is all you need to construct and simulate photonic circuits.
It contains the items you'd find in a lab:

* states (Vacuum, Coherent, SqueezedVacuum, Thermal, etc.)
* transformations (Sgate, BSgate, LossChannel, etc.)
* detectors (PNRDetector, Homodyne, etc.)
* the Circuit class
"""

from .circuit import *
from .states import *
from .gates import *
from .detectors import *
from .abstract import State
