# Copyright 2023 Xanadu Quantum Technologies Inc.

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
This module contains the Homodyne class.
"""

from __future__ import annotations

import numpy as np

from numbers import Number
from typing import Optional, Sequence

from ..states import Number, ConditionalState, State
from .base import MeasurementDevice
from ..circuit_components import CircuitComponent
from mrmustard import settings, math

__all__ = ["Homodyne"]


class Homodyne(MeasurementDevice):
    r""" """

    def __init__(
        self,
        modes: Sequence[int],
    ):
        super().__init__(
            modes=set(modes),
            name="Homodyne",
            sampling_technique=[],
        )
