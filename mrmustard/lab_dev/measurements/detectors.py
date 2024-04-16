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
The classes representing transformations in quantum circuits.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

from mrmustard import math, settings
from .base import Detector
from ...physics.representations import Fock
from ...physics import triples
from ...physics.fock import fock_state
from ..utils import make_parameter, reshape_params

__all__ = ["PNR"]


class PNR(Detector):
    r"""The Photon Number Resolving (PNR) detector.

    Args:
        mode: The mode the this detection is performed on.
        cutoff: The cutoff. If ``None``, it defaults to the value of ``AUTOCUTOFF_MAX_CUTOFF`` of
            ``settings``.
    """

    def __init__(
        self,
        mode: int,
        cutoff: Optional[int] = None,
    ):
        cutoff = cutoff or settings.AUTOCUTOFF_MAX_CUTOFF

        super().__init__(modes=set([mode]), name="PNR")
        self._cutoff = cutoff

    @property
    def cutoff(self) -> int:
        r"""
        The cutoff of this PNR.
        """
        return self._cutoff

    @property
    def representation(self) -> Fock:
        array = [math.outer((f := fock_state(n, self.cutoff)), f) for n in range(self.cutoff + 1)]
        # array = math.concat([[s for s in fock_state(n, self.cutoff)] for n in range(self.cutoff + 1)], 2)
        return Fock(math.concat(array, 2), batched=True)
