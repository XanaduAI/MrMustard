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
from ..states import Number
from ...physics.representations import Fock
from ...physics import triples
from ...physics.fock import fock_state
from ..utils import make_parameter, reshape_params

__all__ = ["PNR"]


class PNR(Detector):
    r"""
    The Photon Number Resolving (PNR) detector.

    PNR detectors have a batched ``Fock`` representation with array of shape
    ``(cutoff, cutoff, cutoff)``. The `n`-th batch , with

    .. code-block::

        >>> from mrmustard.lab_dev import *
        >>> import numpy as np

        >>> state = Number(modes=[0, 1], n=[2, 0], cutoffs=2)
        >>> gate = BSgate([0, 1], theta=np.pi/4)
        >>> proj01 = Number(modes=[0, 1], n=[2, 0]).dual

        >>> # initialize the circuit and specify a custom path
        >>> circuit = Circuit([state, gate, proj01])
        >>> circuit.path = [(1, 2), (0, 1)]

        >>> result = Simulator().run(circuit)

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
        self._cutoff = cutoff or settings.AUTOCUTOFF_MAX_CUTOFF
        super().__init__(
            modes=set([mode]),
            name="PNR",
            meas_op=[Number([mode], n, self.cutoff) for n in range(self.cutoff + 1)],
        )

    @property
    def cutoff(self) -> int:
        r"""
        The cutoff of this PNR.
        """
        return self._cutoff

    # @property
    # def representation(self) -> Fock:
    #     array = [math.outer((f := fock_state(n, self.cutoff)), f) for n in range(self.cutoff + 1)]
    #     return Fock(math.concat(array, 2), batched=True)
