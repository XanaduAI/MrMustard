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

from typing import Optional

from mrmustard import settings
from .base import Detector
from ..states import Number

__all__ = ["PNR"]


class PNR(Detector):
    r"""
    The Photon Number Resolving (PNR) detector.

    For a given cutoff :math:`c`, a PNR detector is a ``Detector`` with measurement operators
    :math:`|0\rangle, |1\rangle, \ldots, |c\rangle`, where :math:`|n\rangle` corresponds to
    the state ``Number([mode], n, cutoff)``.

    .. code-block::

        >>> from mrmustard.lab_dev import *

        >>> # the measurement operators of a PNR detector with cutoff at n=4
        >>> cutoff = 4
        >>> meas_op = [Number([0], n, cutoff) for n in range(cutoff + 1)]

        >>> assert Detector("my_detector", [0], meas_op) == PNR(0, cutoff)

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
