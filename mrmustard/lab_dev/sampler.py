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

# pylint: disable=too-many-branches

"""
This module contains the sampler class.
"""

from __future__ import annotations


__all__ = ["Sampler"]


class Sampler:
    r""" """

    def __init__(self, meas_outcomes, sampling_technique) -> None:
        self._meas_outcomes = meas_outcomes
        self._sampling_technique = sampling_technique

    @property
    def meas_outcomes(self):
        r""" """
        return self._meas_outcomes

    @property
    def sampling_technique(self):
        r""" """
        return self._sampling_technique

    def sample(self):
        r""" """
        pass

    def prob_dist(self):
        r""" """
        pass

    def povms(self):
        r""" """
        pass
