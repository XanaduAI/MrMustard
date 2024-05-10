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
The avaliable transformation.

As for states, Mr Mustard provides a range of built-in transformations, such as:

    .. code-block::

        >>> from mrmustard.lab_dev.transformations import Dgate, Sgate, Attenuator

        >>> # the displacement gate
        >>> dgate = Dgate(modes=[0], x=1)

        >>> # the squeezing gate
        >>> sgate = Sgate(modes=[1, 2], r=0.8)

        >>> # the attenuator channel
        >>> att = Attenuator(modes=[3], transmissivity=0.9)

All these transformations are of one of two types, namely :class:`~mrmustard.lab_dev.transformations.Unitary` or
:class:`~mrmustard.lab_dev.transformations.Channel`.
"""

from .base import *
from .transformations import *
