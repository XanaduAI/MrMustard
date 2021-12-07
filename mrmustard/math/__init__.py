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
The ``math`` module contains low-level functions for performing mathematical operations.

It is recommended that users access the backends using the an instance of the :class:`Math` class rather than the backends themselves.

The Math class is a wrapper that passes the calls to the currently active backend, which is determined by
the ``BACKEND`` parameter in ``mrmustard.settings`` (the default is ``tensorflow``).

The advantage of using the Math class is that the same code can run on different backends, allowing for a
greater degree of flexibility and code reuse.

.. code-block::

    from mrmustard.math import Math
    math = Math()
    math.cos(x)  # tensorflow backend

    from mrmustard import settings
    settings.BACKEND = 'torch'

    math.cos(x)  # torch backend
"""


from mrmustard import settings
import importlib

if importlib.util.find_spec("tensorflow"):
    from mrmustard.math.tensorflow import TFMath
if importlib.util.find_spec("torch"):
    from mrmustard.math.torch import TorchMath


class Math:
    r"""
    This class is a switcher for performing math operations on the currently active backend.
    """

    def __getattribute__(self, name):
        if settings.backend == "tensorflow":
            return object.__getattribute__(TFMath(), name)
        elif settings.backend == "torch":
            return object.__getattribute__(TorchMath(), name)
