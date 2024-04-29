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
The :mod:`~mrmustard.lab_dev` module contains all the objects that would normally be found in a photonic lab, such as:

* Several useful states, such as :class:`~mrmustard.lab_dev.states.Vacuum`,
  :class:`~mrmustard.lab_dev.states.Coherent`, and :class:`~mrmustard.lab_dev.states.SqueezedVacuum`.

* The gates routinely performed in optical experiments, including
  :class:`~mrmustard.lab_dev.transformations.Dgate`, :class:`~mrmustard.lab_dev.transformations.Sgate`,
  and :class:`~mrmustard.lab_dev.transformations.BSgate`.

* Typical noise channels such as the :class:`~mrmustard.lab_dev.transformations.Attenuator`.

* Detectors (currently in development).

Additionally, it enables users to initialize custom states, gates, channels, and measurements.

The ``>>`` operator allows combining these objects together with intuitive, paper-style syntax.

.. code-block::

    >>> from mrmustard.lab_dev import *
    >>> from mrmustard import settings
    >>> import numpy as np

    >>> settings.AUTOCUTOFF_MAX_CUTOFF = 40

    >>> # initialize three single-mode states with one, zero, and zero photons, respectively
    >>> s0 = Number(modes=[0], n=1)
    >>> s1 = Number(modes=[1], n=0)
    >>> s2 = Number(modes=[2], n=0)

    >>> # initialize 50/50 beam splitters between modes (0, 1) and (1, 2)
    >>> bs01 = BSgate(modes=[0, 1], theta=np.pi/4)
    >>> bs12 = BSgate(modes=[1, 2], theta=np.pi/4)

    >>> # simulate the result of an experiment where the three states are sent through
    >>> # the beam splitters
    >>> result = s0 >> s1 >> s2 >> bs01 >> bs12

    >>> # calculate expectation values on the resulting state
    >>> assert np.allclose(result.expectation(Number(modes=[0], n=0)), 0.5)
    >>> assert np.allclose(result.expectation(Number(modes=[0], n=1)), 0.5)
    >>> assert np.allclose(result.expectation(Number(modes=[0], n=2)), 0)
    >>> assert np.allclose(result.expectation(Number(modes=[0, 1, 2], n=[0, 1, 0])), 0.25)

While :mod:`~mrmustard.lab_dev` is designed to be intuitive and easy to use, proficient users can take advantage
of several features to speed up their computations. For example, using
:class:`~mrmustard.lab_dev.circuits.Circuit`\s and the :class:`~mrmustard.lab_dev.simulator.Simulator`,
they can take advantage of built-in tensor network functionality to run circuits more efficiently.

.. code-block::

    >>> from mrmustard.lab_dev import *
    >>> from mrmustard import settings
    >>> import numpy as np

    >>> settings.AUTOCUTOFF_MAX_CUTOFF = 40

    >>> # initialize three single-mode states with one, zero, and zero photons, respectively
    >>> s0 = Number(modes=[0], n=1)
    >>> s1 = Number(modes=[1], n=0)
    >>> s2 = Number(modes=[2], n=0)

    >>> # initialize 50/50 beam splitters between modes (0, 1) and (1, 2)
    >>> bs01 = BSgate(modes=[0, 1], theta=np.pi/4)
    >>> bs12 = BSgate(modes=[1, 2], theta=np.pi/4)

    >>> # initialize projectors for modes 1 and 2
    >>> p1 = Number(modes=[1], n=0).dual
    >>> p2 = Number(modes=[2], n=0).dual

    >>> # initialize a circuit
    >>> circ = Circuit([s0, s1, s2, bs01, bs12, p1, p2])

    >>> # specify a "contraction path" for the circuit above -- i.e., in what order you'd
    >>> # like to combine its components via ``>>``
    >>> circ.path = [(4, 5), (4, 6), (2, 4), (0, 3), (1, 0), (1, 2)]

    >>> # run the circuit -- the `Simulator` will follow the circuit's path
    >>> result = Simulator().run(circ)
    >>> assert result == s0 >> s1 >> s2 >> bs01 >> bs12 >> p1 >> p2

    >>> # every path leads to the same result, but some paths are faster than others.
    >>> # for example, selecting this path leads to a simulation time of about 17ms on a
    >>> # typical laptop ...
    >>> circ.path = [(4, 5), (4, 6), (2, 4), (0, 3), (1, 0), (1, 2)]
    >>> assert result == Simulator().run(circ)

    >>> # ... while with this path, the simulation time drops to about 7ms.
    >>> circ.path = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)]
    >>> assert result == Simulator().run(circ)

Check out our guides to learn more about :mod:`~mrmustard.lab_dev` and its core functionalities:

* The :mod:`~mrmustard.lab_dev.circuit_components` guide introduces the
  :class:`~mrmustard.lab_dev.circuit_components.CircuitComponent` class, which is the parent of
  every state, gate, channel, and measurement object in :mod:`~mrmustard.lab_dev`.
* The :mod:`~mrmustard.lab_dev.states` guide illustrates how to initialize states.
* The :mod:`~mrmustard.lab_dev.transformations` guide shows how to initialize unitary gates and
  channels.
* :mod:`~mrmustard.lab_dev.circuits` tells you everything you need to know about circuits.
* The :mod:`~mrmustard.lab_dev.simulator` page shows how to use the simulator object and discusses
  how to select the best paths for your circuits.
* For more advanced functionality, take a look at the :mod:`~mrmustard.lab_dev.circuit_components`
  module, which contains a series of maps (e.g., the trace-out operation) expressed as circuit
  components.
  """

from .circuit_components import *
from .circuit_components_utils import *
from .circuits import *
from .states import *
from .simulator import *
from .transformations import *
from .wires import *
