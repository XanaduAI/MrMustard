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

r"""
Simulators for quantum circuits.

The simulator allows contracting the components stored in quantum circuits.

.. code-block::

    >>> from mrmustard.lab_dev import *
    >>> import numpy as np

    >>> # initialize a circuit
    >>> state = Number(modes=[0, 1], n=[2, 0], cutoffs=2)
    >>> gate = BSgate([0, 1], theta=np.pi/4)
    >>> proj1 = Number(modes=[1], n=[0]).dual
    >>> circuit = Circuit([state, gate, proj1])

    >>> # run the simulation
    >>> result = Simulator().run(circuit)

    >>> # the simulator returns a component that can be potentially be plugged
    >>> # into another circuit
    >>> assert isinstance(result, CircuitComponent)

The simulation is carried out by contracting the components of the given circuit in pairs,
until only one component is left and returned. In the examples above, the contractions happen
in a "left-to-right" fashion, meaning that the left-most component in the circuit (``state``)
is contracted with the one in its right (``gate``), and finally the resulting component is
contracted with the projector. This provides a simple and convenient way to run simulations,
but for large circuits, different contraction paths may be more efficient.

The ``path`` attribute of ``Circuit``\s allows customising the contraction order and potentially
speeding up the simulation. When a ``path`` of the type ``[(i, j), (l, m), ...]`` is given, the
simulator creates a dictionary of the type ``{0: c0, ..., N: cN}``, where ``[c0, .., cN]``
is the ``circuit.component`` list. Then:

* The two components ``ci`` and ``cj`` in positions ``i`` and ``j`` are contracted. ``ci`` is
  replaced by the resulting component ``cj >> cj``, while ``cj`` is popped.
* The two components ``cl`` and ``cm`` in positions ``l`` and ``m`` are contracted. ``cl`` is
  replaced by the resulting component ``cl >> cm``, while ``cm`` is popped.
* Et cetera.

Below is an example where a circuit is simulated in a "right-to-left" fashion:

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

The setter for ``path`` also validates the path using the ``validate_path`` function of
``Circuit``.
"""

from __future__ import annotations

from .circuit_components import CircuitComponent
from .circuits import Circuit

__all__ = ["Simulator"]


class Simulator:
    r"""
    A simulator for quantum circuits.

    Circuits can be simulated by using the ``run`` method of ``Simulator``:

    .. code-block::

        >>> from mrmustard.lab_dev import *
        >>> import numpy as np

        >>> # initialize a circuit
        >>> state = Number(modes=[0, 1], n=[2, 0], cutoffs=2)
        >>> gate = BSgate([0, 1], theta=np.pi/4)
        >>> proj1 = Number(modes=[1], n=[0]).dual
        >>> circuit = Circuit([state, gate, proj1])

        >>> # run the simulation
        >>> result = Simulator().run(circuit)

        >>> # the simulator returns a component that can be potentially be plugged
        >>> # into another circuit
        >>> assert isinstance(result, CircuitComponent)

    The simulation is carried out by contracting the components of the given circuit in pairs,
    until only one component is left and returned. In the examples above, the contractions happen
    in a "left-to-right" fashion, meaning that the left-most component in the circuit (``state``)
    is contracted with the one in its right (``gate``), and finally the resulting component is
    contracted with the projector. This provides a simple and convenient way to run simulations,
    but for large circuits, different contraction paths may be more efficient.

    The ``path`` attribute of ``Circuit``\s allows customising the contraction order and potentially
    speeding up the simulation. When a ``path`` of the type ``[(i, j), (l, m), ...]`` is given, the
    simulator creates a dictionary of the type ``{0: c0, ..., N: cN}``, where ``[c0, .., cN]``
    is the ``circuit.component`` list. Then:

    * The two components ``ci`` and ``cj`` in positions ``i`` and ``j`` are contracted. ``ci`` is
      replaced by the resulting component ``cj >> cj``, while ``cj`` is popped.
    * The two components ``cl`` and ``cm`` in positions ``l`` and ``m`` are contracted. ``cl`` is
      replaced by the resulting component ``cl >> cm``, while ``cm`` is popped.
    * Et cetera.

    Below is an example where a circuit is simulated in a "right-to-left" fashion:

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

    The setter for ``path`` also validates the path using the ``validate_path`` function of
    ``Circuit``.
    """

    def run(self, circuit: Circuit) -> CircuitComponent:
        r"""
        Runs the simulations of the given circuit.

        Arguments:
            circuit: The circuit to simulate.

        Returns:
            A circuit component representing the entire circuit.

        Raises:
            ValueError: If ``circuit`` has an incomplete path.
        """
        if not circuit.path:
            circuit.make_path()

        if len(circuit.path) != len(circuit) - 1:
            msg = f"``circuit.path`` needs to specify {len(circuit) - 1} contractions, "
            msg += f"found {len(circuit.path)}."
            raise ValueError(msg)

        ret = dict(enumerate(circuit.components))
        for idx0, idx1 in circuit.path:
            ret[idx0] = ret[idx0] >> ret.pop(idx1)

        return list(ret.values())[0]
