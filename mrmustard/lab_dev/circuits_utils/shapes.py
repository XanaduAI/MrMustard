# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Propagate shapes through a list of components.
"""

from mrmustard.lab_dev.transformations import BSgate


def propagate_component_shapes(components, indices_dict):
    r"""Propagates the shape information so that the shapes of the components are better
    than those provided by the auto_shape attribute.

    .. code-block:: python

        >>> from mrmustard.lab_dev import BSgate, Dgate, Coherent, Circuit, SqueezedVacuum

        >>> circ = Circuit([Coherent([0], x=1.0), Dgate([0], 0.1)])
        >>> assert [op.auto_shape() for op in circ] == [(5,), (50,50)]
        >>> circ.propagate_shapes()
        >>> assert [op.auto_shape() for op in circ] == [(5,), (50, 5)]

        >>> circ = Circuit([SqueezedVacuum([0,1], r=[0.5,-0.5]), BSgate([0,1], 0.9)])
        >>> assert [op.auto_shape() for op in circ] == [(6, 6), (50, 50, 50, 50)]
        >>> circ.propagate_shapes()
        >>> assert [op.auto_shape() for op in circ] == [(6, 6), (12, 12, 6, 6)]
    """

    for component in components:
        component.manual_shape = list(component.auto_shape())

    # update the manual_shapes until convergence
    changes = True
    while changes:
        changes = False
        # get shapes from neighbors if needed
        for i, component in enumerate(components):
            for j, indices in indices_dict[i].items():
                for a, b in indices.items():
                    s_ia = components[i].manual_shape[a]
                    s_jb = components[j].manual_shape[b]
                    s = min(s_ia or 1e42, s_jb or 1e42) if (s_ia or s_jb) else None
                    if components[j].manual_shape[b] != s:
                        components[j].manual_shape[b] = s
                        changes = True
                    if components[i].manual_shape[a] != s:
                        components[i].manual_shape[a] = s
                        changes = True

        # propagate through BSgates
        for i, component in enumerate(components):
            if isinstance(component, BSgate):
                a, b, c, d = component.manual_shape
                if c and d:
                    if not a or a > c + d:
                        a = c + d
                        changes = True
                    if not b or b > c + d:
                        b = c + d
                        changes = True
                if a and b:
                    if not c or c > a + b:
                        c = a + b
                        changes = True
                    if not d or d > a + b:
                        d = a + b
                        changes = True

                components[i].manual_shape = [a, b, c, d]
