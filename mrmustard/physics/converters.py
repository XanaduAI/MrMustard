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
This module contains the functions to convert between different representations.
"""
from typing import Iterable, Union, Optional
from mrmustard.physics.representations import Representation, Bargmann, Fock
from mrmustard import math, settings
from mrmustard.utils.typing import Matrix, Vector
from mrmustard.physics.triples import displacement_map_s_parametrized_Abc
from mrmustard.physics.bargmann import complex_gaussian_integral, join_Abc
from mrmustard.lab_dev.states import State


def to_fock(rep: Representation, shape: Optional[Union[int, Iterable[int]]] = None) -> Fock:
    r"""A function to map ``Representation``s to ``Fock`` representations.

    If the given ``rep`` is ``Fock``, this function simply returns ``rep``.

    Args:
        rep: The orginal representation of the object.
        shape: The shape of the returned representation. If ``shape``is given as an ``int``, it is broadcasted
            to all the dimensions. If ``None``, it defaults to the value of ``AUTOCUTOFF_MAX_CUTOFF`` in
            the settings.

    Raises:
        ValueError: If the size of the shape given is not compatible with the representation.

    Returns:
        A ``Fock`` representation object.

    .. code-block::

        >>> from mrmustard.physics.converters import to_fock
        >>> from mrmustard.physics.representations import Bargmann, Fock
        >>> from mrmustard.physics.triples import displacement_gate_Abc

        >>> bargmann = Bargmann(*displacement_gate_Abc(x=0.1, y=[0.2, 0.3]))
        >>> fock = to_fock(bargmann, shape=10)
        >>> assert isinstance(fock, Fock)

    """
    if isinstance(rep, Bargmann):
        len_shape = len(rep.b[0])
        if not shape:
            shape = settings.AUTOCUTOFF_MAX_CUTOFF
        shape = (shape,) * len_shape if isinstance(shape, int) else shape
        if len_shape != len(shape):
            raise ValueError(f"Given shape ``{shape}`` is incompatible with the representation.")

        array = [math.hermite_renormalized(A, b, c, shape) for A, b, c in zip(rep.A, rep.b, rep.c)]
        return Fock(math.astensor(array), batched=True)
    return rep


def to_phase_space(state: State, modes: Union[int, Iterable[int]], s: int = None) -> Union[Matrix, Vector]:
    r"""A function to map states from Bargamann representations to s-parametrized ``phase-space`` representations.

    This function supports only from Bargmann to phase space representation for states.

    Args:
        state: The orginal state object.
        modes: the modes of the state that needs to be transformed into phase space.
        s: the parametrization related to the ordering of creation and annihilation operators in the expression of any operator. :math:`s=0` is the "symmetric" ordering, which is symmetric under the exchange of creation and annihilation operators, :math:`s=-1` is the "normal" ordering, where all the creation operators are on the left and all the annihilation operators are on the right, and :math:`s=1` is the "anti-normal" ordering, which is the vice versa of the normal ordering. By using s-parametrized displacement map to generate the s-parametrized characteristic function :math:`\chi_s = Tr[\rho D_s]`, and then by doing the complex fourier transform, we get the s-parametrized quasi-probaility distribution: :math:`s=0` is the Wigner distribution, :math:`s=-1` is the Husimi Q distribution, and :math:`s=1` is the Glauber P distribution.

    Returns:
        The covariance matrix and means vector in phase space.

    .. code-block::

        >>> from mrmustard.physics.converters import to_phase_space
        >>> from mrmustard.physics.representations import Bargmann
        >>> from mrmustard.physics.triples import displacement_gate_Abc

        >>> bargmann = Bargmann(*displacement_gate_Abc(x=0.1, y=[0.2, 0.3]))
        >>> cov, means = to_phase_space(bargmann)
        >>> assert 1.0 #TODO

    """
    if not isinstance(rep, Bargmann):
        raise ValueError("This converter only works for Bargamnn representation.")

    D_s_map_A, D_s_map_b, D_s_map_c = displacement_map_s_parametrized_Abc(s)
    A,b,c = rep.ansatz.A, rep.ansatz.b, rep.ansatz.c
    new_A, new_b, new_c = complex_gaussian_integral(
        join_Abc((D_s_map_A, D_s_map_b, D_s_map_c), (A, b, c)),
        idx_z=[3, 4],
        idx_zconj=index_pair,
    )

