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
The ``new`` physics module contains the representative theory of the quantum objects and all the physical propety calculations about them related to ``lab_dev``.

The ``Representation`` is defined as the way to describe a quantum object in the basis. Different representations can be used to represent a quantum object: such as the Fock basis (or number basis) represenatation, Bargmann representation (holomorphic quantum computing), phase space representation (e.g. Wigner, Husimi, Glauber and their characteristic functions) and etc.

The internal engine of MM is powered by the computation of the quantum circuits between Fock and Bargmann representations. The theory about Fock and Bargmann representations is illustrated in arXiv:2209.06069v4.
The most important computation between representations is the inner product, which can be realized by gaussian integrals.

Under the level of the representation, the data structure has been defined to store the information to describe the quantum object, which is called ``Ansatz``. 
``Ansatz`` is not only the data, but also includes the basic mathematical operations of the data. Each ``Representation`` has an attribue ``Ansatz``.

For a quantum object, it is also powerful to be able to convert between different representations. We support the conversion from Bargmann to Fock right now for all quantum objects. Especially, we also support the transformations from/to s-parametrized phase space representation and from/to quadrature representation for quantum states.

Check out our guides to learn more about :mod:`~mrmustard.physics` and its core functionalities:

* The :mod:`~mrmustard.physics.ansatze` guide introduces the concept of an ansatz and two pre-defined ansatze.
* The :mod:`~mrmustard.physics.representations` guide how to initialize representations and there are two representations: :class:`~mrmustard.physics.representations.Fock` and :class:`~mrmustard.physics.representations.Bargmann`.
* The :mod:`~mrmustard.physics.converters` contains the conversion functions from any representation to :class:`~mrmustard.physics.representations.Fock` representation. The convert functions from wigner representation to :class:`~mrmustard.physics.representations.Bargmann` representation are stored in :mod:`~mrmustard.physics.bargmann`.
* The :mod:`~mrmustard.physics.triples` contains the data (triple: a matrix, a vector and a scalar) related to the quantum objects in :class:`~mrmustard.physics.representations.Bargmann` representation.
* The :mod:`~mrmustard.physics.gaussian_integrals` contains the real and complex Gaussian integrals functions to support the inner product.
* Other modules are used in the ``lab``, which needs to be rearranged and well documented in the future.
* The functions in the init file of physics needs to be rearranged as well.
"""

from .ansatze import *
from .representations import *
