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

""" This package contains the modules implementing base classes for representations, datas and converter.
"""

from .representation import Representation
from .wigner import Wigner
from .wigner_ket import WignerKet
from .wigner_dm import WignerDM
from .fock import Fock
from .fock_ket import FockKet
from .fock_dm import FockDM
from .bargmann import Bargmann
from .bargmann_ket import BargmannKet
from .bargmann_dm import BargmannDM
from .wavefunctionq import WaveFunctionQ
from .wavefunctionq_ket import WaveFunctionQKet
from .wavefunctionq_dm import WaveFunctionQDM

from .data.data import Data
from .data.matvec_data import MatVecData
from .data.gaussian_data import GaussianData
from .data.qpoly_data import QPolyData
from .data.array_data import ArrayData
from .data.symplectic_data import SymplecticData

from .converter import Converter