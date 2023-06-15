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

from mrmustard.lab.representations import Representation
from mrmustard.typing import Batch, Matrix, Scalar, Vector

class BargmannKet(Representation):

    def __init__(self, A:Batch[Matrix], b:Batch[Vector], c:Batch[Scalar]) -> None:
        super().__init__(A=A, b=b, c=c)
        self.num_modes = self.A.shape[-1] # TODO: BATCH?