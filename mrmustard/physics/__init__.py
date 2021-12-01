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
The physics module contains high-level functions for performing calculations on quantum objects.
It allows for objects in different representation to be used seamlessly in the same calculation.
E.g. :code:`fidelity(A, B)` works whether A and B are in Fock or Gaussian representation or a mix of both.

All the functions are automatically differentiated and can be used in conjunction with an
optimiization routine.
"""
