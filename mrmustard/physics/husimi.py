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

"""
This module contains functions for transforming to the Husimi representation.
"""

from mrmustard import math, settings


def pq_to_aadag(X):
    r"""maps a matrix or vector from the q/p basis to the a/adagger basis"""
    N = X.shape[0] // 2
    R = math.rotmat(N)
    if X.ndim == 2:
        return math.matmul(math.matmul(R, X / settings.HBAR), math.dagger(R))
    if X.ndim == 1:
        return math.matvec(R, X / math.sqrt(settings.HBAR, dtype=X.dtype))
    raise ValueError("Input to complexify must be a matrix or vector")


def wigner_to_husimi(cov, means):
    r"Returns the husimi complex covariance matrix and means vector."
    N = cov.shape[-1] // 2
    sigma = pq_to_aadag(cov)
    beta = pq_to_aadag(means)
    Q = sigma + 0.5 * math.eye(2 * N, dtype=sigma.dtype)
    return Q, beta
