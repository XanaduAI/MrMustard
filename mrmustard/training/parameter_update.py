# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TODO: document this module
"""

from typing import Tuple, Sequence
from mrmustard.math import Math
from mrmustard.typing import Tensor

from .parameter import Trainable

math = Math()


def update_symplectic(grads_and_vars: Sequence[Tuple[Tensor, Trainable]], symplectic_lr: float):
    r"""Updates the symplectic parameters using the given symplectic gradients.
    Implemented from:
        Wang J, Sun H, Fiori S. A Riemannian-steepest-descent approach
        for optimization on the real symplectic group.
        Mathematical Methods in the Applied Sciences. 2018 Jul 30;41(11):4273-86.
    """
    for dS_euclidean, S in grads_and_vars:
        Y = math.euclidean_to_symplectic(S, dS_euclidean)
        YT = math.transpose(Y)
        new_value = math.matmul(
            S, math.expm(-symplectic_lr * YT) @ math.expm(-symplectic_lr * (Y - YT))
        )
        math.assign(S, new_value)


def update_orthogonal(grads_and_vars: Sequence[Tuple[Tensor, Trainable]], orthogonal_lr: float):
    r"""Updates the orthogonal parameters using the given orthogonal gradients.
    Implemented from:
        Fiori S, Bengio Y. Quasi-Geodesic Neural Learning Algorithms
        Over the Orthogonal Group: A Tutorial.
        Journal of Machine Learning Research. 2005 May 1;6(5).
    """
    for dO_euclidean, O in grads_and_vars:
        dO_orthogonal = 0.5 * (
            dO_euclidean - math.matmul(math.matmul(O, math.transpose(dO_euclidean)), O)
        )
        new_value = math.matmul(
            O, math.expm(orthogonal_lr * math.matmul(math.transpose(dO_orthogonal), O))
        )
        math.assign(O, new_value)


def update_euclidean(grads_and_vars: Sequence[Tuple[Tensor, Trainable]], euclidean_lr: float):
    """Updates the parameters using the euclidian gradients."""
    math.euclidean_opt.lr = euclidean_lr
    math.euclidean_opt.apply_gradients(grads_and_vars)


# This dictionary relates each Trainable type to an updater function
param_update_method = {
    "euclidean": update_euclidean,
    "symplectic": update_symplectic,
    "orthogonal": update_orthogonal,
}
