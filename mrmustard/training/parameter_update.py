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
from mrmustard.utils.typing import Tensor

from mrmustard import math
from .parameter import Trainable


def update_symplectic(
    grads_and_vars: Sequence[Tuple[Tensor, Trainable]], symplectic_lr: float
):
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


def update_orthogonal(
    grads_and_vars: Sequence[Tuple[Tensor, Trainable]], orthogonal_lr: float
):
    r"""Updates the orthogonal parameters using the given orthogonal gradients.
    Implemented from:
        Y Yao, F Miatto, N Quesada - arXiv preprint arXiv:2209.06069, 2022.
    """
    for dO_euclidean, O in grads_and_vars:
        Y = math.euclidean_to_unitary(O, math.real(dO_euclidean))
        new_value = math.matmul(O, math.expm(-orthogonal_lr * Y))
        math.assign(O, new_value)


def update_unitary(
    grads_and_vars: Sequence[Tuple[Tensor, Trainable]], unitary_lr: float
):
    r"""Updates the unitary parameters using the given unitary gradients.
    Implemented from:
        Y Yao, F Miatto, N Quesada - arXiv preprint arXiv:2209.06069, 2022.
    """
    for dU_euclidean, U in grads_and_vars:
        Y = math.euclidean_to_unitary(U, dU_euclidean)
        new_value = math.matmul(U, math.expm(-unitary_lr * Y))
        math.assign(U, new_value)


def update_euclidean(
    grads_and_vars: Sequence[Tuple[Tensor, Trainable]], euclidean_lr: float
):
    """Updates the parameters using the euclidian gradients."""
    math.euclidean_opt.lr = euclidean_lr
    math.euclidean_opt.apply_gradients(grads_and_vars)


# This dictionary relates each Trainable type to an updater function
param_update_method = {
    "euclidean": update_euclidean,
    "symplectic": update_symplectic,
    "unitary": update_unitary,
    "orthogonal": update_orthogonal,
}
