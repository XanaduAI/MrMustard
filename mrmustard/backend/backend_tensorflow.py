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

import numpy as np
import tensorflow as tf

from typing import Callable, List, Optional, Sequence, Tuple, Union

from .backend_base import BackendBase
from ..utils.typing import Tensor, Trainable


class BackendTensorflow(BackendBase):
    r"""
    A base class for backends.
    """

    def __init__(self):
        super().__init__(name="tensorflow")

    def hello(self):
        print(f"Hello from {self._name}")

    def abs(self, array: tf.Tensor) -> tf.Tensor:
        return tf.abs(array)

    def any(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.reduce_any(array)

    def arange(self, start: int, limit: int = None, delta: int = 1, dtype=tf.float64) -> tf.Tensor:
        return tf.range(start, limit, delta, dtype=dtype)

    def asnumpy(self, tensor: tf.Tensor) -> Tensor:
        return np.array(tensor)

    def assign(self, tensor: tf.Tensor, value: tf.Tensor) -> tf.Tensor:
        tensor.assign(value)
        return tensor

    def astensor(self, array: Union[np.ndarray, tf.Tensor], dtype=None) -> tf.Tensor:
        return tf.convert_to_tensor(array, dtype=dtype)

    def atleast_1d(self, array: tf.Tensor, dtype=None) -> tf.Tensor:
        return self.cast(tf.reshape(array, [-1]), dtype)

    def cast(self, array: tf.Tensor, dtype=None) -> tf.Tensor:
        if dtype is None:
            return array
        return tf.cast(array, dtype)

    def clip(self, array, a_min, a_max) -> tf.Tensor:
        return tf.clip_by_value(array, a_min, a_max)

    def concat(self, values: Sequence[tf.Tensor], axis: int) -> tf.Tensor:
        return tf.concat(values, axis)

    def conj(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.conj(array)

    def constraint_func(
        self, bounds: Tuple[Optional[float], Optional[float]]
    ) -> Optional[Callable]:
        bounds = (
            -np.inf if bounds[0] is None else bounds[0],
            np.inf if bounds[1] is None else bounds[1],
        )
        if bounds != (-np.inf, np.inf):

            def constraint(x):
                return tf.clip_by_value(x, bounds[0], bounds[1])

        else:
            constraint = None
        return constraint
