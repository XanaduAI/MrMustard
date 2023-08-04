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

from __future__ import annotations

import numpy as np

from abc import ABC, abstractmethod
from typing import List, Union

from mrmustard.typing import Scalar, Vector


class Data(ABC):
    r"""Abstract parent class for types of data encoding a quantum state's representation."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def same( #change this method for the batches
        X: List[Union[List[Vector], List[Scalar]]],
        Y: List[Union[List[Vector], List[Scalar]]],
    ) -> bool:
        r"""Method to compare two sets of elements

        Args:
            X (List[Union[List[Vector], List[Scalar]]]):    list of elements to be compared with ys
            Y (List[Union[List[Vector], List[Scalar]]]):    list of elements to be compared with xs

        Returns:
            (bool) True if all the elements compared are same within numpy's default rtol/atol, 
            False otherwise
        """
        return all([np.allclose(x, y) for x, y in zip(X, Y)])

    @abstractmethod
    def __neg__(self) -> Data:
        raise NotImplementedError()  # prompting override in children

    @abstractmethod
    def __eq__(self, other: Data) -> bool:
        raise NotImplementedError()  # prompting override in children

    @abstractmethod
    def __add__(self, other: Data) -> Data:
        raise NotImplementedError()  # prompting override in children

    def __sub__(self, other: Data) -> Data:
        try:
            return self.__add__(-other)
        except AttributeError as e:
            raise TypeError(f"Cannot subtract {self.__class__} and {other.__class__}.") from e

    @abstractmethod
    def __truediv__(self, other: Union[Scalar, Data]) -> Data:
        raise NotImplementedError()  # prompting override in children

    @abstractmethod
    def __mul__(self, other: Union[Scalar, Data]) -> Data:
        raise NotImplementedError()  # prompting override in children

    def __rmul__(self, other: Scalar) -> Data:
        return self.__mul__(other=other)