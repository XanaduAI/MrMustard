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

mock_scalar = (int, float, complex)
class MockData():
    r""" Mock class for Data objects and any child of Data that is still abstract. """

    def __init__(self, mat=None, vec=None, coeffs=None, array=np.ones(10), cutoffs=None, qs=None) -> None:
        self.mat = mat
        self.vec = vec
        self.coeffs = coeffs
        self.array = array
        self.cutoffs = cutoffs
        self.qs = qs

    def raise_error_if_different_type_and_not_scalar(self, other):
        if (not isinstance(other, self.__class__)) and (not isinstance(other, mock_scalar)):
            raise TypeError()
        
    def raise_error_if_not_scalar(self, other):
        if (not isinstance(other, mock_scalar)):
            raise TypeError()

    def __neg__(self):
        return self.__class__(array= -self.array)

    def __eq__(self, other):
        self.raise_error_if_different_type_and_not_scalar(other)
        return self

    def __add__(self, other):
        self.raise_error_if_different_type_and_not_scalar(other)
        return self.__class__(array = self.array + other.array)

    def __sub__(self, other):
        self.raise_error_if_different_type_and_not_scalar(other)
        return self.__class__(array = self.array - other.array)

    def __truediv__(self, x):
        self.raise_error_if_not_scalar(x)
        return self.__class__(array = self.array / x)

    def __mul__(self, other):
        self.raise_error_if_different_type_and_not_scalar(other)
        return self

    def __rmul__(self, other):
        self.raise_error_if_different_type_and_not_scalar(other)
        return self

    def __and__(self, other):
        self.raise_error_if_different_type_and_not_scalar(other)
        return self





class MockNoCommonAttributesObject():
    r""" Mock placeholder class for an object which has different attributes but same methods. """
    def __init__(self) -> None:
        self.apple = None
        self.pear = None
        self.banana = None




class MockCommonAttributesObject():
    r""" Mock class for Data objects and any child of Data that is still abstract. """

    def __init__(self) -> None:
        self.mat = None
        self.vec = None
        self.coeffs = None
        self.array = None
        self.cutoffs = None

    def __neg__(self):
        pass

    def __eq__(self, other):
        pass

    def __add__(self, other):
        pass

    def __sub__(self, other):
        pass

    def __truediv__(self, other):
        pass

    # def __mul__(self, other):
    #     pass

    # def __rmul__(self, other):
    #     pass

    def __and__(self, other):
        pass
