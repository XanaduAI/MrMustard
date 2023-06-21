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

class MockData():
    r""" Mock class for Data objects and any child of Data that is still abstract. """

    def __init__(self) -> None:
        self.mat = None
        self.vec = None
        self.coeffs = None
        self.array = None
        self.cutoffs = None

    def __neg__(self):
        pass

    def __eq__(self):
        pass

    def __add__(self):
        pass

    def __sub__(self):
        pass

    def __truediv__(self):
        pass

    def __mul__(self):
        pass

    def __rmul__(self):
        pass

    def __and__(self):
        pass


class MockNoCommonAttributeObject():
    r""" Mock placeholder class for an object which has different attributes but same methods. """
    def __init__(self) -> None:
        self.apple = None
        self.pear = None
        self.banana = None

    def __neg__(self):
        pass

    def __eq__(self):
        pass

    def __add__(self):
        pass

    def __sub__(self):
        pass

    def __truediv__(self):
        pass

    def __mul__(self):
        pass

    def __rmul__(self):
        pass

    def __and__(self):
        pass