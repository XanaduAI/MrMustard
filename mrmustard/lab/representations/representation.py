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

from abc import ABC, abstractmethod, abstractproperty

class Representation(ABC):

    def __init__(self):
        super().__init__(self)

    
    @abstractmethod
    def number_means(self):
        pass


    @abstractmethod
    def number_cov(self):
        pass


    @abstractproperty
    def purity(self):
        pass


    @abstractmethod
    def number_stdev(self):
        pass


<<<<<<< HEAD
    @abstractmethod
    @property
    def norm(self):
=======
    @abstractproperty
    def norm():
>>>>>>> a5ddef4 (abstractproperty)
        pass


    @abstractmethod
    def probability(self):
        pass


    @abstractproperty
    def von_neumann_entropy(self):
        pass


    @abstractmethod
    def _repr_markdown_(self):
        pass

     
