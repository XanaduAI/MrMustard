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

"""This file contains the abstract Representation class."""



from abc import ABC, abstractmethod

class Representation:
    
    @classmethod
    def from_repr(cls, representation: Representation):
        try:
            rep = representation.__class__.__qualname__.lower()
            cls.__dict__['from_'+ rep + '_gaussian'](representation)
        except (AttributeError, KeyError):
            cls.__dict__['from_'+ rep + '_ket'](representation)
        except (AttributeError, KeyError):
            cls.__dict__['from_'+ rep + '_densitymatrix'](representation)
        except (AttributeError, KeyError):
            raise ValueError("Cannot convert representation of type {} to {}".format(representation.__class__.__qualname__, self.__class__.__qualname__))

