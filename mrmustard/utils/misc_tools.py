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
This module contains miscellaneous functions which are used accross the code base of MM. 
They can be used as is in any of the classes.
"""

all = ['duck_type_checker', 
       'general_factory']

def duck_type_checker(obj_a:object, obj_b:object) -> bool:
    r""" *If it walks like a duck and it quacks like a duck, then it must be a duck.*

    This function performs type-checking from a duck-type perspective, testing whether both given
    objects share the same set of attributes. If they do, no matter their *type* -as obtained by
    a call to ``isinstance``-, they will be considered to be of the same duck-type.

    Note that we here look at the symmetric difference, hence the exact question we ask is better
    phrased as "*Are both objects of the same type?*" than as "*Is object a of the same type as 
    object b?*".

    Note also that this checks for the exact match between types -via attributes-, this system is 
    **not** meant to work with subclasses since there is a high probability that they might have 
    different attributes.

    Args:
        obj_a:  one of the objects to be checked
        obj_b:  the other object to be checked

    Returns:
        True if both objects have exactly the same attributes, False otherwise.

    Raises:
        TypeError:  If at least one of the objects given can't be compared via duck-typing, i.e. 
        lack attributes. A typical example would be python basic types such as ``int``, ``float``, 
        ``bool``, etc. 
     """
    try:
        set_a = set(obj_a.__dict__.keys())
        set_b = set(obj_b.__dict__.keys())
        return set_a.symmetric_difference(set_b) == set()
    
    except AttributeError as e:
        raise TypeError(
            f"Objects of types {type(obj_a), type(obj_b)} can't be compared via duck-type"
            ) from e



def general_factory(cls, *args, **kwargs) -> object:
    r""" Factory method which generates an instance of cls parameterized by the given arguments.

    Args:
        args:   non-keyword ordered arguments used to parameterize the class instance
        kwargs: keyword arguments used to parameterize the class instance

    Returns:
        An instance of the class cls parametrized by args and kwargs
    """
    return cls(*args, **kwargs)