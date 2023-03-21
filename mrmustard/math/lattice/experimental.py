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


def digits_to_int(digits):
    return sum(digit * 10**i for i, digit in enumerate(digits))


def f(shape, k, index):
    return digits_to_int(index + (k,)) if len(index) == 2 else 99


def binomial_recurrence(shape, k):
    if len(shape) == 1:
        if shape[0] >= k:
            return [(k,)]
        else:
            return []
    result = []
    for i in range(min(shape[0], k) + 1):
        for indices in binomial_recurrence(shape[1:], k - i):
            result.append((i,) + indices)
    return result


# turn this into a generator


def binomial_recurrence_gen(shape, k):
    if len(shape) == 1:
        if shape[0] >= k:
            yield (k,)
        return
    for i in range(min(shape[0], k) + 1):
        for indices in binomial_recurrence(shape[1:], k - i):
            yield (i,) + indices


class BinomialData:
    def __new__(cls, shape, weight, index=()):
        if len(shape) == 1 and weight < shape[0]:
            return (weight, f(shape, weight, index))
        if weight > sum(shape) - len(shape):
            return NoneList(shape)
        if len(shape) > 0:
            return super().__new__(cls)

    def __init__(self, shape, weight, index=()):
        self.shape = shape
        self.index = index
        self.axis = []
        self._indices = None
        for photons in range(min(shape[0], self.weight + 1)):
            # if self.weight + photons == weight:
            #     self.axis.append(f(self.shape, weight, self.index + (photons,)))
            #     break
            self.axis.append(
                BinomialData(shape[1:], self.weight - photons, self.index + (photons,))
            )

    @property
    def weight(self):
        return sum(self.index)

    @property
    def indices(self):
        r"""indexes the current manifold with sum k, of which self.index is part"""
        yield from binomial_recurrence_gen(self.shape, self.k)

    def __getitem__(self, key):
        if type(key) == int:
            return self.axis[key]
        elif type(key) == tuple:
            if len(key) == 1:
                return self.axis[key[0]]
            return self.axis[key[0]][key[1:]]

    def __iter__(self):
        yield from self.axis

    def __repr__(self):
        return str(self.axis)


class NoneList:
    def __init__(self, shape: list[int]):
        self.shape = shape

    def __getitem__(self, key):
        if isinstance(key, int) and key >= self.shape[0]:
            raise IndexError
        elif isinstance(key, tuple):
            if any(k >= s for k, s in zip(key, self.shape)):
                raise IndexError
            return NoneList(self.shape[len(key) :])
        return NoneList(self.shape[1:])

    def __repr__(self):
        return "NL"
