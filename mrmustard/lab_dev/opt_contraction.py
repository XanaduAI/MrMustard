from __future__ import annotations
import heapq
import numpy as np
from functools import lru_cache
from typing import Optional
from random import randint


class Node:
    def __init__(self, elements: tuple[Element,...], path: tuple[tuple[int, int],...] = tuple()):
        self.path = path
        self.elements: dict[int,Element] = {e.id : e for e in elements}
        self.costs = {id : 0 for id in self.elements}

    @property
    def cost(self):
        return sum(self.costs.values())

    @property
    def ids(self):
        return tuple(self.elements.keys())

    def __repr__(self):
        return f'Node({tuple(self.elements.values())}, {self.path}) cost: {self.cost}'

    def __hash__(self):
        return hash(tuple(self.elements.values()))

    def __lt__(self, other):
        return self.cost < other.cost

class Element:
    def __init__(self, id_to_dim: dict[int,int]):
        self.id = randint(0, 2**64-1)
        self.id_to_dim = id_to_dim
        self.ids = set(id_to_dim.keys())

    @property
    def shape(self):
        return tuple(self.id_to_dim.values())

    @property
    def kind(self):
        return 'f' if all(s is not None for s in self.shape) else 'b'

    def __repr__(self):
        return f'{"B" if self.kind == "b" else "F"}({",".join("0" if s is None else str(s) for s in self.shape)})'

    def __hash__(self):
        return hash((tuple(self.ids), self.shape))
    
    def __eq__(self, other):
        return self.kind == other.kind and self.id_to_dim == other.id_to_dim

DEFAULT_DIM = 50

@lru_cache(maxsize=None)
def joinBF(el1: Element, el2: Element) -> tuple[Element, int]:
    common = el1.ids & el2.ids
    only1 = el1.ids - el2.ids
    id_to_dim = {}
    for id in common:
        id_to_dim[id] = el2.id_to_dim[id]
    for id in only1:
        id_to_dim[id] = DEFAULT_DIM if el1.id_to_dim[id] is None else el1.id_to_dim[id]
    el1_f = Element({id:id_to_dim[id] for id in el1.ids})
    cost_to_f = np.prod(list(id_to_dim.values()))
    joined, cost = joinFF(el1_f, el2)
    return joined, cost + cost_to_f
    
@lru_cache(maxsize=None)
def joinFF(el1: Element, el2: Element) -> tuple[Element, int]:
    common = el1.ids & el2.ids
    rest = el1.ids ^ el2.ids
    cost = 1
    for id in common:
        cost *= min(el1.id_to_dim[id], el2.id_to_dim[id])
    for id in rest:
        if id in el1.id_to_dim:
            cost *= el1.id_to_dim[id]
        else:
            cost *= el2.id_to_dim[id]
    return Element({id:DEFAULT_DIM for id in rest}), cost

@lru_cache(maxsize=None)
def joinBB(el1: Element, el2: Element) -> tuple[Element, int]:
    common = el1.ids & el2.ids
    rest = el1.ids ^ el2.ids
    cost = len(common)**3  # what constant?
    return Element({id:None for id in rest}), cost

def join(el1: Element, el2: Element) -> tuple[Element, int]:
    if el1.kind == 'f' and el2.kind == 'f':
        return joinFF(el1, el2) 
    elif el1.kind == 'f' and el2.kind == 'b':
        return joinBF(el2, el1)
    elif el1.kind == 'b' and el2.kind == 'f':
        return joinBF(el1, el2)
    elif el1.kind == 'b' and el2.kind == 'b':
        return joinBB(el1, el2)


def branch_and_bound(elements: tuple[Element,...]) -> Optional[Node]:
    lowest_cost = np.inf
    heap = [Node(elements)]
    while heap:
        print(f'heap size: {len(heap)}', end='\r')
        node = heapq.heappop(heap)
        print(node)
        if node.cost > lowest_cost:
            continue
        if len(node.elements) <= 1:
            lowest_cost = node.cost
            return node
        for i,id1 in enumerate(node.ids[:-1]):
            for j,id2 in enumerate(node.ids[i+1:]):
                el1 = node.elements[id1]
                el2 = node.elements[id2]
                new_el, cost = join(el1, el2)
                new_elements = [el for id,el in node.elements.items() if id not in (id1, id2)]
                new_elements.append(new_el)
                new_node = Node(new_elements, node.path + ((i,j+i+1),))
                new_node.costs[new_el.id] = cost
                heapq.heappush(heap, new_node)
                print('push --->', new_node)
    return None

def main():
    elements = [Element({1: 5, 2: 10}), Element({2: 10, 3: 20})]
    print(branch_and_bound(elements))

def staircase_GBS_2():
    elements = [Element({1: None}), Element({2: None}), Element({1: None, 2: None, 3: None, 4: None}), Element({4:13})]
    print(branch_and_bound(elements))

def staircase_GBS_3():
    elements = [Element({1: None}), Element({2: None}), Element({3:None}), # 3 squeezers
                Element({1: None, 2: None, 4: None, 5: None}), Element({5: None, 3: None, 6: None, 7: None}), # 2 beamspitters
                Element({6:8}), Element({7:8})]  # 2 detectors
    print(branch_and_bound(elements))

def staircase_GBS_4():
    elements = [Element({0: None}), Element({1: None}), Element({2:None}), Element({3:None}), # 4 squeezers
                Element({0:None, 4:None, 1:None, 5:None}),  Element({5: None, 6:None, 2:None, 7:None}),  Element({7:None, 8:None, 3:None, 9:None}), # 3 beamspitters
                Element({6:10}), Element({8:5}), Element({9:7})] # 3 detectors
    print(branch_and_bound(elements))

if __name__ == '__main__':
    print("testing")
    main()
    
    print('now staircase 2')
    staircase_GBS_2()

    print('now staircase 3')
    staircase_GBS_3()

    print('now staircase 4')
    staircase_GBS_4()
