from typing import TypeVar, Union, List, Dict, Optional, Tuple, Union, Sequence, Generator, Callable

# NOTE: when type-annotating with typevars, objects with the same typevars must have the same type
# E.g. in `def f(x: Vector, y: Vector) -> Tensor: ...`
# the type of `x` and the type of `y` are assumed to be the same, even though "Vector" can mean different things.
Scalar = TypeVar("Scalar")
Vector = TypeVar("Vector")
Matrix = TypeVar("Matrix")
Tensor = TypeVar("Tensor")
Array = TypeVar("Array")  # TODO: let mypy know that this is Vector, Matrix, or Tensor
Trainable = TypeVar("Trainable")

Numeric = Union[Scalar, Vector, Matrix, Tensor]

