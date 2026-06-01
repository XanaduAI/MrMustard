.. Documenting type aliases is tricky
.. We use the solution from https://github.com/sphinx-doc/sphinx/issues/10785
.. A class and data directive are used to define the type aliases.
.. Then we use html to hide the class directive.
.. Note: Sphinx 9.0+ adds support for type aliases
.. see https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#directive-autotype

.. currentmodule:: mrmustard.utils.typing

Typing
======

.. raw:: html

    <style>
        dl.class {
            display: none;
        }
    </style>

.. class:: BoolScalar

.. data:: BoolScalar
    :type: BoolScalarValue | BoolArray
    :noindex:

.. class:: BoolScalarValue

.. data:: BoolScalarValue
    :type: bool | np.bool
    :noindex:

.. class:: ComplexMatrix

.. data:: ComplexMatrix
    :type: ComplexArray
    :noindex:

.. class:: ComplexScalar

.. data:: ComplexScalar
    :type: ComplexScalarValue | ComplexArray
    :noindex:

.. class:: ComplexScalarValue

.. data:: ComplexScalarValue
    :type: complex | np.complexfloating
    :noindex:

.. class:: ComplexTensor

.. data:: ComplexTensor
    :type: ComplexArray
    :noindex:

.. class:: ComplexVector

.. data:: ComplexVector
    :type: ComplexArray
    :noindex:

.. class:: IntMatrix

.. data:: IntMatrix
    :type: IntArray
    :noindex:

.. class:: IntScalar

.. data:: IntScalar
    :type: IntScalarValue | IntArray
    :noindex:

.. class:: IntScalarValue

.. data:: IntScalarValue
    :type: int | np.signedinteger
    :noindex:

.. class:: IntTensor

.. data:: IntTensor
    :type: IntArray
    :noindex:

.. class:: IntVector

.. data:: IntVector
    :type: IntArray
    :noindex:

.. class:: Matrix

.. data:: Matrix
    :type: Array
    :noindex:

.. class:: RealMatrix

.. data:: RealMatrix
    :type: RealArray
    :noindex:

.. class:: RealScalar

.. data:: RealScalar
    :type: RealScalarValue | RealArray
    :noindex:

.. class:: RealScalarValue

.. data:: RealScalarValue
    :type: float | np.floating
    :noindex:

.. class:: RealTensor

.. data:: RealTensor
    :type: RealArray
    :noindex:

.. class:: RealVector

.. data:: RealVector
    :type: RealArray
    :noindex:

.. class:: Scalar

.. data:: Scalar
    :type: ScalarValue | Array
    :noindex:

.. class:: ScalarValue

.. data:: ScalarValue
    :type: complex | float | int | np.number
    :noindex:

.. class:: Tensor

.. data:: Tensor
    :type: Array
    :noindex:

.. class:: Trainable

.. data:: Trainable
    :type: TypeVar("Trainable", bound=NDArray[np.number])
    :noindex:

.. class:: Vector

.. data:: Vector
    :type: Array
    :noindex:

