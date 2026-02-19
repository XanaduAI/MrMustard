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

.. class:: Scalar

.. data:: Scalar
    :type: R | C | Z | N
    :noindex:

.. class:: Vector

.. data:: Vector
    :type: np.ndarray[tuple[int], Scalar]
    :noindex:

.. class:: Matrix

.. data:: Matrix
    :type: np.ndarray[tuple[int, int], Scalar]
    :noindex:

.. class:: Tensor

.. data:: Tensor
    :type: np.ndarray[tuple[int, ...], Scalar]
    :noindex:

.. class:: RealVector

.. data:: RealVector
    :type: np.ndarray[tuple[int], R]
    :noindex:

.. class:: ComplexVector

.. data:: ComplexVector
    :type: np.ndarray[tuple[int], C]
    :noindex:

.. class:: IntVector

.. data:: IntVector
    :type: np.ndarray[tuple[int], Z]
    :noindex:

.. class:: UIntVector

.. data:: UIntVector
    :type: np.ndarray[tuple[int], N]
    :noindex:

.. class:: RealMatrix

.. data:: RealMatrix
    :type: np.ndarray[tuple[int, int], R]
    :noindex:

.. class:: ComplexMatrix

.. data:: ComplexMatrix
    :type: np.ndarray[tuple[int, int], C]
    :noindex:

.. class:: IntMatrix

.. data:: IntMatrix
    :type: np.ndarray[tuple[int, int], Z]
    :noindex:

.. class:: UIntMatrix

.. data:: UIntMatrix
    :type: np.ndarray[tuple[int, int], N]
    :noindex:

.. class:: RealTensor

.. data:: RealTensor
    :type: np.ndarray[tuple[int, ...], R]
    :noindex:

.. class:: ComplexTensor

.. data:: ComplexTensor
    :type: np.ndarray[tuple[int, ...], C]
    :noindex:

.. class:: IntTensor

.. data:: IntTensor
    :type: np.ndarray[tuple[int, ...], Z]
    :noindex:

.. class:: UIntTensor

.. data:: UIntTensor
    :type: np.ndarray[tuple[int, ...], N]
    :noindex:
    