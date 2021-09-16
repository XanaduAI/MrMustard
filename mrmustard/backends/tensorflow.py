import numpy as np
import tensorflow as tf
from mrmustard.backends import BackendInterface, Autocast
from thewalrus._hermite_multidimensional import hermite_multidimensional_numba, grad_hermite_multidimensional_numba
from mrmustard._typing import *

#  NOTE: the reason why we have a class with methods and not a namespace with functions
#  is that we want to enforce the interface, in order to ensure compatibility
#  of new backends with the rest of the library.


class Backend(BackendInterface):

    float64 = tf.float64
    float32 = tf.float32
    complex64 = tf.complex64
    complex128 = tf.complex128

    # ~~~~~~~~~
    # Basic ops
    # ~~~~~~~~~

    def abs(self, array: tf.Tensor) -> tf.Tensor:
        return tf.abs(array)

    def arange(self, start: int, limit: int = None, delta: int = 1, dtype=tf.float64) -> tf.Tensor:
        return tf.range(start, limit, delta, dtype=dtype)

    def asnumpy(self, tensor: tf.Tensor) -> Tensor:
        return tensor.numpy()

    def assign(self, array: tf.Tensor, value: tf.Tensor) -> tf.Tensor:
        array.assign(value)
        return array

    def astensor(self, array: Union[np.ndarray, tf.Tensor], dtype=None) -> tf.Tensor:
        return tf.convert_to_tensor(array, dtype=dtype)

    def atleast_1d(self, array: tf.Tensor, dtype=None) -> tf.Tensor:
        return self.cast(tf.reshape(array, [-1]), dtype)

    def cast(self, x: tf.Tensor, dtype=None) -> tf.Tensor:
        if dtype is None:
            return x
        return tf.cast(x, dtype)

    def concat(self, values: Sequence[tf.Tensor], axis: int) -> tf.Tensor:
        return tf.concat(values, axis)

    def conj(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.conj(array)

    def constraint_func(self, bounds: Tuple[Optional[float], Optional[float]]) -> Optional[Callable]:
        bounds = (-np.inf if bounds[0] is None else bounds[0], np.inf if bounds[1] is None else bounds[1])
        if not bounds == (-np.inf, np.inf):
            constraint: Optional[Callable] = lambda x: tf.clip_by_value(x, bounds[0], bounds[1])
        else:
            constraint = None
        return constraint

    @Autocast()
    def convolution(
        self,
        array: tf.Tensor,
        filters: tf.Tensor,
        strides: Optional[List[int]] = None,
        padding="VALID",
        data_format="NWC",
        dilations: Optional[List[int]] = None,
    ) -> tf.Tensor:
        return tf.nn.convolution(array, filters, strides, padding, data_format, dilations)

    def cos(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.cos(array)

    def cosh(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.cosh(array)

    def det(self, a: tf.Tensor) -> tf.Tensor:
        return tf.linalg.det(a)

    def diag(self, array: tf.Tensor, k: int = 0) -> tf.Tensor:
        return tf.linalg.diag(array, k=k)

    def diag_part(self, array: tf.Tensor) -> tf.Tensor:
        return tf.linalg.diag_part(array)

    def einsum(self, string: str, *tensors) -> tf.Tensor:
        return tf.einsum(string, *tensors)

    def exp(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.exp(array)

    def expm(self, matrix: tf.Tensor) -> tf.Tensor:
        return tf.linalg.expm(matrix)

    def eye(self, size: int, dtype=tf.float64) -> tf.Tensor:
        return tf.eye(size, dtype=dtype)

    def gather(self, array: tf.Tensor, indices: tf.Tensor, axis: int = None) -> tf.Tensor:
        return tf.gather(array, indices, axis=axis)

    @tf.custom_gradient
    def getitem(tensor, *, key):
        result = np.array(tensor)[key]

        def grad(dy):
            dL_dtensor = np.zeros_like(tensor)
            dL_dtensor[key] = dy
            return dL_dtensor

        return result, grad

    def hash_tensor(self, tensor: tf.Tensor) -> int:
        try:
            REF = tensor.ref()
        except AttributeError:
            raise TypeError(f"Cannot hash tensor")
        return hash(REF)

    @tf.custom_gradient
    def hermite_renormalized(self, A: tf.Tensor, B: tf.Tensor, C: tf.Tensor, shape: Tuple[int]) -> tf.Tensor:  # TODO this is not ready
        r"""
        Renormalized multidimensional Hermite polynomial given by the "exponential" Taylor series
        of exp(C + Bx - Ax^2) at zero, where the series has `sqrt(n!)` at the denominator rather than `n!`.
        Note the minus sign in front of A.

        Args:
            A: The A matrix.
            B: The B vector.
            C: The C scalar.
            shape: The shape of the final tensor.
        Returns:
            The renormalized Hermite polynomial of given shape.
        """
        poly = tf.numpy_function(hermite_multidimensional_numba, [A, shape, B, C], A.dtype)

        def grad(dLdpoly):
            print(C)
            dpoly_dC, dpoly_dA, dpoly_dB = tf.numpy_function(grad_hermite_multidimensional_numba, [poly, A, B, C], [poly.dtype] * 3)
            ax = tuple(range(dLdpoly.ndim))
            dLdA = self.sum(dLdpoly[..., None, None] * self.conj(dpoly_dA), axes=ax)
            dLdB = self.sum(dLdpoly[..., None] * self.conj(dpoly_dB), axes=ax)
            dLdC = self.sum(dLdpoly * self.conj(dpoly_dC), axes=ax)
            return dLdA, dLdB, dLdC

        return poly, grad

    def imag(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.imag(array)

    def inv(self, a: tf.Tensor) -> tf.Tensor:
        return tf.linalg.inv(a)

    def lgamma(self, x: tf.Tensor) -> tf.Tensor:
        return tf.math.lgamma(x)

    def log(self, x: tf.Tensor) -> tf.Tensor:
        return tf.math.log(x)

    @Autocast()
    def matmul(self, a: tf.Tensor, b: tf.Tensor, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False) -> tf.Tensor:
        return tf.linalg.matmul(a, b, transpose_a, transpose_b, adjoint_a, adjoint_b)

    @Autocast()
    def matvec(self, a: tf.Tensor, b: tf.Tensor, transpose_a=False, adjoint_a=False) -> tf.Tensor:
        return tf.linalg.matvec(a, b, transpose_a, adjoint_a)

    @Autocast()
    def maximum(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        return tf.maximum(a, b)

    @Autocast()
    def minimum(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        return tf.minimum(a, b)

    def new_variable(self, value, bounds: Tuple[Optional[float], Optional[float]], name: str, dtype=tf.float64):
        if value is None:
            value = np.random.normal(0, 1)
        return tf.Variable(value, name=name, dtype=dtype, constraint=self.constraint_func(bounds))

    def new_constant(self, value, name: str, dtype=tf.float64):
        if value is None:
            value = np.random.normal(0, 1)
        return tf.constant(value, dtype=dtype, name=name)

    def norm(self, array: tf.Tensor) -> tf.Tensor:
        "Note that the norm preserves the type of array"
        return tf.linalg.norm(array)

    def ones(self, shape: Sequence[int], dtype=tf.float64) -> tf.Tensor:
        return tf.ones(shape, dtype=dtype)

    def ones_like(self, array: tf.Tensor) -> tf.Tensor:
        return tf.ones_like(array)

    @Autocast()
    def outer(self, array1: tf.Tensor, array2: tf.Tensor) -> tf.Tensor:
        return tf.tensordot(array1, array2, [[], []])

    def pad(self, array: tf.Tensor, paddings: Sequence[Tuple[int, int]], mode="CONSTANT", constant_values=0) -> tf.Tensor:
        return tf.pad(array, paddings, mode, constant_values)

    def pinv(self, array: tf.Tensor) -> tf.Tensor:
        return tf.linalg.pinv(array)

    def real(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.real(array)

    def reshape(self, array: tf.Tensor, shape: Sequence[int]) -> tf.Tensor:
        return tf.reshape(array, shape)

    @tf.custom_gradient
    def setitem(tensor, value, *, key):
        "A differentiable pure equivalent of numpy's tensor[key] = value."
        tensor = np.array(tensor)
        value = np.array(value)
        tensor[key] = value

        def grad(dy):
            dL_dtensor = np.array(dy)
            dL_dtensor[key] = 0.0
            # unbroadcasting the gradient
            implicit_broadcast = list(range(tensor.ndim - value.ndim))
            explicit_broadcast = [tensor.ndim - value.ndim + j for j in range(value.ndim) if value.shape[j] == 1]
            dL_dvalue = np.sum(np.array(dy)[key], axis=tuple(implicit_broadcast + explicit_broadcast))
            dL_dvalue = np.expand_dims(dL_dvalue, [i - len(implicit_broadcast) for i in explicit_broadcast])
            print(dL_dtensor, dL_dvalue)
            return dL_dtensor, dL_dvalue

        return tensor, grad

    def sin(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.sin(array)

    def sinh(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.sinh(array)

    def sqrt(self, x: tf.Tensor, dtype=None) -> tf.Tensor:
        return self.cast(tf.sqrt(x), dtype)

    def sum(self, array: tf.Tensor, axes: Sequence[int] = None):
        return tf.reduce_sum(array, axes)

    @Autocast()
    def tensordot(self, a: tf.Tensor, b: tf.Tensor, axes: List[int]) -> tf.Tensor:
        return tf.tensordot(a, b, axes)

    def tile(self, array: tf.Tensor, repeats: Sequence[int]) -> tf.Tensor:
        return tf.tile(array, repeats)

    def trace(self, array: tf.Tensor, dtype=None) -> tf.Tensor:
        return self.cast(tf.linalg.trace(array), dtype)

    def transpose(self, a: tf.Tensor, perm: Sequence[int] = None) -> tf.Tensor:
        if a is None:
            return None  # TODO: remove and address None inputs where tranpose is used
        return tf.transpose(a, perm)

    @Autocast()
    def update_tensor(self, tensor: tf.Tensor, indices: tf.Tensor, values: tf.Tensor):
        return tf.tensor_scatter_nd_update(tensor, indices, values)

    @Autocast()
    def update_add_tensor(self, tensor: tf.Tensor, indices: tf.Tensor, values: tf.Tensor):
        return tf.tensor_scatter_nd_add(tensor, indices, values)

    def zeros(self, shape: Sequence[int], dtype=tf.float64) -> tf.Tensor:
        return tf.zeros(shape, dtype=dtype)

    def zeros_like(self, array: tf.Tensor) -> tf.Tensor:
        return tf.zeros_like(array)

    # TODO: reassess

    def DefaultEuclideanOptimizer(self) -> tf.keras.optimizers.Optimizer:
        r"""
        Default optimizer for the Euclidean parameters.
        """
        return tf.keras.optimizers.Adam(learning_rate=0.001)

    def loss_and_gradients(self, cost_fn: Callable, parameters: Dict[str, List[Trainable]]) -> Tuple[tf.Tensor, Dict[str, List[tf.Tensor]]]:
        r"""
        Computes the loss and gradients of the given cost function.

        Arguments:
            cost_fn (Callable with no args): The cost function.
            parameters (Dict): The parameters to optimize in three kinds:
                symplectic, orthogonal and euclidean.

        Returns:
            The loss and the gradients.
        """
        with tf.GradientTape() as tape:
            loss = cost_fn()
        gradients = tape.gradient(loss, list(parameters.values()))
        return loss, {p: g for p, g in zip(parameters.keys(), gradients)}

    def eigvals(self, tensor: tf.Tensor) -> Tensor:
        "Returns the eigenvalues of a matrix."
        return tf.linalg.eigvals(tensor)

    def eigvalsh(self, tensor: tf.Tensor) -> Tensor:
        "Returns the eigenvalues of a Real Symmetric or Hermitian matrix."
        return tf.linalg.eigvalsh(tensor)

    def svd(self, tensor: tf.Tensor) -> Tensor:
        "Returns the Singular Value Decomposition of a matrix."
        return tf.linalg.svd(tensor)

    def xlogy(self, x: tf.Tensor, y: tf.Tensor) -> Tensor:
        "Returns 0 if x == 0, and x * log(y) otherwise, elementwise."
        return tf.math.xlogy(x, y)

    def eigh(self, tensor: tf.Tensor) -> Tensor:
        "Returns the eigenvalues and eigenvectors of a matrix."
        return tf.linalg.eigh(tensor)

    def sqrtm(self, tensor: tf.Tensor, rtol=1e-05, atol=1e-08) -> Tensor:
        "Returns the matrix square root of a square matrix, such that sqrt(A) @ sqrt(A) = A."

        # The sqrtm function has issues with matrices that are close to zero, hence we branch
        if np.allclose(tensor, 0, rtol=rtol, atol=atol):
            return self.zeros_like(tensor)
        else:
            return tf.linalg.sqrtm(tensor)
