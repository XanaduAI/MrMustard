import numpy as np
cimport cython
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free

cimport numpy as cnp
cnp.import_array()

from .utils cimport INV_SQRT, SQRT, compute_strides, ndindex_init, ndindex_next


# --------------
# Vanilla Python
# --------------


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def vanilla(
    shape: tuple[int, ...],
    const double complex[:, :] A,
    const double complex[:] b,
    const double complex c,
    stable: bool = False,
    out: np.ndarray | None = None,
) -> np.ndarray:
    r"""
    Vanilla algorithm for calculating the fock representation of a Gaussian tensor.
    This implementation works on flattened tensors and reshapes the tensor before returning.


    The vanilla algorithm implements the flattened version of the following recursion which
    calculates the Fock amplitude at index :math:`k` using a pivot at index :math:`k - 1_i`
    and its neighbours at indices :math:`k - 1_i - 1_j`:

    .. math::

        G_{k} = \frac{1}{\sqrt{k_i}} \left[
                b_{i} G_{k-1_i} + \sum_j A_{ij} \sqrt{k_j - \delta_{ij}} G_{k-1_i-1_j} \right
            ]

    where :math:`1_i` is the vector of zeros with a 1 at index :math:`i`,
    and :math:`\delta_{ij}` is the Kronecker delta.
    In this formula :math:`k` is the vector of indices indexing into the Fock lattice.
    In the implementation the indices are flattened into a single integer index.
    This simplifies the bounds check when calculating the index of the pivot :math:`k-1_i`,
    which need to be done only until the index is smaller than the maximum stride.

    see https://quantum-journal.org/papers/q-2020-11-30-366/ and
    https://arxiv.org/abs/2209.06069 for more details.

    Args:
        shape (tuple[int, ...]): shape of the output tensor
        A (np.ndarray): A matrix of the Bargmann representation
        b (np.ndarray): b vector of the Bargmann representation
        c (complex): vacuum amplitude
        stable (bool): whether to run the stable implementation of `vanilla`.
        out (np.ndarray): if provided, the result will be stored in this tensor.

    Returns:
        np.ndarray: Fock representation of the Gaussian tensor with shape ``shape``
    """
    # convert tuple of ints into a fast C array
    cdef Py_ssize_t shape_len = len(shape)
    cdef int *shape_arr = <int*>malloc(shape_len * sizeof(int))
    cdef Py_ssize_t i
    for i in range(shape_len):
        shape_arr[i] = shape[i]

    cdef int *strides = <int*>malloc(shape_len * sizeof(int))
    compute_strides(shape_arr, strides, shape_len)

    G = np.empty(shape, dtype=np.cdouble) if out is None else out

    # Convert to memory views for nogil access
    cdef const double complex[:, :] A_view = A
    cdef const double complex[:] b_view = b
    cdef double complex[:] G_flat_view = G.reshape(-1)

    if stable:
        vanilla_stable_c(shape_arr, strides, A_view, b_view, c, G_flat_view)
    else:
        vanilla_c(shape_arr, strides, A_view, b_view, c, G_flat_view)

    free(shape_arr)
    free(strides)
    return G


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def vanilla_batched(
    shape: tuple[int, ...],
    const double complex[:, :, :] A,
    const double complex[:, :] b,
    const double complex[:] c,
    stable: bool = False,
    out: np.ndarray | None = None,
) -> np.ndarray:
    r"""
    The batched implementation of `vanilla`.
    Returns the Fock representation of the Gaussian tensor by parallelizing
    the single-instance ``vanilla`` over the batch dimension.

        Args:
        shape (tuple[int, ...]): shape of the output tensor
        A (np.ndarray): A matrix of the Bargmann representation
        b (np.ndarray): b vector of the Bargmann representation
        c (np.ndarray): vacuum amplitude
        out (np.ndarray): if provided, the result will be stored in this tensor.

    Returns:
        np.ndarray: Fock representation of the Gaussian tensor with shape ``shape``
    """
    # convert tuple of ints into a fast C array
    cdef Py_ssize_t shape_len = len(shape)
    cdef int *shape_arr = <int*>malloc(shape_len * sizeof(int))
    cdef Py_ssize_t i
    for i in range(shape_len):
        shape_arr[i] = shape[i]

    cdef int *strides = <int*>malloc(shape_len * sizeof(int))
    compute_strides(shape_arr, strides, shape_len)

    cdef Py_ssize_t batch_size = b.shape[0]
    G = np.empty((batch_size,) + shape, dtype=np.cdouble) if out is None else out

    # Convert to memory views for nogil access
    cdef const double complex[:, :, :] A_view = A
    cdef const double complex[:, :] b_view = b
    cdef const double complex[:] c_view = c
    cdef double complex[:, :] G_flat_view = G.reshape(batch_size, -1)

    cdef Py_ssize_t k
    if stable:
        for k in prange(batch_size, nogil=True):
            vanilla_stable_c(shape_arr, strides, A_view[k], b_view[k], c_view[k], G_flat_view[k])
    else:
        for k in prange(batch_size, nogil=True):
            vanilla_c(shape_arr, strides, A_view[k], b_view[k], c_view[k], G_flat_view[k])
    free(shape_arr)
    free(strides)
    return G


# ---------
# Vanilla C
# ---------


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void vanilla_stable_c(
    const int* shape,
    const int* strides,
    const double complex[:, :] A,
    const double complex[:] b,
    const double complex c,
    double complex[:] G,
) noexcept nogil:
    # set the first element and iterate ndindex
    cdef Py_ssize_t flat_index = 0
    G[flat_index] = c

    cdef int D = b.shape[0]
    cdef int *nd_idx = <int*>malloc(D * sizeof(int))
    ndindex_init(nd_idx, D)

    cdef int is_done = 0
    ndindex_next(nd_idx, &shape[0], D, &is_done)

    cdef int num_pivots
    cdef double complex val, vals
    cdef Py_ssize_t i, j, pivot, temp_idx
    cdef double inv_num_pivots

    while not is_done:
        flat_index += 1
        num_pivots = 0
        vals = 0.0
        for i in range(D):
            if nd_idx[i] == 0:
                continue
            num_pivots += 1
            pivot = flat_index - strides[i]
            val = b[i] * G[pivot]
            # first loop: j < i
            for j in range(i):
                temp_idx = pivot - strides[j]
                if temp_idx >= 0:
                    val += A[i, j] * SQRT[nd_idx[j]] * G[temp_idx]

            # middle calculation: j == i
            temp_idx = pivot - strides[i]
            if temp_idx >= 0:
                val += A[i, i] * SQRT[nd_idx[i] - 1] * G[temp_idx]

            # last loop: j > i
            for j in range(i + 1, D):
                temp_idx = pivot - strides[j]
                if temp_idx >= 0:
                    val += A[i, j] * SQRT[nd_idx[j]] * G[temp_idx]
            vals += val * INV_SQRT[nd_idx[i]]
        inv_num_pivots = 1.0 / num_pivots
        G[flat_index] = vals * inv_num_pivots

        # move to next index
        ndindex_next(nd_idx, &shape[0], D, &is_done)

    # clean up allocated memory
    free(nd_idx)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void vanilla_c(
    const int* shape,
    const int* strides,
    const double complex[:, :] A,
    const double complex[:] b,
    const double complex c,
    double complex[:] G,
) noexcept nogil:
    cdef int D = b.shape[0]
    if D <= 0:
        return

    cdef int *nd_idx = <int*>malloc(D * sizeof(int))
    ndindex_init(nd_idx, D)

    cdef int is_done = 0
    cdef double complex value_at_index
    cdef Py_ssize_t i, j, pivot, s, flat_index, temp_idx

    # set the first element
    G[0] = c

    # Iterate over the indices smaller than max(strides) with pivot bound check.
    # The check is needed only if the flat index is smaller than the largest stride.
    # Afterwards it will be safe to get the pivot by subtracting the first (largest) stride.
    for flat_index in range(1, strides[0]):
        ndindex_next(nd_idx, &shape[0], D, &is_done)

        i = 0
        # calculate (flat) pivot
        for s in range(D):
            pivot = flat_index - strides[s]
            if pivot >= 0:  # if pivot not outside array
                break
            i += 1

        # contribution from pivot
        value_at_index = b[i] * G[pivot]

        # contributions from pivot's lower neighbours
        # note the first is when j=i which needs a -1 in the sqrt from delta_ij
        temp_idx = pivot - strides[i]
        if temp_idx >= 0:
            value_at_index += A[i, i] * SQRT[nd_idx[i] - 1] * G[temp_idx]
        for j in range(i + 1, D):
            temp_idx = pivot - strides[j]
            if temp_idx >= 0:
                value_at_index += A[i, j] * SQRT[nd_idx[j]] * G[pivot - strides[j]]
        G[flat_index] = value_at_index * INV_SQRT[nd_idx[i]]

    # Iterate over the rest of the indices.
    # Now i can always be 0 (largest stride), and we don't need bounds check
    for flat_index in range(strides[0], len(G)):
        ndindex_next(nd_idx, &shape[0], D, &is_done)

        # pivot can be calculated without bounds check
        pivot = flat_index - strides[0]

        # contribution from pivot
        value_at_index = b[0] * G[pivot]

        # contribution from pivot's lower neighbours
        # note the first is when j=0 which needs a -1 in the sqrt from delta_0j
        temp_idx = pivot - strides[0]
        if temp_idx >= 0:
            value_at_index += A[0, 0] * SQRT[nd_idx[0] - 1] * G[pivot - strides[0]]
        for j in range(1, D):
            temp_idx = pivot - strides[j]
            if temp_idx >= 0:
                value_at_index += A[0, j] * SQRT[nd_idx[j]] * G[pivot - strides[j]]
        G[flat_index] = value_at_index * INV_SQRT[nd_idx[0]]

    # clean up allocated memory
    free(nd_idx)


# -------------------
# Vanilla VJPs Python
# -------------------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def vanilla_vjp(
    G: np.ndarray,
    c: np.ndarray,
    dLdG: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, complex]:
    r"""
    Vanilla vjp function. Returns dL/dA, dL/db, dL/dc.

    Args:
        G (np.ndarray): Tensor result of the forward pass
        c (complex): vacuum amplitude
        dLdG (np.ndarray): gradient of the loss with respect to the output tensor

    Returns:
        tuple[np.ndarray, np.ndarray, complex]: dL/dA, dL/db, dL/dc
    """
    shape = G.shape
    cdef Py_ssize_t D = len(shape)
    cdef int *shape_arr = <int*>malloc(D * sizeof(int))
    cdef Py_ssize_t i
    for i in range(D):
        shape_arr[i] = shape[i]

    cdef int *strides = <int*>malloc(D * sizeof(int))
    compute_strides(shape_arr, strides, D)

    cdef const double complex[:] G_view = G.reshape(-1)
    cdef double complex c_view = c
    cdef const double complex[:] dLdG_view = dLdG.reshape(-1)

    cdef double complex[:, :] dLdA = np.zeros((D, D), dtype=np.cdouble)
    cdef double complex[:] dLdb = np.zeros(D, dtype=np.cdouble)
    cdef double complex dLdc = 0

    vanilla_vjp_c(shape_arr, strides, G_view, c_view, dLdG_view, dLdA, dLdb, &dLdc)

    free(shape_arr)
    free(strides)
    return dLdA, dLdb, dLdc


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def vanilla_vjp_batched(
    G: np.ndarray,
    c: np.ndarray,
    dLdG: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Batched implementation of `vanilla_vjp`.
    Returns dL/dA, dL/db, dL/dc by parallelizing the single-instance
    ``vanilla_vjp`` over the batch dimension.

    Args:
        G (np.ndarray): Tensor result of the forward pass with shape `(batch_size,) + shape`.
        c (np.ndarray): Batched vacuum amplitudes with shape `(batch_size,)`.
        dLdG (np.ndarray): Gradient of the loss with respect to the output tensor `G`,
        with the same shape as `G`.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: dL/dA, dL/db, dL/dc.
            dL/dA has shape `(batch_size, D, D)`, dL/db has shape `(batch_size, D)`,
            dL/dc has shape `(batch_size,)`, where D is the number of modes
            (last dimension of G).
    """
    shape = G.shape[1:]
    cdef Py_ssize_t D = len(shape)
    cdef int *shape_arr = <int*>malloc(D * sizeof(int))
    cdef Py_ssize_t i
    for i in range(D):
        shape_arr[i] = shape[i]

    cdef int *strides = <int*>malloc(D * sizeof(int))
    compute_strides(shape_arr, strides, D)

    cdef Py_ssize_t batch_size = G.shape[0]
    cdef const double complex[:, :] G_view = G.reshape((batch_size, -1))
    cdef double complex[:] c_view = c
    cdef const double complex[:, :] dLdG_view = dLdG.reshape((batch_size, -1))

    cdef double complex[:, :, :] dLdA = np.zeros((batch_size, D, D), dtype=np.cdouble)
    cdef double complex[:, :] dLdb = np.zeros((batch_size, D), dtype=np.cdouble)
    cdef double complex[:] dLdc = np.zeros(batch_size, dtype=np.cdouble)

    cdef Py_ssize_t k
    for k in prange(batch_size, nogil=True):
        vanilla_vjp_c(
            shape_arr, strides, G_view[k], c_view[k], dLdG_view[k], dLdA[k], dLdb[k], &dLdc[k]
        )

    free(shape_arr)
    free(strides)
    return dLdA, dLdb, dLdc

# --------------
# Vanilla VJPs C
# --------------


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void vanilla_vjp_c(
    const int* shape,
    const int* strides,
    const double complex[:] G,
    double complex c,
    const double complex[:] dLdG,
    double complex[:, :] dLdA,
    double complex[:] dLdb,
    double complex* dLdc,
) noexcept nogil:
    cdef Py_ssize_t D = dLdA.shape[0]

    # Allocate C arrays for ultra-fast performance
    cdef double complex* dA = <double complex*>malloc(D * D * sizeof(double complex))
    cdef double complex* db = <double complex*>malloc(D * sizeof(double complex))
    cdef double complex* dLdA_tmp = <double complex*>malloc(D * D * sizeof(double complex))

    # Initialize arrays to zero
    cdef Py_ssize_t i
    for i in range(D * D):
        dA[i] = 0.0
        dLdA_tmp[i] = 0.0
    for i in range(D):
        db[i] = 0.0

    cdef int *nd_idx = <int*>malloc(D * sizeof(int))
    ndindex_init(nd_idx, D)

    cdef int is_done = 0
    ndindex_next(nd_idx, &shape[0], D, &is_done)

    cdef Py_ssize_t j, pivot, temp_idx
    cdef Py_ssize_t flat_index = 0
    cdef double complex val
    cdef Py_ssize_t G_len = G.shape[0]

    while not is_done:
        for i in range(D):
            pivot = (flat_index + 1) - strides[i]
            if pivot < 0:
                pivot = G_len + pivot
            db[i] = SQRT[nd_idx[i]] * G[pivot]
            if nd_idx[i] > 1:
                temp_idx = pivot - strides[i]
                if temp_idx < 0:
                    temp_idx = G_len + temp_idx
                dA[i * D + i] = 0.5 * SQRT[nd_idx[i]] * SQRT[nd_idx[i] - 1] * G[temp_idx]
            else:
                dA[i * D + i] = 0.0
            for j in range(i + 1, D):
                temp_idx = pivot - strides[j]
                if temp_idx < 0:
                    temp_idx = G_len + temp_idx
                dA[i * D + j] = SQRT[nd_idx[i]] * SQRT[nd_idx[j]] * G[temp_idx]

        flat_index += 1
        val = dLdG[flat_index]
        for i in range(D):
            for j in range(D):
                dLdA_tmp[i * D + j] += dA[i * D + j] * val
        for i in range(D):
            dLdb[i] = dLdb[i] + db[i] * val

        ndindex_next(nd_idx, &shape[0], D, &is_done)

    free(nd_idx)

    # dLdA - copy from C array to output
    for i in range(D):
        for j in range(D):
            dLdA[i, j] = (dLdA_tmp[i * D + j] + dLdA_tmp[j * D + i]) / 2

    # dLdc
    dLdc[0] = 0
    for i in range(G_len):
        dLdc[0] += G[i] * dLdG[i]
    dLdc[0] /= c

    # Free C arrays
    free(dA)
    free(db)
    free(dLdA_tmp)
