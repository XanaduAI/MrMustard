import numpy as np

cimport cython
cimport numpy as cnp
cnp.import_array()


# ------------------------
# Cached SQRT and INV_SQRT
# ------------------------


np_sqrt = np.sqrt(np.arange(100_000))
np_inv_sqrt = np.zeros_like(np_sqrt)
np_inv_sqrt[1:] = 1 / np_sqrt[1:]

SQRT = cython.declare(cython.double[100_000], np_sqrt)
INV_SQRT = cython.declare(cython.double[100_000], np_inv_sqrt)


# -----------------
# Utility Functions
# -----------------


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void compute_strides(const int* shape, int* strides, int ndim) noexcept nogil:
    """Compute strides for multi-dimensional array indexing"""
    strides[ndim - 1] = 1
    cdef Py_ssize_t i
    for i in range(ndim - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void ndindex_init(int* indices, int ndim) noexcept nogil:
    """Initialize ndindex iterator to all zeros"""
    cdef int i
    for i in range(ndim):
        indices[i] = 0


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void ndindex_next(int* indices, const int* shape, int ndim, int* is_done) noexcept nogil:
    """
    Ultra-fast C-level ndindex iterator. Increments indices in-place.
    Sets is_done[0] = 1 when all indices are exhausted.
    """
    cdef int i
    for i in range(ndim - 1, -1, -1):
        indices[i] += 1
        if indices[i] < shape[i]:
            return
        indices[i] = 0
    is_done[0] = 1
