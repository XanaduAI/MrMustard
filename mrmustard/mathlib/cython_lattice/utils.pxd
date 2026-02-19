import numpy as np
cimport cython
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free

cimport numpy as cnp
cnp.import_array()


# ------------------------
# Cached SQRT and INV_SQRT
# ------------------------


cdef public cython.double[100_000] SQRT
cdef public cython.double[100_000] INV_SQRT


# -----------------
# Utility Functions
# -----------------

cdef void compute_strides(const int* shape, int* strides, int ndim) noexcept nogil
cdef void ndindex_init(int* indices, int ndim) noexcept nogil
cdef void ndindex_next(int* indices, const int* shape, int ndim, int* is_done) noexcept nogil
