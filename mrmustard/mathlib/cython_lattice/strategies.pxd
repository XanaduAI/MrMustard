import numpy as np
cimport cython

cdef extern from "Strategies.hpp" namespace "strategies":
    cdef void compute_beamsplitter(
        int N,
        int M,
        int P,
        int Q,
        double theta,
        double phi,
        double complex *G,
    ) noexcept nogil

    cdef void compute_beamsplitter_stable(
        int N,
        int M,
        int P,
        int Q,
        double theta,
        double phi,
        double complex *G,
    ) noexcept nogil

    cdef void compute_displacement(
        int N,
        int M,
        double complex alpha,
        bint flipped,
        double complex *D,
    ) noexcept nogil

    cdef void compute_homodyne_projector(
        int M,
        int N,
        int P,
        const double complex *A,
        const double complex *b,
        const double complex c,
        double complex *G,
    ) noexcept nogil

    cdef void compute_jacobian_displacement(
        int N,
        int M,
        double complex alpha,
        const double complex *D,
        double complex *jac_alpha,
        double complex *jac_alphac,
    ) noexcept nogil

    cdef void compute_squeezed(
        int N,
        double r,
        double theta,
        double complex *S,
    ) noexcept nogil

    cdef void compute_squeezer(
        int M,
        int N,
        double r,
        double theta,
        double complex *S,
    ) noexcept nogil
