import numpy as np
cimport cython

from cython.parallel cimport prange

cimport numpy as cnp
cnp.import_array()

# -------------------
# Beamsplitter Python
# -------------------


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def beamsplitter(
    tuple shape,
    double theta,
    double phi,
    bint stable = False,
):
    r"""
    Calculates the Fock representation of the beamsplitter.
    It takes advantage of input-output particle conservation (m+n=p+q)
    to avoid one for loop. Inspired from the original implementation in
    the walrus by @ziofil. Here is how the parameters are used in the
    code (see eq. 73-75 in https://arxiv.org/abs/2004.11002):

    .. math::

        A = \begin{bmatrix} 0 & V \\ V^T & 0 \end{bmatrix} \quad
            \text{(BS Bargmann matrix)}

        V = \begin{bmatrix} ct & -st e^{-i\phi} \\ st e^{i\phi} & ct \end{bmatrix} \quad
            \text{(BS unitary)}

    Args:
        shape: The Fock shape of the resulting array.
        theta: The beamsplitter angle.
        phi: The beamsplitter phase.
        stable: Whether to use the stable algorithm.

    Returns:
        The matrix representing the beamsplitter operation.
    """
    G = np.empty(shape, dtype=np.complex128)

    cdef Py_ssize_t M, N, P, Q
    M, N, P, Q = shape

    cdef double complex[:, :, :, :] G_view = G

    if stable:
        with nogil:
            compute_beamsplitter_stable(
                M, N, P, Q,
                theta,
                phi,
                &G_view[0, 0, 0, 0],
            )
    else:
        with nogil:
            compute_beamsplitter(
                M, N, P, Q,
                theta,
                phi,
                &G_view[0, 0, 0, 0],
            )

    return G


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def beamsplitter_batched(
    tuple shape,
    const double[:] theta,
    const double[:] phi,
    bint stable = False,
):
    r"""
    Calculates the Fock representation of the batched beamsplitter.
    It takes advantage of input-output particle conservation (m+n=p+q)
    to avoid one for loop. Inspired from the original implementation in
    the walrus by @ziofil. Here is how the parameters are used in the
    code (see eq. 73-75 in https://arxiv.org/abs/2004.11002):

    .. math::

        A = \begin{bmatrix} 0 & V \\ V^T & 0 \end{bmatrix} \quad
            \text{(BS Bargmann matrix)}

        V = \begin{bmatrix} ct & -st e^{-i\phi} \\ st e^{i\phi} & ct \end{bmatrix} \quad
            \text{(BS unitary)}

    Note: batch dimensions must be flattened.

    Args:
        shape: The Fock shape of the resulting array.
        theta: The batched beamsplitter angles.
        phi: The batched beamsplitter phases.
        stable: Whether to use the stable algorithm.

    Returns:
        The batched matrices representing the beamsplitter operation.
    """
    cdef Py_ssize_t batch_size = np.shape(theta)[0]

    G = np.empty((batch_size, *shape), dtype=np.complex128)

    cdef Py_ssize_t M, N, P, Q
    M, N, P, Q = shape

    cdef double complex[:, :, :, :, :] G_flat_view = G
    cdef Py_ssize_t k

    if stable:
        for k in prange(batch_size, nogil=True):
            compute_beamsplitter_stable(
                M, N, P, Q,
                theta[k],
                phi[k],
                &(G_flat_view[k])[0, 0, 0, 0],
            )
    else:
        for k in prange(batch_size, nogil=True):
            compute_beamsplitter(
                M, N, P, Q,
                theta[k],
                phi[k],
                &(G_flat_view[k])[0, 0, 0, 0],
            )
    return G


# -------------------
# Displacement Python
# -------------------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def displacement(
    tuple cutoffs,
    double complex alpha,
) :
    r"""
    Calculates the matrix elements of the displacement gate using a recurrence relation.

    Args:
        cutoffs: The Fock ladder output-input cutoffs.
        alpha: The displacement magnitude and angle.

    Returns:
        The matrix representing the displacement operation.
    """
    cdef Py_ssize_t N = cutoffs[0]
    cdef Py_ssize_t M = cutoffs[1]
    cdef bint flipped = False

    if N < M:
        N, M = M, N
        flipped = True

    D = np.empty((N, M), dtype=np.complex128)

    cdef double complex[:, :] D_view = D

    with nogil:
        compute_displacement(N, M, alpha, flipped, &D_view[0, 0])

    if flipped:
        D = np.swapaxes(D, -2, -1)

    return D


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def displacement_batched(
    tuple cutoffs,
    const double complex[:] alpha,
) :
    r"""
    Calculates the matrix elements of the displacement gate using a recurrence relation.
    Supports batched alpha values.

    Note: batch dimensions must be flattened.

    Args:
        cutoffs: The Fock ladder output-input cutoffs.
        alpha: The batched displacement magnitude and angles.

    Returns:
        The batch of matrices representing the displacement operation.
    """
    cdef Py_ssize_t batch_size = np.shape(alpha)[0]
    cdef Py_ssize_t N = cutoffs[0]
    cdef Py_ssize_t M = cutoffs[1]
    cdef bint flipped = False

    if N < M:
        N, M = M, N
        flipped = True

    D = np.empty((batch_size, N, M), dtype=np.complex128)

    # for batched
    cdef double complex[:, :, :] D_flat_view = D
    cdef Py_ssize_t k

    for k in prange(batch_size, nogil=True):
        compute_displacement(N, M, alpha[k], flipped, &(D_flat_view[k])[0, 0])

    if flipped:
        D = np.swapaxes(D, -2, -1)

    return D


# -------------------------
# Homodyne Projector Python
# -------------------------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def homodyne_projector(
    int fock_dim,
    const double complex[:, :] A,
    const double complex[:] b,
    double complex c,
    out = None,
):
    r"""
    Fast homodyne projector.

    Args:
        fock_dim: The Fock dimension (cutoff+1) of the remaining wires.
        A: The Bargmann `A` matrix.
        b: The Bargmann `b` vector.
        c: The Bargmann `c` scalar.
        out: An optional output array to write to.
            Should be of shape `(fock_dim, fock_dim, fock_dim)`.

    Returns:
        The resulting `(fock_dim, fock_dim, fock_dim)` array.

    """
    cdef Py_ssize_t N = fock_dim

    out = np.empty((N, N, N), dtype=np.complex128) if out is None else out
    cdef double complex[:] out_view = out.reshape(N * N * N)

    with nogil:
        compute_homodyne_projector(
            N, N, N,
            &A[0, 0],
            &b[0],
            c,
            &out_view[0],
        )
    return out


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def homodyne_projector_batched(
    int fock_dim,
    const double complex[:, :, :] A,
    const double complex[:, :] b,
    const double complex[:] c,
    out = None,
):
    r"""
    Fast homodyne projector for batched inputs.

    Note: batch dimensions must be flattened.

    Args:
        fock_dim: The Fock dimension (cutoff+1) of the remaining wires.
        A: The batched Bargmann `A` matrix.
        b: The batched Bargmann `b` vector.
        c: The batched Bargmann `c` scalar.
        out: An optional output array to write to.
            Should be of shape `(batch_size, fock_dim, fock_dim, fock_dim)`.

    Returns:
        The resulting `(batch_size, fock_dim, fock_dim, fock_dim)` array.

    """
    cdef Py_ssize_t N = fock_dim
    cdef Py_ssize_t batch_size = np.shape(c)[0]

    out = np.empty((batch_size, N, N, N), dtype=np.complex128) if out is None else out
    cdef double complex[:, :] out_view = out.reshape((batch_size, N * N * N))

    for k in prange(batch_size, nogil=True):
        compute_homodyne_projector(
            N, N, N,
            &(A[k])[0, 0],
            &(b[k])[0],
            c[k],
            &(out_view[k])[0],
        )
    return out


# ----------------------------
# Jacobian Displacement Python
# ----------------------------


@cython.boundscheck(False)
def jacobian_displacement(
    D: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    r"""
    Calculates the jacobian of the displacement gate with respect to the complex displacement
    alpha and its conjugate. Both are needed for backprop, as the displacement gate is not a
    holomorphic function, as the Fock amplitudes depend on both alpha and on its conjugate.
    Each jacobian in this case has the same shape as the array D, as the displacement is a scalar.

    Args:
        D: The D(alpha) gate in Fock representation.
        alpha: The alpha parameter of D(alpha).

    Returns:
        The Jacobian of the displacement gate with respect to alpha and alpha_conj.
    """
    shape = D.shape
    batch_shape = shape[:-2]

    jac_alpha = np.zeros(shape, dtype=np.complex128)  # i.e. dD_dalpha for all m,n
    jac_alphac = np.zeros(shape, dtype=np.complex128)  # i.e. dD_dalphac for all m,n

    cdef int N = shape[-2]
    cdef int M = shape[-1]

    # for unbatched
    cdef double complex alpha_view
    cdef const double complex[:, :] D_view
    cdef double complex[:, :] jac_alpha_view
    cdef double complex[:, :] jac_alphac_view

    # for batched
    cdef const double complex[:] alpha_flat_view
    cdef const double complex[:, :, :] D_flat_view
    cdef double complex[:, :, :] jac_alpha_flat_view
    cdef double complex[:, :, :] jac_alphac_flat_view
    cdef Py_ssize_t batch_size
    cdef Py_ssize_t k

    if batch_shape == ():
        D_view = D
        alpha_view = alpha
        jac_alpha_view = jac_alpha
        jac_alphac_view = jac_alphac
        with nogil:
            compute_jacobian_displacement(
                N,
                M,
                alpha_view,
                &D_view[0, 0],
                &jac_alpha_view[0, 0],
                &jac_alphac_view[0, 0],
            )
    else:
        D_flat_view = D.reshape((-1, N, M))
        alpha_flat_view = alpha.reshape(-1)
        jac_alpha_flat_view = jac_alpha.reshape((-1, N, M))
        jac_alphac_flat_view = jac_alphac.reshape((-1, N, M))
        batch_size = D_flat_view.shape[0]
        for k in prange(batch_size, nogil=True):
            compute_jacobian_displacement(
                N,
                M,
                alpha_flat_view[k],
                &(D_flat_view[k])[0, 0],
                &(jac_alpha_flat_view[k])[0, 0],
                &(jac_alphac_flat_view[k])[0, 0],
            )

    return jac_alpha, jac_alphac


# ---------------
# Squeezed Python
# ---------------

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def squeezed(
    int cutoff,
    double r,
    double theta,
) :
    r"""
    Calculates the matrix elements of the single-mode squeezed state using recurrence relations.

    Args:
        cutoff: The Fock cutoff for the ket.
        r: The squeezing magnitude.
        theta: The squeezing angle.

    Returns:
        The matrix representing the squeezing gate.
    """
    cdef Py_ssize_t N = cutoff

    S = np.zeros(cutoff, dtype=np.complex128)

    cdef double complex[:] S_view = S

    with nogil:
        compute_squeezed(N, r, theta, &S_view[0])

    return S


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def squeezed_batched(
    int cutoff,
    const double[:] r,
    const double[:] theta,
) :
    r"""
    Calculates the matrix elements of the single-mode squeezed state using recurrence relations.
    Supports batched r and theta values.

    Note: batch dimensions must be flattened.

    Args:
        cutoff: The Fock cutoff for the ket.
        r: The batched squeezing magnitudes.
        theta: The batched squeezing angles.

    Returns:
        The batch of matrices representing the squeezing gate.
    """
    cdef Py_ssize_t batch_size = np.shape(r)[0]
    cdef Py_ssize_t N = cutoff

    S = np.zeros((batch_size, cutoff), dtype=np.complex128)

    cdef double complex[:, :] S_flat_view = S
    cdef Py_ssize_t k

    for k in prange(batch_size, nogil=True):
        compute_squeezed(N, r[k], theta[k], &(S_flat_view[k])[0])

    return S


# ---------------
# Squeezer Python
# ---------------


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def squeezer(
    tuple shape,
    double r,
    double theta,
) :
    r"""
    Calculates the matrix elements of the squeezing gate using a recurrence relation.
    (See eq. 50-52 in https://arxiv.org/abs/2004.11002)

    Args:
        shape: The Fock cutoffs for the output and input indices.
        r: The squeezing magnitude.
        theta: the squeezing angle.

    Returns:
        The matrix representing the squeezing gate.
    """
    cdef Py_ssize_t M = shape[0]
    cdef Py_ssize_t N = shape[1]

    S = np.zeros(shape, dtype=np.complex128)

    cdef double complex[:, :] S_view = S

    with nogil:
        compute_squeezer(M, N, r, theta, &S_view[0, 0])

    return S


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def squeezer_batched(
    tuple shape,
    const double[:] r,
    const double[:] theta,
) :
    r"""
    Calculates the matrix elements of the squeezing gate using a recurrence relation.
    Supports batched r and theta values.
    (See eq. 50-52 in https://arxiv.org/abs/2004.11002)

    Note: batch dimensions must be flattened.

    Args:
        shape: The Fock cutoffs for the output and input indices.
        r: The batched squeezing magnitudes.
        theta: The batched squeezing angles.

    Returns:
        The batch of matrices representing the squeezing gate.
    """
    cdef Py_ssize_t batch_size = np.shape(r)[0]
    cdef Py_ssize_t M = shape[0]
    cdef Py_ssize_t N = shape[1]

    S = np.zeros((batch_size, *shape), dtype=np.complex128)

    cdef double complex[:, :, :] S_flat_view = S
    cdef Py_ssize_t k

    for k in prange(batch_size, nogil=True):
        compute_squeezer(M, N, r[k], theta[k], &(S_flat_view[k])[0, 0])

    return S
