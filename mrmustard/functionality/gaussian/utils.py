from mrmustard._typing import *
from mrmustard import Backend
backend = Backend()
from mrmustard.experimental import XPTensor


def is_mixed_cov(cov: XPTensor) -> bool:
    r"""
    Returns True if the covariance matrix is mixed, False otherwise.
    """
    return not is_pure_cov(backend.asnumpy(cov.to_xxpp()))


def number_means(cov: XPTensor, means: XPTensor, hbar: float) -> Vector:
    r"""
    Returns the photon number means vector
    given a Wigenr covariance matrix and a means vector.
    Args:
        cov: The Wigner covariance matrix.
        means: The Wigner means vector.
        hbar: The value of the Planck constant.
    Returns:
        The photon number means vector.
    """
    N = means.num_modes
    means = means.to_xxpp()
    cov = cov.to_xxpp()
    return (means[:N] ** 2 + means[N:] ** 2 + backend.diag_part(cov[:N, :N]) + backend.diag_part(cov[N:, N:]) - hbar) / (2 * hbar)


def number_cov(cov: XPTensor, means: XPTensor, hbar: float) -> Matrix:
    r"""
    Returns the photon number covariance matrix
    given a Wigenr covariance matrix and a means vector.
    Args:
        cov: The Wigner covariance matrix.
        means: The Wigner means vector.
        hbar: The value of the Planck constant.
    Returns:
        The photon number covariance matrix.
    """
    N = means.num_modes
    means = means.to_xxpp()
    cov = cov.to_xxpp()
    mCm = cov * means[:, None] * means[None, :]
    dd = backend.diag(backend.diag_part(mCm[:N, :N] + mCm[N:, N:] + mCm[:N, N:] + mCm[N:, :N])) / (2 * hbar ** 2)
    CC = (cov ** 2 + mCm) / (2 * hbar ** 2)
    return CC[:N, :N] + CC[N:, N:] + CC[:N, N:] + CC[N:, :N] + dd - 0.25 * backend.eye(N, dtype=CC.dtype)


def purity(cov: XPTensor, hbar: float) -> Scalar:
    r"""
    Returns the purity of the state with the given covariance matrix.
    Arguments:
        cov (Matrix): the covariance matrix
    Returns:
        float: the purity
    """
    return 1 / backend.sqrt(backend.det((2 / hbar) * cov))


def fidelity(cov1, cov2, means1, means2) -> Scalar:
    r"""
    Returns the fidelity between two states.
    Arguments:
        cov1 (Matrix): the first covariance matrix
        cov2 (Matrix): the second covariance matrix
        means1 (Vector): the first means vector
        means2 (Vector): the second means vector
    Returns:
        float: the fidelity
    """
    return backend.exp(-backend.trace(backend.matmul(backend.transpose(cov1), cov2)))

def join_covs(*covs: XPTensor) -> XPTensor:
    r"""
    Joins the given sequence of covariance matrices along the diagonal.
    Arguments:
        matrices (Sequence[XPTensor]): the sequence of covariance matrices
    Returns:
        XPTensor: the joined covariance matrix
    """
    if any(c.isVector or c.isCoherence for c in covs):
        raise ValueError("Only cov matrices allowed")
    if any(c.additive for c in covs) and any(c.multiplicative for c in covs):
        raise ValueError("Must be either all additive or all multiplicative")
    covs = backend.stack([backend.transpose(c.tensor, (0,2,1,3)) for c in covs], axis=-1)
    cov = backend.diag(covs)  # shape [2,2,N,N,T,T]
    cov = backend.transpose(cov, (0,1,2,4,3,5))  # shape [2,2,N,T,N,T]
    cov = backend.reshape(cov, (2,2,cov.shape[2]*cov.shape[3],cov.shape[4]*cov.shape[5]))  # shape [2,2,N*T,N*T]
    return XPTensor(cov, multiplicative=True)