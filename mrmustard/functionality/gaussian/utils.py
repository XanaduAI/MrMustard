from mrmustard._typing import *
from mrmustard import Backend
backend = Backend()
from mrmustard.experimental import XPTensor


def is_mixed_cov(cov: Matrix) -> bool:
    r"""
    Returns True if the covariance matrix is mixed, False otherwise.
    """
    return not is_pure_cov(backend.asnumpy(cov.to_xxpp()))


# def trace(cov: Matrix, means: Vector, Bmodes: Sequence[int]) -> Tuple[Matrix, Vector]:
#     r"""
#     Returns the covariances and means after discarding the specified modes.
#     Arguments:
#         cov (Matrix): covariance matrix
#         means (Vector): means vector
#         Bmodes (Sequence[int]): modes to discard
#     Returns:
#         Tuple[Matrix, Vector]: the covariance matrix and the means vector after discarding the specified modes
#     """
#     N = len(cov) // 2
#     Aindices = backend.astensor([i for i in range(N) if i not in Bmodes])
#     A_cov_block = backend.gather(backend.gather(cov, Aindices, axis=0), Aindices, axis=1)
#     A_means_vec = backend.gather(means, Aindices)
#     return A_cov_block, A_means_vec


# def partition_cov(cov: Matrix, Amodes: Sequence[int]) -> Tuple[Matrix, Matrix, Matrix]:
#     r"""
#     Partitions the covariance matrix into the A and B subsystems and the AB coherence block.
#     Arguments:
#         cov (Matrix): the covariance matrix
#         Amodes (Sequence[int]): the modes of system A
#     Returns:
#         Tuple[Matrix, Matrix, Matrix]: the cov of A, the cov of B and the AB block
#     """
#     N = cov.shape[-1] // 2
#     Bindices = backend.cast([i for i in range(N) if i not in Amodes] + [i + N for i in range(N) if i not in Amodes], "int32")
#     Aindices = backend.cast(Amodes + [i + N for i in Amodes], "int32")
#     A_block = backend.gather(backend.gather(cov, Aindices, axis=1), Aindices, axis=0)
#     B_block = backend.gather(backend.gather(cov, Bindices, axis=1), Bindices, axis=0)
#     AB_block = backend.gather(backend.gather(cov, Bindices, axis=1), Aindices, axis=0)
#     return A_block, B_block, AB_block


# def partition_means(means: Vector, Amodes: Sequence[int]) -> Tuple[Vector, Vector]:
#     r"""
#     Partitions the means vector into the A and B subsystems.
#     Arguments:
#         means (Vector): the means vector
#         Amodes (Sequence[int]): the modes of system A
#     Returns:
#         Tuple[Vector, Vector]: the means of A and the means of B
#     """
#     N = len(means) // 2
#     Bindices = backend.cast([i for i in range(N) if i not in Amodes] + [i + N for i in range(N) if i not in Amodes], "int32")
#     Aindices = backend.cast(Amodes + [i + N for i in Amodes], "int32")
#     return backend.gather(means, Aindices), backend.gather(means, Bindices)


def purity(cov: Matrix, hbar: float) -> Scalar:
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