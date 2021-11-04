# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from mrmustard.utils.types import *
from thewalrus.quantum import is_pure_cov
from mrmustard.utils.xptensor import XPMatrix, XPVector
from mrmustard import settings
import importlib
from numpy import pi

def _load_backend(backend_name: str):
    "This private function is called by the Settings object to set the math backend in this module"
    Math = importlib.import_module(f"mrmustard.math.{backend_name}").Math
    globals()["math"] = Math()  # setting global variable only in this module's scope

# NOTE: Gaussian functions operate on XPMatrix and XPVector objects


def CPTP(cov: XPMatrix = XPMatrix(like_1=True),
         means: XPVector = XPVector(),
         X:XPMatrix = XPMatrix(like_1=True),
         Y: XPMatrix = XPMatrix(like_0=True),
         d: XPVector = XPVector()) -> Tuple[XPMatrix, XPVector]:
    r"""Returns the cov matrix and means vector of a state after undergoing a CPTP channel, computed as `cov = X \cdot cov \cdot X^T + Y`
    and `d = X \cdot means + d`.
    If the channel is single-mode, `modes` can contain `M` modes to apply the channel to,
    otherwise it must contain as many modes as the number of modes in the channel.

    Args:
        cov (XPMatrix): covariance matrix
        means (XPVector): means vector
        X (XPMatrix): the X matrix of the CPTP channel
        Y (XPMatrix): noise matrix of the CPTP channel
        d (XPVector): displacement vector of the CPTP channel
    Returns:
        Tuple[Matrix, Vector]: the covariance matrix and the means vector of the state after the CPTP channel
    """
    # if single-mode channel, apply to all modes indicated in `modes`
    if X.num_modes == 1:
        X.clone_like(cov)
    if Y.num_modes == 1:
        Y.clone_like(cov)
    if d.num_modes == 1:
        d.clone_like(means)
    return X @ cov @ X.T + Y, X @ means + d


def general_dyne(
    cov: XPMatrix, means: XPVector, proj_cov: XPMatrix, proj_means: XPVector) -> Tuple[Scalar, XPMatrix, XPVector]:
    r"""
    Returns the results of a general dyne measurement.
    Arguments:
        cov (XPMatrix): covariance matrix of the state being measured
        means (XPVector): means vector of the state being measured
        proj_cov (XPMatrix): covariance matrix of the state being projected onto
        proj_means (XPVector): means vector of the state being projected onto (i.e. the measurement outcome)
    Returns:
        Tuple[Scalar, XPMatrix, XPVector]: the outcome probability, the post-measurement cov and means
    """
    # B is the system being measured, A is the leftover
    Amodes = [m for m in cov.outmodes if m not in proj_cov.outmodes]
    Bmodes = proj_cov.outmodes
    A, B, AB = cov[Amodes], cov[Bmodes], cov[Amodes, Bmodes]
    a, b = means[Amodes], means[Bmodes]
    inv = XPMatrix.from_xxpp(math.inv(B.to_xxpp() + proj_cov.to_xxpp()), like_1 = True, modes=(Bmodes, Bmodes))
    new_cov = A - AB @ inv @ AB.T
    m_b = (proj_means - b)
    new_means = a + AB @ (inv @ m_b)
    prob = math.exp((-m_b @ inv @ m_b).to_xxpp()) / (
        pi ** nB * (settings.HBAR ** -nB) * math.sqrt(math.det(B.to_xxpp() + proj_cov.to_xxpp()))
    )  # TODO: check this (hbar part especially)
    return prob, new_cov, new_means


# ~~~~~~~~~
# utilities
# ~~~~~~~~~
def number_means(cov: Matrix, means: Vector, hbar: float) -> Vector:
    r"""
    Returns the photon number means vector
    given a Wigner covariance matrix and a means vector.
    Args:
        cov: The Wigner covariance matrix.
        means: The Wigner means vector.
        hbar: The value of the Planck constant.
    Returns:
        The photon number means vector.
    """
    N = means.shape[-1] // 2
    return (means[:N] ** 2 + means[N:] ** 2 + math.diag_part(cov[:N, :N]) + math.diag_part(cov[N:, N:]) - hbar) / (2 * hbar)


def number_cov(cov: Matrix, means: Vector, hbar: float) -> Matrix:
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
    N = means.shape[-1] // 2
    mCm = cov * means[:, None] * means[None, :]
    dd = math.diag(math.diag_part(mCm[:N, :N] + mCm[N:, N:] + mCm[:N, N:] + mCm[N:, :N])) / (2 * hbar ** 2)
    CC = (cov ** 2 + mCm) / (2 * hbar ** 2)
    return CC[:N, :N] + CC[N:, N:] + CC[:N, N:] + CC[N:, :N] + dd - 0.25 * math.eye(N, dtype=CC.dtype)


def is_mixed_cov(cov: Matrix) -> bool:  # TODO: deprecate
    r"""
    Returns True if the covariance matrix is mixed, False otherwise.
    """
    return not is_pure_cov(math.asnumpy(cov), hbar=settings.HBAR)


def trace(cov: Matrix, means: Vector, Bmodes: Sequence[int]) -> Tuple[Matrix, Vector]:
    r"""
    Returns the covariances and means after discarding the specified modes.
    Arguments:
        cov (Matrix): covariance matrix
        means (Vector): means vector
        Bmodes (Sequence[int]): modes to discard
    Returns:
        Tuple[Matrix, Vector]: the covariance matrix and the means vector after discarding the specified modes
    """
    N = len(cov) // 2
    Aindices = math.astensor([i for i in range(N) if i not in Bmodes])
    A_cov_block = math.gather(math.gather(cov, Aindices, axis=0), Aindices, axis=1)
    A_means_vec = math.gather(means, Aindices)
    return A_cov_block, A_means_vec


def partition_cov(cov: Matrix, Amodes: Sequence[int]) -> Tuple[Matrix, Matrix, Matrix]:
    r"""
    Partitions the covariance matrix into the A and B subsystems and the AB coherence block.
    Arguments:
        cov (Matrix): the covariance matrix
        Amodes (Sequence[int]): the modes of system A
    Returns:
        Tuple[Matrix, Matrix, Matrix]: the cov of A, the cov of B and the AB block
    """
    N = cov.shape[-1] // 2
    Bindices = math.cast([i for i in range(N) if i not in Amodes] + [i + N for i in range(N) if i not in Amodes], "int32")
    Aindices = math.cast(Amodes + [i + N for i in Amodes], "int32")
    A_block = math.gather(math.gather(cov, Aindices, axis=1), Aindices, axis=0)
    B_block = math.gather(math.gather(cov, Bindices, axis=1), Bindices, axis=0)
    AB_block = math.gather(math.gather(cov, Bindices, axis=1), Aindices, axis=0)
    return A_block, B_block, AB_block


def partition_means(means: Vector, Amodes: Sequence[int]) -> Tuple[Vector, Vector]:
    r"""
    Partitions the means vector into the A and B subsystems.
    Arguments:
        means (Vector): the means vector
        Amodes (Sequence[int]): the modes of system A
    Returns:
        Tuple[Vector, Vector]: the means of A and the means of B
    """
    N = len(means) // 2
    Bindices = math.cast([i for i in range(N) if i not in Amodes] + [i + N for i in range(N) if i not in Amodes], "int32")
    Aindices = math.cast(Amodes + [i + N for i in Amodes], "int32")
    return math.gather(means, Aindices), math.gather(means, Bindices)


def purity(cov: Matrix, hbar: float) -> Scalar:
    r"""
    Returns the purity of the state with the given covariance matrix.
    Arguments:
        cov (Matrix): the covariance matrix
    Returns:
        float: the purity
    """
    return 1 / math.sqrt(math.det((2 / hbar) * cov))


def join_covs(covs: Sequence[Matrix]) -> Tuple[Matrix, Vector]:
    r"""
    Joins the given covariance matrices into a single covariance matrix.
    Arguments:
        covs (Sequence[Matrix]): the covariance matrices
    Returns:
        Matrix: the joined covariance matrix
    """
    modes = list(range(len(covs[0]) // 2))
    cov = XPMatrix.from_xxpp(covs[0], modes=(modes, modes), like_1=True)
    for i, c in enumerate(covs[1:]):
        modes = list(range(cov.num_modes, cov.num_modes + c.shape[-1] // 2))
        cov = cov @ XPMatrix.from_xxpp(c, modes=(modes, modes), like_1=True)
    return cov.to_xxpp()


def join_means(means: Sequence[Vector]) -> Vector:
    r"""
    Joins the given means vectors into a single means vector.
    Arguments:
        means (Sequence[Vector]): the means vectors
    Returns:
        Vector: the joined means vector
    """
    mean = XPVector.from_xxpp(means[0], modes=list(range(len(means[0]) // 2)))
    for i, m in enumerate(means[1:]):
        mean = mean + XPVector.from_xxpp(m, modes=list(range(mean.num_modes, mean.num_modes + len(m) // 2)))
    return mean.to_xxpp()
