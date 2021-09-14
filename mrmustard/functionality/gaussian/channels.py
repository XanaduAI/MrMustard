from math import pi, sqrt
from thewalrus.quantum import is_pure_cov
from mrmustard._typing import *
from mrmustard import Backend
from mrmustard.experimental import XPTensor
backend = Backend()



def CPTP(cov: XPTensor, means: XPTensor, modes: Optional[Sequence[int]] = None, X = XPTensor(multiplicative=True), Y = XPTensor(additive=True), d = XPTensor(additive=True)) -> Tuple[XPTensor, XPTensor]:
    r"""Returns the cov matrix and means vector of a state after undergoing a CPTP channel, computed as `cov = X \cdot cov \cdot X^T + Y`
    and `d = X \cdot means + d`.
    If the channel is single-mode, `modes` can contain `M` modes to apply the channel to,
    otherwise it must contain as many modes as the number of modes in the channel.
    It is assumed that X, Y and d are XPTensors in fewer or the same modes as cov and means.

    Args:
        cov (Matrix): covariance matrix
        means (Vector): means vector
        X (Matrix): the X matrix of the CPTP channel
        Y (Matrix): noise matrix of the CPTP channel
        d (Vector): displacement vector of the CPTP channel
        modes (Sequence[int] or None): override which modes the states are in. Othewise it's assumed that X, Y and d are in the same modes as cov and means.
        hbar (float): value of hbar
    Returns:
        Tuple[Matrix, Vector]: the covariance matrix and the means vector of the state after the CPTP channel
    """
    # if single-mode channel, apply to all modes indicated in `modes`
    if len(X.inmodes) == len(X.outmodes) == 1 and len(modes) > 1:
        X = X.clone(modes)
    if len(Y.indmoes) == len(Y.outmodes) == 1 and len(modes) > 1:
        Y = Y.clone(modes)
    if len(d.outmodes) == 1 and len(modes) > 1:
        d = d.clone(modes)
    cov = X @ cov @ X.T + Y
    means = X @ means + d
    return cov, means


def loss_X(transmissivity: Union[Scalar, Vector]) -> Matrix:
    r"""Returns the X matrix for the lossy bosonic channel.
    The full channel is applied to a covariance matrix `\Sigma` as `X\Sigma X^T + Y`.
    """
    D = backend.sqrt(transmissivity)
    return backend.diag(backend.concat([D, D], axis=0))


def loss_Y(transmissivity: Union[Scalar, Vector], hbar: float) -> Matrix:
    r"""Returns the Y (noise) matrix for the lossy bosonic channel.
    The full channel is applied to a covariance matrix `\Sigma` as `X\Sigma X^T + Y`.
    """
    D = (1.0 - transmissivity) * hbar / 2
    return backend.diag(backend.concat([D, D], axis=0))


def thermal_X(nbar: Union[Scalar, Vector]) -> Matrix:
    r"""Returns the X matrix for the thermal lossy channel.
    The full channel is applied to a covariance matrix `\sigma` as `X\sigma X^T + Y`.
    """
    raise NotImplementedError


def thermal_Y(nbar: Union[Scalar, Vector], hbar: float) -> Matrix:
    r"""Returns the Y (noise) matrix for the thermal lossy channel.
    The full channel is applied to a covariance matrix `\sigma` as `X\sigma X^T + Y`.
    """
    raise NotImplementedError


def compose_channels_XYd(X1: XPTensor, Y1: XPTensor, d1: XPTensor, X2: XPTensor, Y2: XPTensor, d2: XPTensor) -> Tuple[XPTensor, XPTensor, XPTensor]:
    r"""Returns the combined X, Y, and d for two CPTP channels.
    Arguments:
        X1 (Matrix): the X matrix of the first CPTP channel
        Y1 (Matrix): the Y matrix of the first CPTP channel
        d1 (Vector): the displacement vector of the first CPTP channel
        X2 (Matrix): the X matrix of the second CPTP channel
        Y2 (Matrix): the Y matrix of the second CPTP channel
        d2 (Vector): the displacement vector of the second CPTP channel
    Returns:
        Tuple[Matrix, Matrix, Vector]: the combined X, Y, and d matrices
    """
    if X1 is None:
        X = X2
    elif X2 is None:
        X = X1
    else:
        X = X2 @ X1
    if Y1 is None:
        Y = Y2
    elif Y2 is None:
        Y = Y1
    else:
        Y = X2 @ Y1 @ X2.T + Y2
    if d1 is None:
        d = d2
    elif d2 is None:
        d = d1
    else:
        d = X2 @ d1 + d2
    return X, Y, d


# ~~~~~~~~~~~~~~~
# non-TP channels
# ~~~~~~~~~~~~~~~


def general_dyne(
    cov: XPTensor, means: XPTensor, proj_cov: XPTensor, proj_means: XPTensor, modes: List[int], hbar: float
) -> Tuple[Scalar, XPTensor, XPTensor]:
    r"""
    Returns the results of a general dyne measurement.
    Arguments:
        cov (Matrix): covariance matrix of the state being measured
        means (Vector): means vector of the state being measured
        proj_cov (Matrix): covariance matrix of the state being projected onto
        proj_means (Vector): means vector of the state being projected onto (i.e. the measurement outcome)
        modes (Sequence[int]): modes being measured
    Returns:
        Tuple[Scalar, Matrix, Vector]: the outcome probability *density*, the post-measurement cov and means vector
    """
    Amodes = [i for i in range(cov.ndim) if i not in modes]
    Bmodes = list(modes)
    A, B, AB = cov[Amodes], cov[Bmodes], cov[Amodes, Bmodes]
    a, b = means[Amodes], means[Bmodes]
    inv = XPTensor(backend.inv(B + proj_cov), multiplicative=True)
    new_cov = A - AB @ inv @ AB.T
    new_means = a + AB @ inv @ (proj_means - b)
    prob = backend.exp(-backend.sum(backend.matvec(inv, proj_means - b) * proj_means - b)) / (
        pi ** B.ndim * (hbar ** -B.ndim) * backend.sqrt(backend.det(B + proj_cov))
    )  # TODO: check this (hbar part especially)
    return prob, new_cov, new_means
