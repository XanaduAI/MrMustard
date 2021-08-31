from mrmustard._typing import *
from mrmustard.abstract import GaussianMeasurement, FockMeasurement, Parametrized, State
from mrmustard.concrete.states import DisplacedSqueezed, Coherent
from mrmustard.plugins import fock, gaussian
from math import pi

__all__ = ["PNRDetector", "ThresholdDetector", "Homodyne", "Heterodyne", "Generaldyne"]


class PNRDetector(Parametrized, FockMeasurement):
    r"""
    Photon Number Resolving detector. If len(modes) > 1 the detector is applied in parallel to all of the modes provided.
    If a parameter is a single float, the parallel instances of the detector share that parameter.
    To apply mode-specific parmeters use a list of floats.
    One can optionally set bounds for each parameter, which the optimizer will respect.
    It can be supplied the full conditional detection probabilities, or it will compute them from
    the quantum efficiency (binomial) and the dark count probability (possonian).
    Arguments:
        conditional_probs (Optional 2d array): if supplied, these probabilities will be used for belief propagation
        efficiency (float or List[float]): list of quantum efficiencies for each detector
        dark_counts (float or List[float]): list of expected dark counts
        max_cutoffs (int or List[int]): largest Fock space cutoffs that the detector should expect
    """

    def __init__(
        self,
        modes: List[int],
        conditional_probs=None,
        efficiency: Union[float, List[float]] = 1.0,
        efficiency_trainable: bool = False,
        efficiency_bounds: Tuple[Optional[float], Optional[float]] = (0.0, 1.0),
        dark_counts: Union[float, List[float]] = 0.0,
        dark_counts_trainable: bool = False,
        dark_counts_bounds: Tuple[Optional[float], Optional[float]] = (0.0, None),
        max_cutoffs: Union[int, List[int]] = 50,
    ):
        if not isinstance(max_cutoffs, Sequence):
            max_cutoffs = [max_cutoffs for m in modes]
        if not isinstance(efficiency, Sequence):
            efficiency = [efficiency for m in modes]
        if not isinstance(dark_counts, Sequence):
            dark_counts = [dark_counts for m in modes]

        super().__init__(
            modes=modes,
            conditional_probs=conditional_probs,
            efficiency=efficiency,
            efficiency_trainable=efficiency_trainable,
            efficiency_bounds=efficiency_bounds,
            dark_counts=dark_counts,
            dark_counts_trainable=dark_counts_trainable,
            dark_counts_bounds=dark_counts_bounds,
            max_cutoffs=max_cutoffs,
        )

        self.recompute_stochastic_channel()

    def recompute_stochastic_channel(self):
        self._stochastic_channel = []
        if self._conditional_probs is not None:
            self._stochastic_channel = self._conditional_probs
        else:
            for cut, qe, dc in zip(self._max_cutoffs, self.efficiency[:], self.dark_counts[:]):
                dark_prior = fock.backend.poisson(max_k=cut, rate=dc)
                condprob = fock.backend.binomial_conditional_prob(success_prob=qe, dim_in=cut, dim_out=cut)
                self._stochastic_channel.append(
                    fock.backend.convolve_probs_1d(condprob, [dark_prior, fock.backend.eye(condprob.shape[1])[0]])
                )


class ThresholdDetector(Parametrized, FockMeasurement):
    r"""
    Threshold detector: any Fock component other than vacuum counts toward a click in the detector.
    If len(modes) > 1 the detector is applied in parallel to all of the modes provided.
    If a parameter is a single float, its value is applied to all of the parallel instances of the detector.
    To apply mode-specific values use a list of floats.
    It can be supplied the full conditional detection probabilities, or it will compute them from
    the quantum efficiency (binomial) and the dark count probability (bernoulli).
    Arguments:
        conditional_probs (Optional 2d array): if supplied, these probabilities will be used for belief propagation
        efficiency (float or List[float]): list of quantum efficiencies for each detector
        dark_count_prob (float or List[float]): list of dark count probabilities for each detector
        max_cutoffs (int or List[int]): largest Fock space cutoffs that the detector should expect
    """

    def __init__(
        self,
        modes: List[int],
        conditional_probs=None,
        efficiency: Union[float, List[float]] = 1.0,
        efficiency_trainable: bool = False,
        efficiency_bounds: Tuple[Optional[float], Optional[float]] = (0.0, 1.0),
        dark_count_prob: Union[float, List[float]] = 0.0,
        dark_count_prob_trainable: bool = False,
        dark_count_prob_bounds: Tuple[Optional[float], Optional[float]] = (0.0, None),
        max_cutoffs: Union[int, List[int]] = 50,
    ):
        if not isinstance(max_cutoffs, Sequence):
            max_cutoffs = [max_cutoffs for m in modes]

        super().__init__(
            modes=modes,
            conditional_probs=conditional_probs,
            efficiency=efficiency,
            efficiency_trainable=efficiency_trainable,
            efficiency_bounds=efficiency_bounds,
            dark_count_prob=dark_count_prob,
            dark_count_prob_trainable=dark_count_prob_trainable,
            dark_count_prob_bounds=dark_count_prob_bounds,
            max_cutoffs=max_cutoffs,
        )

        self.recompute_stochastic_channel()

    def recompute_stochastic_channel(self):
        self._stochastic_channel = []

        if self._conditional_probs is not None:
            self._stochastic_channel = self.conditional_probs
        else:
            for cut, qe, dc in zip(self._max_cutoffs, self.efficiency[:], self.dark_count_probs[:]):
                row1 = ((1.0 - qe) ** fock.backend.arange(cut))[None, :] - dc
                row2 = 1.0 - row1
                rest = fock.backend.zeros((cut - 2, cut), dtype=row1.dtype)
                condprob = fock.backend.concat([row1, row2, rest], axis=0)
                self._stochastic_channel.append(condprob)

    @property
    def stochastic_channel(self) -> List[Matrix]:
        if self._stochastic_channel is None:
            fock.stochastic_channel()
        return self._stochastic_channel


class Generaldyne(Parametrized, GaussianMeasurement):
    r"""
    General dyne measurement.
    """

    def __init__(self, modes: List[int], project_onto: State):
        assert len(modes) * 2 == project_onto.cov.shape[-1] == project_onto.means.shape[-1]
        super().__init__(modes=modes, project_onto=project_onto, hbar=project_onto.hbar)

    def recompute_project_onto(self, project_onto: State) -> State:
        return project_onto


class Homodyne(Parametrized, GaussianMeasurement):
    r"""
    Homodyne measurement on a given list of modes.
    """

    def __init__(
        self,
        modes: List[int],
        quadrature_angles: Union[Scalar, Vector],
        results: Union[Scalar, Vector],
        squeezing: float = 10.0,
        hbar: float = 2.0,
    ):
        r"""
        Args:
            modes (list of ints): modes of the measurement
            quadrature_angles (float or vector): angle(s) of the quadrature axes of the measurement
            results (float or vector): result(s) of the measurement on each axis
            squeezing (float): amount of squeezing of the measurement (default 10.0, ideally infinite)
        """

        super().__init__(
            modes=modes,
            quadrature_angles=quadrature_angles,
            results=results,
            squeezing=gaussian.backend.astensor(squeezing, "float64"),
            hbar=hbar,
        )
        self._project_onto = self.recompute_project_onto(quadrature_angles, results)

    def recompute_project_onto(self, quadrature_angles: Union[Scalar, Vector], results: Union[Scalar, Vector]) -> State:
        quadrature_angles = gaussian.backend.astensor(quadrature_angles, "float64")
        results = gaussian.backend.astensor(results, "float64")
        x = results * gaussian.backend.cos(quadrature_angles)
        y = results * gaussian.backend.sin(quadrature_angles)
        return DisplacedSqueezed(r=self._squeezing, phi=quadrature_angles, x=x, y=y, hbar=self._hbar)


class Heterodyne(Parametrized, GaussianMeasurement):
    r"""
    Heterodyne measurement on a given mode.
    """

    def __init__(self, modes: List[int], x: Union[Scalar, Vector], y: Union[Scalar, Vector], hbar: float = 2.0):
        r"""
        Args:
            mode: modes of the measurement
            x: x-coordinates of the measurement
            y: y-coordinates of the measurement
        """
        super().__init__(modes=modes, x=x, y=y, hbar=hbar)
        self._project_onto = self.recompute_project_onto(x, y)

    def recompute_project_onto(self, x: Union[Scalar, Vector], y: Union[Scalar, Vector]) -> State:
        return Coherent(x=x, y=y, hbar=self._hbar)
