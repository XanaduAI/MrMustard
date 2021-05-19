from abc import ABC, abstractmethod
import numpy as np
from typing import List
from mrmustard._circuit import DetectorInterface
from mrmustard._backends import MathBackendInterface, utils
from scipy.stats import poisson


class Detector(DetectorInterface):
    r"""
    Base class for photon detectors. It implements a conditional probability P(out|in) as a stochastic matrix,
    so that an input prob distribution P(in) is transformed to P(out) via belief propagation.
    """
    _math_backend: MathBackendInterface

    def __init__(self, modes: List[int]):
        self.modes = modes

    def __call__(self, fock_probs):
        cutoffs = [fock_probs.shape[m] for m in self.modes]
        for mode in self.modes:
            if cutoffs[mode] > self._stochastic_channel.shape[1]:
                raise IndexError(
                    f"This detector does not support so many input photons in mode {mode} (you could increase max_input_photons or reduce the cutoff)"
                )

        detector_probs = fock_probs
        for mode in self.modes:
            detector_probs = self._math_backend.tensordot(
                detector_probs,
                self._stochastic_channel[: cutoffs[mode], : cutoffs[mode]],
                [[mode], [1]],
            )
            indices = list(range(fock_probs.ndim - 1))
            indices.insert(mode, fock_probs.ndim - 1)
            detector_probs = self._math_backend.transpose(detector_probs, indices)
        return detector_probs


class PNR(Detector):
    r"""
    Photon Number Resolving detector. It can be given the full conditional detection probabilities,
    or it can compute them from the quantum efficiency (binomial) and the dark count probability (possonian).
    """

    def __init__(
        self,
        modes: List[int],
        conditional_probs=None,
        quantum_efficiency: float = 1.0,
        dark_count_prob: float = 0.0,
        max_input_photons: int = 50,
    ):
        super().__init__(modes)
        self.quantum_efficiency = quantum_efficiency
        self.dark_count_prob = dark_count_prob
        self.max_input_photons = max_input_photons
        cutoff = self.max_input_photons + 1

        if conditional_probs is not None:
            self._stochastic_channel = conditional_probs
        else:
            dark_prior = poisson.pmf(np.arange(cutoff), dark_count_prob)
            condprob = utils.binomial_conditional_prob(
                success_prob=quantum_efficiency, dim_in=cutoff, dim_out=cutoff
            )
            self._stochastic_channel = self._math_backend.convolve_probs_1d(
                condprob, [dark_prior, np.identity(condprob.shape[1])[0]]
            )


class APD(Detector):
    pass
