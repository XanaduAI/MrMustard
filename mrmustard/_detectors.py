from abc import ABC, abstractmethod
import numpy as np
from typing import List
from mrmustard._circuit import DetectorInterface
from mrmustard._backends import MathBackendInterface, utils
from scipy.stats import poisson


class DetectorBackendInterface(ABC):
    @abstractmethod
    def convolve_probs_1d(prob, other_probs: List): pass


class Detector(DetectorInterface):
    _math_backend: MathBackendInterface
    _detector_backend: DetectorBackendInterface

    def __init__(self, mode: int):
        self.mode = mode

    def __call__(self, fock_probs):
        cutoff = fock_probs.shape[self.mode]
        if cutoff > self._stochastic_channel.shape[0]:
            raise IndexError('This detector does not support so many input photons (you should probably increase max_input_photons)')
        return self._math_backend.replace_index(fock_probs, self._stochastic_channel[:cutoff, :cutoff], self.mode)


class PNR(Detector):
    r"""
    Photon Number Resolving detector. It can be given the full conditional detection probabilities,
    or it can compute them from the quantum efficiency (binomial) and the dark count probability (possonian).
    """
    def __init__(self, mode: int, conditional_probs=None, quantum_efficiency: float = 1.0, dark_count_prob: float = 0.0, max_input_photons: int = 20):
        super().__init__(mode)
        self.quantum_efficiency = quantum_efficiency
        self.dark_count_prob = dark_count_prob
        self.max_input_photons = max_input_photons
        cutoff = self.max_input_photons + 1

        if conditional_probs is not None:
            self._stochastic_channel = conditional_probs
        else:
            dark_prior = poisson.pmf(np.arange(cutoff), dark_count_prob)
            condprob = utils.binomial_conditional_prob(success_prob=quantum_efficiency, dim_in=cutoff, dim_out=cutoff)
            self._stochastic_channel = self._detector_backend.convolve_probs_1d(condprob, [np.identity(cutoff)[0], dark_prior])


class APD(Detector):
    pass

    
