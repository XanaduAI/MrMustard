from abc import ABC
from mrmustard.typing import *
from mrmustard.plugins import FockPlugin, GaussianPlugin
from mrmustard.abstract import State, Parametrized


class GaussianMeasurement(ABC, Parametrized):
    r"""
    A Gaussian general-dyne measurement.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, state: State, projecto_onto: State) -> Tuple[Scalar, State]:
        r"""
        Applies a general-dyne Gaussian measurement to the state, i.e. it projects
        onto the state with given cov and outcome means vector.
        Args:
            state: the state to be measured
            projecto_onto: the Gaussian state to project onto
        Returns:
            the measurement probabilities and the remaining post-measurement state (if any)
        """
        prob, cov, means = self._gaussian.general_dyne(state.cov, state.means, projecto_onto.cov, projecto_onto.means, self._modes)
        remaining_modes = [m for m in range(state.num_modes) if m not in self._modes]
        remaining_state = State(len(remaining_modes), state.hbar, self._gaussian.is_mixed(cov))
        if len(remaining_modes) > 0:
            remaining_state.cov = cov
            remaining_state.means = means
        return prob, remaining_state


class FockMeasurement(ABC, Parametrized):
    r"""
    A Fock measurement projecting onto a Fock measurement pattern.
    It works by representing the state in the Fock basis and then applying
    a stochastic channel matrix P(meas|n) to the Fock probabilities (belief propagation).
    It outputs the measurement probabilities and the remaining post-measurement state (if any)
    in the Fock basis.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def project(self, state: State, cutoffs: Sequence[int], measurement: Sequence[Optional[int]]) -> State:
        r"""
        Projects the state onto a Fock measurement in the form [a,b,c,...] where integers
        indicate the Fock measurement on that mode and None indicates no projection on that mode.

        Returns the renormalized state (in the Fock basis) in the unmeasured modes
        and the measurement probability.
        """
        if (len(cutoffs) != state.num_modes) or (len(measurement) != state.num_modes):
            raise ValueError("the length of cutoffs/measurements does not match the number of modes")
        dm = state.dm(cutoffs=cutoffs)
        measured = 0
        for mode, (stoch, meas) in enumerate(zip(self._stochastic_channel, measurement)):
            if meas is not None:
                # put both indices last and compute sum_m P(meas|m)rho_mm for every meas
                last = [mode - measured, mode + state.num_modes - 2 * measured]
                perm = list(set(range(dm.ndim)).difference(last)) + last
                dm = self._backend.transpose(dm, perm)
                dm = self._backend.diag(dm)
                dm = self._backend.tensordot(dm, stoch[meas, : dm.shape[-1]], [[-1], [0]], dtype=dm.dtype)
                measured += 1
        prob = self._backend.sum(self._backend.all_diagonals(dm, real=False))
        return dm / prob, self._backend.abs(prob)

    def apply_stochastic_channel(self, stochastic_channel, fock_probs: State):
        cutoffs = [fock_probs.shape[m] for m in self._modes]
        for i, mode in enumerate(self._modes):
            if cutoffs[mode] > stochastic_channel[i].shape[1]:
                raise IndexError(
                    f"This detector does not support so many input photons in mode {mode} (you could increase max_input_photons or reduce the cutoff)"
                )
        detector_probs = fock_probs
        for i, mode in enumerate(self._modes):
            detector_probs = self._backend.tensordot(
                detector_probs,
                stochastic_channel[i][: cutoffs[mode], : cutoffs[mode]],
                [[mode], [1]],
            )
            indices = list(range(fock_probs.ndim - 1))
            indices.insert(mode, fock_probs.ndim - 1)
            detector_probs = self._backend.transpose(detector_probs, indices)
        return detector_probs

    def __call__(self, state: State, measurement_: Matrix, outcome: Vector) -> Tuple[Scalar, State]:
