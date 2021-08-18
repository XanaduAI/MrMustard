from abc import ABC
from mrmustard._typing import *
from mrmustard import FockPlugin, GaussianPlugin
from mrmustard.abstract.state import State


# TODO: the recompute_project_onto trick is there because measurements are treated differently from gates: the parameters
# are assumed to be mostly constant and they can be called with additional kwargs if we want the internal representation to be recomputed.
# However, we should find the how to make them work the same way, i.e. use xxx_trainable and xxx_bounds for all of the measurement parameters.
# This is a problem due to the generality of Generaldyne: for homodyne and heterodyne we could already do it, as they
# depend on euclidean parameters, rather than on a Gaussian state.

class GaussianMeasurement(ABC):
    r"""
    A Gaussian general-dyne measurement.
    """
    _gaussian = GaussianPlugin()

    def __call__(self, state: State, **kwargs) -> Tuple[Scalar, State]:
        r"""
        Applies a general-dyne Gaussian measurement to the state, i.e. it projects
        onto the state with given cov and outcome means vector.
        Args:
            state (State): the state to be measured.
            kwargs (optional): same arguments as in the init, but specify if they are different
            from the arguments supplied at init time (e.g. for training the measurement).
        Returns:
            (float, state) The measurement probabilities and the remaining post-measurement state.
            Note that the post-measurement state is trivial if all modes are measured.
        """
        assert self._hbar == state.hbar
        if len(kwargs) > 0:
            self._project_onto = self.recompute_project_onto(**kwargs)
        prob, cov, means = self._gaussian.general_dyne(state.cov, state.means, self._project_onto.cov, self._project_onto.means, self._modes, self._project_onto.hbar)
        remaining_modes = [m for m in range(state.num_modes) if m not in self._modes]
        remaining_state = State(len(remaining_modes), state.hbar, self._gaussian.is_mixed_cov(cov))
        if len(remaining_modes) > 0:
            remaining_state.cov = cov
            remaining_state.means = means
        return prob, remaining_state

    def recompute_project_onto(self, **kwargs) -> State: ...


# TODO: push backend methods into the fock plugin
class FockMeasurement(ABC):
    r"""
    A Fock measurement projecting onto a Fock measurement pattern.
    It works by representing the state in the Fock basis and then applying
    a stochastic channel matrix P(meas|n) to the Fock probabilities (belief propagation).
    It outputs the measurement probabilities and the remaining post-measurement state (if any)
    in the Fock basis.
    """

    _fock = FockPlugin()

    def project(self, state: State, cutoffs: Sequence[int], measurement: Sequence[Optional[int]]) -> Tuple[State, Tensor]:
        r"""
        Projects the state onto a Fock measurement in the form [a,b,c,...] where integers
        indicate the Fock measurement on that mode and None indicates no projection on that mode.

        Returns the measurement probability and the renormalized state (in the Fock basis) in the unmeasured modes.
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
                dm = self._fock._backend.transpose(dm, perm)
                dm = self._fock._backend.diag_part(dm)
                dm = self._fock._backend.tensordot(dm, stoch[meas, : dm.shape[-1]], [[-1], [0]])
                measured += 1
        prob = self._fock._backend.sum(self._fock._backend.all_diagonals(dm, real=False))
        return self._fock._backend.abs(prob), dm / prob

    def apply_stochastic_channel(self, stochastic_channel, fock_probs: Tensor) -> Tensor:
        cutoffs = [fock_probs.shape[m] for m in self._modes]
        for i, mode in enumerate(self._modes):
            if cutoffs[mode] > stochastic_channel[i].shape[1]:
                raise IndexError(
                    f"This detector does not support so many input photons in mode {mode} (you could increase max_input_photons or reduce the cutoff)"
                )
        detector_probs = fock_probs
        for i, mode in enumerate(self._modes):
            detector_probs = self._fock._backend.tensordot(
                detector_probs,
                stochastic_channel[i][: cutoffs[mode], : cutoffs[mode]],
                [[mode], [1]],
            )
            indices = list(range(fock_probs.ndim - 1))
            indices.insert(mode, fock_probs.ndim - 1)
            detector_probs = self._fock._backend.transpose(detector_probs, indices)
        return detector_probs

    def __call__( self, state: State, cutoffs: List[int], outcomes: Optional[Sequence[Optional[int]]]=None) -> Tuple[Tensor, Tensor]:
        fock_probs = state.fock_probabilities(cutoffs)
        all_probs = self.apply_stochastic_channel(self._stochastic_channel, fock_probs)
        if outcomes is None:
            return all_probs
        else:
            probs, dm = self.project(state, cutoffs, outcomes)
            return dm, probs

    def recompute_stochastic_channel(self, **kwargs) -> State: ...