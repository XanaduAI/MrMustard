import tensorflow as tf
from typing import Sequence
from mrmustard.core.plugins.fockplugin import FockPlugin

fock = FockPlugin()


@tf.custom_gradient
def to_fock(A: tf.Tensor, B: tf.Tensor, C: tf.Tensor, cutoffs: Sequence[int]) -> tf.Tensor:
    r"""
    Tensorflow implementation of the `to_fock` function.
    Returns the fock state given the A, B, C matrices and a list of cutoff indices.
        Args:
            A: The A matrix.
            B: The B vector.
            C: The C scalar.
            cutoffs: The cutoff indices.
        Returns:
            The fock state.
    """
    state = fock.fock_representation(A, B, C, cutoffs)

    def grad(dy):
        return fock.fock_representation_gradient(dy, state, A, B, C, cutoffs)

    return state, grad
