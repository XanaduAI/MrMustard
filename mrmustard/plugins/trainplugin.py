import numpy as np
from scipy.linalg import expm
from mrmustard.typing import *
from mrmustard.backends import BackendInterface


class TrainPlugin:

    backend: BackendInterface

    def __init__(self):
        self.euclidean_opt = self.__class__.backend.DefaultEuclideanOptimizer()

    def value(self, tensor: Tensor) -> Tensor:
        return self.backend.value(tensor)

    def update_symplectic(self, symplectic_grads: Sequence[Tensor], symplectic_params: Sequence[Tensor], symplectic_lr: float):
        for S, dS_riemann in zip(symplectic_params, symplectic_grads):
            Y = self.backend.riemann_to_symplectic(S, dS_riemann)
            YT = self.backend.transpose(Y)
            S = S @ self.backend.expm(-symplectic_lr * YT) @ self.backend.expm(-symplectic_lr * (Y - YT))

    def update_orthogonal(self, orthogonal_grads: Sequence[Tensor], orthogonal_params: Sequence[Trainable], orthogonal_lr: float):
        for O, dO_riemann in zip(orthogonal_params, orthogonal_grads):
            D = 0.5 * (dO_riemann - self.backend.matmul(self.backend.matmul(O, self.backend.transpose(dO_riemann)), O))
            O = O @ self.backend.expm(orthogonal_lr * self.backend.matmul(self.backend.transpose(D), O))

    def update_euclidean(self, euclidean_grads: Sequence[Tensor], euclidean_params: Sequence[Trainable], euclidean_lr: float):
        self.euclidean_opt.lr = euclidean_lr
        self.euclidean_opt.apply_gradients(zip(euclidean_grads, euclidean_params))

    def extract_parameters(self, items: Sequence[Trainable], kind: str) -> Dict[str, Trainable]:
        r"""
        Extracts the parameters of the given kind from the given items.
        Arguments:
            items (Sequence[Trainable]): The items to extract the parameters from
            kind (str): The kind of parameters to extract. Can be "symplectic", "orthogonal", or "euclidean".
        Returns:
            parameters (List[Trainable]): The extracted parameters
        """
        params_dict = dict()
        for item in items:
            try:
                for p in getattr(item, kind + '_parameters'):
                    if (hash := self.backend.hash_tensor(p)) not in params:
                        params_dict[hash] = p
            except TypeError:  # make sure hash_tensor raises a TypeError when the tensor is not hashable
                continue
        return {kind : list(params_dict.values())}

    def loss_and_gradients(self, cost_fn: Callable, params: dict):
        r"""
        Computes the loss and gradients of the cost function with respect to the parameters.
        The dictionary has three keys: "symplectic", "orthogonal", and "euclidean", to maintain
        the information of the different parameter types.

        Arguments:
            cost_fn (Callable): The cost function to be minimized
            params (dict): A dictionary of parameters to be optimized
        
        Returns:
            loss (float): The cost function of the current parameters
            gradients (dict): A dictionary of gradients of the cost function with respect to the parameters
        """
        loss, grads = self.backend.loss_and_gradients(cost_fn, params)  # delegate entirely to backend
        return loss, grads