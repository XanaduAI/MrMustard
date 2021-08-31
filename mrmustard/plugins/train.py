import numpy as np
from scipy.linalg import expm
from mrmustard import Backend
from mrmustard._typing import *

backend = Backend()
euclidean_opt = backend.DefaultEuclideanOptimizer()


def new_variable(value, bounds: Tuple[Optional[float], Optional[float]], name: str) -> Trainable:
    r"""
    Returns a new trainable variable from the current backend
    with initial value set by `value` and bounds set by `bounds`.
    Arguments:
        value (float): The initial value of the variable
        bounds (Tuple[float, float]): The bounds of the variable
        name (str): The name of the variable
    Returns:
        variable (Trainable): The new variable
    """
    return backend.new_variable(value, bounds, name)


def new_constant(value, name: str) -> Tensor:
    r"""
    Returns a new constant (non-trainable) tensor from the current backend
    with initial value set by `value`.
    Arguments:
        value (numeric): The initial value of the tensor
        name (str): The name of the constant
    Returns:
        tensor (Tensor): The new constant tensor
    """
    return backend.new_constant(value, name)


def new_symplectic(num_modes: int) -> Tensor:
    r"""
    Returns a new symplectic matrix from the current backend
    with `num_modes` modes.
    Arguments:
        num_modes (int): The number of modes in the symplectic matrix
    Returns:
        tensor (Tensor): The new symplectic matrix
    """
    return backend.random_symplectic(num_modes)


def new_orthogonal(num_modes: int) -> Tensor:
    return backend.random_orthogonal(num_modes)


def numeric(tensor: Tensor) -> Tensor:
    return backend.asnumpy(tensor)


def update_symplectic(symplectic_params: Sequence[Trainable], symplectic_grads: Sequence[Tensor], symplectic_lr: float):
    for S, dS_riemann in zip(symplectic_params, symplectic_grads):
        Y = backend.riemann_to_symplectic(S, dS_riemann)
        YT = backend.transpose(Y)
        new_value = backend.matmul(S, backend.expm(-symplectic_lr * YT) @ backend.expm(-symplectic_lr * (Y - YT)))
        backend.assign(S, new_value)


def update_orthogonal(orthogonal_params: Sequence[Trainable], orthogonal_grads: Sequence[Tensor], orthogonal_lr: float):
    for O, dO_riemann in zip(orthogonal_params, orthogonal_grads):
        D = 0.5 * (dO_riemann - backend.matmul(backend.matmul(O, backend.transpose(dO_riemann)), O))
        new_value = backend.matmul(O, backend.expm(orthogonal_lr * backend.matmul(backend.transpose(D), O)))
        backend.assign(O, new_value)


def update_euclidean(euclidean_params: Sequence[Trainable], euclidean_grads: Sequence[Tensor], euclidean_lr: float):
    # backend.update_euclidean(euclidean_params, euclidean_grads, euclidean_lr)
    euclidean_opt.lr = euclidean_lr
    euclidean_opt.apply_gradients(zip(euclidean_grads, euclidean_params))


def extract_parameters(items: Sequence, kind: str) -> List[Trainable]:
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
            for p in item.trainable_parameters[kind]:
                if (hash := backend.hash_tensor(p)) not in params_dict:
                    params_dict[hash] = p
        except TypeError:  # NOTE: make sure hash_tensor raises a TypeError when the tensor is not hashable
            continue
    return list(params_dict.values())


def loss_and_gradients(cost_fn: Callable, params: dict) -> Tuple[Tensor, Dict[str, Tensor]]:
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
    loss, grads = backend.loss_and_gradients(cost_fn, params)  # delegate entirely to backend
    return loss, grads
