from functools import partial  # pragma: no cover

import jax  # pragma: no cover
import jax.numpy as jnp  # pragma: no cover
import numpy as np  # pragma: no cover

from .lattice import strategies  # pragma: no cover

# from mrmustard.math.backend_manager import BackendManager

# math = BackendManager()


@jax.custom_vjp
@partial(jax.jit, static_argnames=["shape"])
def hermite_renormalized_unbatched(
    A: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    shape: tuple[int],
) -> jnp.ndarray:
    shape = tuple(shape)
    G = jax.pure_callback(
        lambda A, b, c: strategies.stable_numba(shape, np.array(A), np.array(b), np.array(c)),
        jax.ShapeDtypeStruct(shape, jnp.complex128),
        A,
        b,
        c,
    )
    return G


def hermite_renormalized_unbatched_fwd(A, b, c, shape):
    G = jax.pure_callback(
        lambda A, b, c: strategies.stable_numba(shape, np.array(A), np.array(b), np.array(c)),
        jax.ShapeDtypeStruct(shape, jnp.complex128),
        A,
        b,
        c,
    )

    return G, (G, A, b, c)


def hermite_renormalized_unbatched_bwd(res, g):
    G, A, b, c = res
    dLdA, dLdB, dLdC = jax.pure_callback(
        lambda G, c, g: strategies.vanilla_vjp_numba(np.array(G), np.array(c), np.array(g)),
        (
            jax.ShapeDtypeStruct(A.shape, jnp.complex128),
            jax.ShapeDtypeStruct(b.shape, jnp.complex128),
            jax.ShapeDtypeStruct(c.shape, jnp.complex128),
        ),
        G,
        c,
        g,
    )
    return dLdA, dLdB, dLdC, None


hermite_renormalized_unbatched.defvjp(
    hermite_renormalized_unbatched_fwd, hermite_renormalized_unbatched_bwd
)
