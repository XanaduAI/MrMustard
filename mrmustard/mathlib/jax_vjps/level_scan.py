"""Hermite Renormalized JAX Implementation with Level-Scan Architecture.

Computes the renormalized multidimensional Hermite polynomial tensor using a level-by-level
recurrence via ``lax.scan``. This function accepts arbitrary batch dimensions.

Batch-Last Memory Layout (Why and How)
--------------------------------------
Regardless of how many batch dimensions the inputs have, the computation of a step always proceeds
with a **single flattened batch axis in the rightmost (innermost) position** of each internal
array:

  - ``G``:  ``(total_size, B)``    (B = product of all batch dims)
  - ``A``:  ``(D, D, B)``
  - ``b``:  ``(D, B)``

This is critical for performance to prevent cache thrashing. This implements a hard-coded
batching scheme in the step processing, instead of ``jax.vmap`` due to how ``jax.vmap`` interacts
with ``lax.scan``:

1. **vmap inserts transpose+copy chains.** JAX's scan batching rule always places the batch
   dimension at axis 0 of the carry (hardcoded in ``jax/_src/lax/control_flow/loops.py``,
   ``_scan_batching_rule``). The gather output layout ``(P, B)`` doesn't match the compute layout
   ``(B, P)``, forcing XLA to insert a number of transpose+copy operations for each scan step. The
   single (unbatched) version has zero such operations.

2. **vmap creates strided gathers.** When ``lax.scan``'s carry ``G`` has shape ``(B, total_size)``
   (batch-first, row-major), gathering index ``j`` across all batch elements reads addresses ``(b *
   total_size + j) * 8`` for ``b = 0..B-1``. These are separated by ``total_size * 8`` bytes --
   megabytes apart -- producing maximally non-coalesced memory transactions.

3. **Batch-last fixes both problems.** With ``G`` shaped ``(total_size, B)`` (batch-last,
   row-major), gathering index ``j`` reads ``G[j, :]`` -- a contiguous ``B * 8`` byte chunk. This is
   perfectly coalesced memory operations. Since all arrays share the batch-last convention, no
   transpose+copy chains are needed inside the scan loop.

4. **vmap's ``in_axes``/``out_axes`` cannot fix this.** These parameters only control where the
   batch axis sits on function inputs/outputs. They don't affect the carry layout inside
   ``lax.scan``, which is hardcoded to  batch-at-axis-0. Column-major storage would also solve this,
   but JAX/XLA doesn't expose layout control to user code.

5. **Unbatched inputs use B=1.** When no batch dimensions are present, a singleton batch axis is
   added so the same code path handles both cases. XLA optimizes away the trivial trailing dimension
   efficiently.


Step Size Tuning
----------------
The ``max_step_size`` parameter controls how levels are split into scan steps. Each ``lax.scan``
step processes a fixed-size batch of ``effective_max`` elements. When ``max_step_size=None``
(default), each level is one step, and ``effective_max`` equals the largest level. Since level sizes
follow a bell curve, most steps process far fewer elements than ``effective_max``, wasting work on
padding.

Setting ``max_step_size`` caps the elements per step. The primary mechanism is **padding waste
reduction**:

  - Each step executes gather/compute/scatter for ``effective_max`` elements, regardless of how many
    are real vs padding. Padded elements still trigger full cache-line fetches (128 bytes on NVIDIA
    GPUs) from scattered memory locations, wasting HBM bandwidth and ALU cycles for zero useful
    work.
  - With ``max_step_size=None`` for 4D cutoff 50, ``effective_max=83,350`` but the median level has
    only 22,100 elements -- **62% of all gather/scatter work is pure waste**.
  - Setting ``max_step_size=16384`` reduces padding waste to ~24%, cutting total wasted scattered
    memory traffic by roughly 2x.

Note on carry overhead: despite ``lax.scan`` semantically passing the full G array as an immutable
carry between steps, XLA's buffer assignment pass **aliases the carry buffer in-place**. The scatter
operation ``G.at[indices].set(values, unique_indices=True)`` compiles to an in-place write.
Increasing the number of steps therefore does NOT incur meaningful carry traffic -- the cost is
primarily the small per-step loop overhead.

The runtime curve is U-shaped as a function of ``max_step_size``:
  - **Too large** (or None): high padding waste in gather/scatter operations.
  - **Too small** (e.g., 512): thousands of steps with per-step loop overhead accumulating, plus
    reduced GPU occupancy from tiny kernels.

The optimal ``max_step_size`` is hardware-dependent. The scattered gather/scatter operations suffer
from cache-line amplification (~16x on NVIDIA GPUs: 128-byte cache lines fetching 8-byte complex64
elements). GPUs with larger L2 caches (A100: 40 MB, H100: 50 MB) may tolerate more padding waste
before it becomes the bottleneck, shifting the optimum toward larger step sizes. The default of
``None`` is likely near-optimal on higher-end GPUs for most realistic use cases. If
hardware-specific tuning is desired, it is recommended to sweep values in powers of 2.
"""

from functools import partial
from typing import Any, overload

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import ArrayLike
from numpy.typing import NDArray

# =============================================================================
# Complex128 Scatter Bug Workaround
# =============================================================================
#
# XLA's GPU backend has a known bug (jax#4115, jax#24872, open since 2020) where scatter-set for
# 128-bit types (complex128) is lowered to a sequential while loop instead of a parallel scatter
# kernel. This is because there is no native 128-bit atomic on NVIDIA GPUs, and XLA's lowering for
# scatter-set does not decompose complex128 into two independent float64 writes — even when
# unique_indices=True guarantees no conflicts.
#
# Workaround: when the input dtype is complex128, split into two float64 arrays (G_re, G_im). When
# processing data, gathered values are recombined to complex128 for the arithmetic (which runs at
# full speed), then split back to float64 only for the scatter writes. This avoids the
# serialized scatter while keeping all compute in native complex128.


# =============================================================================
# Precomputation (runs at trace time, in NumPy)
# =============================================================================


def _precompute_level_scan(shape: tuple[int, ...]) -> dict[str, NDArray | int] | None:
    """Enumerate all multi-indices, grouped by level.

    The levels are defined by the sum of the index values.

    Args:
        shape: The cutoff for each mode.

    Returns:
        Dictionary containing all multi-indices grouped by level, the positions of the start of the
        levels, the size of the grouping of the level.  It also contains metadata that is used
        within the computation of the strides used, the number of dimensions used, the total number
        indices enumerated and the number of levels.
    """
    D = len(shape)
    total_size = int(np.prod(shape))

    for s in shape:
        if s > 255:
            raise ValueError(f"Cutoff {s} exceeds uint8 max (255).")

    # Compute strides for shape
    strides = np.zeros(D, dtype=np.int32)
    strides[-1] = 1
    for i in range(D - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]

    # Compute an array of all multi-indices
    grids = np.meshgrid(*(np.arange(s, dtype=np.uint8) for s in shape), indexing="ij")
    all_idx = np.stack([g.ravel() for g in grids], axis=-1)

    # Compute the levels of the multi-indices
    levels = all_idx.astype(np.int32).sum(axis=1)
    max_level = int(levels.max()) if total_size > 0 else 0

    if max_level == 0:
        return None

    # Sorting the levels array will give indicies grouped by level
    order = np.argsort(levels, kind="stable")
    sorted_levels = levels[order]

    # Find the pointers for the start and end points of the levels
    level_starts_all = np.searchsorted(sorted_levels, np.arange(max_level + 1), side="left")
    level_ends_all = np.searchsorted(sorted_levels, np.arange(max_level + 1), side="right")

    # Remove level zero
    skip = int(level_ends_all[0])
    all_data = all_idx[order[skip:]]

    # Convert level boundaries to start points and sizes of the equal level region
    level_sizes = (level_ends_all[1:] - level_starts_all[1:]).astype(np.int32)
    level_starts = np.zeros(max_level, dtype=np.int32)
    level_starts[1:] = np.cumsum(level_sizes[:-1])

    return {
        "all_data": all_data,
        "level_starts": level_starts,
        "level_sizes": level_sizes,
        "strides": strides,
        "D": D,
        "total_size": total_size,
        "num_levels": max_level,
    }


def _build_scan_schedule(
    level_starts: NDArray[np.integer],
    level_sizes: NDArray[np.integer],
    all_data: NDArray[np.integer],
    max_step_size: int | None,
) -> dict[str, Any]:
    """Partition the precomputed level geometry into scan steps.

    The ``max_step_size`` parameter controls padding waste. Smaller values reduce wasted
    gather/scatter on padded elements at the cost of more loop iterations. XLA aliases the carry
    buffer in-place, so additional steps do NOT incur carry-copy overhead. See the module docstring
    for details.

    Args:
        level_starts: Start positions in all_data for levels.
        level_sizes: Number of entries in all_data for levels.
        all_data: The multi-index data sorted by levels.
        max_step_size: The largest step size to construct.

    Returns:
        Dictionary of padded multi-index data with step starting positions and sizes.


    """
    D = all_data.shape[1]

    if max_step_size is None:
        step_starts = level_starts.copy()
        step_sizes = level_sizes.copy()
    else:
        n_subs = np.maximum(1, np.ceil(level_sizes / max_step_size).astype(np.int32))
        total_subs = int(n_subs.sum())

        expanded_starts = np.repeat(level_starts, n_subs)
        expanded_sizes = np.repeat(level_sizes, n_subs)

        cumulative = np.repeat(np.cumsum(n_subs) - n_subs, n_subs)
        local_idx = np.arange(total_subs) - cumulative

        step_starts = (expanded_starts + local_idx * max_step_size).astype(np.int32)
        step_sizes = np.minimum(expanded_sizes - local_idx * max_step_size, max_step_size).astype(
            np.int32
        )

    effective_max = int(max(step_sizes))

    pad_needed = effective_max
    all_data_padded = np.zeros((all_data.shape[0] + pad_needed, D), dtype=np.uint8)
    all_data_padded[: all_data.shape[0]] = all_data

    return {
        "all_data_padded": all_data_padded,
        "step_starts": np.asarray(step_starts, dtype=np.int32),
        "step_sizes": np.asarray(step_sizes, dtype=np.int32),
        "effective_max": effective_max,
        "num_steps": len(step_starts),
    }


# =============================================================================
# Recurrence Kernel
# =============================================================================


@overload
def _process_step_single_pivot(
    G: jax.Array,
    multi_idx: jax.Array,
    valid_mask: jax.Array,
    A: jax.Array,
    b: jax.Array,
    strides: jax.Array,
) -> jax.Array: ...


@overload
def _process_step_single_pivot(
    G: tuple[jax.Array, jax.Array],
    multi_idx: jax.Array,
    valid_mask: jax.Array,
    A: jax.Array,
    b: jax.Array,
    strides: jax.Array,
) -> tuple[jax.Array, jax.Array]: ...


def _process_step_single_pivot(
    G: jax.Array | tuple[jax.Array, jax.Array],
    multi_idx: jax.Array,
    valid_mask: jax.Array,
    A: jax.Array,
    b: jax.Array,
    strides: jax.Array,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Apply the Hermite recurrence, using a single pivot point.

    This kernel applies a step of the recurrence relationship at the index given by ``multi_idx``.
    The implementation is meant for applying many in parallel.  However, this only makes sense to do
    in parallel if the values of ``multi_idx`` are all in the same level, that is, they all have the
    same total value for the summation of the indices.  The code is written with this in mind and
    not doing so may result in errors. The code also expects the multi_idx values to all be unique
    except for any padding values indicated by the valid_mask. Violations of either of these
    requirements will result in undefined behaviour.

    This implementation is of the 'vanilla' type where only a single pivot contributes to the
    result.  The pivot is chosen to be the one closest in memory to the point ``multi_idx`` to
    assist with cache residency.

    Arrays ``G``, ``A`` and ``b`` use **batch-last layout**; the batch dimension B is the rightmost
    (innermost) dimension of these arrays. In row-major (C-contiguous) storage, the rightmost
    dimension varies fastest in memory. This means that gathering index ``j`` across all batch
    elements reads ``G[j, :]`` -- a contiguous ``B * 8`` byte chunk -- producing coalesced GPU
    memory transactions.

    **Pivot dimension choice.** The pivot dimension is chosen as the **last positive dimension** of
    each multi-index. In row-major layout with strides [c^{D-1}, ..., c, 1], this means the parent
    lookup is at distance strides[-1] = 1, yielding nearly-coalesced memory access.  This is the
    best choice for similar reasons to the batch-last approach.

    **Complex128 scatter workaround.** When the carry ``G`` is a tuple ``(G_re, G_im)`` of float64
    arrays (indicating complex128 mode), the function gathers from both arrays and recombines to
    complex128 for the arithmetic, then splits back to float64 only for the scatter writes.
    This avoids the XLA GPU bug where complex128 scatter-set compiles to a sequential while loop to
    ensure atomic updates (typically ~1000x slower than float64 scatter-set). All other operations
    (gathers, multiplies, reductions) run at full speed in native complex128. See the module-level
    comment for the full diagnosis.

    Args:
        G: (total_size, B) array, or tuple of two (total_size, B) float64 arrays.  Flat tensor with
            batch-last layout. When a tuple ``(G_re, G_im)``, the function operates in split-carry
            mode for complex128 on GPUs.
        multi_idx: (P, D) uint8. Multi-indices to compute (shared across all batch elements).
        valid_mask: (P,) bool. True for real elements, False for padding.
        A: (D, D, B) dtype. Data matrix in batch-last layout.
        b: (D, B) dtype. Recurrence vector in batch-last layout.
        strides: (D,) int32. Row-major strides for G to compute neighbours.

    Returns:
        Same type as ``G``: a single array, or a tuple ``(G_re, G_im)``.
    """
    # Detect split-carry mode (complex128 workaround).
    use_split = isinstance(G, tuple)
    real_dtype = G[0].dtype if use_split else jnp.real(G).dtype

    D = multi_idx.shape[-1]
    P = multi_idx.shape[0]

    # --- Index computation (no batch dependence -- shared across all B) ---
    flat_idxs = (multi_idx * strides).sum(axis=-1)  # (P,)
    is_padded = ~valid_mask  # (P,)
    real_mask = valid_mask.astype(real_dtype)  # (P,)

    # Pivot = last positive dimension. In row-major layout the last dimension has stride 1, so the
    # parent G[k - e_pivot] is at the adjacent memory location for the most common case.
    pivot_modes = D - 1 - jnp.argmax(multi_idx[:, ::-1] > 0, axis=-1)  # (P,)

    pivot_flats = flat_idxs - strides[pivot_modes]  # (P,)
    pivot_flats = jnp.where(is_padded, 0, pivot_flats)

    pivot_mask = (jnp.arange(D) == pivot_modes[:, None]).astype(jnp.uint8)
    pidx = multi_idx - pivot_mask  # (P, D)
    pidx = jnp.where(is_padded[:, None], jnp.zeros_like(pidx), pidx)

    nflats = pivot_flats[:, None] - strides[None, :]  # (P, D)
    nflats_clamped = jnp.clip(nflats, 0)  # (P, D)
    nsqrt = jnp.sqrt(pidx.astype(real_dtype))  # (P, D)

    P_range = jnp.arange(P)
    ki = pidx[P_range, pivot_modes] + 1  # (P,)
    inv_sk = 1.0 / jnp.sqrt(ki.astype(real_dtype))  # (P,)
    inv_sk = inv_sk * real_mask  # (P,)

    # --- Gathers from G: each reads a contiguous (B,) chunk ---
    # In split-carry mode, gather from float64 arrays and recombine to complex128. Gathers are fast
    # for all dtypes (no atomics needed for reads), so the recombination lets subsequent compute use
    # native complex128 multiply/reduce kernels.
    if use_split:
        pivot_vals = G[0][pivot_flats] + 1j * G[1][pivot_flats]  # (P, B)
        neighbor_vals = G[0][nflats_clamped] + 1j * G[1][nflats_clamped]  # (P, D, B)
    else:
        pivot_vals = G[pivot_flats]  # (P, B)
        neighbor_vals = G[nflats_clamped]  # (P, D, B)

    # --- b contribution: b is (D, B), indexing with pivot_modes gives (P, B) ---
    b_vals = b[pivot_modes]  # (P, B)
    contrib = b_vals * pivot_vals  # (P, B)

    # --- Neighbor contribution ---
    # A is (D, D, B), indexing first dim with pivot_modes gives (P, D, B). nsqrt is (P, D) --
    # broadcast over B via nsqrt[:, :, None]. neighbor_vals is (P, D, B). Element-wise multiply and
    # sum over D (axis=1) gives (P, B).
    A_rows = A[pivot_modes]  # (P, D, B)
    neighbor_contrib = jnp.sum(
        A_rows * nsqrt[:, :, None] * neighbor_vals,
        axis=1,  # (P, B)
    )

    values = (contrib + neighbor_contrib) * inv_sk[:, None]  # (P, B)

    # --- Scatter: each writes a contiguous (B,) chunk ---
    # In split-carry mode, split the complex128 result back to float64 for the scatter writes. This
    # is the ONLY reason for the split: XLA's complex128 scatter-set compiles to a sequential while
    # loop (~1000x slower on GPU), whereas float64 scatter-set uses a fast parallel GPU kernel.
    if use_split:
        current_re = G[0][flat_idxs]  # (P, B)
        current_im = G[1][flat_idxs]  # (P, B)
        safe_re = jnp.where(is_padded[:, None], current_re, values.real)
        safe_im = jnp.where(is_padded[:, None], current_im, values.imag)
        return (
            G[0].at[flat_idxs].set(safe_re, unique_indices=True),
            G[1].at[flat_idxs].set(safe_im, unique_indices=True),
        )

    current_vals = G[flat_idxs]  # (P, B)
    safe_values = jnp.where(is_padded[:, None], current_vals, values)
    return G.at[flat_idxs].set(safe_values, unique_indices=True)


@partial(jax.jit, static_argnums=(3, 4))
def _hermite_renormalized_core(
    A: jax.Array,
    b: jax.Array,
    c: ArrayLike,
    shape: tuple[int, ...],
    max_step_size: int | None = None,
) -> jax.Array:
    """Core (non-sharded) Hermite renormalized computation.

    This is the internal implementation that handles a single batch (or sub-batch when called from
    ``shard_map``). See ``hermite_renormalized`` for the full public API documentation.
    """
    A = jnp.asarray(A)
    b = jnp.asarray(b)
    c = jnp.asarray(c)
    out_dtype = jnp.result_type(A, b, c)

    D = len(shape)

    # Compute the batch shape from A's leading dimensions. A's last two axes are (D, D); everything
    # before that is batch.
    batch_shape = jnp.broadcast_shapes(A.shape[:-2], b.shape[:-1], c.shape)
    A = jnp.broadcast_to(A, batch_shape + A.shape[-2:])
    b = jnp.broadcast_to(b, batch_shape + b.shape[-1:])
    c = jnp.broadcast_to(c, batch_shape)

    # Flatten all batch dimensions into a single axis B. For unbatched inputs (batch_shape = ()), B
    # = 1 and a singleton dimension is added. This allows a single code path for all cases.
    A_flat = A.reshape(-1, D, D)  # (..., D, D) -> (B, D, D)
    b_flat = b.reshape(-1, D)  # (..., D)    -> (B, D)
    c_flat = c.reshape(-1)  # (...)       -> (B,)
    B = c_flat.shape[0]

    # Transpose to batch-last layout ONCE at entry. Batch-last means the batch axis is rightmost
    # (innermost in row-major), so gathering a single index j reads G[j, :] -- a contiguous B*8 byte
    # chunk. Similarly, A[pivot_modes] yields (P, D, B) with B contiguous, and b[pivot_modes] yields
    # (P, B) with B contiguous. See the module docstring "Batch-Last Memory Layout" for full
    # rationale.
    A_bl = jnp.moveaxis(A_flat, 0, -1)  # (B, D, D) -> (D, D, B)
    b_bl = jnp.moveaxis(b_flat, 0, -1)  # (B, D)    -> (D, B)

    total_size = np.prod(shape)

    # --- Trivial cases ---
    if total_size == 1:
        G = jnp.zeros((1, B), dtype=out_dtype)
        G = G.at[0, :].set(c_flat)
        # (c) Transpose back and reshape to (..., *shape)
        return G.T.reshape(*batch_shape, *shape)

    geom = _precompute_level_scan(shape)

    # If there is only one level, there is no recurrence to compute.
    if geom is None:
        G = jnp.zeros((total_size, B), dtype=out_dtype)
        G = G.at[0, :].set(c_flat)
        return G.T.reshape(*batch_shape, *shape)

    # --- Build scan schedule ---
    schedule = _build_scan_schedule(
        geom["level_starts"], geom["level_sizes"], geom["all_data"], max_step_size
    )

    EMS = schedule["effective_max"]

    j_all_data = jnp.array(schedule["all_data_padded"])
    j_step_starts = jnp.array(schedule["step_starts"])
    j_step_sizes = jnp.array(schedule["step_sizes"])
    j_strides = jnp.array(geom["strides"])

    idx_range = jnp.arange(EMS, dtype=jnp.int32)

    # --- Initialise the scan carry ---
    # For complex128 on GPU, the carry is a tuple (G_re, G_im) of float64 arrays to avoid the
    # serialized scatter-set bug. Otherwise, it is a single array. _process_step_single_pivot
    # detects which mode to operate in via isinstance(G, tuple). See the module-level comment for
    # full details.
    if (out_dtype == jnp.complex128) and (jax.default_backend() != "cpu"):
        G_re = jnp.zeros((total_size, B), dtype=jnp.float64)
        G_im = jnp.zeros((total_size, B), dtype=jnp.float64)
        G_re = G_re.at[0, :].set(c_flat.real)
        G_im = G_im.at[0, :].set(c_flat.imag)
        G_init = (G_re, G_im)
    else:
        G_init = jnp.zeros((total_size, B), dtype=out_dtype)
        G_init = G_init.at[0, :].set(c_flat)

    def scan_step(G, step_info):
        start, size = step_info
        multi_idx = lax.dynamic_slice(j_all_data, (start, jnp.int32(0)), (EMS, D))
        valid_mask = idx_range < size
        G = _process_step_single_pivot(G, multi_idx, valid_mask, A_bl, b_bl, j_strides)
        return G, None

    G, _ = lax.scan(scan_step, G_init, (j_step_starts, j_step_sizes))

    # --- Recombine (split path) and reshape to output ---
    # G is (total_size, B) or a tuple of two such arrays. Transpose to batch-first (B, total_size)
    # then reshape to (*batch_shape, *shape) to restore the original batch structure. For unbatched
    # inputs (batch_shape = ()), this is just (*shape).
    if isinstance(G, tuple):
        G_re, G_im = G
        G = (G_re + 1j * G_im).astype(out_dtype)

    return G.T.reshape(*batch_shape, *shape)


@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def _sharded_hermite_renormalized(
    A: jax.Array,
    b: jax.Array,
    c: jax.Array,
    shape: tuple[int, ...],
    max_step_size: int | None,
    batch_shape: tuple[int, ...],
    n_shards: int,
) -> jax.Array:
    """Distribute the batch across multiple devices via shard_map.

    Each device independently runs the full level-scan recurrence on its sub-batch,
    with zero cross-device communication.
    """
    D = len(shape)

    # Flatten to 1D batch for even sharding across devices.
    A_flat = jnp.broadcast_to(A, (*batch_shape, D, D)).reshape(-1, D, D)
    b_flat = jnp.broadcast_to(b, (*batch_shape, D)).reshape(-1, D)
    c_flat = jnp.broadcast_to(c, batch_shape).reshape(-1)

    devices = np.array(jax.devices()[:n_shards])
    mesh = Mesh(devices, axis_names=("dev",))
    sharding = NamedSharding(mesh, P("dev"))

    A_s = jax.device_put(A_flat, sharding)
    b_s = jax.device_put(b_flat, sharding)
    c_s = jax.device_put(c_flat, sharding)

    G = shard_map(
        lambda a, b_, c_: _hermite_renormalized_core(a, b_, c_, shape, max_step_size),
        mesh=mesh,
        in_specs=(P("dev"), P("dev"), P("dev")),
        out_specs=P("dev"),
    )(A_s, b_s, c_s)

    return G.reshape(*batch_shape, *shape)


# =============================================================================
# Auto Sub-Batch Size
# =============================================================================


def _auto_sub_batch_size(batch_size: int) -> int | None:
    """Auto-select sub-batch granularity based on backend and available devices.

    Args:
        batch_size : Total number of batch elements.

    Returns:
        The sub-batch size per shard, or ``None`` to disable sharding.

    Notes:
        - **CPU**: returns 4 (optimal for SIMD data coalescing on x86 SSE/AVX).
        - **Multi-GPU**: returns ``batch_size // n_gpus`` to split evenly across all GPUs.
        - **Single GPU**: returns ``None`` (no sharding -- GPU parallelism is handled at the
          warp/thread level within a single device).
    """
    backend = jax.default_backend()
    n_devices = len(jax.devices())

    if backend == "cpu":
        return 4  # SIMD coalescing on x86 (SSE/AVX)

    if n_devices > 1:
        # Multi-GPU: split evenly across all GPUs
        sub = batch_size // n_devices
        return max(sub, 1)

    # Single GPU: no splitting, full batch in one shot
    return None


# =============================================================================
# Public API
# =============================================================================


def hermite_renormalized(
    A: jax.Array,
    b: jax.Array,
    c: ArrayLike,
    shape: tuple[int, ...],
    max_step_size: int | None = -1,
    sub_batch_size: int | None = -1,
) -> jax.Array:
    """Compute the renormalized multidimensional Hermite polynomial tensor.

    This currently only uses the "vanilla" implementation with a single pivot contribution.

    Supports arbitrary batch dimensions with shape (...). (Potentially empty).

    When sharding is applicable (multiple devices and sufficient batch size), the batch dimension is
    automatically distributed across devices for parallel execution with zero cross-device
    communication. The ``sub_batch_size`` parameter controls how the batch is partitioned.

    This implementation uses uint8 for indexing the output array and hence the maximum size that can
    be requested for any dimension within shape is 255.

    **CPU setup (for multi-device sharding)** If a multi-device setup is to be simulated on CPU,
    virtual CPU devices must be created before importing JAX.  This can be done by setting the
    environment variable::

        XLA_FLAGS="--xla_force_host_platform_device_count=N"

    Here, ``N`` is the number of devices.

    Args:
        A : Data matrix. Shape (..., D, D).
        b : Data vector. Shape (..., D).
        c : Normalisation constant. Shape (...).
        shape : [static] Output shape per element, e.g. ``(cutoff,) * D``.  Maximum index size 255.
        max_step_size : [static] Controls padding waste in the scan loop. Smaller values reduce
            wasted gather/scatter on padded elements but increase the number of loop iterations
            (with negligible per-step overhead since XLA aliases the carry buffer in-place). None
            means no maximum.  -1 means backend default: None for GPU and 2048 for CPU. See module
            docstring for tuning guidance.
        sub_batch_size : [static] Controls how the batch dimension is partitioned across devices for
            parallel execution. The number of shards is derived as ``ceil(batch_size /
            sub_batch_size)``, clamped to the available device count.

            - ``-1`` (default): auto-detect based on backend. On CPU, uses 4 (optimal for SIMD data
              coalescing on x86 SSE/AVX). On a single GPU, disables sharding (GPU parallelism is
              handled at the warp/thread level). On multiple GPUs, splits evenly across all devices.
            - ``None``: disable sharding entirely (run the full batch on a single device).
            - Explicit ``int``: user override for tuning. E.g., ``sub_batch_size=8`` with a batch of
              24 yields 3 shards (one per device if 3+ devices are available).

    Returns:
        G : Matrix of coefficients.  Shape (..., *shape).

    Notes:
        **Internal batch handling.** Regardless of input shape, the function:

        (a) Computes the batch shape from A, b and c.
        (b) Flattens all batch dimensions into a single axis B.
        (c) Transposes to batch-_last_ layout: ``G`` is ``(total_size, B)``, ``A`` is ``(D, D, B)``,
            ``b`` is ``(D, B)``.
        (d) Runs the recurrence via ``lax.scan`` with memory contiguous batch-last gathers and scatters.
        (e) Transposes back to batch-first and reshapes to ``(..., *shape)``.

        This ensures that every gather/scatter inside the scan loop reads/writes contiguous memory along
        the batch dimension, which is critical for performance to avoid cache thrashing. Using
        ``jax.vmap`` would produce a batch-first layout for the scan carry, causing strided
        (non-coalesced) memory access and generates extra transpose+copy operations per scan step. See
        the module docstring for the full analysis with HLO verification.

        **Multi-device sharding.** When ``sub_batch_size`` enables sharding (and multiple devices are
        available), the batch is distributed across devices via ``shard_map``. Each device independently
        runs the full level-scan recurrence on its sub-batch, with zero cross-device communication.
        On CPU with 12 cores, typical speedups are 1.6--2.3x for batch sizes of 12--24.

        **Gradient support**: compatible with ``jax.grad`` for all input shapes.

    Examples:
        Single call::

            G = hermite_renormalized(A, b, c, shape=(20, 20, 20))

        Batched call (8 independent problems)::

            G = hermite_renormalized(A_batch, b_batch, c_batch, shape=(20, 20, 20))
            # A_batch.shape == (8, 3, 3), b_batch.shape == (8, 3), c_batch.shape == (8,)
            # G.shape == (8, 20, 20, 20)

        Multi-dimensional batch::

            G = hermite_renormalized(A_grid, b_grid, c_grid, shape=(10, 10))
            # A_grid.shape == (4, 5, 2, 2), b_grid.shape == (4, 5, 2), c_grid.shape == (4, 5)
            # G.shape == (4, 5, 10, 10)

        Multi-device sharding (auto)::

            # Batch of 12, auto-selects sub_batch_size=4 on CPU (3 devices)
            G = hermite_renormalized(A_batch, b_batch, c_batch, (80, 80, 80))

        Explicit sub-batch size::

            G = hermite_renormalized(A_batch, b_batch, c_batch, (80, 80, 80), sub_batch_size=8)
    """
    if max_step_size == -1:
        max_step_size = 2048 if jax.default_backend() == "cpu" else None

    A = jnp.asarray(A)
    b = jnp.asarray(b)
    c = jnp.asarray(c)

    # Compute the batch shape from A's leading dimensions.
    batch_shape = jnp.broadcast_shapes(A.shape[:-2], b.shape[:-1], c.shape)
    batch_size = int(np.prod(batch_shape)) if batch_shape else 1

    # --- Resolve sub_batch_size and determine sharding ---
    if sub_batch_size == -1:
        sub_batch_size = _auto_sub_batch_size(batch_size) if batch_size > 1 else None

    n_available = len(jax.devices())

    if sub_batch_size is not None and sub_batch_size > 0 and batch_size > 1 and n_available > 1:
        n_shards = min(n_available, max(1, -(-batch_size // sub_batch_size)))  # ceil division
        # Ensure batch_size is evenly divisible by n_shards
        while batch_size % n_shards != 0 and n_shards > 1:
            n_shards -= 1

        if n_shards > 1:
            return _sharded_hermite_renormalized(
                A, b, c, shape, max_step_size, batch_shape, n_shards
            )

    # --- Fall through to non-sharded computation ---
    return _hermite_renormalized_core(A, b, c, shape, max_step_size)
