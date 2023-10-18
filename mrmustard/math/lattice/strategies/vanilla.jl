using MultiFloats
T = Float64x4
SQRT = [0.0; [sqrt(T(i)) for i in 1:100000]]

function vanilla(
    A::AbstractMatrix{Complex{Float64}},
    b::AbstractVector{Complex{Float64}},
    c::Complex{Float64},
    shape::AbstractVector{Int64}
    )
    """Vanilla Fock-Bargmann strategy. Fills the tensor by iterating over all indices
    in ndindex (i.e. CartesianIndices) order.
    Both the input and output of this function have dtype Complex{Float64},
    but Complex{Float64x4} is used intermediately to postpone the nummerical blowup
    that results from the instable recurrence relation.

    Args:
        A: A matrix of the Fock-Bargmann representation
        b: B vector of the Fock-Bargmann representation
        c: vacuum amplitude
        shape: shape of the output tensor

    Returns:
        Array{Complex{Float64}}: Fock representation of the Gaussian tensor with shape ``shape``
    """

    shape = Tuple(shape)

    path = CartesianIndices(shape)

    G = Array{Complex{T}}(undef,shape) # initialize empty array with high precision values
    G[first(path)] = c

    for idx in Iterators.drop(path, 1)
        update_Fock_array!(G, A, b, idx)
    end

    return Complex{Float64}.(G) # convert back to lower precision
end

function update_Fock_array!(
    G::AbstractArray{Complex{T}},
    A::AbstractMatrix{Complex{Float64}},
    b::AbstractVector{Complex{Float64}},
    idx::CartesianIndex,
    )

    i, pivot = get_pivot(idx)

    @views temp = b[i] * G[pivot]
    @inbounds for (j, neighbor) in get_neighbors(pivot)
        @views temp += A[i, j] * G[neighbor] * SQRT[pivot[j]]
    end

    G[idx] = temp / SQRT[idx[i]]
end

function get_pivot(idx::CartesianIndex)
    """returns a single idx where the first non-one value of idx has been lowered.
    E.g. (1,1,5,3) -> (1,1,4,3)"""
    @inbounds @simd for i in 1:length(idx)
        if idx[i] > 1
            return i, CartesianIndex(ntuple(j -> j == i ? idx[j] - 1 : idx[j], length(idx)))
        end
    end
    return 1, idx
end

function get_neighbors(idx::CartesianIndex{N}) where {N}
    """Does the same as get_neighbors, but skips over the first "start" iterations"""
    return ((i, CartesianIndex(ntuple(d -> d == i ? idx[d] - 1 : idx[d], Val(N))))
            for i in 1:N if idx[i] > 1)
end