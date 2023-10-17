module HelperFunctions

using MultiFloats

const dtypes_dict = Dict(128 => Float64, 256 => Float64x2, 384 => Float64x3, 512 => Float64x4)
const SQRT_dict = Dict(k => [0;sqrt.(dtypes_dict[k].(1:999))] for k in keys(dtypes_dict))
function get_dtype(precision_bits)
    if !(precision_bits in keys(dtypes_dict))
        error("The possible values for precision_bits are ", keys(dtypes))
    end
    return dtypes_dict[precision_bits]
end

function repeat_twice(params)
    pivot = Vector{Int64}(undef, 2 * length(params))
    for (i, val) in enumerate(params)
        pivot[2 * i - 1] = val
        pivot[2 * i] = val
    end
    return pivot
end

function construct_dict_params(cutoffs)
    M = length(cutoffs)
    indices = Dict{Int64, Vector{Tuple}}()
    for sum_params in 0:sum(cutoffs)-1
        indices[sum_params] = Vector{Tuple}()
    end

    for params in CartesianIndices(cutoffs)
        params_tup = Tuple(params)
        push!(indices[sum(params_tup) - M], params_tup)
    end
    return indices
end

end # end module