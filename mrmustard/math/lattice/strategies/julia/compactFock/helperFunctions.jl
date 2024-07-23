module CompactFock_HelperFunctions

function repeat_twice(params)
    """
    Args:
        params: [a,b,c,...]
    Returns:
        [a,a,b,b,c,c,...]
    """
    pivot = Vector{Int64}(undef, 2 * length(params))
    for (i, val) in enumerate(params)
        pivot[2 * i - 1] = val
        pivot[2 * i] = val
    end
    return pivot
end

function construct_dict_params(cutoffs)
    """
    Args:
        cutoffs (tuple): upper bounds for the number of photons in each mode
    Returns:
        Dict: all possible values for (a,b,c,...), grouped in lists according to their sum a+b+c+...
    """
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
