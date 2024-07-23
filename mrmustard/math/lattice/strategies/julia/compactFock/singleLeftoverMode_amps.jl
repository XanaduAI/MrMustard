module LeftoverModeAmps

import ..GetPrecision
import ..CompactFock_HelperFunctions

function write_block!(i, arr_write, write, arr_read_pivot, read_GB, G_in, GB, A, K_i, cutoff_leftoverMode, SQRT)
    """
    Apply the recurrence relation to blocks of Fock amplitudes (of shape cutoff_leftoverMode x cutoff_leftoverMode)
    (cfr. algorithm 2 of https://doi.org/10.22331/q-2023-08-29-1097)
    """
    m, n = 1, 1
    A_adapted = A[i, 3:end]
    G_in_adapted = G_in[1, 1, :]
    arr_write[1, 1, write...] = (GB[1, 1, i] + sum(A_adapted .* G_in_adapted)) / K_i[i - 2]

    m = 1
    A_adapted = A[i, 2:end]
    for n in 2:cutoff_leftoverMode
        G_in_adapted = vcat(arr_read_pivot[1, n - 1, read_GB...] * SQRT[n], G_in[1, n, :])
        arr_write[1, n, write...] = (GB[1, n, i] + sum(A_adapted .* G_in_adapted)) / K_i[i - 2]
    end

    n = 1
    A_adapted = vcat(A[i, 1], A[i, 3:end])
    for m in 2:cutoff_leftoverMode
        G_in_adapted = vcat(arr_read_pivot[m - 1, 1, read_GB...] * SQRT[m], G_in[m, 1, :])
        arr_write[m, 1, write...] = (GB[m, 1, i] + sum(A_adapted .* G_in_adapted)) / K_i[i - 2]
    end

    A_adapted = A[i, :]
    for m in 2:cutoff_leftoverMode
        for n in 2:cutoff_leftoverMode
            G_in_adapted = vcat(arr_read_pivot[m - 1, n, read_GB...] * SQRT[m], arr_read_pivot[m, n - 1, read_GB...] * SQRT[n], G_in[m, n, :])
            arr_write[m, n, write...] = (GB[m, n, i] + sum(A_adapted .* G_in_adapted)) / K_i[i - 2]
        end
    end
end

function use_offDiag_pivot!(A, B, M, cutoff_leftoverMode, cutoffs_tail, params, d, arr0, arr2, arr1010, arr1001, arr1, T, SQRT)
    """Given params=(a,b,c,...), apply the recurrence relation for the pivots
    [a+1,a,b,b,c,c,...] / [a,a,b+1,b,c,c,...] / [a,a,b,b,c+1,c,...] / ..."""
    pivot = CompactFock_HelperFunctions.repeat_twice(params)
    pivot[2 * d - 1] += 1
    K_l = SQRT[pivot] # julia indexing counters extra zero in SQRT
    K_i = SQRT[pivot .+ 1] # julia indexing counters extra zero in SQRT
    G_in = zeros(Complex{T}, cutoff_leftoverMode, cutoff_leftoverMode, 2 * M)

    ########## READ ##########
    read_GB = tuple(2 * d - 1, params...)
    GB = zeros(Complex{T}, cutoff_leftoverMode, cutoff_leftoverMode, length(B))
    for m in 1:cutoff_leftoverMode
        for n in 1:cutoff_leftoverMode
            GB[m, n, :] = arr1[m, n, read_GB...] .* B
        end
    end

    # Array0
    G_in[:, :, 2 * d - 1] .= arr0[:, :, params...]


    # read from Array2
    if params[d] > 1
        params_adapted = collect(params)
        params_adapted[d] -= 1
        G_in[:, :, 2 * d] .= arr2[:, :, d, params_adapted...] # read block
    end

    # read from Array11
    for i in d+1:M  # i>d
        if params[i] > 1
            params_adapted = collect(params)
            params_adapted[i] -= 1
            G_in[:, :, 2 * i - 1] .= arr1001[:, :, d, i - d, params_adapted...] # read block
            G_in[:, :, 2 * i] .= arr1010[:, :, d, i - d, params_adapted...] # read block
        end
    end

    ########## WRITE ##########
    for m in 1:cutoff_leftoverMode
        for n in 1:cutoff_leftoverMode
            G_in[m, n, :] .*= K_l
        end
    end

    # Array0
    write = collect(params)
    write[d] += 1
    write_block!(2 * d + 2, arr0, write, arr1, read_GB, G_in, GB, A, K_i, cutoff_leftoverMode, SQRT)

    # Array2
    if params[d] + 1 < cutoffs_tail[d]
        write = (d, params...)
        write_block!(2 * d + 1, arr2, write, arr1, read_GB, G_in, GB, A, K_i, cutoff_leftoverMode, SQRT)
    end

    # Array11
    for i in d+1:M
        if params[i] < cutoffs_tail[i]
            write = (d, i - d, params...)
            write_block!(2 * i + 1, arr1010, write, arr1, read_GB, G_in, GB, A, K_i, cutoff_leftoverMode, SQRT)
            write_block!(2 * i + 2, arr1001, write, arr1, read_GB, G_in, GB, A, K_i, cutoff_leftoverMode, SQRT)
        end
    end
end

function use_diag_pivot!(A, B, M, cutoff_leftoverMode, cutoffs_tail, params, arr0, arr1, T, SQRT)
    """Given params=(a,b,c,...), apply the recurrence relation for the pivot [a,a,b,b,c,c...]"""
    pivot = CompactFock_HelperFunctions.repeat_twice(params)
    K_l = SQRT[pivot] # julia indexing counters extra zero in SQRT
    K_i = SQRT[pivot .+ 1] # julia indexing counters extra zero in SQRT
    G_in = zeros(Complex{T}, cutoff_leftoverMode, cutoff_leftoverMode, 2*M)

    ########## READ ##########
    read_GB = params
    GB = zeros(Complex{T}, cutoff_leftoverMode, cutoff_leftoverMode, length(B))
    for m in 1:cutoff_leftoverMode
        for n in 1:cutoff_leftoverMode
            GB[m, n, :] = arr0[m, n, read_GB...] .* B
        end
    end

    # Array1
    for i in 1:2*M
        if params[(i-1)÷2+1] > 1
            params_adapted = collect(params)
            params_adapted[(i-1)÷2+1] -= 1
            G_in[:, :, i] .= arr1[:, :, i+1-2*((i-1) % 2), params_adapted...] # read block
        end
    end

    ########## WRITE ##########
    for m in 1:cutoff_leftoverMode
        for n in 1:cutoff_leftoverMode
            G_in[m, n, :] .*= K_l
        end
    end

    # Array1
    for i in 1:2*M
        if params[(i-1)÷2+1] < cutoffs_tail[(i-1)÷2+1]
            # this if statement prevents a few elements from being written that will never be read
            if i ≠ 2 || params[1] + 1 < cutoffs_tail[1]
                write = tuple(i,params...)
                write_block!(i + 2, arr1, write, arr0, read_GB, G_in, GB, A, K_i, cutoff_leftoverMode, SQRT)
            end
        end
    end
end

function fill_firstMode_PNRzero!(arr0,A,B,M,cutoff_leftoverMode,SQRT)
    # fill first mode for all PNR detections equal to zero
    one_tuple = tuple(fill(1,M-1)...)
    for m in 1:cutoff_leftoverMode-1
        arr0[m + 1, 1, one_tuple...] = (arr0[m, 1, one_tuple...] * B[1]) / SQRT[m + 1]
        if m != 1
            arr0[m + 1, 1, one_tuple...] += (SQRT[m] * A[1, 1] * arr0[m - 1, 1, one_tuple...]) / SQRT[m + 1]
        end
    end
    for m in 1:cutoff_leftoverMode
        for n in 1:cutoff_leftoverMode-1
            arr0[m, n + 1, one_tuple...] = (arr0[m, n, one_tuple...] * B[2]) / SQRT[n + 1]
            if m != 1
                arr0[m, n + 1, one_tuple...] += (SQRT[m] * A[2, 1] * arr0[m - 1, n, one_tuple...]) / SQRT[n + 1]
            end
            if n != 1
                arr0[m, n + 1, one_tuple...] += (SQRT[n] * A[2, 2] * arr0[m, n - 1, one_tuple...]) / SQRT[n + 1]
            end
        end
    end
end

function fock_1leftoverMode_amps(
    A::AbstractMatrix{Complex{Float64}},
    B::AbstractVector{Complex{Float64}},
    G0::Complex{Float64},
    cutoffs::Tuple,
    precision_bits::Int64
    )
    """Returns the density matrices in the upper, undetected mode of a circuit when all other modes are PNR detected
    according to algorithm 2 of https://doi.org/10.22331/q-2023-08-29-1097
    Args:
        A, B, G0: required input for recurrence relation
        cutoffs: upper bounds for the number of photons in each mode
        precision_bits: number of bits used to represent a single Fock amplitude
    Returns:
        Submatrices of the Fock representation. Each submatrix contains Fock indices of a certain type.
        arr0 --> type: [a,a,b,b,c,c...]
        arr2 --> type: [a+2,a,b,b,c,c...] / [a,a,b+2,b,c,c...] / ...
        arr1010 --> type: [a+1,a,b+1,b,c,c,...] / [a+1,a,b,b,c+1,c,...] / [a,a,b+1,b,c+1,c,...] / ...
        arr1001 --> type: [a+1,a,b,b+1,c,c,...] / [a+1,a,b,b,c,c+1,...] / [a,a,b+1,b,c,c+1,...] / ...
        arr1 --> type: [a+1,a,b,b,c,c...] / [a,a+1,b,b,c,c...] / [a,a,b+1,b,c,c...] / ...
    """

    T = GetPrecision.get_dtype(precision_bits)
    SQRT = GetPrecision.SQRT_dict[precision_bits]

    M = length(cutoffs)
    cutoff_leftoverMode = cutoffs[1]
    cutoffs_tail = cutoffs[2:end]

    arr0 = zeros(Complex{T}, cutoff_leftoverMode, cutoff_leftoverMode, cutoffs_tail...)
    arr0[fill(1,M+1)...] = G0
    arr2 = zeros(Complex{T}, cutoff_leftoverMode, cutoff_leftoverMode, M - 1, cutoffs_tail...)
    arr1 = zeros(Complex{T}, cutoff_leftoverMode, cutoff_leftoverMode, 2 * (M - 1), cutoffs_tail...)
    arr1010 = zeros(Complex{T}, cutoff_leftoverMode, cutoff_leftoverMode, M - 1, M - 2, cutoffs_tail...)
    arr1001 = zeros(Complex{T}, cutoff_leftoverMode, cutoff_leftoverMode, M - 1, M - 2, cutoffs_tail...)


    # fill first mode for all PNR detections equal to zero
    fill_firstMode_PNRzero!(arr0,A,B,M,cutoff_leftoverMode,SQRT)

    dict_params = CompactFock_HelperFunctions.construct_dict_params(cutoffs_tail)
    for sum_params in 0:sum(cutoffs_tail)-1
        for params in dict_params[sum_params]
            # diagonal pivots: aa,bb,cc,...
            if (cutoffs_tail[1] == 1) || (params[1] < cutoffs_tail[1]) # julia indexing!
                use_diag_pivot!(A, B, M - 1, cutoff_leftoverMode, cutoffs_tail, params, arr0, arr1, T, SQRT)
            end
            # off-diagonal pivots: d=1: (a+1)a,bb,cc,... | d=2: 00,(b+1)b,cc,... | d=3: 00,00,(c+1)c,... | ...
            for d in 1:M - 1
                if all(params[1:d-1] .== 1) && (params[d] < cutoffs_tail[d])
                    use_offDiag_pivot!(A,B,M - 1,cutoff_leftoverMode,cutoffs_tail,params,d,arr0,arr2,arr1010,arr1001,arr1,T,SQRT)
                end
            end
        end
    end

    return Complex{Float64}.(arr0), Complex{Float64}.(arr2), Complex{Float64}.(arr1010), Complex{Float64}.(arr1001), Complex{Float64}.(arr1)
end

end # end module
