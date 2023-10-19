module LeftoverModeGrad

import ..GetPrecision
import ..CompactFock_HelperFunctions

function calc_dA_dB(m, n, i, arr_read_pivot, read_GB, G_in_adapted, A_adapted, B, K_i, K_l_adapted, arr_read_pivot_dA, G_in_dA_adapted, arr_read_pivot_dB, G_in_dB_adapted, l_range)
    """Apply eqs. 16 & 17 (of https://doi.org/10.22331/q-2023-08-29-1097) for a single Fock amplitude"""
    dA = arr_read_pivot_dA[m, n, read_GB...,:,:] .* B[i]
    dB = arr_read_pivot_dB[m, n, read_GB...,:] .* B[i]
    dB[i] += arr_read_pivot[m, n, read_GB...]
    for (l_prime, l) in enumerate(l_range)
        dA += (K_l_adapted[l_prime] * A_adapted[l_prime]) .* G_in_dA_adapted[l_prime,:,:]
        dB += (K_l_adapted[l_prime] * A_adapted[l_prime]) .* G_in_dB_adapted[l_prime,:]
        dA[i, l] += G_in_adapted[l_prime]
    end
    return dA ./ K_i[i - 2], dB ./ K_i[i - 2]
end

function write_block_grad(i, write, arr_read_pivot, read_GB, G_in, A, B, K_i, K_l, cutoff_leftoverMode, arr_write_dA, arr_read_pivot_dA, G_in_dA, arr_write_dB, arr_read_pivot_dB, G_in_dB, SQRT)
    """
    Apply eqs. 16 & 17 (of https://doi.org/10.22331/q-2023-08-29-1097)
    to blocks of Fock amplitudes (of shape cutoff_leftoverMode x cutoff_leftoverMode)
    """
    m, n = 1, 1
    l_range = 3:size(A)[2]
    A_adapted = A[i, 3:end]
    G_in_adapted = G_in[1, 1, :]
    G_in_dA_adapted = G_in_dA[1,1,:,:,:]
    G_in_dB_adapted = G_in_dB[1,1,:,:]
    K_l_adapted = K_l
    arr_write_dA[1, 1, write...,:,:], arr_write_dB[1, 1, write...,:] = calc_dA_dB(m, n, i, arr_read_pivot, read_GB, G_in_adapted, A_adapted, B, K_i, K_l_adapted, arr_read_pivot_dA, G_in_dA_adapted, arr_read_pivot_dB, G_in_dB_adapted, l_range)

    m = 1
    l_range = 2:size(A)[2]
    A_adapted = A[i, 2:end]
    for n in 2:cutoff_leftoverMode
        K_l_adapted = vcat(SQRT[n], K_l)
        G_in_adapted = vcat(arr_read_pivot[1, n - 1, read_GB...] * SQRT[n], G_in[1, n, :])
        G_in_dA_adapted = vcat(reshape(arr_read_pivot_dA[1, n - 1, read_GB...,:,:], (1, size(A)...)), G_in_dA[1,n,:,:,:])
        G_in_dB_adapted = vcat(reshape(arr_read_pivot_dB[1, n - 1, read_GB...,:], (1, size(B)...)), G_in_dB[1,n,:,:])
        arr_write_dA[1, n, write...,:,:], arr_write_dB[1, n, write...,:] = calc_dA_dB(m, n, i, arr_read_pivot, read_GB, G_in_adapted, A_adapted, B, K_i, K_l_adapted, arr_read_pivot_dA, G_in_dA_adapted, arr_read_pivot_dB, G_in_dB_adapted, l_range)
    end

    n = 1
    l_range = [2:size(A)[2]...]
    l_range[1] = 1
    A_adapted = vcat(A[i, 1], A[i, 3:end])
    for m in 2:cutoff_leftoverMode
        K_l_adapted = vcat(SQRT[m], K_l)
        G_in_adapted = vcat(arr_read_pivot[m - 1, 1, read_GB...] * SQRT[m], G_in[m, 1, :])
        G_in_dA_adapted = vcat(reshape(arr_read_pivot_dA[m - 1, 1, read_GB...,:,:], (1, size(A)...)), G_in_dA[m, 1,:,:,:])
        G_in_dB_adapted = vcat(reshape(arr_read_pivot_dB[m - 1, 1, read_GB...,:], (1, size(B)...)), G_in_dB[m, 1,:,:])
        arr_write_dA[m, 1, write...,:,:], arr_write_dB[m, 1, write...,:] = calc_dA_dB(m, n, i, arr_read_pivot, read_GB, G_in_adapted, A_adapted, B, K_i, K_l_adapted, arr_read_pivot_dA, G_in_dA_adapted, arr_read_pivot_dB, G_in_dB_adapted, l_range)
    end

    l_range = 1:size(A)[2]
    A_adapted = A[i, :]
    for m in 2:cutoff_leftoverMode
        for n in 2:cutoff_leftoverMode
            K_l_adapted = vcat(SQRT[m], SQRT[n], K_l)
            G_in_adapted = vcat(arr_read_pivot[m - 1, n, read_GB...] * SQRT[m], arr_read_pivot[m, n - 1, read_GB...] * SQRT[n], G_in[m, n, :])
            G_in_dA_adapted = vcat(reshape(arr_read_pivot_dA[m - 1, n, read_GB...,:,:], (1, size(A)...)), reshape(arr_read_pivot_dA[m, n - 1, read_GB...,:,:], (1, size(A)...)), G_in_dA[m, n,:,:,:])
            G_in_dB_adapted = vcat(reshape(arr_read_pivot_dB[m - 1, n, read_GB...,:], (1, size(B)...)), reshape(arr_read_pivot_dB[m, n - 1, read_GB...,:], (1, size(B)...)), G_in_dB[m, n,:,:])
            arr_write_dA[m, n, write...,:,:], arr_write_dB[m, n, write...,:] = calc_dA_dB(m, n, i, arr_read_pivot, read_GB, G_in_adapted, A_adapted, B, K_i, K_l_adapted, arr_read_pivot_dA, G_in_dA_adapted, arr_read_pivot_dB, G_in_dB_adapted, l_range)
        end
    end
    return arr_write_dA, arr_write_dB
end

function read_block(arr_write, arr_write_dA, arr_write_dB, idx_write, arr_read, arr_read_dA, arr_read_dB, idx_read_tail)
    """
    Read the blocks of Fock amplitudes (of shape cutoff_leftoverMode x cutoff_leftoverMode)
    and their derivatives w.r.t A and B and write them to G_in, G_in_dA, G_in_dB
    """
    arr_write[:, :, idx_write] .= arr_read[:, :, idx_read_tail...]
    arr_write_dA[:, :, idx_write, :, :] .= arr_read_dA[:, :, idx_read_tail...,:, :]
    arr_write_dB[:, :, idx_write, :] .= arr_read_dB[:, :, idx_read_tail...,:]
    return arr_write, arr_write_dA, arr_write_dB
end


function use_offDiag_pivot_grad!(A, B, M, cutoff_leftoverMode, cutoffs_tail, params, d, arr0, arr2, arr1010, arr1001, arr1, arr0_dA, arr2_dA, arr1010_dA, arr1001_dA, arr1_dA, arr0_dB, arr2_dB, arr1010_dB, arr1001_dB, arr1_dB, T, SQRT)
    """Given params=(a,b,c,...), apply the eqs. 16 & 17 (of https://doi.org/10.22331/q-2023-08-29-1097)
    for the pivots [a+1,a,b,b,c,c,...] / [a,a,b+1,b,c,c,...] / [a,a,b,b,c+1,c,...] / ..."""
    pivot = CompactFock_HelperFunctions.repeat_twice(params)
    pivot[2 * d - 1] += 1
    K_l = SQRT[pivot] # julia indexing counters extra zero in SQRT
    K_i = SQRT[pivot .+ 1] # julia indexing counters extra zero in SQRT
    G_in = zeros(Complex{T}, cutoff_leftoverMode, cutoff_leftoverMode, 2 * M)
    G_in_dA = zeros(Complex{T}, size(G_in)..., size(A)...)
    G_in_dB = zeros(Complex{T}, size(G_in)..., size(B)...)

    ########## READ ##########
    read_GB = (2 * d - 1, params...)
    GB = zeros(Complex{T}, cutoff_leftoverMode, cutoff_leftoverMode, length(B))
    for m in 1:cutoff_leftoverMode
        for n in 1:cutoff_leftoverMode
            GB[m, n, :] = arr1[m, n, read_GB...] .* B
        end
    end

    # Array0
    G_in, G_in_dA, G_in_dB = read_block(G_in, G_in_dA, G_in_dB, 2 * d - 1, arr0, arr0_dA, arr0_dB, params)

    # read from Array2
    if params[d] > 1
        params_adapted = collect(params)
        params_adapted[d] -= 1
        G_in, G_in_dA, G_in_dB = read_block(G_in, G_in_dA, G_in_dB, 2 * d, arr2, arr2_dA, arr2_dB, (d,params_adapted...))
    end

    # read from Array11
    for i in d+1:M  # i>d
        if params[i] > 1
            params_adapted = collect(params)
            params_adapted[i] -= 1
            G_in, G_in_dA, G_in_dB = read_block(G_in, G_in_dA, G_in_dB, 2 * i - 1, arr1001, arr1001_dA, arr1001_dB, (d, i - d,params_adapted...))
            G_in, G_in_dA, G_in_dB = read_block(G_in, G_in_dA, G_in_dB, 2 * i, arr1010, arr1010_dA, arr1010_dB, (d, i - d,params_adapted...))
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
    arr0_dA, arr0_dB = write_block_grad(2 * d + 2, write, arr1, read_GB, G_in, A, B, K_i, K_l, cutoff_leftoverMode, arr0_dA, arr1_dA, G_in_dA, arr0_dB, arr1_dB, G_in_dB, SQRT)

    # Array2
    if params[d] + 1 < cutoffs_tail[d]
        write = (d, params...)
        arr2_dA, arr2_dB = write_block_grad(2 * d + 1, write, arr1, read_GB, G_in, A, B, K_i, K_l, cutoff_leftoverMode, arr2_dA, arr1_dA, G_in_dA, arr2_dB, arr1_dB, G_in_dB, SQRT)
    end

    # Array11
    for i in d+1:M
        if params[i] < cutoffs_tail[i]
            write = (d, i - d, params...)
            arr1010_dA, arr1010_dB = write_block_grad(2 * i + 1, write, arr1, read_GB, G_in, A, B, K_i, K_l, cutoff_leftoverMode, arr1010_dA, arr1_dA, G_in_dA, arr1010_dB, arr1_dB, G_in_dB, SQRT)
            arr1001_dA, arr1001_dB = write_block_grad(2 * i + 2, write, arr1, read_GB, G_in, A, B, K_i, K_l, cutoff_leftoverMode, arr1001_dA, arr1_dA, G_in_dA, arr1001_dB, arr1_dB, G_in_dB, SQRT)
        end
    end
end

function use_diag_pivot_grad!(A, B, M, cutoff_leftoverMode, cutoffs_tail, params, arr0, arr1, arr0_dA, arr1_dA, arr0_dB, arr1_dB, T, SQRT)
    pivot = CompactFock_HelperFunctions.repeat_twice(params)
    """Given params=(a,b,c,...), apply the eqs. 16 & 17 (of https://doi.org/10.22331/q-2023-08-29-1097)
     for the pivot [a,a,b,b,c,c...]"""
    K_l = SQRT[pivot] # julia indexing counters extra zero in SQRT
    K_i = SQRT[pivot .+ 1] # julia indexing counters extra zero in SQRT
    G_in = zeros(Complex{T}, cutoff_leftoverMode, cutoff_leftoverMode, 2*M)
    G_in_dA = zeros(Complex{T}, size(G_in)..., size(A)...)
    G_in_dB = zeros(Complex{T}, size(G_in)..., size(B)...)

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
            read = tuple(i+1-2*((i-1) % 2), params_adapted...)
            G_in, G_in_dA, G_in_dB = read_block(G_in, G_in_dA, G_in_dB, i, arr1, arr1_dA, arr1_dB, read)
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
                arr1_dA, arr1_dB = write_block_grad(i + 2, write, arr0, read_GB, G_in, A, B, K_i, K_l, cutoff_leftoverMode, arr1_dA, arr0_dA, G_in_dA, arr1_dB, arr0_dB, G_in_dB, SQRT)
            end
        end
    end
end

function fill_firstMode_PNRzero!(arr0,arr0_dA,arr0_dB,A,B,M,cutoff_leftoverMode,SQRT)
    """fill first mode when all PNR detection values are equal to zero"""
    one_tuple = tuple(fill(1,M-1)...)

    for m in 1:cutoff_leftoverMode - 1
        arr0_dA[m + 1, 1, one_tuple..., :, :] = arr0_dA[m, 1, one_tuple..., :, :] .* B[1]
        arr0_dB[m + 1, 1, one_tuple..., :] = arr0_dB[m, 1, one_tuple..., :] .* B[1]
        arr0_dB[m + 1, 1, one_tuple..., 1] += arr0[m, 1, one_tuple...] 
        if m != 1
            arr0_dA[m + 1, 1, one_tuple..., :, :] += SQRT[m] .* A[1, 1] .* arr0_dA[m - 1, 1, one_tuple..., :, :]
            arr0_dA[m + 1, 1, one_tuple..., 1, 1] += SQRT[m] .* arr0[m - 1, 1, one_tuple...] 
            arr0_dB[m + 1, 1, one_tuple..., :] += SQRT[m] .* A[1, 1] .* arr0_dB[m - 1, 1, one_tuple..., :]
        end
        arr0_dA[m + 1, 1, one_tuple..., :, :] ./= SQRT[m + 1]
        arr0_dB[m + 1, 1, one_tuple..., :] ./= SQRT[m + 1]
    end

    for m in 1:cutoff_leftoverMode
        for n in 1:cutoff_leftoverMode - 1
            arr0_dA[m, n + 1, one_tuple..., :, :] = arr0_dA[m, n, one_tuple..., :, :] .* B[2]
            arr0_dB[m, n + 1, one_tuple..., :] = arr0_dB[m, n, one_tuple..., :] .* B[2]
            arr0_dB[m, n + 1, one_tuple..., 2] += arr0[m, n, one_tuple...]
            if m != 1
                arr0_dA[m, n + 1, one_tuple..., :, :] += SQRT[m] .* A[2, 1] .* arr0_dA[m - 1, n, one_tuple..., :, :]
                arr0_dA[m, n + 1, one_tuple..., 2, 1] += SQRT[m] .* arr0[m - 1, n, one_tuple...]
                arr0_dB[m, n + 1, one_tuple..., :] += SQRT[m] .* A[2, 1] .* arr0_dB[m - 1, n, one_tuple..., :]
                
            end
            if n != 1
                arr0_dA[m, n + 1, one_tuple..., :, :] += SQRT[n] .* A[2, 2] .* arr0_dA[m, n - 1, one_tuple..., :, :]
                arr0_dA[m, n + 1, one_tuple..., 2, 2] += SQRT[n] .* arr0[m, n - 1, one_tuple...]
                arr0_dB[m, n + 1, one_tuple..., :] += SQRT[n] .* A[2, 2] .* arr0_dB[m, n - 1, one_tuple..., :]
            end
            arr0_dA[m, n + 1, one_tuple..., :, :] ./= SQRT[n + 1]
            arr0_dB[m, n + 1, one_tuple..., :] ./= SQRT[n + 1]
        end
    end
end

function fock_1leftoverMode_grad(
    A::AbstractMatrix{Complex{Float64}}, 
    B::AbstractVector{Complex{Float64}},
    arr0::AbstractArray{Complex{Float64}}, 
    arr2::AbstractArray{Complex{Float64}}, 
    arr1010::AbstractArray{Complex{Float64}}, 
    arr1001::AbstractArray{Complex{Float64}}, 
    arr1::AbstractArray{Complex{Float64}},
    precision_bits::Int64
    )
    """Returns the gradients of the density matrices in the upper, undetected mode of a circuit when all other modes
    are PNR detected (according to algorithm 2 of https://doi.org/10.22331/q-2023-08-29-1097)
    Args:
        A, B: required input for recurrence relation
        Submatrices of the Fock representation. Each submatrix contains Fock indices of a certain type.
            arr0 --> type: [a,a,b,b,c,c...]
            arr2 --> type: [a+2,a,b,b,c,c...] / [a,a,b+2,b,c,c...] / ...
            arr1010 --> type: [a+1,a,b+1,b,c,c,...] / [a+1,a,b,b,c+1,c,...] / [a,a,b+1,b,c+1,c,...] / ...
            arr1001 --> type: [a+1,a,b,b+1,c,c,...] / [a+1,a,b,b,c,c+1,...] / [a,a,b+1,b,c,c+1,...] / ...
            arr1 --> type: [a+1,a,b,b,c,c...] / [a,a+1,b,b,c,c...] / [a,a,b+1,b,c,c...] / ...
        precision_bits: number of bits used to represent a single Fock amplitude
    Returns:
        arr0_dA, arr0_dB: derivatives of arr0 w.r.t A and B
    """
    
    T = GetPrecision.get_dtype(precision_bits)
    SQRT = GetPrecision.SQRT_dict[precision_bits]
    
    cutoffs = size(arr0)[2:end]
    M = length(cutoffs)
    cutoff_leftoverMode = cutoffs[1]
    cutoffs_tail = cutoffs[2:end]

    arr0_dA = zeros(Complex{T}, size(arr0)..., size(A)...)
    arr2_dA = zeros(Complex{T}, size(arr2)..., size(A)...)
    arr1010_dA = zeros(Complex{T}, size(arr1010)..., size(A)...)
    arr1001_dA = zeros(Complex{T}, size(arr1001)..., size(A)...)
    arr1_dA = zeros(Complex{T}, size(arr1)..., size(A)...)
    arr0_dB = zeros(Complex{T}, size(arr0)..., size(B)...)
    arr2_dB = zeros(Complex{T}, size(arr2)..., size(B)...)
    arr1010_dB = zeros(Complex{T}, size(arr1010)..., size(B)...)
    arr1001_dB = zeros(Complex{T}, size(arr1001)..., size(B)...)
    arr1_dB = zeros(Complex{T}, size(arr1)..., size(B)...)

    fill_firstMode_PNRzero!(arr0,arr0_dA,arr0_dB,A,B,M,cutoff_leftoverMode,SQRT)

    dict_params = CompactFock_HelperFunctions.construct_dict_params(cutoffs_tail)
    for sum_params in 0:sum(cutoffs_tail)-1
        for params in dict_params[sum_params]
            # diagonal pivots: aa,bb,cc,...
            if (cutoffs_tail[1] == 1) || (params[1] < cutoffs_tail[1]) # julia indexing!
                use_diag_pivot_grad!(A,B,M - 1,cutoff_leftoverMode,cutoffs_tail,params,arr0,arr1,arr0_dA,arr1_dA,arr0_dB,arr1_dB,T,SQRT)
            end
            # off-diagonal pivots: d=1: (a+1)a,bb,cc,... | d=2: 00,(b+1)b,cc,... | d=3: 00,00,(c+1)c,... | ...
            for d in 1:M - 1
                if all(params[1:d-1] .== 1) && (params[d] < cutoffs_tail[d])
                    use_offDiag_pivot_grad!(A,B,M - 1,cutoff_leftoverMode,cutoffs_tail,params,d,arr0,arr2,arr1010,arr1001,arr1,arr0_dA,arr2_dA,arr1010_dA,arr1001_dA,arr1_dA,arr0_dB,arr2_dB,arr1010_dB,arr1001_dB,arr1_dB,T,SQRT)
                end
            end
        end
    end

    return Complex{Float64}.(arr0_dA), Complex{Float64}.(arr0_dB)
end

end # end module
