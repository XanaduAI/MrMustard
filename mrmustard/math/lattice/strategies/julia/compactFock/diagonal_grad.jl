module DiagonalGrad

import ..GetPrecision
import ..CompactFock_HelperFunctions

function calc_dA_dB(i, G_in_dA, G_in_dB, G_in, A, B, K_l, K_i, M, pivot_val, pivot_val_dA, pivot_val_dB)
    """Calculate the derivatives of a single Fock amplitude w.r.t A and B.
    Args:
        i (int): the element of the multidim index that is increased
        G_in, G_in_dA, G_in_dB (array, array, array): all Fock amplitudes from the 'read' group in the recurrence relation and their derivatives w.r.t. A and B
        A, B (array, vector): required input for recurrence relation (given by mrmustard.physics.fock.ABC)
        K_l, K_i (vector, vector): SQRT[pivot], SQRT[pivot + 1]
        M (int): number of modes
        pivot_val, pivot_val_dA, pivot_val_dB (array, array, array): Fock amplitude at the position of the pivot and its derivatives w.r.t. A and B
    """
    dA = pivot_val_dA .* B[i]
    dB = pivot_val_dB .* B[i]
    dB[i] += pivot_val
    for l in 1:2*M
        dA += K_l[l] * A[i, l] * G_in_dA[l,:,:]
        dB += K_l[l] * A[i, l] * G_in_dB[l,:]
        dA[i, l] += G_in[l]
    end
    return dA ./ K_i[i], dB ./ K_i[i]
end
                                
function use_offDiag_pivot_grad!(A, B, M, cutoffs, params, d, arr0, arr2, arr1010, arr1001, arr1,
    arr0_dA, arr2_dA, arr1010_dA, arr1001_dA, arr1_dA, arr0_dB, arr2_dB, arr1010_dB, arr1001_dB, arr1_dB, T, SQRT)
    """Given params=(a,b,c,...), apply the eqs. 16 & 17 (of https://doi.org/10.22331/q-2023-08-29-1097)
    for the pivots [a+1,a,b,b,c,c,...] / [a,a,b+1,b,c,c,...] / [a,a,b,b,c+1,c,...] / ..."""
    
    pivot = CompactFock_HelperFunctions.repeat_twice(params)
    pivot[2 * d - 1] += 1
    K_l = SQRT[pivot] # julia indexing counters extra zero in SQRT
    K_i = SQRT[pivot .+ 1] # julia indexing counters extra zero in SQRT
    G_in = zeros(Complex{T}, 2 * M)
    G_in_dA = zeros(Complex{T}, 2*M, size(A)...)
    G_in_dB = zeros(Complex{T}, 2*M, size(B)...)

    ########## READ ##########
    pivot_val = arr1[2 * d - 1, params...]
    pivot_val_dA = arr1_dA[2 * d - 1, params...,:,:]
    pivot_val_dB = arr1_dB[2 * d - 1, params...,:]

    # Array0
    G_in[2 * d - 1] = arr0[params...]
    G_in_dA[2 * d - 1,:,:] = arr0_dA[params...,:,:]
    G_in_dB[2 * d - 1,:] = arr0_dB[params...,:]

    # read from Array2
    if params[d] > 1
        params_adapted = collect(params)
        params_adapted[d] -= 1
        G_in[2 * d] = arr2[d, params_adapted...]
        G_in_dA[2 * d,:,:] = arr2_dA[d, params_adapted...,:,:]
        G_in_dB[2 * d,:] = arr2_dB[d, params_adapted...,:]
    end

    # read from Array11
    for i in d+1:M # i>d
        if params[i] > 1
            params_adapted = collect(params)
            params_adapted[i] -= 1
            G_in[2 * i - 1] = arr1001[d, i - d, params_adapted...]
            G_in_dA[2 * i - 1,:,:] = arr1001_dA[d, i - d, params_adapted...,:,:]
            G_in_dB[2 * i - 1,:] = arr1001_dB[d, i - d, params_adapted...,:]
            G_in[2 * i] = arr1010[d, i - d, params_adapted...]
            G_in_dA[2 * i,:,:] = arr1010_dA[d, i - d, params_adapted...,:,:]
            G_in_dB[2 * i,:] = arr1010_dB[d, i - d, params_adapted...,:]
        end
    end

    ########## WRITE ##########
    G_in .*= K_l

    # Array0
    params_adapted = collect(params)
    params_adapted[d] += 1
    arr0_dA[params_adapted...,:,:], arr0_dB[params_adapted...,:] = calc_dA_dB(2 * d, G_in_dA, G_in_dB, G_in, A, B, K_l, K_i, M, pivot_val, pivot_val_dA, pivot_val_dB)

    # Array2
    if params[d] + 1 < cutoffs[d]
        arr2_dA[d, params...,:,:], arr2_dB[d, params...,:] = calc_dA_dB(2 * d - 1, G_in_dA, G_in_dB, G_in, A, B, K_l, K_i, M, pivot_val, pivot_val_dA, pivot_val_dB)
    end

    # Array11
    for i in d+1:M # i>d
        if params[i] < cutoffs[i]
            arr1010_dA[d, i - d, params...,:,:], arr1010_dB[d, i - d, params...,:] = calc_dA_dB(2 * i - 1, G_in_dA, G_in_dB, G_in, A, B, K_l, K_i, M, pivot_val, pivot_val_dA, pivot_val_dB)
            arr1001_dA[d, i - d, params...,:,:], arr1001_dB[d, i - d, params...,:] = calc_dA_dB(2 * i, G_in_dA, G_in_dB, G_in, A, B, K_l, K_i, M, pivot_val, pivot_val_dA, pivot_val_dB)
        end
    end
end
function use_diag_pivot_grad!(A, B, M, cutoffs, params, arr0, arr1, arr0_dA, arr1_dA, arr0_dB, arr1_dB, T, SQRT)
    """Given params=(a,b,c,...), apply the eqs. 16 & 17 (of https://doi.org/10.22331/q-2023-08-29-1097)
     for the pivot [a,a,b,b,c,c...]"""
    pivot = CompactFock_HelperFunctions.repeat_twice(params)
    K_l = SQRT[pivot] # julia indexing counters extra zero in SQRT
    K_i = SQRT[pivot .+ 1] # julia indexing counters extra zero in SQRT
    G_in = zeros(Complex{T}, 2*M)
    G_in_dA = zeros(Complex{T}, 2*M, size(A)...)
    G_in_dB = zeros(Complex{T}, 2*M, size(B)...)

    ########## READ ##########
    pivot_val = arr0[params...]
    pivot_val_dA = arr0_dA[params...,:,:]
    pivot_val_dB = arr0_dB[params...,:]

    # Array1
    for i in 1:2*M
        if params[(i-1)÷2+1] > 1
            i_staggered = i+1-2*((i-1) % 2)
            params_adapted = collect(params)
            params_adapted[(i-1)÷2+1] -= 1
            G_in[i] = arr1[i_staggered, params_adapted...]
            G_in_dA[i,:,:] = arr1_dA[i_staggered, params_adapted...,:,:]
            G_in_dB[i,:] = arr1_dB[i_staggered, params_adapted...,:]
        end
    end

    ########## WRITE ##########
    G_in .*= K_l

    # Array1
    for i in 1:2*M
        if params[(i-1)÷2+1] < cutoffs[(i-1)÷2+1]
            # This if statement prevents a few elements from being written that will never be read
            if i ≠ 2 || params[1] + 1 < cutoffs[1]
                arr1_dA[i, params...,:,:], arr1_dB[i, params...,:] = calc_dA_dB(i, G_in_dA, G_in_dB, G_in, A, B, K_l, K_i, M, pivot_val, pivot_val_dA, pivot_val_dB)
            end
        end
    end
end
function fock_diagonal_grad(
    A::AbstractMatrix{Complex{Float64}}, 
    B::AbstractVector{Complex{Float64}},
    arr0::AbstractArray{Complex{Float64}}, 
    arr2::AbstractArray{Complex{Float64}}, 
    arr1010::AbstractArray{Complex{Float64}}, 
    arr1001::AbstractArray{Complex{Float64}}, 
    arr1::AbstractArray{Complex{Float64}},
    precision_bits::Int64
    )
    """Returns the gradients of the PNR probabilities of a mixed state according to algorithm 1 of
    https://doi.org/10.22331/q-2023-08-29-1097
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

    cutoffs = size(arr0)
    M = length(cutoffs)

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

    dict_params = CompactFock_HelperFunctions.construct_dict_params(cutoffs)
    for sum_params in 0:sum(cutoffs)-1
        for params in dict_params[sum_params]
            # diagonal pivots: aa,bb,cc,...
            if (cutoffs[1] == 1) || (params[1] < cutoffs[1]) # julia indexing!
                use_diag_pivot_grad!(A, B, M, cutoffs, params, arr0, arr1, arr0_dA, arr1_dA, arr0_dB, arr1_dB, T, SQRT)
            end
            # off-diagonal pivots: d=1: (a+1)a,bb,cc,... | d=2: 00,(b+1)b,cc,... | d=3: 00,00,(c+1)c,... | ...
            for d in 1:M
                if all(params[1:d-1] .== 1) && (params[d] < cutoffs[d])  # julia indexing!
                    use_offDiag_pivot_grad!(A,B,M,cutoffs,params,d,arr0,arr2,arr1010,arr1001,arr1,arr0_dA,arr2_dA,arr1010_dA,arr1001_dA,arr1_dA,arr0_dB,arr2_dB,arr1010_dB,arr1001_dB,arr1_dB, T, SQRT)
                end
            end
        end
    end

    return Complex{Float64}.(arr0_dA), Complex{Float64}.(arr0_dB)
end

end # end module