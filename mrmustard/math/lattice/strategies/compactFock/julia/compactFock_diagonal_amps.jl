module DiagonalAmps

import ..HelperFunctions

function use_offDiag_pivot!(A, B, M, cutoffs, params, d, arr0, arr2, arr1010, arr1001, arr1, T, SQRT)
    pivot = HelperFunctions.repeat_twice(params)
    pivot[2 * d - 1] += 1

    K_l = SQRT[pivot] # julia indexing counters extra zero in SQRT
    K_i = SQRT[pivot .+ 1] # julia indexing counters extra zero in SQRT
    
    G_in = zeros(Complex{T}, 2 * M)

    ########## READ ##########
    GB = arr1[2 * d - 1, params...] .* B

    # Array0
    G_in[2 * d - 1] = arr0[params...]

    # read from Array2
    if params[d] > 1
        params_adapted = collect(params)
        params_adapted[d] -= 1
        G_in[2 * d] = arr2[d, params_adapted...]
    end

    # read from Array11
    for i in d+1:M # i>d
        if params[i] > 1
            params_adapted = collect(params)
            params_adapted[i] -= 1
            G_in[2 * i - 1] = arr1001[d, i - d, params_adapted...]
            G_in[2 * i] = arr1010[d, i - d, params_adapted...]
        end
    end

    ########## WRITE ##########
    G_in .*= K_l

    # Array0
    params_adapted = collect(params)
    params_adapted[d] += 1
    arr0[params_adapted...] = (GB[2 * d] .+ sum(A[2 * d,:] .* G_in)) / K_i[2 * d]

    # Array2
    if params[d] + 1 < cutoffs[d]
        arr2[d, params...] = (GB[2 * d - 1] .+ sum(A[2 * d - 1,:] .* G_in)) / K_i[2 * d - 1]
    end

    # Array11
    for i in d+1:M # i>d
        if params[i] < cutoffs[i]
            arr1010[d, i - d, params...] = (GB[2 * i - 1] .+ sum(A[2 * i - 1,:] .* G_in)) / K_i[2 * i - 1]
            arr1001[d, i - d, params...] = (GB[2 * i] .+ sum(A[2 * i,:] .* G_in)) / K_i[2 * i]
        end
    end
end

function use_diag_pivot!(A, B, M, cutoffs, params, arr0, arr1, T, SQRT)
    pivot = HelperFunctions.repeat_twice(params)
    K_l = SQRT[pivot] # julia indexing counters extra zero in SQRT
    K_i = SQRT[pivot .+ 1] # julia indexing counters extra zero in SQRT
    G_in = zeros(Complex{T}, 2*M)

    ########## READ ##########
    GB = arr0[params...] .* B

    # Array1
    for i in 1:2*M
        if params[(i-1)÷2+1] > 1
            params_adapted = collect(params)
            params_adapted[(i-1)÷2+1] -= 1
            G_in[i] = arr1[i+1-2*((i-1) % 2), params_adapted...]
        end
    end

    ########## WRITE ##########
    G_in .*= K_l

    # Array1
    for i in 1:2*M
        if params[(i-1)÷2+1] < cutoffs[(i-1)÷2+1]
            # this prevents a few elements from being written that will never be read
            if i ≠ 2 || params[1] + 1 < cutoffs[1]
                arr1[i, params...] = (GB[i] .+ sum(A[i,:] .* G_in)) / K_i[i]
            end
        end
    end
end

function fock_diagonal_amps(
    A::AbstractMatrix{Complex{Float64}}, 
    B::AbstractVector{Complex{Float64}}, 
    G0::Complex{Float64}, 
    cutoffs::Tuple,
    precision_bits::Int64
    )
    
    T = HelperFunctions.get_dtype(precision_bits)
    SQRT = HelperFunctions.SQRT_dict[precision_bits]

    M = length(cutoffs)

    arr0 = zeros(Complex{T}, cutoffs)
    arr0[fill(1,M)...] = G0
    arr2 = zeros(Complex{T}, M, cutoffs...)
    arr1 = zeros(Complex{T}, 2*M, cutoffs...)
    arr1010 = zeros(Complex{T}, M, M - 1, cutoffs...)
    arr1001 = zeros(Complex{T}, M, M - 1, cutoffs...)


    dict_params = HelperFunctions.construct_dict_params(cutoffs)
    for sum_params in 0:sum(cutoffs)-1
        for params in dict_params[sum_params]
            # diagonal pivots: aa,bb,cc,...
            if (cutoffs[1] == 1) || (params[1] < cutoffs[1]) # julia indexing!
                use_diag_pivot!(A, B, M, cutoffs, params, arr0, arr1, T, SQRT)
            end
            # off-diagonal pivots: d=1: (a+1)a,bb,cc,... | d=2: 00,(b+1)b,cc,... | d=3: 00,00,(c+1)c,... | ...
            for d in 1:M
                if all(params[1:d-1] .== 1) && (params[d] < cutoffs[d])  # julia indexing!
                    use_offDiag_pivot!(A, B, M, cutoffs, params, d, arr0, arr2, arr1010, arr1001, arr1, T, SQRT)
                end
            end
        end
    end
    return Complex{Float64}.(arr0), Complex{Float64}.(arr2), Complex{Float64}.(arr1010), Complex{Float64}.(arr1001), Complex{Float64}.(arr1)
end

end # end module