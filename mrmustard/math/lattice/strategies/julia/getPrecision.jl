module GetPrecision

using MultiFloats

# possible dtypes
const dtypes_dict = Dict(128 => Float64, 256 => Float64x2, 384 => Float64x3, 512 => Float64x4)

# pre-calculate SQRT values
const SQRT_dict = Dict(k => [0;sqrt.(dtypes_dict[k].(1:999))] for k in keys(dtypes_dict))

function get_dtype(precision_bits)
    if !(precision_bits in keys(dtypes_dict))
        error("The possible values for precision_bits are ", keys(dtypes))
    end
    return dtypes_dict[precision_bits]
end

end # end module