
export init_params, init_params!

const DFMParams = ComponentArray{Float64}
DFMParams(x::DFMParams; kwargs...)::DFMParams = DFMParams(; x..., kwargs...)

function _make_loading(blk::CommonComponents, nobserved::Integer)
    Matrix{Float64}(undef, nobserved, blk.size)
end

function _make_loading(blk::IdiosyncraticComponents, nobserved::Integer)
    nobserved == blk.size || throw(DimensionMismatch("Size of idiosyncratic components block ($(blk.size)) does not match number of observed variables ($nobserved)."))
    Vector{Float64}(undef, nobserved)
end


@inline init_params(any) = init_params!(DFMParams(), any)

function init_params!(p::DFMParams, blk::CommonComponents)
    return DFMParams(p;
        coefs=Array{Float64}(undef, blk.size, blk.size, blk.order),
        covar=Array{Float64}(undef, nshocks(blk), nshocks(blk))
    )
end

function init_params!(p::DFMParams, blk::IdiosyncraticComponents)
    # matrices are diagonal, so keep only diagonal in 1d-array
    return DFMParams(p;
        coefs=Array{Float64}(undef, blk.size, blk.order),
        covar=Array{Float64}(undef, nshocks(blk))
    )
end

function init_params!(p::DFMParams, blk::ObservedBlock)
    loadings = Pair{Symbol,AbstractArray}[]
    for (name, vars) = blk.comp2vars
        Λ_name = Symbol(:Λ_, name)
        Λ_block = blk.components[name]
        push!(loadings, Λ_name => _make_loading(Λ_block, length(vars)))
    end
    return DFMParams(p;
        mean=Array{Float64}(undef, blk.size),
        loadings...,
        covar=Array{Float64}(undef, nshocks(blk))
    )
end

function init_params!(p::DFMParams, m::DFMModel)
    params = []
    push!(params, :observed => init_params(m.observed_block))
    for (name, block) in m.components
        push!(params, name => init_params(block))
    end
    return DFMParams(p; params...)
end
