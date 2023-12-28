##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################

export init_params, init_params!

const DFMParams{T<:Real} = ComponentArray{T}
# DFMParams(x::DFMParams{T}; kwargs...)::DFMParams{T} where {T} = DFMParams{T}(; x..., kwargs...)

function _make_loading(blk::CommonComponents, nobserved::Integer, T::Type{<:Real}=Float64)
    Matrix{T}(undef, nobserved, blk.size)
end

function _make_loading(blk::IdiosyncraticComponents, nobserved::Integer, T::Type{<:Real}=Float64)
    nobserved == blk.size || throw(DimensionMismatch("Size of idiosyncratic components block ($(blk.size)) does not match number of observed variables ($nobserved)."))
    Vector{T}(undef, nobserved)
end

@inline init_params(any::DFMBlockOrModel, T::Type{<:Real}=Float64) = init_params!(DFMParams{T}(), any)

function init_params!(p::DFMParams{T}, blk::CommonComponents) where {T<:Real}
    return DFMParams{T}(; p...,
        # mean = zeros(blk.size)
        coefs=Array{T}(undef, blk.size, blk.size, blk.order),
        covar=Array{T}(undef, nshocks(blk), nshocks(blk))
    )
end

function init_params!(p::DFMParams{T}, blk::IdiosyncraticComponents) where {T<:Real}
    # matrices are diagonal, so keep only diagonal in 1d-array
    return DFMParams{T}(; p...,
        # mean = zeros(blk.size)
        coefs=Array{T}(undef, blk.size, blk.order),
        covar=Array{T}(undef, nshocks(blk))
    )
end

function init_params!(p::DFMParams{T}, blk::ObservedBlock) where {T<:Real}
    loadings = Pair{Symbol,AbstractArray}[]
    for (name, vars) = blk.comp2vars
        block = blk.components[name]
        # idiosyncratic components and shocks don't get loadings (they're all ones)
        block isa IdiosyncraticComponents && continue
        push!(loadings, name => _make_loading(block, length(vars), T))
    end
    return DFMParams{T}(; p...,
        mean=DFMParams{T}(; (v.name => 0 for v in endog(blk))...),
        loadings=DFMParams{T}(; loadings...),
        covar=Array{T}(undef, nshocks(blk))
    )
end

function init_params!(p::DFMParams{T}, m::DFMModel) where {T<:Real}
    params = []
    push!(params, :observed => init_params(m.observed_block, T))
    for (name, block) in m.components
        push!(params, name => init_params(block, T))
    end
    return fill!(DFMParams{T}(; p..., params...), 0.0)
end

export get_covariance
get_covariance(::CommonComponents, p::DFMParams) = Symmetric(p.covar)
get_covariance(::IdiosyncraticComponents, p::DFMParams) = Diagonal(p.covar)
get_covariance(::ObservedBlock, p::DFMParams) = Diagonal(p.covar)
function get_covariance(m::DFMModel, p::DFMParams)
    shks = shocks(m)
    nshks = length(shks)
    COV = zeros(nshks, nshks)
    AX = Axis{_enumerate_vars(shks)}
    C = ComponentArray(COV, AX(), AX())
    blk = m.observed_block
    covar = get_covariance(blk, p.observed)
    C[blk.shks, blk.shks] = covar
    isdiagonal = length(covar) == 1 || covar isa Diagonal
    for (name, blk) in m.components
        covar = get_covariance(blk, getproperty(p, name))
        C[blk.shks, blk.shks] = covar
        isdiagonal = isdiagonal && (length(covar) == 1 || covar isa Diagonal)
    end
    return isdiagonal ? Diagonal(COV) : Symmetric(COV)
end

