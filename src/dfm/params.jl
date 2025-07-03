##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

export init_params, init_params!

const DFMParams{T<:Real} = ComponentArray{T}
# DFMParams(x::DFMParams{T}; kwargs...)::DFMParams{T} where {T} = DFMParams{T}(; x..., kwargs...)

function _make_loading(blk::CommonComponents, crefs::NamedList{_BlockComponentRef}, T::Type{<:Real}=Float64)
    nobserved = length(crefs)
    all_BlockRef(crefs) && return Matrix{T}(undef, nobserved, blk.size)
    nnz = sum(n_comp_refs, crefs.vals)
    @assert (0 < nnz < nobserved * blk.size) "Unexpected number of non-zeros in loading."
    return Vector{T}(undef, nnz)
end

# function _make_loading(blk::IdiosyncraticComponents, vars_comprefs::NamedList{_BlockComponentRef}, T::Type{<:Real}=Float64)
#     nobserved = length(vars_comprefs)
#     nobserved == blk.size || throw(DimensionMismatch("Size of idiosyncratic components block ($(blk.size)) does not match number of observed variables ($nobserved)."))
#     Vector{T}(undef, nobserved)
# end

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
    for (blkname, vars_comprefs) = blk.comp2vars
        block = blk.components[blkname]
        # idiosyncratic components and shocks don't get loadings (they're all ones)
        block isa IdiosyncraticComponents && continue
        push!(loadings, blkname => _make_loading(block, vars_comprefs, T))
    end
    return DFMParams{T}(; p...,
        mean=DFMParams{T}(; (v.name => 0 for v in endog(blk))...),
        loadings=DFMParams{T}(; loadings...),
        covar=Array{T}(undef, nshocks(blk))
    )
end

function init_params!(p::DFMParams{T}, m::DFMModel) where {T<:Real}
    params = []
    for (name, block) in m.observed
        push!(params, name => init_params(block, T))
    end
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
    COV = zeros(eltype(p), nshks, nshks)
    AX = Axis{_enumerate_vars(shks)}
    C = ComponentArray(COV, AX(), AX())
    isdiagonal = false
    for (name, blk) in m.observed
        covar = get_covariance(blk, getproperty(p, name))
        C[blk.shks, blk.shks] = covar
        isdiagonal = isdiagonal && (length(covar) <= 1 || covar isa Diagonal)
    end
    for (name, blk) in m.components
        covar = get_covariance(blk, getproperty(p, name))
        C[blk.shks, blk.shks] = covar
        isdiagonal = isdiagonal && (length(covar) == 1 || covar isa Diagonal)
    end
    return isdiagonal ? Diagonal(COV) : Symmetric(COV)
end

set_covariance!(p::DFMParams, ::ObservedBlock, val) = (p.covar[:] = diag(val); val)
set_covariance!(p::DFMParams, ::IdiosyncraticComponents, val) = (p.covar[:] = diag(val); val)
set_covariance!(p::DFMParams, ::CommonComponents, val) = (p.covar[:, :] = Symmetric(val); val)
