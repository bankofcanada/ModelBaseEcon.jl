##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

#########################################################################
#  Declaration of API for working with model parameters 
#  for the purpose of estimation

"""Return the total number of parameters in the given model instance."""
function nparams end

"""Gather the model parameters from the given model instance into a new `Vector` instance."""
function pack_params end
# generic implementation that falls on pack_params!
pack_params(args...) = pack_params!(Vector{Float64}(undef, nparams(args...)), args...)

"""Gather the model parameters from the given model instance into the given `Vector`."""
function pack_params! end

"""Update the model parameters into the given model instance from the given `Vector`."""
function unpack_params! end

export nparams, pack_params, pack_params!, unpack_params!

#########################################################################
# Implementation of the API for DFMModel and factor blocks

const _Inds = AbstractVector{Int}

mutable struct FBEstimData
    ready::Bool
    loadings::Bool
    loadings_inds::_Inds
    arcoefs::Bool
    arcoefs_inds::Vector{_Inds}
    covariance::Bool
    covariance_inds::_Inds
    FBEstimData(fb::ARFactorBlock) =
        new(false, !isa(fb, IdiosyncraticComponents), [],
            true, [], true, [])
end

mutable struct DFMEstimData
    ready::Bool
    mean::BitVector
    mean_inds::_Inds
    covariance::BitVector
    covariance_inds::_Inds
    factorblocks::LittleDict{Symbol,FBEstimData}
    DFMEstimData(m::DFMModel) =
        new(false, trues(nobserved(m)), [], trues(nobservedshocks(m)^2), [],
            LittleDict(b.name => FBEstimData(b) for b in _blocks(m)))
end

#= 
function setproperty!(ed::DFMEstimData, name::Core.Symbol, value)
    if name in (:mean, :covariance)
        setfield!(ed, :ready, false)
    end
    setfield!(ed, name, value)
end

function setproperty!(ed::FBEstimData, name::Core.Symbol, value)
    if name in (:loadings, :arcoefs, :covariance)
        setfield!(ed, :ready, false)
    end
    setfield!(ed, name, value)
end
=#

function Base.getproperty(ed::DFMEstimData, name::Symbol)
    if !hasfield(typeof(ed), name)
        fb = get(ed.factorblocks, name, nothing)
        if fb !== nothing
            return fb
        end
    end
    return getfield(ed, name)
end

export new_estimdata
new_estimdata(m::DFMModel) = DFMEstimData(m)

nparams(ed::DFMEstimData, m::DFMModel) =
    sum(ed.mean) + sum(ed.covariance) +
    sum(nparams(args...) for args in zip(_blocks(ed), _blocks(m)))

nparams(ed::FBEstimData, ic::IdiosyncraticComponents) =
    ed.arcoefs * ic.nfactors * ic.order +
    ed.covariance * ic.nfactors

nparams(ed::FBEstimData, fb::FactorBlock) =
    ed.loadings * fb.nobserved * fb.nfactors +
    ed.arcoefs * fb.nfactors * fb.nfactors * fb.order +
    ed.covariance * fb.nfactors * fb.nfactors

_new_inds(num, offset) = (ret = offset[] .+ (1:num); offset[] += num; ret)

function prepare_estimdata!(ed::FBEstimData, fb::IdiosyncraticComponents, offset=Ref(0))
    ed.arcoefs_inds = ed.arcoefs ? [_new_inds(fb.nfactors, offset) for _ = 1:fb.order] : []
    ed.covariance_inds = ed.covariance ? _new_inds(fb.nfactors, offset) : []
    ed.ready = true
    return ed
end

function prepare_estimdata!(ed::FBEstimData, fb::FactorBlock, offset=Ref(0))
    ed.loadings_inds = ed.loadings ? _new_inds(fb.nobserved * fb.nfactors, offset) : []
    ed.arcoefs_inds = ed.arcoefs ? [_new_inds(fb.nfactors^2, offset) for _ = 1:fb.order] : []
    ed.covariance_inds = ed.covariance ? _new_inds(fb.nfactors^2, offset) : []
    return ed
end

function prepare_estimdata!(ed::DFMEstimData, m::DFMModel, offset=Ref(0))
    num = sum(ed.mean)
    ed.mean_inds = num > 0 ? _new_inds(num, offset) : []
    num = sum(ed.covariance)
    ed.covariance_inds = num > 0 ? _new_inds(num, offset) : []
    for (fb, fed) in zip(_blocks(m), _blocks(ed))
        prepare_estimdata!(fed, fb, offset)
    end
    ed.ready = true
    return ed
end

function pack_params!(vec::AbstractVector{Float64}, ed::FBEstimData, fb::ARFactorBlock)
    if !isempty(ed.loadings)
        vec[ed.loadings_inds] .= fb.loadings[:]
    end
    if ed.arcoefs
        for i = 1:fb.order
            vec[ed.arcoefs_inds[i]] .= fb.arcoefs[i][:]
        end
    end
    if !isempty(ed.covariance)
        vec[ed.covariance_inds] .= fb.covariance[:]
    end
    return vec
end

function pack_params!(vec::AbstractVector{Float64}, ed::FBEstimData, fb::IdiosyncraticComponents)
    if ed.arcoefs
        for i = 1:fb.order
            vec[ed.arcoefs_inds[i]] .= fb.arcoefs[i].diag
        end
    end
    if !isempty(ed.covariance)
        vec[ed.covariance_inds] .= fb.covariance.diag
    end
    return vec
end

function pack_params!(vec::AbstractVector{Float64}, ed::DFMEstimData, m::DFMModel)
    if !(ed.ready && all(x.ready for x in _blocks(ed)))
        prepare_estimdata!(ed, m)
    end
    if !isempty(ed.mean_inds)
        vec[ed.mean_inds] .= m.mean[ed.mean]
    end
    if !isempty(ed.covariance)
        vec[ed.covariance_inds] .= m.covariance[][ed.covariance]
    end
    for (fed, fb) in zip(_blocks(ed), _blocks(m))
        pack_params!(vec, fed, fb)
    end
    return vec
end

function unpack_params!(fb::ARFactorBlock, ed::FBEstimData, vec::AbstractVector{Float64})
    if !isempty(ed.loadings)
        fb.loadings[:] .= vec[ed.loadings_inds]
    end
    if ed.arcoefs
        for i = 1:fb.order
            fb.arcoefs[i][:] .= vec[ed.arcoefs_inds[i]]
        end
    end
    if !isempty(ed.covariance)
        fb.covariance[:] .= vec[ed.covariance_inds]
    end
    return fb

end

function unpack_params!(fb::IdiosyncraticComponents, ed::FBEstimData, vec::AbstractVector{Float64})
    if ed.arcoefs
        for i = 1:fb.order
            fb.arcoefs[i].diag .= vec[ed.arcoefs_inds[i]]
        end
    end
    if !isempty(ed.covariance)
        fb.covariance.diag .= vec[ed.covariance_inds]
    end
    return fb

end

function unpack_params!(m::DFMModel, ed::DFMEstimData, vec::AbstractVector{Float64})
    if !isempty(ed.mean_inds)
        m.mean[ed.mean] .= vec[ed.mean_inds]
    end
    if !isempty(ed.covariance)
        m.covariance[][ed.covariance] .= vec[ed.covariance_inds]
    end
    for (fed, fb) in zip(_blocks(ed), _blocks(m))
        unpack_params!(fb, fed, vec)
    end
    return m
end


