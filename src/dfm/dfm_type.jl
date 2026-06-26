##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################
#
# DFM{T} wrapper + DFM-level accessors. Split out so it can be included after
# utils/params/evals (it references DFMParams etc.).
##################################################################################

export DFM
mutable struct DFM{T} <: AbstractModel
    model::DFMModel
    params::DFMParams{T}
end
DFM(name::Sym=:dfm, T::Type{<:Real}=Float64) = DFM{T}(DFMModel(name), DFMParams{T}())

@inline ismixfreq(dfm::DFM) = ismixfreq(dfm.model)

eval_resid(point::AbstractMatrix, dfm::DFM) = eval_resid(point, dfm.model, dfm.params)
eval_RJ(point::AbstractMatrix, dfm::DFM) = eval_RJ(point, dfm.model, dfm.params)
eval_R!(R::AbstractVector, point::AbstractMatrix, dfm::DFM) = eval_R!(R, point, dfm.model, dfm.params)
eval_RJ!(R::AbstractVector, J::AbstractMatrix, point::AbstractMatrix, dfm::DFM) = eval_RJ!(R, J, point, dfm.model, dfm.params)
add_components!(dfm::DFM, args...; kwargs...) = (add_components!(dfm.model, args...; kwargs...); dfm)
map_loadings!(dfm::DFM, args...) = (map_loadings!(dfm.model, args...); dfm)
add_shocks!(dfm::DFM, args...) = (add_shocks!(dfm.model, args...); dfm)
add_observed!(dfm::DFM, args...; kwargs...) = (add_observed!(dfm.model, args...; kwargs...); dfm)
initialize_dfm!(dfm::DFM, args...; kwargs...) = (initialize_dfm!(dfm.model, args...; kwargs...); dfm.params = init_params(dfm.model); dfm)

lags(dfm::DFM) = lags(dfm.model)
leads(dfm::DFM) = leads(dfm.model)

get_covariance(dfm::DFM) = get_covariance(dfm.model, dfm.params)
function get_covariance(dfm::DFM, B::Sym)
    model = dfm.model
    if haskey(model.observed, B)
        return get_covariance(model.observed[B], getproperty(dfm.params, B))
    else
        return get_covariance(model.components[B], getproperty(dfm.params, B))
    end
end

get_covariance(dfm::DFM, V::Val) = get_covariance(dfm.model, dfm.params, V)
set_covariance!(dfm::DFM, COV::AbstractMatrix, V::Val) = set_covariance!(dfm.params, dfm.model, COV, V)

for f in (:observed, :states, :shocks, :endog, :exog, :varshks, :allvars)
    nf = Symbol("n", f)
    @eval begin
        $f(dfm::DFM) = $f(dfm.model)
        $nf(dfm::DFM) = $nf(dfm.model)
    end
end

nstates_with_lags(m::DFM) = nstates_with_lags(m.model)
nstates_with_lags(m::DFMModel) = sum(nstates_with_lags, values(m.components), init=0)
nstates_with_lags((n, b)::Pair{Symbol,<:DFMBlock}) = nstates_with_lags(b)
nstates_with_lags(::ObservedBlock) = 0
nstates_with_lags(b::ComponentsBlock) = nstates(b) * lags(b)

states_with_lags(m::DFM) = states_with_lags(m.model)
states_with_lags(m::DFMModel) = mapfoldl(states_with_lags, append!, values(m.components), init=Symbol[])
states_with_lags((n, b)::Pair{Symbol,<:DFMBlock}) = states_with_lags(b)
states_with_lags(::ObservedBlock) = Symbol[]
function states_with_lags(blk::ComponentsBlock)
    return [_make_lag_name(v, lags(blk) - l) for l = 1:lags(blk) for v in states(blk)]
end

get_mean(dfm::DFM) = get_mean!(Vector{eltype(dfm.params)}(undef, nobserved(dfm)), dfm)
get_mean!(x::AbstractVector, dfm::DFM) = get_mean!(x, dfm.model, dfm.params)
set_mean!(dfm::DFM, mu::AbstractVector) = set_mean!(dfm.params, dfm.model, mu)

get_loading(dfm::DFM) = get_loading(dfm.model, dfm.params)
get_loading!(x::AbstractMatrix, dfm::DFM) = get_loading!(x, dfm.model, dfm.params)
set_loading!(dfm::DFM, x::AbstractMatrix) = set_loading!(dfm.params, dfm.model, x)

get_transition(dfm::DFM) = get_transition(dfm.model, dfm.params)
get_transition!(x::AbstractMatrix, dfm::DFM) = get_transition!(x, dfm.model, dfm.params)
set_transition!(dfm::DFM, T::AbstractMatrix) = set_transition!(dfm.params, dfm.model, T)

export states_with_lags, nstates_with_lags
