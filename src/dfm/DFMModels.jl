##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################

module DFMModels

import ..ModelVariable
import ..shocks
import ..nshocks
import ..allvars
import ..nallvars

import ..eval_resid
import ..eval_RJ
import ..eval_R!
# import ..eval_RJ!

import ..to_shock

import ..AbstractModel

using LinearAlgebra
using OrderedCollections
using ComponentArrays
using SparseArrays
using FillArrays

####################################################

const LittleDictVec{K,V} = LittleDict{K,V,Vector{K},Vector{V}}
const Sym = Union{AbstractString,Symbol,ModelVariable}
const LikeVec{T} = Union{Vector{T},NTuple{N,T} where {N},NamedTuple{NT,NTuple{N,T} where {N}} where {NT}}
const SymVec = LikeVec{<:Sym}
const DiagonalF64 = Diagonal{Float64,Vector{Float64}}
const SymmetricF64 = Symmetric{Float64,Matrix{Float64}}

####################################################

"""
A Dynamic Factor Model (DFM) with n blocks is a state-space model with 
the following structure: 

```
    x[t] = μ + Λ₁ F₁[t] + ⋯ + Λₙ Fₙ[t] + η[t] 
    F₁[t] = A₁₁ F₁[t-1] + ⋯ + A₁ₚ₁ F₁[t-p₁] + u₁[t]
    . . . 
    Fₙ[t] = Aₙ₁ Fₙ[t-1] + ⋯ + Aₙₚₙ Fₙ[t-pₙ] + uₙ[t]
```

where 
  * `x[t]` is a vector of observed variables
  * `μ` is a vector of unconditional means
  * `Fᵢ[t]` is the i-th block a factors, which follow an VAR process of order `pᵢ`
  * `η[t]` is a block of observation shocks
  * `uᵢ[t]` is a vector of state shocks associated with the i-th block of factors
  * `Aᵢⱼ` is the VAR coefficients matrix for the i-th block of factors for its
    j-th lag
  * `Λᵢ` is the loadings matrix for the i-th block of factors. It has non-zero
    rows only for the observed variables that load the i-th factor block.

Notes
  * Some of the factor blocks may be "dense" and some may be "diagonal". The dense
    factor blocks represent common components, while the diagonal ones represent
    idiosyncratic components.
  * An observed variable must either load an idiosyncratic component from one
    diagonal factor block, or it must have an observation shock, but not both.

[`DFMModel`](@ref) represents the model. It contains an observation block and a
collection of components blocks.

[`DFMBlock`](@ref) is an abstract type representing blocks of equations in the.

[`DFMObsBlock`](@ref) `<: DFMBlock` represents a block of observation equations.

[`DFMComponentBlock`](@ref)`{TYPE} <: DFMBlock` represents factor blocks.
`TYPE` can be either `Dense` or `Diagonal`.

"""
DFMModels

####################################################

export DFMModel, DFMParams
export DFMBlock, ComponentsBlock, ObservedBlock, CommonComponents, IdiosyncraticComponents
export observed, nobserved, states, nstates
export varshks, nvarshks, endog, nendog, exog, nexog
export lags, leads

####################################################

"""Abstract type for all DFM blocks"""
abstract type DFMBlock end

"""
    ComponentBlock{TYPE} <: DFMBlock

A struct representing a block of latent components in a DFM model.
See also [`CommonComponents`](@ref), [`IdiosyncraticComponents`](@ref) and
[`ObservedBlock`](@ref).
"""
mutable struct ComponentsBlock{TYPE} <: DFMBlock
    vars::Vector{ModelVariable}
    shks::Vector{ModelVariable}
    size::Int
    order::Int
end

function ComponentsBlock{TYPE}(name::Sym, size::Integer, order::Integer) where {TYPE}
    vars = _make_factor_names(name, size)
    shks = [to_shock(Symbol(var, "_shk")) for var in vars]
    return ComponentsBlock{TYPE}(vars, shks, Int(size), Int(order))
end

"""
    CommonComponents = ComponentBlock{:Dense}

A struct type representing factors that are common to a block of observed
variables. The loadings matrix, the trnasition matrices, and the shocks
covariance matrix for this type of block are all dense matrices. See also
[`IdiosyncraticComponents`](@ref).
"""
const CommonComponents = ComponentsBlock{:Dense}
CommonComponents(name::Sym, size::Integer=1; order::Integer=1) = CommonComponents(name, size, order)

"""
    IdiosyncraticComponents = ComponentBlock{:Diagonal}

A struct type representing factors that are common to a block of observed
variables. The loadings matrix, the trnasition matrices, and the shocks
covariance matrix for this type of block are all diagonal. See also
[`CommonComponents`](@ref).
"""
const IdiosyncraticComponents = ComponentsBlock{:Diagonal}
IdiosyncraticComponents(or::Integer=1; order::Integer=or) = IdiosyncraticComponents("", 0, order)


"""
    ObservedBlock <: DFMBlock

A struct representing the observed variables in a DFM model.
See also [`ComponentBlock`](@ref).
"""
mutable struct ObservedBlock <: DFMBlock
    vars::Vector{ModelVariable}
    shks::Vector{ModelVariable}
    size::Int
    # order is always 0
    components::LittleDictVec{Symbol,ComponentsBlock}
    var2comps::LittleDictVec{Symbol,Vector{Symbol}}
    comp2vars::LittleDictVec{Symbol,Vector{Symbol}}
    var2shk::LittleDictVec{Symbol,Symbol}
end

@inline ObservedBlock() = ObservedBlock(
    ModelVariable[], ModelVariable[], 0,
    LittleDictVec{Symbol,ComponentsBlock}(),
    LittleDictVec{Symbol,Vector{Symbol}}(),
    LittleDictVec{Symbol,Vector{Symbol}}(),
    LittleDictVec{Symbol,Symbol}(),
)

"""
    DFMModel

A struct representing a DFM model. It contains an [`ObservedBlock`](@ref) and 
a collection of [`ComponentBlock`](@ref)s. 
"""
mutable struct DFMModel <: AbstractModel
    name::Symbol
    _state::Symbol
    observed_block::ObservedBlock
    components::LittleDictVec{Symbol,ComponentsBlock}
end
DFMModel(name::Sym=:dfm) = DFMModel(Symbol(name), :new, ObservedBlock(), LittleDict{Symbol,ComponentsBlock}())

const DFMBlockOrModel = Union{DFMModel,DFMBlock}

## ##########################################################################
#    functions 

# info related to state-space representation of model
@inline observed(::ComponentsBlock) = ModelVariable[]
@inline nobserved(::ComponentsBlock) = 0
@inline observed(b::ObservedBlock) = b.vars
@inline nobserved(b::ObservedBlock) = b.size
@inline observed(m::DFMModel) = observed(m.observed_block)
@inline nobserved(m::DFMModel) = nobserved(m.observed_block)

@inline states(::ObservedBlock) = ModelVariable[]
@inline nstates(::ObservedBlock) = 0
@inline states(b::ComponentsBlock) = b.vars
@inline nstates(b::ComponentsBlock) = b.size
# @inline states(m::DFMModel) = mapfoldl(states, append!, values(m.components), init=copy(states(m.observed)))
# @inline nstates(m::DFMModel) = sum(nstates, values(m.components), init=nstates(m.observed))

# info used in eval_XYZ functions

@inline leads(::DFMBlockOrModel) = 0
@inline lags(::ObservedBlock) = 0
@inline lags(b::ComponentsBlock) = b.order
@inline lags(m::DFMModel) = maximum(lags, values(m.components))

@inline varshks(bm::DFMBlockOrModel) = [endog(bm); exog(bm); shocks(bm)]
@inline nvarshks(bm::DFMBlockOrModel) = nendog(bm) + nexog(bm) + nshocks(bm)

@inline shocks(b::DFMBlock) = b.shks
@inline nshocks(b::ComponentsBlock) = b.size
@inline shocks(b::ObservedBlock) = b.shks
@inline nshocks(b::ObservedBlock) = length(b.shks)
# @inline shocks(m::DFMModel) = mapfoldl(shocks, append!, values(m.components), init=copy(shocks(m.observed_block)))
# @inline nshocks(m::DFMModel) = sum(shocks, values(m.components), init=nshocks(m.observed_block))

@inline endog(b::DFMBlock) = b.vars
@inline nendog(b::DFMBlock) = b.size

@inline exog(::ComponentsBlock) = ModelVariable[]
@inline nexog(::ComponentsBlock) = 0
@inline exog(b::ObservedBlock) = mapfoldl(endog, append!, values(b.components), init=ModelVariable[])
@inline nexog(b::ObservedBlock) = sum(nendog, values(b.components))
@inline exog(::DFMModel) = ModelVariable[]
@inline nexog(::DFMModel) = 0

@inline allvars(bm::DFMBlockOrModel) = varshks(bm)
@inline nallvars(bm::DFMBlockOrModel) = nvarshks(bm)

for f = (:states, :shocks, :endog)
    nf = Symbol("n", f)
    @eval begin
        @inline $f(m::DFMModel) = mapfoldl($f, append!, values(m.components), init=copy($f(m.observed_block)))
        @inline $nf(m::DFMModel) = sum($nf, values(m.components), init=$nf(m.observed_block))
    end
end


## ##########################################################################
#    user interface to setup the dfm model 

#  Create an empty DFM model 
#       m = DFMModel(<name>)
#  Add factors -- common and idiosyncratic components
#       add_components!(m, 
#           F = CommonComponents("F", 2, 1), 
#           ic = IdiosyncraticComponents()
#       )
#  Add observed variables and map them to which components they load
#       map_loadings!(m
#           [:a, :b] => :F,
#           :a => :ic
#       )
#  In this example the model has two observed variables.
#  Both variables load the common factor F (two factors, VAR(1))
#  Only :a has an idiosyncratic component, so :b needs an observation shock. 
#  Add shocks
#       add_shocks!(m, :b)
#  This will add a shock :b_shk associated with the observation equation for :b.
#  Syntax :b => :b_shock_name can be used to customize the shock's name.
#  When the model is fully defined, call initialize
#       initialize_dfm!(m)


"""
    add_components!(m::DFMModel; name = component, ...)
    add_components!(m::DFMModel, :name => component, ...)

Add the given component blocks to the model with their given names. 
"""
function add_components! end
export add_components!

@inline add_components!(m::DFMModel; kwargs...) = add_components!(m, kwargs...)
function add_components!(m::DFMModel, args::Pair{<:Sym,<:ComponentsBlock}...)
    for (nm, df) in args
        nm = Symbol(nm)
        haskey(m.components, nm) && @warn "Replacing block $(df.name)"
        push!(m.components, nm => df)
    end
    return m
end

"""
    map_loadings!(m::DFMModel, :var => [:comp1, ...], ...)
    map_loadings!(m::DFMModel, [:var1, ...] => [:comp1, ...], ...)

Specify which observed variables load which components in the given DFM model or
observed block.

All names that appear on the left of "=>" (either on their own, or in a vector)
will be added as observed variables in the model. Any names that appear on the
right of "=>" must be names of component blocks that had been already added to
the model by previous calls to [`add_components!`](@ref).

"""
function map_loadings! end
export map_loadings!

function map_loadings!(m::DFMModel, args::Pair...)
    obs = m.observed_block
    ocomps = obs.components
    mcomps = m.components
    for (vars, comp_names) in args
        if !isa(comp_names, Vector{Symbol})
            comp_names = _tosymvec(comp_names)
        end
        # check that the component names are valid
        missing_names = setdiff(comp_names, keys(mcomps))
        if !isempty(missing_names)
            throw(ArgumentError("Component blocks by these names are missing from the model: $((missing_names...,))"))
        end
        # add references to the component blocks to the observed block
        for cn in comp_names
            ocomps[cn] = mcomps[cn]
        end
        # update our map
        if vars isa Sym
            tmp = get!(obs.var2comps, Symbol(vars), Symbol[])
            append!(tmp, comp_names)
        else
            for var in vars
                tmp = get!(obs.var2comps, Symbol(var), Symbol[])
                append!(tmp, comp_names)
            end
        end
    end
    return m
end

export add_shocks!
add_shocks!(m::DFMModel, args...) = (add_shocks!(m.observed_block, args...); m)
add_shocks!(b::ObservedBlock) = b
function add_shocks!(b::ObservedBlock, var::Sym, args...)
    push!(b.var2shk, Symbol(var) => Symbol(var, "_shk"))
    return add_shocks!(b, args...)
end
function add_shocks!(b::ObservedBlock, pair::Pair, args...)
    var = Symbol(pair.first)
    shk = Symbol(pair.second)
    push!(b.var2shk, var => shk)
    return add_shocks!(b, args...)
end

function _init_observed!(b::ObservedBlock)
    # remove duplicates in components mentioned in the loadings map
    foreach(unique!, values(b.var2comps))
    # add variables and shocks mentioned in the loadings and shocks maps
    empty!(b.vars)
    append!(b.vars, keys(b.var2comps))
    append!(b.vars, keys(b.var2shk))
    unique!(b.vars)
    b.size = length(b.vars)
    empty!(b.shks)
    append!(b.shks, to_shock.(values(b.var2shk)))
    # build the inverse loadings map
    empty!(b.comp2vars)
    for (var, components) in b.var2comps
        for blk in components
            push!(get!(b.comp2vars, blk, Symbol[]), var)
        end
    end
    # remove any duplicates in var-lists
    foreach(unique!, values(b.comp2vars))
    # resize idiosyncratic blocks as needed
    for (name, block) in b.components
        vars = b.comp2vars[name]
        if isempty(vars)
            @warn "No variables are loading the components in $name"
            continue
        end
        block isa IdiosyncraticComponents || continue
        block.size = length(vars)
        block.vars = map(v -> Symbol(v, "_cor"), vars)
        block.shks = map(v -> to_shock(Symbol(v, "_shk")), vars)
    end
    return b
end

"""
    initialize_dfm!(m::DFMModel)

Initialize the internal data structures of a `DFMModel` instance after it is done
receiving inputs from the model developer. Also perform checks for the integrity 
of the model provided by the model developer.
"""
function initialize_dfm!(m::DFMModel)
    _init_observed!(m.observed_block)
    m._state = :ready
    return m
end
export initialize_dfm!

include("params.jl")
include("evals.jl")
include("utils.jl")

export DFM
mutable struct DFM <: AbstractModel
    model::DFMModel
    params::DFMParams
end
DFM(T::Type{<:Real}=Float64) = DFM(DFMModel(), DFMParams{T}())

eval_resid(point::AbstractMatrix, dfm::DFM) = eval_resid(point, dfm.model, dfm.params)
eval_RJ(point::AbstractMatrix, dfm::DFM) = eval_RJ(point, dfm.model, dfm.params)
eval_R!(R::AbstractVector, point::AbstractMatrix, dfm::DFM) = eval_R!(R, point, dfm.model, dfm.params)
eval_RJ!(R::AbstractVector, J::AbstractMatrix, point::AbstractMatrix, dfm::DFM) = eval_RJ!(R, J, point, dfm.model, dfm.params)
add_components!(dfm::DFM; kwargs...) = (add_components!(dfm.model, kwargs...); dfm)
add_components!(dfm::DFM, args...) = (add_components!(dfm.model, args...); dfm)
map_loadings!(dfm::DFM, args::Pair...) = (map_loadings!(dfm.model, args...); dfm)
add_shocks!(dfm::DFM, args...) = (add_shocks!(dfm.model, args...); dfm)
initialize_dfm!(dfm::DFM) = (initialize_dfm!(dfm.model); dfm.params = init_params(dfm.model); dfm)

lags(dfm::DFM) = lags(dfm.model)
leads(dfm::DFM) = leads(dfm.model)

get_covariance(dfm::DFM) = get_covariance(dfm.model, dfm.params)
get_covariance(dfm::DFM, blk::Sym) = get_covariance(dfm, Val(Symbol(blk)))
get_covariance(dfm::DFM, ::Val{:observed}) = get_covariance(dfm.model.observed_block, dfm.params.observed)
get_covariance(dfm::DFM, v::Val{B}) where {B} = (@nospecialize(v); get_covariance(dfm.model.components[B], getproperty(dfm.params, B)))

for f in (:observed, :states, :shocks, :endog, :exog, :varshks, :allvars)
    nf = Symbol("n", f)
    @eval begin
        $f(dfm::DFM) = $f(dfm.model)
        $nf(dfm::DFM) = $nf(dfm.model)
    end
end

end
