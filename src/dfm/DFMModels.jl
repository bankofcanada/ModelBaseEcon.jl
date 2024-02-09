##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2024, Bank of Canada
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
const NamedList{V} = LittleDictVec{Symbol,V}

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

abstract type MixedFrequency end

"""
    ComponentBlock{TYPE,MIXEDFREQ} <: DFMBlock

A struct representing a block of latent components in a DFM model.
See also [`CommonComponents`](@ref), [`IdiosyncraticComponents`](@ref) and
[`ObservedBlock`](@ref).

If MIXEDFREQ is not specified it defaults to a single-frequency model. `:MQ` means a model where
some observed variables are Monthly and the rest are Quarterly. The type of each observed variable is
determined by its idiosyncratic component.
"""
mutable struct ComponentsBlock{TYPE,MF<:MixedFrequency} <: DFMBlock
    vars::Vector{ModelVariable}
    shks::Vector{ModelVariable}
    size::Int
    order::Int
    nlags::Int
end

function ComponentsBlock{TYPE,MF}(names::SymVec, size::Integer, order::Integer, nlags::Integer) where {TYPE,MF}
    if length(names) != size
        throw(ArgumentError("Number of names ($(length(names))) must match factor size ($(size))."))
    end
    vars = _tosymvec(names)
    shks = _make_shocks(names)
    return ComponentsBlock{TYPE,MF}(vars, shks, Int(size), Int(order), Int(nlags))
end

function ComponentsBlock{TYPE,MF}(name::Sym, size::Integer, order::Integer, nlags::Integer) where {TYPE,MF}
    vars = _make_factor_names(name, size)
    shks = _make_shocks(vars)
    return ComponentsBlock{TYPE,MF}(vars, shks, Int(size), Int(order), Int(nlags))
end

struct NoMixFreq <: MixedFrequency end
struct MixFreq{WHICH} <: MixedFrequency end


mf_ncoefs(MF::Type{<:MixedFrequency}) = length(mf_coefs(MF))
mf_coefs(::Type{NoMixFreq}) = (1,)
mf_coefs(::Type{MixFreq{:MQ}}) = (1, 2, 3, 2, 1)

function MixFreq(WHICH::Symbol, blk::ComponentsBlock{TYPE,NoMixFreq}) where {TYPE}
    MF = MixFreq{WHICH}
    ComponentsBlock{TYPE,MF}(blk.vars, blk.shks, blk.size, blk.order, max(blk.nlags, mf_ncoefs(MF)))
end
export MixFreq, NoMixFreq

"""
    CommonComponents = ComponentBlock{:Dense}

A struct type representing factors that are common to a block of observed
variables. The loadings matrix, the trnasition matrices, and the shocks
covariance matrix for this type of block are all dense matrices. See also
[`IdiosyncraticComponents`](@ref).
"""
const CommonComponents{MF<:MixedFrequency} = ComponentsBlock{:Dense,MF}
CommonComponents(name, size::Integer=name isa LikeVec ? length(name) : 1; order::Integer=1, nlags::Integer=order) = CommonComponents{NoMixFreq}(name, size, order, nlags)

"""
    IdiosyncraticComponents = ComponentBlock{:Diagonal}

A struct type representing factors that are common to a block of observed
variables. The loadings matrix, the trnasition matrices, and the shocks
covariance matrix for this type of block are all diagonal. See also
[`CommonComponents`](@ref).
"""
const IdiosyncraticComponents{MF<:MixedFrequency} = ComponentsBlock{:Diagonal,MF}
IdiosyncraticComponents(or::Integer=1; order::Integer=or, nlags::Integer=order) = IdiosyncraticComponents{NoMixFreq}("", 0, order, nlags)

########## 
###  Data structure and algorithms needed to keep track of which observed variable loads which component
###  The components are organized in blocks.
###  An observed can load the entire block (all components in it) or some components from a block.
###  In both cases, the entire block is needed, plus information specifying all or which components are loaded.

"abstract type for a reference to components"
abstract type _BlockComponentRef{ALL,N,NAMES} end
"reference to an entire block (all components in it)"
struct _BlockRef{N,NAMES} <: _BlockComponentRef{true,N,NAMES}
    names::NTuple{N,Symbol}
    _BlockRef(names::SymVec) = (N = length(names); NAMES = ((Symbol(n) for n in names)...,); new{N,NAMES}(NAMES))
end
"reference to some, but not all, components in a block"
struct _CompRef{N,NAMES} <: _BlockComponentRef{false,N,NAMES}
    names::NTuple{N,Symbol}
    inds::Vector{Int}
    _CompRef(names::SymVec) = (N = length(names); NAMES = ((Symbol(n) for n in names)...,); new{N,NAMES}(NAMES, Int[]))
end

comp_ref(::_BlockComponentRef{ALL,N,NAMES}) where {ALL,N,NAMES} = _BlockRef(NAMES)
comp_ref(::IdiosyncraticComponents) = _BlockRef(())
comp_ref(b::CommonComponents) = _BlockRef(b.vars)
comp_ref(b::CommonComponents, comp::Sym) = comp_ref(_CompRef(b.vars), Val(Symbol(comp)))
comp_ref(c::_BlockComponentRef, comp::Sym) = comp_ref(c, Val(Symbol(comp)))
@generated function comp_ref(c::_BlockRef{N,NAMES}, ::Val{comp}) where {N,NAMES,comp}
    if N == 0 || comp in NAMES
        return :(c)
    end
    return :(throw(ArgumentError(string(comp) * " is not a component.")))
end
@generated function comp_ref(c::_CompRef{N,NAMES}, ::Val{comp}) where {N,NAMES,comp}
    ind = 1
    while ind <= N && NAMES[ind] != comp
        ind += 1
    end
    ind > N && return :(throw(ArgumentError(string(comp) * " is not a component.")))
    return quote
        $ind in c.inds && return c
        length(c.inds) == N - 1 && return _BlockRef(NAMES)
        sort!(push!(c.inds, $ind))
        return c
    end
end

_n_comp_refs(::_BlockRef{0}) = error("Cannot determine the number of referenced components")
_n_comp_refs(::_BlockRef{N}) where {N} = N
_n_comp_refs(c::_CompRef) = length(c.inds)

_inds_comp_refs(r::_BlockRef) = 1:_n_comp_refs(r)
_inds_comp_refs(r::_CompRef) = r.inds

Base.show(io::IO, c::_BlockComponentRef) = show(io, MIME"text/plain"(), c)
Base.show(io::IO, ::MIME"text/plain", c::_BlockRef{N,NAMES}) where {N,NAMES} = print(io, NAMES)
Base.show(io::IO, ::MIME"text/plain", c::_CompRef{N,NAMES}) where {N,NAMES} = print(io, NAMES[c.inds])

"""
    ObservedBlock <: DFMBlock

A struct representing the observed variables in a DFM model.
See also [`ComponentBlock`](@ref).
"""
mutable struct ObservedBlock{MF<:MixedFrequency} <: DFMBlock
    vars::Vector{ModelVariable}
    shks::Vector{ModelVariable}
    size::Int
    # order is always 0
    components::NamedList{ComponentsBlock}
    var2comps::NamedList{NamedList{_BlockComponentRef}}
    comp2vars::NamedList{NamedList{_BlockComponentRef}}
    var2shk::NamedList{Symbol}
end

@inline ObservedBlock(MF::Type{<:MixedFrequency}=NoMixFreq) = ObservedBlock{MF}(
    ModelVariable[], ModelVariable[], 0,
    NamedList{ComponentsBlock}(),
    NamedList{NamedList{_BlockComponentRef}}(),
    NamedList{NamedList{_BlockComponentRef}}(),
    NamedList{Symbol}(),
)

function MixFreq(WHICH::Symbol, blk::ObservedBlock{NoMixFreq})
    MF = MixFreq{WHICH}
    ObservedBlock{MF}((getfield(blk, fn) for fn in fieldnames(typeof(blk)))...,)
end

"""
    DFMModel

A struct representing a DFM model. It contains an [`ObservedBlock`](@ref) and 
a collection of [`ComponentBlock`](@ref)s. 
"""
mutable struct DFMModel <: AbstractModel
    name::Symbol
    _state::Symbol
    observed::NamedList{ObservedBlock}
    components::NamedList{ComponentsBlock}
end
DFMModel(name::Sym=:dfm) = DFMModel(Symbol(name), :new, NamedList{ObservedBlock}(), LittleDict{Symbol,ComponentsBlock}())

const DFMBlockOrModel = Union{DFMModel,DFMBlock}

## ##########################################################################
#    functions 

# info related to state-space representation of model
@inline observed(::ComponentsBlock) = ModelVariable[]
@inline nobserved(::ComponentsBlock) = 0
@inline observed(b::ObservedBlock) = b.vars
@inline nobserved(b::ObservedBlock) = b.size

@inline observed(m::DFMModel) = mapfoldl(observed, append!, values(m.observed), init=ModelVariable[])
@inline nobserved(m::DFMModel) = sum(nobserved, values(m.observed), init=0)

@inline states(::ObservedBlock) = ModelVariable[]
@inline nstates(::ObservedBlock) = 0
@inline states(b::ComponentsBlock) = b.vars
@inline nstates(b::ComponentsBlock) = b.size

@inline states(m::DFMModel) = mapfoldl(states, append!, values(m.components), init=ModelVariable[])
@inline nstates(m::DFMModel) = sum(nstates, values(m.components), init=0)

# info used in eval_XYZ functions

@inline leads(::DFMBlockOrModel) = 0
@inline lags(::ObservedBlock{MF}) where MF = mf_ncoefs(MF) - 1
@inline lags(b::ComponentsBlock) = b.nlags
@inline lags(m::DFMModel) = maximum(lags, values(m.components))

@inline order(b::ComponentsBlock) = b.order

@inline varshks(bm::DFMBlockOrModel) = [endog(bm); exog(bm); shocks(bm)]
@inline nvarshks(bm::DFMBlockOrModel) = nendog(bm) + nexog(bm) + nshocks(bm)

@inline shocks(b::DFMBlock) = b.shks
@inline nshocks(b::ComponentsBlock) = b.size
@inline shocks(b::ObservedBlock) = b.shks
@inline nshocks(b::ObservedBlock) = length(b.shks)

add_state_shocks(m::DFMModel; init=ModelVariable[]) = mapfoldl(shocks, append!, values(m.components); init)
add_nstate_shocks(m::DFMModel; init=0) = sum(nshocks, values(m.components); init)
add_observed_shocks(m::DFMModel; init=ModelVariable[]) = mapfoldl(shocks, append!, values(m.observed); init)
add_nobserved_shocks(m::DFMModel; init=0) = sum(nshocks, values(m.observed); init)
@inline shocks(m::DFMModel) = (shks = add_observed_shocks(m); shks = add_state_shocks(m; init=shks); shks)
@inline nshocks(m::DFMModel) = add_nobserved_shocks(m) + add_nstate_shocks(m)

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

endog(m::DFMModel) = [observed(m); states(m)]
nendog(m::DFMModel) = nobserved(m) + nstates(m)


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


add_observed!(m::DFMModel; kwargs...) = add_observed!(m, kwargs...)
function add_observed!(m::DFMModel, args...)
    for a in args
        add_observed!(m, a)
    end
    return m    
end
function add_observed!(m::DFMModel, (bname, blk)::Pair{<:Sym, <:ObservedBlock})
    m.observed[bname] = blk
    return m
end

function add_observed!(m::DFMModel, (bname, vnames)::Pair{<:Sym, <:SymVec})
    add_observed_vars!(get!(m.observed, bname, ObservedBlock()), vnames)    
    return m
end

function add_observed!(m::DFMModel, (bname, var)::Pair{<:Sym, <:Sym})
    add_observed_vars!(get!(m.observed, bname, ObservedBlock()), (var,))
    return m
end

function add_observed_vars!(b::ObservedBlock, vars::SymVec)
    v2c = b.var2comps
    for var in vars
        get!(v2c, Symbol(var), NamedList{_BlockComponentRef}())
    end
    return b
end

const dobn = :observed  # default observed block name

function add_observed!(m::DFMModel, varnames::SymVec)
    o = m.observed
    if isempty(o) 
        push!(o, dobn => add_observed_vars!(ObservedBlock(), varnames))
        return m
    end
    if (length(o) == 1 && haskey(o, dobn))
        add_observed_vars!(o[dobn], varnames)    
        return m
    end
    error("Observed block not specified. Use `add_observed(m, block_name => (vars, ...))`.")
end
export add_observed!


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
        haskey(m.components, nm) && @warn "Replacing block $(nm)"
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

_add_var2comp_ref(v2c::NamedList{NamedList{_BlockComponentRef}}, vars::SymVec, blk_name::Sym, blk::ComponentsBlock, comp::Sym...) = foreach(v -> _add_var2comp_ref(v2c, v, blk_name, blk, comp...), vars)
function _add_var2comp_ref(v2c::NamedList{NamedList{_BlockComponentRef}}, var::Sym, blk_name::Sym, blk::ComponentsBlock, comp::Sym...)
    tmp = get!(v2c, Symbol(var), NamedList{_BlockComponentRef}())
    if haskey(tmp, blk_name)
        tmp[blk_name] = comp_ref(tmp[blk_name], comp...)
    else
        tmp[blk_name] = comp_ref(blk, comp...)
    end
end

_add_var2comp_ref(observed::NamedList{ObservedBlock}, var::Sym, blk_name::Sym, blk::ComponentsBlock, comp::Sym...) = _add_var2comp_ref(observed, (var,), blk_name, blk, comp...)
function _add_var2comp_ref(observed::NamedList{ObservedBlock}, vars::SymVec, blk_name::Sym, blk::ComponentsBlock, comp::Sym...)
    if isempty(observed)
        # no observed block - create a default one
        push!(observed, dobn => ObservedBlock())
    end
    if length(observed) == 1
        # if there's only one observed block, all variables go into it
        obnm, oblk = first(observed)
        oblk.components[blk_name] = blk
        _add_var2comp_ref(oblk.var2comps, vars, blk_name, blk, comp...)
        return
    end
    # multiple observed blocks - no new variables allowed
    not_done = trues(length(vars))
    for (obnm, oblk) in observed
        refd = false
        v2c = oblk.var2comps
        for (i,var) in enumerate(vars)
            not_done[i] || continue
            sv = Symbol(var)
            haskey(v2c, sv) || continue
            _add_var2comp_ref(v2c, sv, blk_name, blk, comp...)
            refd = true
            not_done[i] = false
        end
        if refd
            oblk.components[blk_name] = blk
        end
    end
    if any(not_done)
        not_assigned = ((v for (i,v) in enumerate(vars) if not_done[i])...,)
        error("Variables not assigned to an observed block: $(not_assigned)")
    end
    return
end


function map_loadings!(m::DFMModel, args::Pair...)
    # obs = m.observed
    # ocomps = obs.components
    mcomps = m.components
    mobs = m.observed
    for (vars, comp_names) in args
        comp_names = _tosymvec(comp_names)
        # add references to the component blocks to the observed block
        for cn in comp_names
            blk = get(mcomps, cn, nothing)
            if !isnothing(blk)
                _add_var2comp_ref(mobs, vars, cn, blk)
                continue
            end
            # cn name is not an entire block. Check component names
            found = false
            for (bname, blk) in mcomps
                if cn in endog(blk)
                    _add_var2comp_ref(mobs, vars, bname, blk, cn)
                    found = true
                    break
                end
            end
            found && continue
            throw(ArgumentError("Component $cn not found in the model."))
        end
    end
    return m
end

export add_shocks!
# add_shocks!(b::ObservedBlock) = b
@inline function add_shocks!(b::ObservedBlock, var::Sym)
    push!(b.var2shk, var => _make_shock(var))
    return b
end
function add_shocks!(b::ObservedBlock, pair::Pair{<:Sym,<:Sym})
    var = Symbol(pair.first)
    shk = Symbol(pair.second)
    push!(b.var2shk, var => shk)
    return b
end

add_shocks!(m::DFMModel, args...) = add_shocks!(m, args)
function add_shocks!(m::DFMModel, args)
    mobs = m.observed
    if isempty(mobs)
        push!(mobs, dobn => ObservedBlock())
    end
    if length(mobs) == 1
        _, oblk = first(mobs)
        for a in args
            add_shocks!(oblk, a)
        end
        return m
    end
    for a in args
        sv = a isa Sym ? Symbol(a) : Symbol(a[1])
        found = false
        for (_, oblk) in mobs
            if haskey(oblk.var2comps, sv)
                add_shocks!(oblk, a)
                found = true
                break
            end
        end
        if !found
            error("Variable $sv is not assigned to any observed block.")
        end
    end
    return m
end


add_shocks!(b::ObservedBlock, var::Union{Sym,Pair{<:Sym,<:Sym}}, vars...) = add_shocks!(add_shocks!(b, var), vars...)
add_shocks!(b::ObservedBlock, vars::LikeVec) = add_shocks!(b, vars...)

function _check_ic_shk(b::ObservedBlock)
    v2s = b.var2shk
    v2c = b.var2comps
    for var in b.vars
        x = haskey(v2s, var)
        comps = getindex(v2c, var)
        for (n, c) in b.components
            c isa IdiosyncraticComponents || continue
            n ∈ comps || continue
            x = x + 1
        end
        x == 1 && continue
        if x == 0
            @warn "$var has neither shock nor idiosyncratic component."
        else
            @warn "$var has more than one shock or idiosyncratic components."
        end
    end
    return b
end

function _init_observed!(b::ObservedBlock)
    # add variables and shocks mentioned in the loadings and shocks maps
    empty!(b.vars)
    append!(b.vars, keys(b.var2comps))
    append!(b.vars, keys(b.var2shk))
    unique!(b.vars)
    b.size = length(b.vars)
    empty!(b.shks)
    append!(b.shks, values(b.var2shk))
    # build the inverse loadings map
    empty!(b.comp2vars)
    for (varname, blkcomprefs) in b.var2comps
        for (blkname, c) in blkcomprefs
            tmp = get!(b.comp2vars, blkname, NamedList{_BlockComponentRef}())
            push!(tmp, varname => c)
        end
    end
    # resize idiosyncratic blocks as needed
    for (name, block) in b.components
        vars = keys(b.comp2vars[name])
        if isempty(vars)
            @warn "No variables are loading the components in $name"
            continue
        end
        block isa IdiosyncraticComponents || continue
        block.size = length(vars)
        block.vars = Symbol[Symbol(v, "_cor") for v in vars]
        block.shks = _make_shocks(block.vars)
    end
    # _check_ic_shk(b)
    return b
end

"""
    initialize_dfm!(m::DFMModel)

Initialize the internal data structures of a `DFMModel` instance after it is done
receiving inputs from the model developer. Also perform checks for the integrity 
of the model provided by the model developer.
"""
function initialize_dfm!(m::DFMModel)
    for oblk in values(m.observed)
        _init_observed!(oblk)
    end
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
add_observed!(dfm::DFM, args...) = (add_observed!(dfm.model, args...); dfm)
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

states_with_lags(m::DFM) = states_with_lags(m.model)
states_with_lags(m::DFMModel) =
    mapfoldl(append!, values(m.components), init=Symbol[]) do blk
        s = states(blk)
        ret = copy(s)
        for l = 1:lags(blk)-1
            ret = [(Symbol(v, "_lag_", l) for v in s)..., ret...]
        end
        ret
    end


end
