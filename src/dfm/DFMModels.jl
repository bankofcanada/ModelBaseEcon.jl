##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
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

[`DFMBlock`](@ref) is an abstract type representing a block of equations in a
[`DFMModel`](@ref).

[`DFMObsBlock`](@ref) `<: DFMBlock` represents a block of observation equations.

[`ComponentsBlock`](@ref)`<: DFMBlock` represents a block of latent
components.

"""
DFMModels

####################################################

export DFMModel, DFMParams
export DFMBlock, ComponentsBlock, ObservedBlock, CommonComponents, IdiosyncraticComponents
export observed, nobserved, states, nstates
export varshks, nvarshks, endog, nendog, exog, nexog
export lags, leads, order

####################################################

"""Abstract type for all DFM blocks"""
abstract type DFMBlock end

"""Abstract type for types providing concrete values for the `MIXEDFREQ`
type-parameter of [`ComponentsBlock`](@ref)"""
abstract type MixedFrequency end

"""
    ComponentsBlock{TYPE,MIXEDFREQ} <: DFMBlock

A struct representing a block of latent components in a DFM model.
See also [`CommonComponents`](@ref), [`IdiosyncraticComponents`](@ref) and
[`ObservedBlock`](@ref).

 `TYPE` can be either `:Dense` (for a `CommonComponents` block) or
`:Diagonal` (for `IdiosyncraticComponents` block).

`MIXEDFREQ<:MixedFrequency` specifies if some, or all, variables in the
block run at a lower frequency than the rest of the DFM model.
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

"""
    struct NoMixFreq <: MixedFrequency

Indicates that all variables in the given block run at the DFM's highest frequency.
"""
struct NoMixFreq <: MixedFrequency end

"""
    struct MixFreq{WHICH} <: MixedFrequency

Indicates that some or all variables in the given block run at a lower frequency
than the DFM's highest frequency. The relationship between the two frequencies
is specified by the type-parameter `WHICH`.

For example, `MixFreq{:MQ}` indicates that this block runs at a quarterly
frequency in a model in which the base frequency is Monthly.

"""
struct MixFreq{WHICH} <: MixedFrequency end

"""
    mf_ncoefs(MF::Type{<:MixedFrequency})

Returns the number of coefficients in the mixing constraint for the given
subtype of `MixedFrequency`.
"""
mf_ncoefs(MF::Type{<:MixedFrequency}) = length(mf_coefs(MF))

"""
    mf_coefs(MF::Type{<:MixedFrequency})

Returns the coefficients of the mixing constraint for the given subtype of
`MixedFrequency`.
"""
function mf_coefs end

Base.@propagate_inbounds mf_coefs(::Type{NoMixFreq}, i=:) = @inbounds (1,)[i]
Base.@propagate_inbounds mf_coefs(::Type{MixFreq{:MQ}}, i=:) = (1, 2, 3, 2, 1)[i]

"""
    MixFreq(mf, blk)

Convenience wrapper that re-construct the given block as a mixed-frequency block. 
The given block must be a `NoMixFreq` block.
"""
function MixFreq(WHICH::Symbol, blk::ComponentsBlock{TYPE,NoMixFreq}) where {TYPE}
    MF = MixFreq{WHICH}
    ComponentsBlock{TYPE,MF}(blk.vars, blk.shks, blk.size, blk.order, max(blk.nlags, mf_ncoefs(MF)))
end
export MixFreq, NoMixFreq

"""
    CommonComponents{MF<:MixedFrequency} = ComponentBlock{:Dense,MF}

A struct type representing factors that are common to a block of observed
variables. The loadings matrix, the transition matrices, and the shocks
covariance matrix for this type of block are all dense matrices. See also
[`IdiosyncraticComponents`](@ref).

    CommonComponents(name, size; order, nlags)

Create an instance. `name` is mandatory and can be a single name (string or
symbol) or a list of names. The second positional argument, `size`, is optional
and defaults to the number of names.

If you provide a single name and size greater than 1 then the names of 
the variables are generated from the given name with superscript digits.

The `order` named argument is the VAR order of the components in this block. The
default is 1. 

The `nlags` named argument typically equals `order`, however it may be different
in mixed-frequency models in the block with the lower frequency. Normally this
would be handled automatically and there is no need for the user to specify
`nlags` directly.

"""
CommonComponents

const CommonComponents{MF<:MixedFrequency} = ComponentsBlock{:Dense,MF}
CommonComponents(name, size::Integer=name isa LikeVec ? length(name) : 1; order::Integer=1, nlags::Integer=order) = CommonComponents{NoMixFreq}(name, size, order, nlags)

"""
    IdiosyncraticComponents{MF<:MixedFrequency} = ComponentBlock{:Diagonal, MF}

A struct type representing factors that are common to a block of observed
variables. The loadings matrix, the transition matrices, and the shocks
covariance matrix for this type of block are all diagonal. See also
[`CommonComponents`](@ref).
"""
IdiosyncraticComponents

const IdiosyncraticComponents{MF<:MixedFrequency} = ComponentsBlock{:Diagonal,MF}
IdiosyncraticComponents(or::Integer=1; order::Integer=or, nlags::Integer=order) = IdiosyncraticComponents{NoMixFreq}("", 0, order, nlags)

########## 
###  Data structure and algorithms needed to keep track of which observed variable loads which component
###  The components are organized in blocks.
###  An observed can load the entire block (all components in it) or some components from a block.
###  In both cases, the entire block is needed, plus information specifying all or which components are loaded.

"""
    abstract type _BlockComponentRef{ALL,N,NAMES} end

Abstract type for a reference to components.

This is used internally when creating a map of references between observed and
latent blocks, that is which variables in each observed block load on which
variables in which latent blocks.

* `ALL` is `true` or `false` indicating whether the entire block is being
  referenced, or only some of the components in it.
* `N` is the number of components in the block being referenced.
* `NAMES` is a tuple of `Symbol`s with the names being referenced.

Invariant: `N == length(NAMES)`. 

Convention: `N=0` and `NAMES=()` means that all components in the block are
being referenced. This is needed when we don't know the names of the components,
but we do know that they're all referenced.

Convention: all derived types have inner constructors that take just a list of
`NAMES` to construct the instance.

See also: [`comp_ref`](@ref), [`n_comp_refs`](@ref), [`inds_comp_refs`](@ref)

"""
abstract type _BlockComponentRef{ALL,N,NAMES} end

"""
    struct _BlockRef{N,NAMES} <: _BlockComponentRef{true,N,NAMES} ... end
        
A specific [`_BlockComponentRef`](@ref) type indicating that all names in the
latent block are being referenced.
"""
struct _BlockRef{N,NAMES} <: _BlockComponentRef{true,N,NAMES}
    function _BlockRef(names::SymVec)
        N = length(names)
        NAMES = ((Symbol(n) for n in names)...,)
        return new{N,NAMES}()
    end
end

"""
    struct _CompRef{N,NAMES} <: _BlockComponentRef{false,N,NAMES} ... end
    
A specific [`_BlockComponentRef`](@ref) that some, but not all, names in the
latent block are being referenced.
"""
struct _CompRef{N,NAMES} <: _BlockComponentRef{false,N,NAMES}
    # `NAMES` are all names in the latent block. 
    inds::Vector{Int}  # maintains the indices in `NAMES` being referenced
    function _CompRef(names::SymVec)
        N = length(names)
        NAMES = ((Symbol(n) for n in names)...,)
        new{N,NAMES}(Int[])
    end
end

"""
    struct _NoCompRef{N,NAMES} <: _BlockComponentRef{false,N,NAMES} end
    
A specific [`_BlockComponentRef`](@ref) indicating that no names in the given
block are being referenced.
"""
struct _NoCompRef{N,NAMES} <: _BlockComponentRef{false,N,NAMES}
    function _NoCompRef(names::SymVec)
        N = length(names)
        NAMES = ((Symbol(n) for n in names)...,)
        new{N,NAMES}()
    end
end

"""
    comp_ref(::ComponentsBlock)
    comp_ref(::ComponentsBlock, ::Sym)
    comp_ref(::_BlockComponentRef)
    comp_ref(::_BlockComponentRef, ::Sym)

Function used internally to "register" a reference to a component block, either
for the entire block or for a specific  component within the block.
Returns an instance of [`_BlockComponentRef`](@ref).
"""
function comp_ref end

# reference a variable in a block of idiosyncratic components. 
comp_ref(::IdiosyncraticComponents) = _BlockRef(())  # names are not known yet
# reference a block of common components - either the entire block ...
comp_ref(b::CommonComponents) = _BlockRef(b.vars)
# ... or add a component to the reference list
comp_ref(b::CommonComponents, comp::Sym) = comp_ref(_CompRef(b.vars), Val(Symbol(comp)))

# ???
comp_ref(::_BlockComponentRef{ALL,N,NAMES}) where {ALL,N,NAMES} = _BlockRef(NAMES)

# method to add a component to an existing reference
comp_ref(c::_BlockComponentRef, comp::Sym) = comp_ref(c, Val(Symbol(comp)))
# add a component to a reference to the entire block - it's a no-op, but we check that the component exists within the block
@generated function comp_ref(c::_BlockRef{N,NAMES}, ::Val{comp}) where {N,NAMES,comp}
    if N == 0 || comp in NAMES
        return :(c)
    end
    return :(throw(ArgumentError(string(comp) * " is not a component.")))
end
# add a component to a partial reference.
@generated function comp_ref(c::_CompRef{N,NAMES}, ::Val{comp}) where {N,NAMES,comp}
    # find the index of the component; if not found - error
    # if the component is already referenced, do nothing
    # if this is the last component, change the reference to a _BlockRef, 
    # i.e. a reference to the entire block
    # otherwise, push the index into the list
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

n_comp_refs(::_BlockRef{0}) = error("Cannot determine the number of referenced components.")
n_comp_refs(::_BlockRef{N}) where {N} = N
n_comp_refs(c::_CompRef) = length(c.inds)
n_comp_refs(::_NoCompRef) = 0

inds_comp_refs(c::_BlockRef) = 1:n_comp_refs(c)
inds_comp_refs(c::_CompRef) = c.inds
inds_comp_refs(r::_NoCompRef) = Int[]

vars_comp_refs(::_BlockRef{0}) = error("Cannot determine the names of referenced components.")
vars_comp_refs(::_BlockRef{N,NAMES}) where {N,NAMES} = NAMES
vars_comp_refs(c::_CompRef{N,NAMES}) where {N,NAMES} = NAMES[c.inds]
vars_comp_refs(::_NoCompRef) = Symbol[]

Base.show(io::IO, c::_BlockComponentRef) = show(io, MIME"text/plain"(), c)
Base.show(io::IO, ::MIME"text/plain", c::_BlockRef{N,NAMES}) where {N,NAMES} = print(io, NAMES)
Base.show(io::IO, ::MIME"text/plain", c::_CompRef{N,NAMES}) where {N,NAMES} = print(io, NAMES[c.inds])
Base.show(io::IO, ::MIME"text/plain", c::_NoCompRef{N,NAMES}) where {N,NAMES} = print(io, "∅")

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

mf_coefs(::ObservedBlock{MF}) where {MF} = mf_coefs(MF)
mf_ncoefs(::ObservedBlock{MF}) where {MF} = mf_ncoefs(MF)

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
@inline lags(::ObservedBlock{MF}) where {MF} = mf_ncoefs(MF) - 1
@inline lags(b::ComponentsBlock) = b.nlags
@inline lags(m::DFMModel) = maximum(lags, values(m.components); init=0)

@inline order(b::ComponentsBlock) = b.order

@inline varshks(bm::DFMBlockOrModel) = [endog(bm); exog(bm); shocks(bm)]
@inline nvarshks(bm::DFMBlockOrModel) = nendog(bm) + nexog(bm) + nshocks(bm)

@inline shocks(b::DFMBlock) = b.shks
@inline nshocks(b::ComponentsBlock) = b.size
@inline shocks(b::ObservedBlock) = b.shks
@inline nshocks(b::ObservedBlock) = length(b.shks)

collect_state_shocks(m::DFMModel; init=ModelVariable[]) = mapfoldl(shocks, append!, values(m.components); init)
count_state_shocks(m::DFMModel; init=0) = sum(nshocks, values(m.components); init)
collect_observed_shocks(m::DFMModel; init=ModelVariable[]) = mapfoldl(shocks, append!, values(m.observed); init)
count_observed_shocks(m::DFMModel; init=0) = sum(nshocks, values(m.observed); init)
@inline shocks(m::DFMModel) = collect_state_shocks(m; init=collect_observed_shocks(m))
@inline nshocks(m::DFMModel) = count_observed_shocks(m) + count_state_shocks(m)

@inline endog(b::DFMBlock) = b.vars
@inline nendog(b::DFMBlock) = b.size

@inline exog(::ComponentsBlock) = ModelVariable[]
@inline nexog(::ComponentsBlock) = 0
@inline exog(::DFMModel) = ModelVariable[]
@inline nexog(::DFMModel) = 0

# implementation of exog for ObservedBlock is more complicated because each 
# variable may reference some but not all components in a block
_comp_exog(crefs) = unique!(mapreduce(vars_comp_refs, append!, values(crefs), init=ModelVariable[]))
exog(b::ObservedBlock) = mapfoldl(_comp_exog, append!, values(b.comp2vars), init=ModelVariable[])
nexog(b::ObservedBlock) = sum(length ∘ _comp_exog, values(b.comp2vars))

@inline allvars(bm::DFMBlockOrModel) = varshks(bm)
@inline nallvars(bm::DFMBlockOrModel) = nvarshks(bm)

endog(m::DFMModel) = [observed(m); states(m)]
nendog(m::DFMModel) = nobserved(m) + nstates(m)

################################################################################

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
Add observed variable and observed blocks to a DFM model. 
    
    add_observed!(m; name = observed_block, ...)
    
`name` is the name of the observed block. Use this method if an instance of
`ObservedBlock` is available. N.B. `observed_block` is added to the model
without copying, so any changes (like adding variables or shocks) would be
reflected in the same instance.
    
    add_observed!(m; name = vars, ...)

`name` is the name of the observed block and `vars` is a collection of variables
to be added to the block with the given name. If the block doesn't already exist
a new `ObservedBlock` is created and added to the model `m`.

    add_observed!(m, vars)

A shortcut for models with one observed block. A new `ObservedBlock` with a
default name is created, if one does not already exist. If `m` already contains
and observed block, `vars` added to it. It is an error to call this method if
`m` already has more than one observed block.

"""
function add_observed! end

add_observed!(m::DFMModel; kwargs...) = add_observed!(m, [kwargs...])
add_observed!(m::DFMModel, arg, args...) = add_observed!(m, [arg, args...])
function add_observed!(m::DFMModel, vec::AbstractVector)
    for v in vec
        add_observed!(m, v)
    end
    return m
end

#  method to be called like this: add_observed!(m, :blk_name => obs_blk)
function add_observed!(m::DFMModel, (bname, blk)::Pair{<:Sym,<:ObservedBlock})
    m.observed[bname] = blk
    return m
end

"""
    b = add_observed_vars!(b, vars)

Add variables to the given [`ObservedBlock`](@ref) `b`. `vars` is a `Vector` of
variable names, which can be specified as string or symbols. Variables in `vars`
that already exist in `b` are silently ignored.

"""
function add_observed_vars!(b::ObservedBlock, vars::SymVec)
    v2c = b.var2comps
    for var in vars
        get!(v2c, Symbol(var), NamedList{_BlockComponentRef}())
    end
    return b
end

#  method to be called like this: add_observed!(m, :blk_name => (:var1, :var2))
function add_observed!(m::DFMModel, (bname, vnames)::Pair{<:Sym,<:SymVec})
    add_observed_vars!(get!(m.observed, bname, ObservedBlock()), vnames)
    return m
end

#  method to be called like this: add_observed!(m, :blk_name => :var1)
function add_observed!(m::DFMModel, (bname, var)::Pair{<:Sym,<:Sym})
    add_observed_vars!(get!(m.observed, bname, ObservedBlock()), (var,))
    return m
end

const _dobn = :observed  # default observed block name

function add_observed!(m::DFMModel, varnames::SymVec)
    o = m.observed
    if isempty(o)
        push!(o, _dobn => add_observed_vars!(ObservedBlock(), varnames))
        return m
    end
    if length(o) == 1
        add_observed_vars!(first(values(o)), varnames)
        return m
    end
    error("In a model with more than one observed block the name of the block must be explicitly given.")
end
export add_observed!

################################################################################

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

################################################################################

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
        push!(observed, _dobn => ObservedBlock())
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
        for (i, var) in enumerate(vars)
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
        not_assigned = [v for (i, v) in enumerate(vars) if not_done[i]]
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
            found || error("Component $cn not found in the model.")
        end
    end
    return m
end

################################################################################

"""
    add_shocks!(m, :var, ...)
    add_shocks!(m, :var => :shk, ...)

Add shocks to the given *observed* variables. The shock name can be given
explicitly (using the `=>` syntax) or, if omitted, it will be generated
automatically from the variable name.

If the observed variable already exists, the shock is added to the  
observed block that contains the variable. If the variable already has a shock
it is an error.

If the observed variable does not exist, then the behaviour depends on whether
the model contains one or many observed blocks. If one observed block, the
variable and the shock are added. If many observed blocks, then it is an error.

Note that components blocks handle their shocks automatically, so this is only
necessary for observed shocks. Observed shocks are not automatic because some
observed variables may have idiosyncratic components.

"""
function add_shocks! end
export add_shocks!

add_shocks!(b::Union{DFMModel,ObservedBlock}, arg, args...) = add_shocks!(b, [arg, args...])
function add_shocks!(b::Union{DFMModel,ObservedBlock}, args::AbstractVector)
    for arg in args
        add_shocks!(b, arg)
    end
    return b
end

# method where the variable is given and a default shock name is made for it
add_shocks!(b::ObservedBlock, var::Sym) = add_shocks!(b, var => _make_shock(var))

# method where the variable and its shock name are given 
function add_shocks!(b::ObservedBlock, varshk::Pair{<:Sym,<:Sym})
    v2s = b.var2shk
    (var, shk) = varshk
    var = Symbol(var)
    if haskey(v2s, var)
        error("Variable `$var` already has a shock: `$(v2s[var])`.")
    end
    v2s[var] = Symbol(shk)
    return b
end

add_shocks!(m::DFMModel, var::Sym) = add_shocks!(m, var => _make_shock(var))
function add_shocks!(m::DFMModel, varshk::Pair{<:Sym,<:Sym})
    mobs = m.observed
    if isempty(mobs)
        # no observed blocks -- create the default one
        mobs[_dobn] = add_shocks!(ObservedBlock(), varshk)
        return m
    end
    if length(mobs) == 1
        # one observed block -- use it
        oblk = first(values(mobs))
        add_shocks!(oblk, varshk)
        return m
    end
    # more than one observed block - we find the one that has the variable
    var = varshk.first
    var = Symbol(var)
    for oblk in values(mobs)
        if haskey(oblk.var2comps, var) || haskey(oblk.var2shk, var)
            add_shocks!(oblk, varshk)
            return m
        end
    end
    error("Variable `$var` not found in any observed block.")
end

################################################################################

# add variables and shocks. Also updates IdiosyncraticComponents with 
# variables from the given block that load on it
function _init_observed_pass1!(b::ObservedBlock)
    # add variables and shocks mentioned in the loadings and shocks maps
    empty!(b.vars)
    append!(b.vars, keys(b.var2comps))
    append!(b.vars, keys(b.var2shk))
    unique!(b.vars)
    b.size = length(b.vars)
    empty!(b.shks)
    append!(b.shks, values(b.var2shk))
    # add idiosyncratic components referenced in this observed block to 
    # their idiosyncratic block
    for (ic_name, ic_blk) in b.components
        ic_blk isa IdiosyncraticComponents || continue
        any_loaders = false
        for (var, crefs) in b.var2comps
            if haskey(crefs, ic_name)
                any_loaders = true
                push!(ic_blk.vars, _make_ic_name(var))
            end
        end
        any_loaders || error("Internal error: $ic_name mentioned but no observed variable loads it.")
        ic_blk.size = length(ic_blk.vars)
        ic_blk.shks = _make_shocks(ic_blk.vars)
    end
end

# assuming _init_observed_pass1! was done,
# here we finalize the connectivity maps
# update var2comps for idiosyncratic components
# create the inverse map, comp2vars
function _init_observed_pass2!(b::ObservedBlock)
    # update the var2comp map for idiosyncratic components
    for (var, crefs) in pairs(b.var2comps)
        for cn in keys(crefs)
            cr = crefs[cn]
            if cr isa _BlockRef{0}
                comp = b.components[cn]
                crefs[cn] = comp_ref(_CompRef(comp.vars), _make_ic_name(var))
            end
        end
    end
    # build the inverse loadings map
    b_c2v = b.comp2vars
    empty!(b_c2v)
    for cn in keys(b.components)
        tmp = NamedList{_BlockComponentRef}()
        for vn in b.vars
            tmp[vn] = _NoCompRef(b.vars)
        end
        b_c2v[cn] = tmp
    end
    for (var, crefs) in b.var2comps
        for (bname, c) in crefs
            b_c2v[bname][var] = c
        end
    end
end


"""
    check_dfm(m)

Verify the consistency of internal data structures. Throw and `ErrorException`
with an appropriate error message if any problem is found.

"""
function check_dfm(m::DFMModel)
    if m._state != :ready
        error("Model must be initialized first.")
    end
    m._state = :check
    # check observed blocks
    if isempty(m.observed)
        error("Model does not have any observed variables.")
    end
    for (onm, oblk) in pairs(m.observed)
        # check for duplicate variables (in previous blocks only)
        for (onm1, oblk1) in pairs(m.observed)
            onm1 == onm && break
            dups = intersect(oblk.vars, oblk1.vars)
            if !isempty(dups)
                error("""Duplicate variable(s) found: $("(`"*join(dups, "`,`")*"`)") in blocks `$(onm)` and `$(onm1)`""")
            end
        end
        # check for observed that don't load anything
        if isempty(oblk.components)
            error("Observed block does not load any components: $(onm)")
        end
        # each variable must either have a shock or an idiosyncratic 
        # component (autocorrelated shock), but not both. check for violation
        ic_names = [k for (k, v) in pairs(oblk.components) if v isa IdiosyncraticComponents]
        for var in oblk.vars
            has_comps = haskey(oblk.var2comps, var) && !isempty(oblk.var2comps[var])
            has_shk = haskey(oblk.var2shk, var)
            if !has_comps && !has_shk
                error("Variable `$var` in block `$onm`` does not load any component and does not have a shock.")
            end
            if has_comps
                nic = has_shk + sum(Base.Fix2(in, ic_names), keys(oblk.var2comps[var]))
                if nic == 0
                    error("Variable `$var` in block `$onm` has neither a shock nor an idiosyncratic component.")
                elseif nic > 1
                    @warn("Variable `$var` in block `$onm` has more than one shock or idiosyncratic components.")
                end
            end
        end
    end
    # make sure there are no duplicate variables (this is an internal check)
    varshks(m) == unique(varshks(m)) || error("Duplicate variables or shocks")
    m._state = :ready
    return m
end

"""
    initialize_dfm!(m; check=true)

Initialize the internal data structures of a `DFM` or `DFMModel` instance after
it is done receiving inputs from the model developer. Also perform checks for
the integrity of the model provided by the model developer.
"""
function initialize_dfm!(m::DFMModel; check=true)
    m._state = :new
    # empty idiosyncratic blocks, so they can be re-populated from scratch
    for cblk in values(m.components)
        cblk isa IdiosyncraticComponents || continue
        empty!(cblk.vars)
        empty!(cblk.shks)
    end
    for oblk in values(m.observed)
        # add variables and shocks. also, update idiosyncratic component blocks
        _init_observed_pass1!(oblk)
    end
    for oblk in values(m.observed)
        # update the var2comps map and create the comp2vars map
        _init_observed_pass2!(oblk)
    end
    m._state = :ready
    return check ? check_dfm(m) : m
end
export initialize_dfm!

################################################################################

include("params.jl")
include("evals.jl")
include("utils.jl")

################################################################################

export DFM
mutable struct DFM <: AbstractModel
    model::DFMModel
    params::DFMParams
end
DFM(name::Sym=:dfm, T::Type{<:Real}=Float64) = DFM(DFMModel(name), DFMParams{T}())

eval_resid(point::AbstractMatrix, dfm::DFM) = eval_resid(point, dfm.model, dfm.params)
eval_RJ(point::AbstractMatrix, dfm::DFM) = eval_RJ(point, dfm.model, dfm.params)
eval_R!(R::AbstractVector, point::AbstractMatrix, dfm::DFM) = eval_R!(R, point, dfm.model, dfm.params)
eval_RJ!(R::AbstractVector, J::AbstractMatrix, point::AbstractMatrix, dfm::DFM) = eval_RJ!(R, J, point, dfm.model, dfm.params)
add_components!(dfm::DFM; kwargs...) = (add_components!(dfm.model, kwargs...); dfm)
add_components!(dfm::DFM, args...) = (add_components!(dfm.model, args...); dfm)
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
# states_with_lags((n, b)::Pair{Symbol,<:DFMBlock}) = states_with_lags(b)
states_with_lags(::ObservedBlock) = Symbol[]
function states_with_lags(blk::ComponentsBlock)
    return [_make_lag_name(v, lags(blk)-l) for l = 1:lags(blk) for v in states(blk)]
end

get_mean(dfm::DFM) = get_mean!(Vector{eltype(dfm.params)}(undef, nobserved(dfm)), dfm)
get_mean!(x::AbstractVector, dfm::DFM) = get_mean!(x, dfm.model, dfm.params)

get_loading(dfm::DFM) = get_loading!(Matrix{eltype(dfm.params)}(undef, nobserved(dfm), nstates_with_lags(dfm)), dfm)
get_loading!(x::AbstractMatrix, dfm::DFM) = get_loading!(x, dfm.model, dfm.params)

get_transition(dfm::DFM) = get_transition!(Matrix{eltype(dfm.params)}(undef, nstates_with_lags(dfm), nstates_with_lags(dfm)), dfm)
get_transition!(x::AbstractMatrix, dfm::DFM) = get_transition!(x, dfm.model, dfm.params)

export states_with_lags, nstates_with_lags

include("constraints.jl")

end

