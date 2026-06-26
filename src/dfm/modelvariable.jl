##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################
#
# Self-contained slice of the `ModelVariable` machinery needed by the DFM
# subsystem. The DFM imports `ModelVariable`, `to_shock`, `isshock`, `shocks`,
# `nshocks`, `allvars`, `nallvars`, `eval_resid`/`eval_RJ`/`eval_R!`,
# `AbstractModel` from the parent module. The Symbolics-based core has none of
# these (it works in plain `Symbol`s and routes equation models through a kernel
# pipeline that the DFM does not use), so we provide the minimal faithful subset
# here.
##################################################################################

export ModelVariable, ModelSymbol
export to_shock, isshock
export AbstractModel
export shocks, nshocks, allvars, nallvars
export eval_resid, eval_RJ, eval_R!, eval_RJ!

"""Abstract supertype for DFM models (mirrors `ModelBaseEcon.AbstractModel`)."""
abstract type AbstractModel end

const variable_types = (:var, :shock, :exog)

"""
    struct ModelVariable

A `Symbol`-like model-variable wrapper carrying a variable type (`:var`,
`:shock`, `:exog`). Restricted port of `ModelBaseEcon.ModelVariable` — the DFM
only uses the name and the shock flag, so the transformation / steady-state
metadata of the full type is dropped.
"""
struct ModelVariable
    name::Symbol
    vr_type::Symbol   # one of :var, :shock, :exog
    function ModelVariable(n, vt)
        n isa Symbol || error("Variable name must be a Symbol, not a $(typeof(n))")
        vt ∈ variable_types || error("Unknown variable type $vt. Expected one of $variable_types")
        new(n, vt)
    end
end

const ModelSymbol = ModelVariable

ModelVariable(s::Symbol) = ModelVariable(s, :var)
ModelVariable(s::AbstractString) = ModelVariable(Symbol(s), :var)
ModelVariable(v::ModelVariable) = v

# !!! must not update v.name.
_update_vrtype(v::ModelVariable, vr_type::Symbol) = ModelVariable(v.name, vr_type)

"""
    to_shock(v)

Make a shock `ModelVariable` from `v`.
"""
to_shock(v) = _update_vrtype(convert(ModelVariable, v), :shock)

"""
    isshock(v)

Return `true` if the given `ModelVariable` is a shock, otherwise `false`.
"""
isshock(v::ModelVariable) = v.vr_type == :shock

Core.Symbol(v::ModelVariable) = v.name
Base.convert(::Type{Symbol}, v::ModelVariable) = v.name
Base.convert(::Type{ModelVariable}, v::Symbol) = ModelVariable(v)
Base.convert(::Type{ModelVariable}, v::AbstractString) = ModelVariable(Symbol(v))
Base.:(==)(a::ModelVariable, b::ModelVariable) = a.name == b.name
Base.:(==)(a::ModelVariable, b::Symbol) = a.name == b
Base.:(==)(a::Symbol, b::ModelVariable) = a == b.name

# The hash must match the hash of the symbol, so that a ModelVariable can index a
# Dict/LittleDict with Symbol keys interchangeably.
Base.hash(v::ModelVariable, h::UInt) = hash(v.name, h)
Base.hash(v::ModelVariable) = hash(v.name)

function Base.show(io::IO, v::ModelVariable)
    if get(io, :compact, false)
        print(io, v.name)
    else
        type = v.vr_type == :var ? "" : "@$(v.vr_type) "
        print(io, type, v.name)
    end
end

#  Generic eval_* names that the DFM specializes (see evals.jl / DFM methods).
#  Declared as bare functions so the DFM methods can extend them.
function eval_resid end
function eval_RJ end
function eval_R! end
function eval_RJ! end
