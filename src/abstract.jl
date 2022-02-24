##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

abstract type AbstractEquation end

# equations must have these fields: expr, vinds, vsyms, eval_resid, eval_RJ
for fn in (:expr, :vinds, :vsyms, :eval_resid, :eval_RJ)
    local qnfn = QuoteNode(fn)
    eval(quote
        @inline $fn(eqn::AbstractEquation) = getfield(eqn, $qnfn)
    end)
end

# equations might have these fields. If not, we provide defaults
@inline flags(eqn::AbstractEquation) = hasfield(typeof(eqn), :flags) ? getfield(eqn, :flags) : nothing
@inline flag(eqn::AbstractEquation, f::Symbol) = (flgs = flags(eqn); hasfield(typeof(flgs), f) ? getfield(flgs, f) : false)
@inline doc(eqn::AbstractEquation) = :doc in fieldnames(typeof(eqn)) ? getfield(eqn, :doc) : ""

# @inline eval_resid(eqn::AbstractEquation, x) = eval_resid(eqn)(x)
# @inline eval_RJ(eqn::AbstractEquation, x) = eval_RJ(eqn)(x)

#
function Base.show(io::IO, eqn::AbstractEquation)
    flagstr = ""
    flgs = flags(eqn)
    for f in fieldnames(typeof(flgs))
        if getfield(flgs, f)
            flagstr *= "@$(f) "
        end
    end
    docstr = ""
    if !isempty(doc(eqn)) && !get(io, :compact, false)
        docstr = "\"$(doc(eqn))\" "
    end
    print(io, docstr, flagstr, expr(eqn))
end

Base.:(==)(e1::AbstractEquation, e2::AbstractEquation) = flags(e1) == flags(e2) && expr(e1) == expr(e2)
Base.hash(e::AbstractEquation, h::UInt) = hash((flags(e), expr(e)), h)

abstract type AbstractModel end

# a subtype of AbstractModel is expected to have a number of fields.
# If it doesn't, the creater of the new model type must define the
# access methods that follow.

@inline variables(m::AM) where {AM<:AbstractModel} = getfield(m, :variables)
@inline nvariables(m::AM) where {AM<:AbstractModel} = length(variables(m))

@inline shocks(m::AM) where {AM<:AbstractModel} = getfield(m, :shocks)
@inline nshocks(m::AM) where {AM<:AbstractModel} = length(shocks(m))

@inline allvars(m::AM) where {AM<:AbstractModel} = vcat(variables(m), shocks(m))
@inline nallvars(m::AM) where {AM<:AbstractModel} = length(variables(m)) + length(shocks(m))

@inline sstate(m::AM) where {AM<:AbstractModel} = getfield(m, :sstate)

@inline parameters(m::AM) where {AM<:AbstractModel} = getfield(m, :parameters)

@inline equations(m::AM) where {AM<:AbstractModel} = getfield(m, :equations)
@inline nequations(m::AM) where {AM<:AbstractModel} = length(equations(m))

@inline alleqns(m::AM) where {AM<:AbstractModel} = getfield(m, :equations)
@inline nalleqns(m::AM) where {AM<:AbstractModel} = length(equations(m))

export parameters
export variables, nvariables
export shocks, nshocks
export equations, nequations
export sstate
#######


# @inline moduleof(f::Function) = parentmodule(f)
@inline moduleof(e::AbstractEquation) = parentmodule(eval_resid(e))
function moduleof(m::AbstractModel)
    eqns = equations(m)
    if isempty(eqns)
        error("Unable to determine the module containing the given model. Try adding equations to it and call `@initialize`.")
    end
    return moduleof(first(eqns))
end
