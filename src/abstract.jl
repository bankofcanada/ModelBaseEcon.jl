##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020, Bank of Canada
# All rights reserved.
##################################################################################

abstract type AbstractEquation end

# equations must have these fields: expr, vinds, vsyms, eval_resid, eval_RJ
for fn in (:expr, :vinds, :vsyms, :eval_resid, :eval_RJ)
    local qnfn = QuoteNode(fn)
    quote
        @inline $fn(eqn::AbstractEquation) = getfield(eqn, $qnfn)
    end |> eval
end

const default_eqn_type = :any

# equations might have these fields. If not, we provide defaults
@inline type(eqn::AbstractEquation) = :type in fieldnames(typeof(eqn)) ? getfield(eqn, :type) : default_eqn_type
@inline doc(eqn::AbstractEquation) = :doc in fieldnames(typeof(eqn)) ? getfield(eqn, :doc) : ""

@inline eval_resid(eqn::AbstractEquation, x) = eval_resid(eqn)(x)
@inline eval_RJ(eqn::AbstractEquation, x) = eval_RJ(eqn)(x)

#
function Base.show(io::IO, eqn::AbstractEquation)
    if get(io, :compact, false)
        return print(io, expr(eqn))
    end
    docstr = isempty(doc(eqn)) ? "" : "\"$(doc(eqn))\" "
    typestr = type(eqn) === default_eqn_type ? "" : "$(type(eqn)) "
    print(io, docstr, typestr, expr(eqn))
end

Base.:(==)(e1::AbstractEquation, e2::AbstractEquation) = type(e1) == type(e1) && expr(e1) == expr(e2)
Base.hash(e::AbstractEquation, h::UInt) = hash((type(e), expr(e)), h)

abstract type AbstractModel end

# a subtype of AbstractModel is expected to have a number of fields.
# If it doesn't, the creater of the new model type must define the
# access methods that follow.

@inline variables(m::AM) where AM <: AbstractModel = getfield(m, :variables)
@inline nvariables(m::AM) where AM <: AbstractModel = length(variables(m))

@inline shocks(m::AM) where AM <: AbstractModel = getfield(m, :shocks)
@inline nshocks(m::AM) where AM <: AbstractModel = length(shocks(m))

@inline allvars(m::AM) where AM <: AbstractModel = vcat(variables(m), shocks(m))
@inline nallvars(m::AM) where AM <: AbstractModel = length(variables(m)) + length(shocks(m))

@inline sstate(m::AM) where AM <: AbstractModel = getfield(m, :sstate)

@inline parameters(m::AM) where AM <: AbstractModel = getfield(m, :parameters)

@inline equations(m::AM) where AM <: AbstractModel = getfield(m, :equations)
@inline nequations(m::AM) where AM <: AbstractModel = length(equations(m))

@inline alleqns(m::AM) where AM <: AbstractModel = getfield(m, :equations)
@inline nalleqns(m::AM) where AM <: AbstractModel = length(equations(m))

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
