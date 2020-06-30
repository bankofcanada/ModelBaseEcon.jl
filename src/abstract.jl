

abstract type AbstractEquation end

# all equations should have `expr`,`vinds`, `vsyms` and `eval_resid`, `eval_RJ`
for fn in (:expr, :vinds, :vsyms, :eval_resid, :eval_RJ)
    local qnfn = QuoteNode(fn)
    quote
        @inline $fn(eqn::AE) where AE <: AbstractEquation = getfield(eqn, $qnfn) # $qnfn ∈ fieldnames(AE) ? getfield(eqn, $qnfn) : error("Must overload `$fn(::$(AE))`")
    end |> eval
end

# 
Base.show(io::IO, eqn::AbstractEquation) = print(io, expr(eqn))


abstract type AbstractModel end

# a subtype of AbstractModel is expected to have a number of fields.
# If it doesn't, the creater of the new model type must define the
# access methods that follow.

@inline variables(m::AM) where AM <: AbstractModel = getfield(m, :variables)
@inline nvariables(m::AM) where AM <: AbstractModel = length(variables(m))

@inline shocks(m::AM) where AM <: AbstractModel = getfield(m, :shocks)
@inline nshocks(m::AM) where AM <: AbstractModel = length(shocks(m))

@inline auxvars(m::AM) where AM <: AbstractModel = getfield(m, :auxvars)
@inline nauxvars(m::AM) where AM <: AbstractModel = length(auxvars(m))

@inline unknowns(m::AM) where AM <: AbstractModel = vcat(variables(m), shocks(m), auxvars(m))
@inline nunknonws(m::AM) where AM <: AbstractModel = length(variables(m)) + length(shocks(m)) + length(auxvars(m))

@inline sstate(m::AM) where AM <: AbstractModel = getfield(m, :sstate)

@inline parameters(m::AM) where AM <: AbstractModel = getfield(m, :parameters)

