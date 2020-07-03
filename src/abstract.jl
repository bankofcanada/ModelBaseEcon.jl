

abstract type AbstractEquation end

# all equations should have `expr`,`vinds`, `vsyms` and `eval_resid`, `eval_RJ`
for fn in (:expr, :vinds, :vsyms, :eval_resid, :eval_RJ)
    local qnfn = QuoteNode(fn)
    quote
        @inline $fn(eqn::AE) where AE <: AbstractEquation = getfield(eqn, $qnfn) # $qnfn ∈ fieldnames(AE) ? getfield(eqn, $qnfn) : error("Must overload `$fn(::$(AE))`")
    end |> eval
end

@inline eval_resid(eqn::AE, x) where AE <: AbstractEquation = eval_resid(eqn)(x)
@inline eval_RJ(eqn::AE, x) where AE <: AbstractEquation = eval_RJ(eqn)(x)

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

@inline allvars(m::AM) where AM <: AbstractModel = vcat(variables(m), shocks(m))
@inline nallvars(m::AM) where AM <: AbstractModel = length(variables(m)) + length(shocks(m))

@inline sstate(m::AM) where AM <: AbstractModel = getfield(m, :sstate)

@inline parameters(m::AM) where AM <: AbstractModel = getfield(m, :parameters)

@inline equations(m::AM) where AM <: AbstractModel = getfield(m, :equations)
@inline nequations(m::AM) where AM <: AbstractModel = length(equations(m))

@inline alleqns(m::AM) where AM <: AbstractModel = getfield(m, :equations)
@inline nalleqns(m::AM) where AM <: AbstractModel = length(equations(m))

#######


@inline moduleof(f::Function) = parentmodule(f)
@inline moduleof(e::AE) where AE <: AbstractEquation = parentmodule(eval_resid(e))
function moduleof(m::AbstractModel)
    eqns = equations(m)
    if isempty(eqns)
        error("Unable to determine the module containing the given model. Try adding equations to it and call `@initialize`.")
    end
    return moduleof(first(eqns))
end
