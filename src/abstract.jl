

abstract type AbstractEquation end

# all equations should have `expr`,`vinds`, `vsyms` and `eval_resid`, `eval_RJ`
for fn in (:expr, :vinds, :vsyms, :eval_resid, :eval_RJ)
    local qnfn = QuoteNode(fn)
    quote
        @inline $fn(eqn::AE) where AE <: AbstractEquation = $qnfn ∈ fieldnames(AE) ? getfield(eqn, $qnfn) : error("Must overload `$fn(::$(AE))`")
    end |> eval
end

# 
Base.show(io::IO, eqn::AbstractEquation) = print(io, expr(eqn))


abstract type AbstractModel end

