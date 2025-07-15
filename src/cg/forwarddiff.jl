##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################


module DerivsFD

using OrderedCollections
using ForwardDiff
using DiffResults


import ..LittleDict
import ..LittleDictVec

struct ModelBaseEconTag end

"""
    precompilefuncs(resid, RJ, resid_param, N::Int)

Pre-compiles the given `resid` and `RJ` functions together
with the dual-number arithmetic required by ForwardDiff.

!!! warning
    Internal function. Do not call directly

"""
function precompilefuncs(resid, RJ, resid_param, N::Int)
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing

    tagtype = ModelBaseEconTag
    dual = ForwardDiff.Dual{tagtype,Float64,N}
    duals = Vector{dual}

    precompile(resid, (Vector{Float64},)) || error("precompile")
    precompile(resid, (duals,)) || error("precompile")
    precompile(RJ, (Vector{Float64},)) || error("precompile")

    # We precompile a version of the "function barrier" for the inital types
    # of the parameters. This is a good apprixmimation of what will be evaluated
    # in practice. If a user updates the parameter to a different type, a new version
    # of the function barrier will have to be compiled but this should be fairly rare in
    # practice.
    type_params = typeof.(values(resid.params))
    if !isempty(type_params)
        precompile(resid_param, (duals, type_params...)) || error("precompile")
    end

    return nothing
end

# """
#     funcsyms(mod::Module)

# Create a pair of identifiers that does not conflict with existing identifiers in
# the given module.

#. !!! warning
#     Internal function. Do not call directly.

# ### Implementation (for developers)
# We need two identifiers `resid_N` and `RJ_N` where "N" is some integer number.
# The first is going to be the name of the function that evaluates the equation
# and the second is going to be the name of the function that evaluates both the
# equation and its gradient.
# """
# function funcsyms end

# function funcsyms(mod::Module, eqn_name::Symbol, args...)
#     iterator = 1
#     fn1 = Symbol("resid_", eqn_name)
#     fn2 = Symbol("RJ_", eqn_name)
#     fn3 = Symbol("resid_param_", eqn_name)
#     while isdefined(mod, fn1) || isdefined(Main, fn1)
#         iterator += 1
#         fn1 = Symbol("resid_", eqn_name, "_", iterator)
#         fn2 = Symbol("RJ_", eqn_name, "_", iterator)
#         fn3 = Symbol("resid_param_", eqn_name, "_", iterator)
#     end
#     return fn1, fn2, fn3
# end

function funcsyms(mod, eqn_name::Symbol, expr::Expr, tssyms, sssyms, psyms)
    eqn_data = (expr, collect(tssyms), collect(sssyms), collect(psyms))
    myhash = @static UInt == UInt64 ? 0x2270e9673a0822b5 : 0x2ce87a13
    myhash = Base.hash(eqn_data, myhash)
    he = mod._hashed_expressions
    hits = get!(he, myhash, valtype(he)())
    ind = indexin([eqn_data], hits)[1]
    if isnothing(ind)
        push!(hits, eqn_data)
        ind = 1
    end
    fn1 = Symbol("resid_", eqn_name, "_", ind, "_", myhash)
    fn2 = Symbol("RJ_", eqn_name, "_", ind, "_", myhash)
    fn3 = Symbol("resid_param_", eqn_name, "_", ind, "_", myhash)
    return fn1, fn2, fn3
end

const MAX_CHUNK_SIZE = 4

# Used to avoid specializing the ForwardDiff functions on
# every equation.
struct FunctionWrapper <: Function
    f::Function
end
(f::FunctionWrapper)(x) = f.f(x)

"""
    makefuncs(expr, tssyms, sssyms, psyms, mod)

Create two functions that evaluate the residual and its gradient for the given
expression.

!!! warning
    Internal function. Do not call directly.

### Arguments
- `expr`: the expression
- `tssyms`: list of time series variable symbols
- `sssyms`: list of steady state symbols
- `psyms`: list of parameter symbols

### Return value
Return a quote block to be evaluated in the module where the model is being
defined. The quote block contains definitions of the residual function (as a
callable `EquationEvaluator` instance) and a second function that evaluates both
the residual and its gradient (as a callable `EquationGradient` instance).
"""
function makefuncs(eqn_name, expr, tssyms, sssyms, psyms, mod)
    nargs = length(tssyms) + length(sssyms)
    chunk = min(nargs, MAX_CHUNK_SIZE)
    fn1, fn2, fn3 = funcsyms(mod, eqn_name, expr, tssyms, sssyms, psyms)
    if isdefined(mod, fn1) && isdefined(mod, fn2) && isdefined(mod, fn3)
        return mod.eval(:(($fn1, $fn2, $fn3, $chunk)))
    end
    x = gensym("x")
    has_psyms = !isempty(psyms)
    # This is the expression that goes inside the body of the "outer" function.
    # If the equation has no parameters, then we just unpack x and evaluate the expressions
    # Otherwise, we unpack the parameters (which have unknown types) and pass it
    # to another function that acts like a function barrier where the types are known.
    psym_expr = if has_psyms
        quote
            ($(psyms...),) = values(ee.params)
            $fn3($x, $(psyms...))
        end
    else
        quote
            ($(tssyms...), $(sssyms...),) = $x
            $expr
        end
    end
    # The expression for the function barrier
    fn3_expr = if has_psyms
        quote
            function $fn3($x, $(psyms...))
                ($(tssyms...), $(sssyms...),) = $x
                $expr
            end
        end
    else
        :(const $fn3 = nothing)
    end
    return mod.eval(quote
        function (ee::EquationEvaluator{$(QuoteNode(fn1))})($x::Vector{<:Real})
            $psym_expr
        end
        const $fn1 = EquationEvaluator{$(QuoteNode(fn1))}(UInt(0),
            $(@__MODULE__).LittleDict(Symbol[$(QuoteNode.(psyms)...)], fill!(Vector{Any}(undef, $(length(psyms))), nothing)))
        const $fn2 = EquationGradient($FunctionWrapper($fn1), $nargs, Val($chunk))
        $fn3_expr
        ($fn1, $fn2, $fn3, $chunk)
    end)
end

"""
    initfuncs(mod::Module)

Initialize the given module before creating functions that evaluate residuals
and thier gradients.

!!! warning
    Internal function. Do not call directly.

### Implementation (for developers)
Declare the necessary types in the module where the model is being defined.
There are two such types. First is `EquationEvaluator`, which is callable and
stores a collection of parameters. The call will be defined in
[`makefuncs`](@ref) and will evaluate the residual. The other type is
`EquationGradient`, which is also callable and stores the `EquationEvaluator`
together with a `DiffResult` and a `GradientConfig` used by `ForwardDiff`. Its
call is defined here and computes the residual and the gradient.
"""
function initfuncs(mod::Module)
    if !isdefined(mod, :EquationEvaluator)
        mod.eval(quote
            const _hashed_expressions = Dict{UInt,Vector{Tuple{Expr,Vector{Symbol},Vector{Symbol},Vector{Symbol}}}}()
            struct EquationEvaluator{FN} <: Function
                rev::Ref{UInt}
                params::$(@__MODULE__).LittleDictVec{Symbol,Any}
            end
            struct EquationGradient{DR,CFG} <: Function
                fn1::Function
                dr::DR
                cfg::CFG
            end
            EquationGradient(fn1::Function, nargs::Int, ::Val{N}) where {N} = EquationGradient(fn1,
                $(@__MODULE__).DiffResults.DiffResult(zero(Float64), zeros(Float64, nargs)),
                $(@__MODULE__).ForwardDiff.GradientConfig(fn1, zeros(Float64, nargs), $(@__MODULE__).ForwardDiff.Chunk{N}(), $ModelBaseEconTag()))
            function (s::EquationGradient)(x::Vector{Float64})
                $(@__MODULE__).ForwardDiff.gradient!(s.dr, s.fn1, x, s.cfg)
                return s.dr.value, s.dr.derivs[1]
            end
        end)
    end
    return nothing
end

end
