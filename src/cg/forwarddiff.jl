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

const MAX_CHUNK_SIZE = 4

#------------------------------------------------------------------------------
# Used to avoid specializing the ForwardDiff functions on
# every equation.
struct FunctionWrapper <: Function
    f::Function
end
(f::FunctionWrapper)(x) = f.f(x)

import ..EquationEvaluator
# struct EquationEvaluatorFD{FN} <: AbstractEquationEvaluator
#     rev::Ref{UInt}
#     params::LittleDictVec{Symbol,Any}
# end

struct EquationGradientFD{DR<:DiffResults.DiffResult,CFG<:ForwardDiff.GradientConfig} <: Function
    fn1::FunctionWrapper
    dr::DR
    cfg::CFG
end

function EquationGradientFD(fn1::Function, nargs::Int, ::Val{N}) where {N}
    return EquationGradientFD(FunctionWrapper(fn1),
        DiffResults.DiffResult(zero(Float64), zeros(Float64, nargs)),
        ForwardDiff.GradientConfig(fn1, zeros(Float64, nargs), ForwardDiff.Chunk{N}(), ModelBaseEconTag()))
end

function (s::EquationGradientFD)(x::Vector{Float64})
    ForwardDiff.gradient!(s.dr, s.fn1, x, s.cfg)
    return s.dr.value, s.dr.derivs[1]
end

function (s::EquationGradientFD)(J::AbstractVector{Float64}, x::Vector{Float64})
    ForwardDiff.gradient!(s.dr, s.fn1, x, s.cfg)
    copyto!(J, s.dr.derivs)
    return s.dr.value, s.dr.derivs[1]
end

import .._update_eqn_params!
_update_eqn_params!(@nospecialize(ee::EquationGradientFD), params) = _update_eqn_params!(ee.fn1.f, params)

import .._unpack_args_expr
import .._unpack_pars_expr
import ..funcsyms
#------------------------------------------------------------------------------

const myhash = @static UInt == UInt64 ? 0x2270e9673a0822b5 : 0x2ce87a13

"""
    makefuncs(eqn_name, expr, tssyms, sssyms, psyms, mod)

Create two functions that evaluate the residual and its gradient for the given
expression.

!!! warning
    Internal function. Do not call directly.

### Arguments
- `expr`: the residual expression
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
    fn1, fn2, fn3 = funcsyms(eqn_name, expr, tssyms, sssyms, psyms, mod, 
        myhash, ("resid_", "RJ_", "resid_param_"))
    if isdefined(mod, fn1) && isdefined(mod, fn2) && isdefined(mod, fn3)
        return mod.eval(:(($fn1, $fn2, $fn3, $chunk)))
    end
    x = gensym("x")
    # If the equation has no parameters, then we just unpack x and evaluate the expressions
    # Otherwise, we unpack the parameters (which have unknown types) and pass it
    # to another function that acts like a function barrier where the types are known.
    return mod.eval(quote
        function (ee::EquationEvaluatorFD{$(QuoteNode(fn1))})($x::Vector{<:Real})
            $(_unpack_args_expr(x, tssyms, sssyms))
            $(_unpack_pars_expr(:ee, psyms))
            $fn3($x, $(psyms...))
        end
        const $fn1 = EquationEvaluatorFD{$(QuoteNode(fn1))}(UInt(0),
            $(@__MODULE__).LittleDict(Symbol[$(QuoteNode.(psyms)...)],
                fill!(Vector{Any}(undef, $(length(psyms))), nothing)))
        const $fn2 = EquationGradientFD($fn1, $nargs, Val($chunk))
        function $fn3($x::Vector{<:Real}, $(psyms...))
            $(_unpack_args_expr(x, tssyms, sssyms))
            $expr
        end
        ($fn1, $fn2, $fn3, $chunk)
    end)
end

function _initfuncs_exprs!(exprs::Vector{Expr}, mod::Module)
    if !isdefined(mod, :EquationEvaluatorFD)
        push!(exprs, quote
            struct EquationEvaluatorFD{FN} <: ModelBaseEcon.EquationEvaluator
                rev::Ref{UInt}
                params::ModelBaseEcon.LittleDictVec{Symbol,Any}
            end
        end)
    end
    if !isdefined(mod, :EquationGradientFD)
        push!(exprs, quote
            import ModelBaseEcon.$(nameof(@__MODULE__)).EquationGradientFD
        end)
    end
    return exprs
end

end
