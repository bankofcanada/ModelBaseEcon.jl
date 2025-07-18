##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################


module DerivsSym

using OrderedCollections
using Symbolics
using SymbolicUtils


import ..EquationEvaluator
import .._update_eqn_params!
import .._unpack_args_expr
import .._unpack_pars_expr
import ..funcsyms

#------------------------------------------------------------------------------

const myhash = @static UInt == UInt64 ? 0xca19b034b699d744 : 0xd2f14686
                       
function _unpack_grad(J, grad)
    ex = Expr(:block)
    ind = 0
    for (ind, g) in zip(Iterators.product(axes(grad)...), grad)
        ass = Expr(:(=), Expr(:ref, J, ind...), g)
        push!(ex.args, ass)
    end
    return Expr(:block, :(@inbounds $ex))
end

function make_res_grad_expr(expr, tssyms, sssyms, psyms, mod)
    # the residual `expr` comes to us packaged in a block with a source line
    if Meta.isexpr(expr, :block) && (length(expr.args) == 2)
        src, resid = expr.args
    else
        src, resid = :nothing, expr
    end        
    svars = map(Symbolics.variable, Iterators.flatten((tssyms, sssyms)))
    # dump(resid)   # for debugging when Symbolics.jl complains
    sexpr = simplify(parse_expr_to_symbolic(resid, mod))
    sgrad = simplify.(Symbolics.gradient(sexpr, svars))
    sym_resid = Expr(:block, src, Symbolics.toexpr(sexpr))
    sym_grad = Symbolics.toexpr.(sgrad)
    return sym_resid, sym_grad
end

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
function makefuncs(eqn_name, expr, tssyms, sssyms, psyms, mod::Module)
    fn1, fn2, fn3, fn4 = funcsyms(eqn_name, expr, tssyms, sssyms, psyms, mod, 
    myhash, ("resid_", "RJ_", "resid_param_", "RJ_param_"))
    if isdefined(mod, fn1) && isdefined(mod, fn2) && isdefined(mod, fn3)
        return mod.eval(:(($fn1, $fn2, $fn3, $fn4)))
    end
    nvars = length(tssyms) + length(sssyms)
    x = gensym("x")
    G = gensym("G")
    R = gensym("R")
    ee = gensym("ee")
    resid, grad = make_res_grad_expr(expr, tssyms, sssyms, psyms, mod)
    # If the equation has no parameters, then we just unpack x and evaluate the expressions
    # Otherwise, we unpack the parameters (which have unknown types) and pass it
    # to another function that acts like a function barrier where the types are known.
    return mod.eval(quote
        function ($ee::EquationEvaluatorSym{$(QuoteNode(fn1))})($x::Vector{<:Real})
            $(_unpack_args_expr(x, tssyms, sssyms))
            $(_unpack_pars_expr(ee, psyms))
            $fn3($x, $(psyms...))
        end
        const $fn1 = EquationEvaluatorSym{$(QuoteNode(fn1))}(UInt(0),
            $(@__MODULE__).LittleDict(Symbol[$(QuoteNode.(psyms)...)],
                fill!(Vector{Any}(undef, $(length(psyms))), nothing)))

        function ($ee::GradientEvaluatorSym{$(QuoteNode(fn2))})($x::Vector{<:Real})
            $(_unpack_args_expr(x, tssyms, sssyms))
            $(_unpack_pars_expr(ee, psyms))
            $R = $fn4($ee.G, $x, $(psyms...))
            $R, $ee.G
        end
        const $fn2 = GradientEvaluatorSym{$(QuoteNode(fn2))}(UInt(0),
            $(@__MODULE__).LittleDict(Symbol[$(QuoteNode.(psyms)...)],
                fill!(Vector{Any}(undef, $(length(psyms))), nothing)), 
                Vector{Float64}(undef, $nvars))
        
        function $fn3($x::Vector{<:Real}, $(psyms...))
            $(_unpack_args_expr(x, tssyms, sssyms))
            $resid
        end

        function $fn4($G::Vector{T}, $x::Vector{T}, $(psyms...)) where {T <: Real}
            $(_unpack_args_expr(x, tssyms, sssyms))
            $(_unpack_grad(G, grad))
            $resid
        end

        ($fn1, $fn2, $fn3, $fn4)
    end)

    error("Not ready")
end

function initfuncs(mod::Module)
    expr = Expr(:block)
    if !isdefined(mod, :_hashed_expressions)
        push!(expr.args, quote
            const _hashed_expressions = Dict{UInt,Vector{Tuple{Expr,Vector{Symbol},Vector{Symbol},Vector{Symbol}}}}()
        end)
    end
    if !isdefined(mod, :EquationEvaluatorSym)
        push!(expr.args, quote
            struct EquationEvaluatorSym{FN} <: ModelBaseEcon.EquationEvaluator
                rev::Ref{UInt}
                params::ModelBaseEcon.LittleDictVec{Symbol,Any}
            end
        end)
    end
    if !isdefined(mod, :GradientEvaluatorSym)
        push!(expr.args, quote
            struct GradientEvaluatorSym{FN} <: ModelBaseEcon.EquationEvaluator
                rev::Ref{UInt}
                params::ModelBaseEcon.LittleDictVec{Symbol,Any}
                G::Vector{Float64}
            end
        end)
    end
    mod.eval(expr)
    return nothing
end

end
