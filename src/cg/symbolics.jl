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


import ..MacroTools
import ..ModelBaseEcon
import ..EquationEvaluator
import .._update_eqn_params!
import .._unpack_args_expr
import .._unpack_pars_expr
import ..funcsyms

#------------------------------------------------------------------------------

const myhash = @static UInt == UInt64 ? 0xca19b034b699d744 : 0xd2f14686

function _unpack_array_pars_expr(ee, psyms, mod::Module)
    ex = Expr(:block)
    symmod = isdefined(mod, :_Sym) ? mod._Sym : mod
    for sym in psyms
        if isdefined(symmod, sym)
            foo = getfield(symmod, sym)
            if foo isa Array
                push!(ex.args, :(@assert axes($sym) == $(axes(foo))))
                for idx in Iterators.product(axes(foo)...)
                    bar = Symbolics.tosymbol(foo[idx...])
                    push!(ex.args, :($bar = $sym[$(idx...)]))
                end
            else
                error("Can't handle params of type $(typeof(foo))")
            end
        end
    end
    return isempty(ex.args) ? [] : [:(@inbounds $ex)]
end

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
    symmod = isdefined(mod, :_Sym) ? mod._Sym : mod
    # the residual `expr` comes to us packaged in a block with a source line
    if Meta.isexpr(expr, :block) && (length(expr.args) == 2)
        src, resid = expr.args
    else
        src, resid = :nothing, expr
    end
    svars = map(Symbolics.variable, Iterators.flatten((tssyms, sssyms)))
    # dump(resid)   # for debugging when Symbolics.jl complains
    sexpr = simplify(parse_expr_to_symbolic(resid, symmod))
    sgrad = simplify.(Symbolics.gradient(sexpr, svars))
    sym_resid = Symbolics.toexpr(sexpr)
    if src !== :nothing
        sym_resid = Expr(:block, src, sym_resid)
    end
    sym_grad = Symbolics.toexpr.(sgrad)
    return sym_resid, sym_grad
end

function _makefuncs_expr(eqn_name, expr, tssyms, sssyms, psyms, mod::Module)
    fn1, fn2, fn3, fn4 = funcsyms(eqn_name, expr, tssyms, sssyms, psyms, mod,
        myhash, ("resid_", "RJ_", "resid_param_", "RJ_param_"))
    if isdefined(mod, fn1) && isdefined(mod, fn2) && isdefined(mod, fn3)
        return :(($fn1, $fn2, $fn3, $fn4))
    end
    nvars = length(tssyms) + length(sssyms)
    x = Symbol("#x#")
    G = Symbol("#G#")
    R = Symbol("#R#")
    ee = Symbol("#e#")
    resid, grad = make_res_grad_expr(expr, tssyms, sssyms, psyms, mod)
    # dump(resid)   # for debugging
    # dump(grad)    # for debugging
    # If the equation has no parameters, then we just unpack x and evaluate the expressions
    # Otherwise, we unpack the parameters (which have unknown types) and pass it
    # to another function that acts like a function barrier where the types are known.
    return quote
        function ($ee::EquationEvaluatorSym{$(QuoteNode(fn1))})($x::Vector{<:Real})
            # $(_unpack_args_expr(x, tssyms, sssyms))
            $(_unpack_pars_expr(ee, psyms).args...)
            return $fn3($x, $(psyms...))
        end
        const $fn1 = EquationEvaluatorSym{$(QuoteNode(fn1))}(UInt(0),
            ModelBaseEcon.LittleDict(Symbol[$(QuoteNode.(psyms)...)],
                fill!(Vector{Any}(undef, $(length(psyms))), nothing)),
            # $(Meta.quot(resid)),
        )

        function ($ee::GradientEvaluatorSym{$(QuoteNode(fn2))})($x::Vector{<:Real})
            # $(_unpack_args_expr(x, tssyms, sssyms))
            $(_unpack_pars_expr(ee, psyms).args...)
            $R = $fn4($ee.G, $x, $(psyms...))
            $R, $ee.G
        end
        const $fn2 = GradientEvaluatorSym{$(QuoteNode(fn2))}(UInt(0),
            ModelBaseEcon.LittleDict(Symbol[$(QuoteNode.(psyms)...)],
                fill!(Vector{Any}(undef, $(length(psyms))), nothing)),
            # $(Meta.quot(resid)), [$(Meta.quot.(grad)...)],
            Vector{Float64}(undef, $nvars))

        function $fn3($x::Vector{<:Real}, $(psyms...))
            $(_unpack_array_pars_expr(ee, psyms, mod)...)
            $(_unpack_args_expr(x, tssyms, sssyms))
            return $resid
        end

        function $fn4($G::Vector{<:Real}, $x::Vector{<:Real}, $(psyms...))
            $(_unpack_array_pars_expr(ee, psyms, mod)...)
            $(_unpack_args_expr(x, tssyms, sssyms))
            $(_unpack_grad(G, grad))
            return $resid
        end

        ($fn1, $fn2, $fn3, $fn4)
    end
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
    return mod.eval(_makefuncs_expr(eqn_name, expr, tssyms, sssyms, psyms, mod))
end

function _initfuncs_exprs!(exprs::Vector{Expr}, mod::Module)
    if !isdefined(mod, :_Sym)
        push!(exprs, :(baremodule _Sym
        import Base
        import ModelBaseEcon
        import ModelBaseEcon.DerivsSym.Symbolics
        end))
    end
    if !isdefined(mod, :EquationEvaluatorSym)
        push!(exprs, quote
            struct EquationEvaluatorSym{FN} <: ModelBaseEcon.EquationEvaluator
                rev::Ref{UInt}
                params::ModelBaseEcon.LittleDictVec{Symbol,Any}
                # resid::Expr
            end
        end)
    end
    if !isdefined(mod, :GradientEvaluatorSym)
        push!(exprs, quote
            struct GradientEvaluatorSym{FN} <: ModelBaseEcon.EquationEvaluator
                rev::Ref{UInt}
                params::ModelBaseEcon.LittleDictVec{Symbol,Any}
                # resid::Expr
                # grad::Vector
                G::Vector{Float64}
            end
        end)
    end
    return exprs
end

end
