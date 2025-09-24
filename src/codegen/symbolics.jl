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
using SparseArrays

import ..MacroTools
import ..ModelBaseEcon

import ..AbstractModel
import ..CodeCache
import ..runandcache_expr
import .._cc_comment
# import .._cc_newline

import ..EquationEvaluator
import .._update_eqn_params!
import .._unpack_args_expr
import .._unpack_pars_expr
import ..funcsyms

#------------------------------------------------------------------------------

_simplify(x) = SymbolicUtils.Fixpoint(simplify)(x)

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
    for (ind, g) in zip(Iterators.product(axes(grad)...), grad)
        assignment = Expr(:(=), Expr(:ref, J, ind...), g)
        push!(ex.args, assignment)
    end
    return Expr(:block, :(@assert length($J) == $(length(grad))), :(@inbounds $ex))
end

function _unpack_derivs(D, derivs)
    exprs = [:(@assert length($D) == $(length(derivs)))]
    for (o, der) in enumerate(derivs)
        push!(exprs, :(@assert length($D[$o]) == $(length(der))))
        ex = Expr(:block)
        for (ind,expr) in zip(der.nzind, der.nzval)
            assignment = Expr(:(=), Expr(:ref, Expr(:ref, D, o), ind), expr)
            push!(ex.args, assignment)
        end
        push!(exprs, :(@inbounds $ex))
    end
    return exprs
end


function next_deriv(svec, svars)
    nexpr = length(svec)
    nvars = length(svars)
    idxmap = LinearIndices(map(Base.OneTo, (nexpr, nvars)))
    ret = SparseVector{Symbolics.Num,Int}(undef, length(idxmap))
    for (idx, sexpr) = enumerate(svec)
        iszero(sexpr) && continue
        sgrad = map(_simplify, Symbolics.gradient(_simplify(sexpr), svars))
        # println.(sgrad)
        for (jdx, gexpr) = enumerate(sgrad)
            iszero(gexpr) && continue
            setindex!(ret, gexpr, idxmap[idx, jdx])
        end
    end
    return ret
end


function make_res_grad_expr(expr, tssyms, sssyms, psyms, mod)
    symmod = isdefined(mod, :_Sym) ? mod._Sym : mod
    # the residual `expr` comes to us packaged in a block with a source line
    if Meta.isexpr(expr, :block) && (length(expr.args) == 2)
        src, resid = expr.args
    else
        src, resid = :nothing, expr
    end
    # prepare Symbolics variables
    svars = map(Symbolics.variable, Iterators.flatten((tssyms, sssyms)))
    # dump(resid)   # for debugging when Symbolics.jl complains
    sresid = _simplify(parse_expr_to_symbolic(resid, symmod))   # Symbolics residual
    jresid = Symbolics.toexpr(sresid)                           # Julia residual
    if src !== :nothing
        jresid = Expr(:block, src, jresid)
    end
    # first order derivatives
    sgrad = map(_simplify, Symbolics.gradient(sresid, svars))   # Symbolics gradient
    jgrad = Symbolics.toexpr.(sgrad)                            # Julia gradient
    # higher order derivatives
    max_hod_order = isdefined(mod, :max_hod_order) ? mod.max_hod_order : 1
    sderivs = [sparsevec(sgrad)]
    jderivs = SparseVector{<:Any,Int}[SparseVector(length(jgrad), collect(1:length(jgrad)), jgrad)]
    order = 1
    while order < max_hod_order
        deriv = next_deriv(sderivs[order], svars)
        push!(sderivs, deriv)
        push!(jderivs, SparseVector(deriv.n, deriv.nzind, Symbolics.toexpr.(deriv.nzval)))
        order = order + 1
        @assert order == length(sderivs) == length(jderivs)
    end
    return jresid, jgrad, jderivs
end


function _makefuncs_exprs!(exprs::Vector, eqn_name, expr, tssyms, sssyms, psyms, mod::Module)
    fn1, fn2, fn3, fn4, fn5, fn6 = funcsyms(eqn_name, expr, tssyms, sssyms, psyms, mod,
        myhash, ("resid", "RJ", "resid_param", "RJ_param", "HOD", "HOD_param"))
    need_hod = isdefined(mod, :max_hod_order) && mod.max_hod_order > 1
    if need_hod && all(f -> isdefined(mod, f), [fn1, fn2, fn3, fn4, fn5, fn6])
        return push!(exprs, :(($fn1, $fn2, $fn3, $fn4, $fn5, $fn6)))
    end
    if !need_hod && all(f -> isdefined(mod, f), [fn1, fn2, fn3, fn4])
        return push!(exprs, :(($fn1, $fn2, $fn3, $fn4)))
    end
    nvars = length(tssyms) + length(sssyms)
    # npars = length(psyms)
    x = Symbol("#x#")
    G = Symbol("#G#")
    R = Symbol("#R#")
    ee = Symbol("#e#")
    resid, grad, derivs = make_res_grad_expr(expr, tssyms, sssyms, psyms, mod)
    # If the equation has no parameters, then we just unpack x and evaluate the expressions
    # Otherwise, we unpack the parameters (which have unknown types) and pass it
    # to another function that acts like a function barrier where the types are known.
    push!(exprs, :(
        function ($ee::EquationEvaluatorSym{$(QuoteNode(fn1))})($x::Vector{<:Real})
            # $(_unpack_args_expr(x, tssyms, sssyms))
            $(_unpack_pars_expr(ee, psyms).args...)
            return $fn3($x, $(psyms...))
        end
    ))
    push!(exprs, :(
        const $fn1 = EquationEvaluatorSym{$(QuoteNode(fn1))}(UInt(0),
            LittleDict(Symbol[$(QuoteNode.(psyms)...)],
                fill!(Vector{Any}(undef, $(length(psyms))), nothing)),
            # $(Meta.quot(resid)),
        )
    ))
    push!(exprs, :(
        function ($ee::GradientEvaluatorSym{$(QuoteNode(fn2))})($x::Vector{<:Real})
            # $(_unpack_args_expr(x, tssyms, sssyms))
            $(_unpack_pars_expr(ee, psyms).args...)
            $R = $fn4($ee.G, $x, $(psyms...))
            $R, $ee.G
        end
    ))
    push!(exprs, :(
        const $fn2 = GradientEvaluatorSym{$(QuoteNode(fn2))}(UInt(0),
            LittleDict(Symbol[$(QuoteNode.(psyms)...)],
                fill!(Vector{Any}(undef, $(length(psyms))), nothing)),
            # $(Meta.quot(resid)), [$(Meta.quot.(grad)...)],
            Vector{Float64}(undef, $nvars))
    ))
    push!(exprs, :(
        function $fn3($x::Vector{<:Real}, $(psyms...))
            $(_unpack_array_pars_expr(ee, psyms, mod)...)
            $(_unpack_args_expr(x, tssyms, sssyms))
            return $resid
        end
    ))
    push!(exprs, :(
        function $fn4($G::Vector{<:Real}, $x::Vector{<:Real}, $(psyms...))
            $(_unpack_array_pars_expr(ee, psyms, mod)...)
            $(_unpack_args_expr(x, tssyms, sssyms))
            $(_unpack_grad(G, grad))
            return $resid
        end
    ))
    push!(exprs, :(@assert precompile($fn1, (Vector{Float64},))))
    push!(exprs, :(@assert precompile($fn2, (Vector{Float64},))))
    if !need_hod
        return push!(exprs, :(($fn1, $fn2, $fn3, $fn4)))
    end
    hod_order = mod.max_hod_order
    push!(exprs, :(
        function ($ee::HODEvaluatorSym{$(QuoteNode(fn5))})($x::Vector{<:Real})
            # $(_unpack_args_expr(x, tssyms, sssyms))
            $(_unpack_pars_expr(ee, psyms).args...)
            $R = $fn6($ee.Derivs, $x, $(psyms...))
            $R, $ee.Derivs
        end
    ))
    push!(exprs, :(
        const $fn5 = HODEvaluatorSym{$(QuoteNode(fn5))}(UInt(0),
            LittleDict(Symbol[$(QuoteNode.(psyms)...)],
                fill!(Vector{Any}(undef, $(length(psyms))), nothing)),
            # $(Meta.quot(resid)), [$(Meta.quot.(grad)...)],
            SparseVector{Float64,Int}[
                SparseVector{Float64,Int}(undef, $nvars^i) for i = 1:$hod_order
            ])
    ))
    D = Symbol("#D#")
    push!(exprs, :(
        function $fn6($D::Vector{<:SparseVector}, $x::Vector{<:Real}, $(psyms...))
            $(_unpack_array_pars_expr(ee, psyms, mod)...)
            $(_unpack_args_expr(x, tssyms, sssyms))
            $(_unpack_derivs(D, derivs)...)
            return $resid
        end
    ))
    return push!(exprs, :(($fn1, $fn2, $fn3, $fn4, $fn5, $fn6)))
end


function makefuncs(eqn_name, expr, tssyms, sssyms, psyms, mod::Module)
    mod = invokelatest(mod._module, Val(:symbolics))
    E = Expr(:block)
    _makefuncs_exprs!(E.args, eqn_name, expr, tssyms, sssyms, psyms, mod)
    return Core.eval(mod, E)
end

function _initfuncs_exprs!(exprs::Vector, mod::Module)
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
    if !isdefined(mod, :HODEvaluatorSym)
        push!(exprs, quote
            struct HODEvaluatorSym{FN} <: ModelBaseEcon.EquationEvaluator
                rev::Ref{UInt}
                params::ModelBaseEcon.LittleDictVec{Symbol,Any}
                # resid::Expr
                # grad::Vector
                Derivs::Vector{SparseVector{Float64,Int}}
            end
        end)
    end
    return exprs
end

## =====================================

function _initcc(CC::CodeCache, model::AbstractModel)
    DMOD = nameof(@__MODULE__)
    if !isdefined(CC.cmod, :ModelBaseEcon)
        runandcache_expr(CC, quote
            using ModelBaseEcon
            using SparseArrays
            # using StateSpaceEcon
            import ModelBaseEcon.LittleDict
            import ModelBaseEcon.LittleDictVec
            import ModelBaseEcon.$DMOD.Symbolics
        end)
    end

    if !isdefined(CC.cmod, :_Sym)
        runandcache_expr(CC, :(const _Sym = @__MODULE__))
        # runandcache_expr(CC, :(baremodule _Sym
        # import Base
        # import ModelBaseEcon
        # import ModelBaseEcon.$DMOD.Symbolics
        # end))
    end

    # Symbolics needs to know about array-valued parameters, if any
    if any(pv.value isa AbstractArray for (p, pv) in model.parameters)
        _cc_comment(CC, "Define symbols for array-valued parameters ")
        E = Expr(:block)
        for (p, pv) in model.parameters
            if pv.value isa AbstractArray
                expr = :(@eval _Sym const $p = Symbolics.variables($(QuoteNode(p)), $(axes(pv.value)...)))
                push!(E.args, expr)
            end
        end
        runandcache_expr(CC, E)
    end

end


end
