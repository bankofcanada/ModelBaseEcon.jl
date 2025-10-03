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

using ..SimpleTensors

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
    exprs = []
    for (o, der) in derivs
        ex = Expr(:block)
        i = 1
        idx = ntuple(one, ndims(der))
        for (ii, expr) in zip(der.data.nzind, der.data.nzval)
            while i < ii
                i = i + 1
                idx = SimpleTensors.next_sym_idx(der.dim, idx...)
            end
            assignment = Expr(:(=), Expr(:ref, Expr(:ref, D, o), idx...), expr)
            push!(ex.args, assignment)
        end
        isempty(ex.args) && break
        push!(exprs, :(@assert length($D[$o]) == $(length(der))))
        push!(exprs, :(@inbounds $ex))
    end
    return exprs
end


function next_deriv(sder::SymmetricTensor{Symbolics.Num,N}, svars) where N
    nvars = length(svars)
    ret = SymmetricTensor{Symbolics.Num,N + 1}(nvars, Val(:sparse))
    iszero(sder) && return ret  # shortcut
    for idx in SymmetricIndices(sder)
        sexpr = sder[idx...]
        iszero(sexpr) && continue
        sgrad = map(_simplify, Symbolics.gradient(_simplify(sexpr), svars))
        for (jdx, gexpr) in enumerate(sgrad)
            iszero(gexpr) && continue
            ret[idx..., jdx] = gexpr
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
    # sderivs = [sparsevec(sgrad)]
    # jderivs = SparseVector{<:Any,Int}[SparseVector(length(jgrad), collect(1:length(jgrad)), jgrad)]
    jderivs = derivs_container(Any, max_hod_order, length(svars))
    push!(jderivs[0].data.nzind, 1)
    push!(jderivs[0].data.nzval, jresid)
    sder = SymmetricTensor(sgrad)
    _copytoexpr!(jderivs[1].data, sder.data)
    for order = 2:max_hod_order
        sder = next_deriv(sder, svars)
        _copytoexpr!(jderivs[order].data, sder.data)
    end
    return jresid, jgrad, jderivs
end

function _copytoexpr!(dest::SparseVector, src::SparseVector{T,Int}) where T
    @boundscheck @assert size(dest) == size(src)
    nnz = SparseArrays.nnz(src)
    resize!(dest.nzind, nnz)
    copyto!(dest.nzind, src.nzind)
    resize!(dest.nzval, nnz)
    if T === Expr
        dest.nzval .= src.nzval
    elseif T === Symbolics.Num
        dest.nzval .= Symbolics.toexpr.(src.nzval)
    else
        error("Unexpected element type $T")
    end
    return dest
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
            $R = $fn6($ee.derivs, $x, $(psyms...))
            $ee.derivs
        end
    ))
    push!(exprs, :(
        const $fn5 = HODEvaluatorSym{$(QuoteNode(fn5))}(UInt(0),
            LittleDict(Symbol[$(QuoteNode.(psyms)...)],
                fill!(Vector{Any}(undef, $(length(psyms))), nothing)),
            # $(Meta.quot(resid)), [$(Meta.quot.(grad)...)],
            derivs_container(Float64, $hod_order, $nvars))
    ))
    D = Symbol("#D#")
    push!(exprs, :(
        function $fn6($D::DerivsContainer{<:Real}, $x::Vector{<:Real}, $(psyms...))
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
                derivs::DerivsContainer{Float64}
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
            using ModelBaseEcon.SimpleTensors
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
