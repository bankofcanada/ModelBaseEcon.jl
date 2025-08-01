##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

iscacheuptodate(cachefile::Nothing, modelfile::AbstractString) = false
iscacheuptodate(cachefile::AbstractString, modelfile::AbstractString) = isfile(modelfile) && (mtime(modelfile) < mtime(cachefile))

# F <: Nothing means that we're not writing to a file, just running the expressions in the code module 
# F <: IOStream means that we're writing to a file and running the expressions in the code module 
mutable struct CodeCache{F<:Union{Nothing,IOStream}}
    const cf::F           # the cache file stream, or `nothing`
    cfn::String     # filename of the code file
    sfn::Symbol     # filename of the source where the currently processed equation was written
    cmod::Module    # the code cache module
    mmod::Module    # the model Module
    codegen::Val
    CodeCache(::Nothing) = new{Nothing}(nothing, "", Symbol())
    function CodeCache(f::AbstractString)
        cf = open(f, "w")
        return new{typeof(cf)}(cf, string(f), Symbol())
    end
end

function initcc!(CC::CodeCache, mmod::Module, codegen::Symbol)
    CC.mmod = mmod
    CC.codegen = Val(codegen)
    if isnothing(CC.cf)
        # not writing to file.
        # will run the code generation in the model module
        CC.cmod = mmod
    else
        cmod_name = Symbol(splitext(basename(CC.cfn))[1])
        # write to file
        println(CC.cf, "# =================================================================== #")
        println(CC.cf, "#  This file contains code generated automatically by ModelBaseEcon.  #")
        println(CC.cf, "# =================================================================== #")
        println(CC.cf, "module ", cmod_name, "\n")
        # create code generation module
        CC.cmod = Core.eval(mmod, :(module $cmod_name end))
    end
    # initialize the code module
    if CC.codegen == Val(:symbolics)
        runandcache_expr(CC, _striplines(CC, quote
            using ModelBaseEcon
            using StateSpaceEcon
            import ModelBaseEcon.LittleDict
            import Symbolics
            function thismodule()
                return @__MODULE__
            end
        end))
    else
        # throw(NotImplementedError("Code caching with codegen=$(QuoteNode(codegen))"))
    end
    # initialize the model module
    if !isdefined(CC.mmod, :thismodule)
        Core.eval(CC.mmod, :(function thismodule()
            return @__MODULE__
        end))
    end
    return CC
end

function _striplines(CC::CodeCache, expr)
    dropline(a) = (a isa LineNumberNode) && (a.file != CC.sfn)
    ret = MacroTools.postwalk(expr) do x
        if (x isa Expr)
            if x.head === :macrocall && length(x.args) >= 2
                return Expr(x.head, x.args[1], nothing, filter(!dropline, x.args[3:end])...)
            else
                return Expr(x.head, filter(!dropline, x.args)...)
            end
        end
        return x
    end
    return ret
end


_cc_newline(CC::CodeCache{Nothing}) = nothing
_cc_newline(CC::CodeCache) = println(CC.cf)

_cc_comment(CC::CodeCache{Nothing}, comment) = nothing
_cc_comment(CC::CodeCache, comment) = println(CC.cf, "# ", rpad(comment, 78, "="))

"The given expression is eval'ed in the code module and is also written into to cache file"
function runandcache_expr end
_do_runandcache_expr(CC::CodeCache{Nothing}, expr) = Core.eval(CC.cmod, expr)
_do_runandcache_expr(CC::CodeCache, expr) = (println(CC.cf, expr); Core.eval(CC.cmod, expr))
function runandcache_expr(CC::CodeCache, expr::Expr; striplines=true, unblock=true)
    if striplines
        expr = _striplines(CC, expr)
    end
    if unblock && (expr.head == :block)
        for ex in expr.args
            ex isa LineNumberNode || _do_runandcache_expr(CC, ex)
        end
    else
        _do_runandcache_expr(CC, expr)
    end
    _cc_newline(CC)
end

closecc!(CC::CodeCache{Nothing}) = CC
function closecc!(CC::CodeCache)
    println(CC.cf, "end # module ", nameof(CC.cmod), "\n")
    close(CC.cf)
    return CC
end
