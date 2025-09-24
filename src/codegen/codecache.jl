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
    sfn::Union{Nothing,Symbol}     # filename of the source where the currently processed equation was written
    cmod::Module    # the code cache module
    mmod::Module    # the model Module
    codegen::Val
    CodeCache(::Nothing=nothing) = new{Nothing}(nothing, "", nothing)
    function CodeCache(f::AbstractString)
        cf = open(f, "w")
        return new{typeof(cf)}(cf, string(f), nothing)
    end
end

CodeCache(model::AbstractModel, mmod::Union{Nothing,Module}=nothing) = CodeCache(nothing, model, mmod)
function CodeCache(fn::Union{Nothing,AbstractString}, model::AbstractModel, mmod::Union{Nothing,Module}=nothing)
    CC = CodeCache(fn)
    codegen = model.options.codegen
    CC.codegen = Val(codegen)
    if hasmethod(model._module, (Val{codegen},))
        CC.mmod = model._module(Val(:model))
        CC.cmod = model._module(Val(codegen))
    elseif isdefined(mmod, :_module) && hasmethod(mmod._module, (Val{codegen},))
        # allow for a new model in the same module
        CC.mmod = mmod._module(Val(:model))
        CC.cmod = mmod._module(Val(codegen))
        model._module = mmod._module
        _initcc(CC, model)  # prepared the existing cmod for use with model
    else
        isnothing(mmod) && error("Module of model must be supplied.")
        initcc!(CC, mmod, model) # create new cmod inside mmod and prepare it for use with model
    end
    return CC
end

_derivs_mod(CC::CodeCache) = _derivs_mod(CC.codegen)

function _initcc(CC::CodeCache, model::AbstractModel)
    DMOD = _derivs_mod(CC)
    # import modules used for code generation
    DMOD._initcc(CC, model)

    # define ModelVariables in the code module -- needed if caching for equations codes
    # if !isnothing(CC.cf)
    _cc_comment(CC, "Define ModelVariable instances for model variables ")
    local E = Expr(:block)
    for vars in (model.variables, model.shocks, model.auxvars)
        for v in vars
            push!(E.args, :(const $(v.name) =
                ModelVariable($(v.doc), $(QuoteNode(v.name)),
                    $(QuoteNode(v.vr_type)), $(QuoteNode(v.tr_type)),
                    $(QuoteNode(v.ss_type)))))
        end
    end
    runandcache_expr(CC, E)
    # end
end

function initcc!(CC::CodeCache, mmod::Module, model::AbstractModel)
    codegen = model.options.codegen
    CC.mmod = mmod

    # initialize the model module
    if !isdefined(CC.mmod, :_module)
        Core.eval(CC.mmod, Expr(:block,
            Expr(:(=), :(_module(s::Symbol)), :(_module(Val(s)))),
            Expr(:(=), :(_module(::Val{:model}=Val(:model))), :($(CC.mmod))),
        ))
    end
    model._module = CC.mmod._module

    # startup a new code generation module
    CC.codegen = Val(codegen)
    if isnothing(CC.cf)
        # default code module name
        cmod_name = Symbol(:_, codegen)
        # not writing to file.
    else
        # code module name from filename
        cmod_name = Symbol(splitext(basename(CC.cfn))[1])
        # write to file
        println(CC.cf, "# =================================================================== #")
        println(CC.cf, "#  This file contains code generated automatically by ModelBaseEcon.  #")
        println(CC.cf, "# =================================================================== #")
        println(CC.cf, "module ", cmod_name, "\n")
    end
    # create code generation module
    CC.cmod = Core.eval(mmod, :(module $cmod_name end))

    runandcache_expr(CC, Expr(:block,
        :(import .._module),
        Expr(:(=), :(_module(::$(typeof(CC.codegen)))), nameof(CC.cmod)),
        :(const max_hod_order = $(getoption(model, :max_hod_order, 1)))
    ))

    # prepare CC.cmod for the current model
    _initcc(CC, model)

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
