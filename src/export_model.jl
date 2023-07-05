##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020, Bank of Canada
# All rights reserved.
##################################################################################

export export_model

_check_name(name) = Base.isidentifier(name) ? true : throw(ArgumentError("Model name must be a valid Julia identifier."))

"""
    export_model(model, name, file::IO)
    export_model(model, name, path::String)

Export the model into a module file. The `name` parameter is used for the name
of the module as well as the module file. The module file is created in the
directory specified by the optional third argument.
"""
function export_model(model::Model, name::AbstractString, path::AbstractString=".")
    if !endswith(path, ".jl")
        path = joinpath(path, name * ".jl")
    end
    open(path, "w") do fd
        export_model(model, name, IOContext(fd, :compact => false, :limit => false))
    end
    return nothing
end

function export_model(model::Model, name::AbstractString, fio::IO)
    _check_name(name)
    println(fio, "module ", name)
    println(fio)
    println(fio, "using ModelBaseEcon")
    println(fio)
    println(fio, "const model = Model()")
    println(fio)

    function _print_modified_options(opts, default, prefix)
        for (ok, ov) in pairs(opts)
            dv = getoption(default, ok, :not_a_default)
            if ov isa Options && dv isa Options
                _print_modified_options(ov, dv, prefix * "$ok.")
            elseif dv === :not_a_default || dv != ov
                println(fio, prefix, ok, " = ", repr(ov))
            end
        end
    end

    println(fio, "# options")
    _print_modified_options(model.options, defaultoptions, "model.")
    println(fio)

    println(fio, "# flags")
    for fld in fieldnames(ModelFlags)
        fval = getfield(model.flags, fld)
        if fval != getfield(ModelFlags(), fld)
            println(fio, "model.", fld, " = ", fval)
        end
    end
    println(fio)

    if !isempty(parameters(model))
        println(fio, "@parameters model begin")
        for (n, p) in model.parameters
            if typeof(p.value) <: AbstractModel || typeof(p.value) <: Parameters
                @warn """The parameter "$n" is a $(typeof(p.value)) struct and is being exported as nothing.
                         The resulting model may not compile."""
                println(fio, "    ", """# the parameter :$n was a $(typeof(p.value)) which is not a supproted type.""")
                println(fio, "    ", n, " = ", nothing)
            else
                println(fio, "    ", n, " = ", p)
            end
        end
        println(fio, "end # parameters")
        println(fio)
    end

    allvars = model.allvars
    if !isempty(allvars)
        println(fio, "@variables model begin")
        has_exog = false
        has_shocks = false
        for v in allvars
            if isexog(v)
                has_exog = true
            elseif isshock(v)
                has_shocks = true
            else
                println(fio, "    ", v)
            end
        end
        println(fio, "end # variables")
        println(fio)
        if has_exog
            println(fio, "@exogenous model begin")
            for v in allvars
                if isexog(v)
                    doc = ifelse(isempty(v.doc), "", v.doc * " ")
                    println(fio, "    ", doc, v.name)
                end
            end
            println(fio, "end # exogenous")
            println(fio)
        end
        if has_shocks
            println(fio, "@shocks model begin")
            for v in allvars
                if isshock(v)
                    println(fio, "    ", v)
                end
            end
            println(fio, "end # shocks")
            println(fio)
        end
    end

    if !isempty(model.autoexogenize)
        println(fio, "@autoexogenize model begin")
        vars = collect(keys(model.autoexogenize))
        sort!(vars)
        for var in vars
            println(fio, "    ", var, " = ", model.autoexogenize[var])
        end
        println(fio, "end # autoexogenize")
        println(fio)
    end

    alleqns = model.alleqns
    if !isempty(alleqns)
        println(fio, "@equations model begin")
        for eqn_pair in alleqns
            str = sprint(print, eqn_pair[2], context=fio, sizehint=0)
            str = replace(str, r"(\s*\".*\"\n)" => s"\1    ") 
            println(fio, "    ", unescape_string(str))
        end
        println(fio, "end # equations")
        println(fio)
    end

    println(fio, "@initialize model")

    sd = sstate(model)
    for cons in sd.constraints
        println(fio)
        cons_string = export_equation(cons)
        println(fio, "@steadystate model ", cons_string)
    end

    println(fio)
    println(fio, "newmodel() = deepcopy(model)")
    println(fio)

    println(fio)
    println(fio, "end # module ", name)

    return nothing
end

    
