

export export_model

function _print_modified_options(opts, default, prefix, fio)
    for (ok, ov) in pairs(opts)
        dv = getoption(default, ok, :not_a_default)
        if ov isa Options && dv isa Options
            _print_modified_options(ov, dv, prefix * "$ok.", fio)
        elseif dv == :not_a_default || dv != ov
            println(fio, prefix, ok, " = ", ov)
        end
    end
end

"""
    export_model(model::Model, name::String, path::String=".")

Export the model into a module file. The `name` parameter is used for the name
of the module as well as the module file. The module file is created in the
directory specified by the optional third argument.
"""
function export_model(m::Model, name::AbstractString, path::AbstractString=".")
    open(joinpath(path, name * ".jl"), "w") do fd
        fio = IOContext(fd, :compact => false, :limit => false)
        println(fio, "module ", name)
        println(fio)
        println(fio, "using ModelBaseEcon")
        println(fio)
        println(fio, "const model = Model()")
        println(fio)

        println(fio, "# options")
        _print_modified_options(m.options, defaultoptions, "model.", fio)
        println(fio)

        println(fio, "# flags")
        for fld in fieldnames(ModelFlags)
            fval = getfield(m.flags, fld)
            if fval != getfield(ModelFlags(), fld)
                println(fio, "model.", fld, " = ", fval)
            end
        end
        println(fio)

        if !isempty(parameters(m))
            println(fio, "@parameters model begin")
            for pn in (sort ∘ collect ∘ keys)(m.parameters)
                println(fio, "    ", pn, " = ", m.parameters[pn])
            end
            println(fio, "end # parameters")
            println(fio)
        end

        if !isempty(variables(m))
            println(fio, "@variables model begin")
            for v in variables(m)
                println(fio, "    ", v)
            end
            println(fio, "end # variables")
            println(fio)
        end

        if !isempty(shocks(m))
            println(fio, "@shocks model begin")
            for v in shocks(m)
                println(fio, "    ", v)
            end
            println(fio, "end # shocks")
            println(fio)
        end
        
        if !isempty(m.autoexogenize)
            println(fio, "@autoexogenize model begin")
            for (k, v) in pairs(m.autoexogenize)
                println(fio, "    ", k, " = ", v)
            end
            println(fio, "end # autoexogenize")
            println(fio)
        end
        
        if !isempty(equations(m))
            println(fio, "@equations model begin")
            for eqn in equations(m)
                str = sprint(print, eqn, context=fio, sizehint=0)
                println(fio, replace(str, r"(^|\n)" => s"\1    "))
            end
            println(fio, "end # equations")
            println(fio)
        end

        println(fio, "@initialize model")
        println(fio)
        println(fio, "end # module ", name)
        println(fio)
    end
end
