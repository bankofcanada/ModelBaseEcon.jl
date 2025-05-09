##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################


function Base.show(io::IO, dfm::DFM)
    show(io, dfm.model)
    if dfm.model._state == :ready
        println(io, "Parametrization")
        for name1 in propertynames(dfm.params)
            if true
                println(io, "  ", name1, ": ", getproperty(dfm.params, name1))
            else
                println(io, "  ", name1)
                p = getproperty(dfm.params, name1)
                for name2 in propertynames(p)
                    println(io, "    ", name2, ": ", getproperty(p, name2))
                end
            end
        end
        # print(io, dfm.params)
    end
end

function Base.show(io::IO, model::DFMModel)
    if model._state != :ready
        println(io, "DFMModel ", model.name, " (not ready)")
        return
    end
    println(io, "DFMModel ", model.name,)
    if length(model.observed) == 0
        println(io, "  No observed variables")
    else
        for (name, block) in model.observed
            if ismixfreq(block)
                println(io, "  ", typeof(block), " ", name)
            else
                println(io, "  ", nameof(typeof(block)), " ", name)
            end
            print_padded(io, "    ", block.vars .=> block.shks)
        end
    end
    if length(model.components) == 0
        println(io, "  No latent variables")
    else
        for (name, block) in model.components
            if ismixfreq(block)
                println(io, "  ", typeof(block), " ", name)
            else
                println(io, "  ", nameof(typeof(block)), " ", name)
            end
            print_padded(io, "    ", block.vars .=> block.shks)
        end
    end
    if !isempty(model.observed)
        println(io, "  Loadings map")
        for (name, block) in model.observed
            for (var, refs) in block.var2comps
                print_padded(io, "     $var ~", Iterators.map(DFMModels.vars_comp_refs, values(refs))..., delim=" + ")
            end
        end
    end
end

function print_padded(io, heading, all_things, more_things...; delim=", ", maxchars=displaysize(io)[2])
    padding = length(heading) + 1
    # split all_things into first and the rest
    first_thing, other_things... = all_things
    # start a new line with the first thing
    line = heading * " " * sprint(print, first_thing; context=io)
    for thing in Iterators.flatten((other_things, more_things...))
        s = sprint(print, thing; context=io, sizehint=0)
        # is there room for this thing on this line?
        if length(line) + length(s) + length(delim) + 1 >= maxchars
            # output this line
            println(io, line, delim)
            # start a new line with padding and this thing
            line = lpad(s, padding)
        else
            # add this thing to this line
            line = line * delim * s
        end
    end
    println(io, line)
end


