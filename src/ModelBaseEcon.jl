##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

"""
    ModelBaseEcon

This package is part of the StateSpaceEcon ecosystem.
It provides the basic elements needed for model definition.
StateSpaceEcon works with model objects defined with ModelBaseEcon.
"""
module ModelBaseEcon

using OrderedCollections
using MacroTools
using SparseArrays
using DiffResults
using ForwardDiff
using Printf

# Note: The full type specification for `LittleDict` has 4 parameters.
# Specifying only the first two in a struct as:
# struct S
#    a::LittleDict{Symbol, Int}
# end
# will be treated by Julia as a non-concrete field.
# Therefore, define a small helper typealias to use in structs:
const LittleDictVec{K,V} = LittleDict{K,V,Vector{K},Vector{V}}

# The Timer submodule
include("Timer.jl")
using .Timer
export @timer, inittimer, stoptimer, printtimer

# The Options submodule
include("Options.jl")

# The "misc" - various types and functions
include("misc.jl")

# NOTE: The order of inclusions matters.
include("abstract.jl")
include("parameters.jl")
include("evaluation.jl")
include("transformations.jl")
include("variables.jl")
include("equation.jl")
include("steadystate.jl")
include("metafuncs.jl")
include("model.jl")
include("export_model.jl")
include("linearize.jl")
include("precompile.jl")

"""
    @using_example name

Load models from the package examples/ folder.
The `@load_example` version is deprecated - stop using it now.
"""
macro using_example(name)
    examples_path = joinpath(dirname(pathof(@__MODULE__)), "..", "examples")
    return quote
        push!(LOAD_PATH, $(examples_path))
        using $(name)
        pop!(LOAD_PATH)
        $(name)
    end |> esc
end

" Deprecated. Use `@using_example` instead."
macro load_example(name)
    Base.depwarn("Use `@using_example` instead.", Symbol("@load_example"))
    return esc(:(@using_example $name))
end
export @using_example, @load_example



end # module
