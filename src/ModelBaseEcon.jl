"""
    ModelBaseEcon

This package is part of the StateSpaceEcon ecosystem. 
It contains the basic elements needed for model definition.
StateSpaceEcon works with model objects defined with ModelBaseEcon.
"""
module ModelBaseEcon

using SparseArrays
using DiffResults
using ForwardDiff
using Printf

# The Timer submodule
include("Timer.jl")
using .Timer
export @timer, inittimer, stoptimer, printtimer

# The Options submodule
include("Options.jl")
using .OptionsMod
export Options, getoption, getoption!, setoption!

# The "misc" - various types and functions
include("misc.jl")

# NOTE: The order of inclusions matters.
include("abstract.jl")
include("evaluation.jl")
include("equation.jl")
include("steadystate.jl")
include("model.jl")

end # module
