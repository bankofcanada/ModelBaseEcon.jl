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

include("Timer.jl")
using .Timer
export @timer, inittimer, stoptimer, printtimer

include("Options.jl")
using .OptionsMod
export Options, getoption, getoption!, setoption!

end # module
