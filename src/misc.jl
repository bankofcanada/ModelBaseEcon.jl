##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

###########################################################
# Part 1: Error handling

"""
    abstract type ModelErrorBase <: Exception end

Abstract error type, base for specific error types used in ModelBaseEcon.

# Implementation (note for developers)

When implementing a derived error type, override two functions:
  * `msg(e::SomeModelError)` returning a string with the error message;
  * `hint(e::SomeModelError)` returning a string containing a suggestion of how
    to fix the problem. Optional, if not implemented for a type, the fallback
    implementation returns an empty string.

"""
abstract type ModelErrorBase <: Exception end
# export ModelErrorBase
"""
    msg(::ModelErrorBase)

Return the error message - a description of what went wrong.
"""
function msg end

"""
    hint(::ModelErrorBase)

Return the hint message - a suggestion of how the problem might be fixed.
"""
hint(::ModelErrorBase) = ""

# TODO: check if this is strictly necessary to have, test efficiency
# Helper function for getting an indexed item out of an ordered dict
Base.get(i::Integer, d::Union{OrderedDict,LittleDict}) = d[collect(keys(d))[i]]

function Base.showerror(io::IO, me::ME) where {ME<:ModelErrorBase}
    # MEstr = split("$(ME)", ".")[end]
    # println(io, MEstr, ": ", msg(me))
    println(io, ME, ": ", msg(me))
    h = hint(me)
    if !isempty(h)
        println(io, "    ", h)
    end
end

struct ModelError <: ModelErrorBase
    msg
end
ModelError() = ModelError("Unknown error")
msg(e::ModelError) = e.msg


"""
    modelerror(ME::Type{<:ModelErrorBase}, args...; kwargs...)

Raise an exception derived from [`ModelErrorBase`](@ref).
"""
modelerror(ME::Type{<:ModelErrorBase}=ModelError, args...; kwargs...) = throw(ME(args...; kwargs...))
modelerror(msg::AbstractString) = modelerror(ModelError, msg)

"""
    struct ModelNotInitError <: ModelErrorBase

Specific error type used when there's an attempt to use a Model object that
has not been initialized.
"""
struct ModelNotInitError <: ModelErrorBase end
msg(::ModelNotInitError) = "Model not ready to use."
hint(::ModelNotInitError) = "Call `@initialize model` first."
# export ModelNotInitError

"""
    struct NotImplementedError <: ModelErrorBase

Specific error type used when a feature is planned but not yet implemented. 
"""
struct NotImplementedError <: ModelErrorBase
    descr
end
msg(fe::NotImplementedError) = "Feature not implemented: $(fe.descr)."
# export NotImplementedError

struct EvalDataNotFound <: ModelErrorBase
    which::Symbol
end
msg(e::EvalDataNotFound) = "Evaluation data for :$(e.which) not found."
hint(e::EvalDataNotFound) = "Try calling `$(e.which)!(model)`."

struct SolverDataNotFound <: ModelErrorBase
    which::Symbol
end
msg(e::SolverDataNotFound) = "Solver data for :$(e.which) not found."
hint(e::SolverDataNotFound) = "Try calling `solve!(model, :$(e.which))`."


