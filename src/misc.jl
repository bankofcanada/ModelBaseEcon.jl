#

###########################################################
# Part 1: Error handling

"""
    ModelErrorBase

Abstract error type, base for specific error types used in ModelBaseEcon.

# Implementation (note for developers)

When implementing a derived error type, override two functions: 
  * [`msg(e::SomeModelError)`](@ref) returning a string with the error message;
  * [`hint(e::SomeModelError)`](@ref) returning a string containing a suggestion
    of how to fix the problem. Optional, if not implemented for a type, the fallback
    implementation returns an empty string.

"""
abstract type ModelErrorBase <: Exception end

"""
    msg(::ME) where ME <: ModelErrorBase

Return the error message - a description of what went wrong.
"""
msg(::ME) where ME <: ModelErrorBase = "Unknown error"

"""
    hint(::ME) where ME <: ModelErrorBase

Return the hint message - a suggestion of how the problem might be fixed.
"""
hint(::ME) where ME <: ModelErrorBase = ""

function Base.showerror(io::IO, me::ME) where ME <: ModelErrorBase
    # MEstr = split("$(ME)", ".")[end]
    # println(io, MEstr, ": ", msg(me))
    println(io, ME, ": ", msg(me))
    h = hint(me)
    if !isempty(h)
        println(io, "    ", h)
    end
end


"""
    struct ModelError <: ModelErrorBase
    
Concrete error type used when no specific error description is available.
"""
struct ModelError <: ModelErrorBase end

@inline modelerror(ME::Type{<:ModelErrorBase}=ModelError, args...; kwargs...) = throw(ME(args...; kwargs...))


"""
    struct ModelNotInitError <: ModelErrorBase

Specific error type used when there's an attempt to use a Model object that
has not been initialized.
"""
struct ModelNotInitError <: ModelErrorBase end
msg(::ModelNotInitError) = "Model not ready to use."
hint(::ModelNotInitError) = "Call `@initialize model` first."


"""
    struct NotImplementedError <: ModelErrorBase

Specific error type used when a feature is planned but not yet implemented. 
"""
struct NotImplementedError <: ModelErrorBase 
    descr
end
msg(fe::NotImplementedError) = "Feature not implemented: $(fe.descr)."


"""
    struct EqnNotReadyError <: ModelErrorBase

Concrete error type used to indicate that a given equation has not been prepared
for use in the model yet.
"""
struct EqnNotReadyError <: ModelErrorBase end
msg(::EqnNotReadyError) = "Equation not ready to use."
hint(::EqnNotReadyError) = "Call `@initialize model` or `add_equation!()` first."

###########################################################
# Part 2: Equations evaluation












