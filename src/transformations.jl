##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020, Bank of Canada
# All rights reserved.
##################################################################################

export Transformation, NoTransform, LogTransform, NegLogTransform, transformation, inverse_transformation

"""
    transformation(::Type{<:Transformation})

Return a `Function` that will be substituted into the model equations and will be
called to transform the input data before solving. See also
[`inverse_transformation`](@ref).

It is expected that `transformation(T) ∘ inverse_transformation(T) == identity`
and `inverse_transformation(T) ∘ transformation(T) == identity`, but these is
not verified.

"""
function transformation end

"""
    inverse_transformation(::Type{<:Transformation})

Return a `Function` that will be called to transform the simulation data after solving. See also
[`transformation`](@ref).

It is expected that `transformation(T) ∘ inverse_transformation(T) == identity`
and `inverse_transformation(T) ∘ transformation(T) == identity`, but these is
not verified.

"""
function inverse_transformation end

"""
    abstract type Transformation end

The base class for all variable transformations.
"""
abstract type Transformation end
transformation(T::Type{<:Transformation}) = error("Transformation of type $T is not defined.")
inverse_transformation(T::Type{<:Transformation}) = error("Inverse transformation of type $T is not defined.")

"""
    NoTransform <: Transformation

The identity transformation.
"""
struct NoTransform <: Transformation end
transformation(::Type{NoTransform}) = Base.identity
inverse_transformation(::Type{NoTransform}) = Base.identity

"""
    LogTransform <: Transformation

The `log` transformation. The inverse is of course `exp`. This is the default
for variables declared with `@log`.
"""
struct LogTransform <: Transformation end
transformation(::Type{LogTransform}) = Base.log
inverse_transformation(::Type{LogTransform}) = Base.exp

"""
    NegLogTransform <: Transformation

The `log(-x)`, with the inverse being `-exp(x)`. Use this when the variable is
negative with exponential behaviour (toward -∞).
"""
struct NegLogTransform <: Transformation end
"logm(x) = log(-x)" @inline logm(x) = log(-x)
"mexp(x) = -exp(x)" @inline mexp(x) = -exp(x)
transformation(::Type{NegLogTransform}) = logm
inverse_transformation(::Type{NegLogTransform}) = mexp

export logm, mexp
