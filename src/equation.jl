##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

"""
struct EqnNotReadyError <: ModelErrorBase

Concrete error type used to indicate that a given equation has not been prepared
for use in the model yet.
"""
struct EqnNotReadyError <: ModelErrorBase end
msg(::EqnNotReadyError) = "Equation not ready to use."
hint(::EqnNotReadyError) = "Call `@initialize model` or `add_equation!()` first."

###############################################
# 

# Equation expressions typed by the user are of course valid equations, however
# during processing we use recursive algorithms, with the bottom of the recursion
# being a Number or a Symbol.  So we need a type that allows these.
const ExtExpr = Union{Expr,Symbol,Number}

# Placeholder evaluation function to use in Equation construction while it is
# being created
eqnnotready(x...) = throw(EqnNotReadyError())

"""
    mutable struct EqnFlags ⋯ end

Holds information about the equation. Flags can be specified in the model
definition by annotating the equation with `@<flag>` (insert the flag you want
to raise in place of `<flag>`). Multiple flags may be applied to the same
equation.

Supported flags:
 * `@log lhs = rhs` instructs the model parser to make the residual
   `log(lhs / rhs)`. Normally the residual is `lhs - rhs`.
 * `@lin lhs = rhs` marks the equation for selective linearization.

"""
mutable struct EqnFlags
    lin::Bool
    log::Bool
    EqnFlags() = new(false, false)
    EqnFlags(lin, log) = new(lin, log)
end

Base.hash(f::EqnFlags, h::UInt) = hash(((f.:($flag) for flag in fieldnames(EqnFlags))...,), h)
Base.:(==)(f1::EqnFlags, f2::EqnFlags) = all(f1.:($flag) == f2.:($flag) for flag in fieldnames(EqnFlags))

export Equation
"""
    struct Equation <: AbstractEquation ⋯ end

Data type representing a single equation in a model.

Equations are defined in [`@equations`](@ref) blocks. The actual equation
instances are later created with [`@initialize`](@ref) and stored within
the model object.

Equation flags can be specified by annotating the equation definition with one
or more `@<flag>`. See [`EqnFlags`](@ref) for details.

Each equation has two functions associated with it, one which computes the
residual and the other computes both the residual and the gradient . Usually
there's no need to users to call these functions directly. They are used
internally by the solvers.
"""
struct Equation <: AbstractEquation
    ### Implementation note 
    # During the phase of definition of the Model, this type simply stores the expression
    # entered by the user. During @initialize(), the full data structure is constructed.
    # We need this, because the construction of the equation requires information from
    # the Model object, which may not be available at the time the equation expression
    # is first read.
    doc::String
    flags::EqnFlags
    "The original expression entered by the user"
    expr::ExtExpr      # original expression
    """
    The residual expression computed from [`expr`](@ref). It is used in the
    evaluation functions. Mentions of known identifiers are replaced by other
    symbols and mapping of the symbol and the original is recorded
    """
    resid::Expr     # residual expression
    "references to time series variables"
    tsrefs::LittleDictVec{Tuple{ModelSymbol, Int}, Symbol}
    "references to steady states of variables"
    ssrefs::LittleDictVec{ModelSymbol, Symbol}
    "references to parameter values"
    prefs::LittleDictVec{Symbol, Symbol}
    "A callable (function) evaluating the residual. Argument is a vector of Float64 same lenght as `vinds`"
    eval_resid::Function  # function evaluating the residual
    "A callable (function) evaluating the (residual, gradient) pair. Argument is a vector of Float64 same lenght as `vinds`"
    eval_RJ::Function     # Function evaluating the residual and its gradient
end

# 
# dummy constructor - just stores the expresstion without any processing
Equation(expr::ExtExpr) = Equation("", EqnFlags(), expr, Expr(:block), 
                                    LittleDict{Tuple{ModelSymbol, Int}, Symbol}(),
                                    LittleDict{ModelSymbol, Symbol}(), 
                                    LittleDict{Symbol, Symbol}(),
                                    eqnnotready, eqnnotready)


function Base.getproperty(eqn::Equation, sym::Symbol)
    if sym == :maxlag
        tsrefs = getfield(eqn, :tsrefs)
        return isempty(tsrefs) ? 0 : -minimum(v -> v[2], keys(tsrefs))
    elseif sym == :maxlead
        tsrefs = getfield(eqn, :tsrefs)
        return isempty(tsrefs) ? 0 : maximum(v -> v[2], keys(tsrefs))
    else
        return getfield(eqn, sym)
    end
end

# Allows us to pass a Number of a Symbol or a raw Expr to calls where Equation is expected.
Base.convert(::Type{Equation}, e::ExtExpr) = Equation(e)
