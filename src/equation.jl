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

"Placeholder evaluation function to use in Equation costruction while it is being created"
@inline eqnnotready(x...) = throw(EqnNotReadyError())

"""
    mutable struct EqnFlags ... end

Holds information about the equation. Flags can be specified in the model
definition by annotating the equation with `@<flag>` (insert the flag you want
to raise in place of `<flag>`).

Supported flags:
 * `@log lhs = rhs` instructs the model parser to make the residual
   `loc(lhs / rhs)`. Normally the residual is computed as `lhs - rhs`.
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
    struct Equation <: AbstractEquation

Data structure representing a single equation in our state space model.

### Methods (for users)
  * TODO

### Implementation (for developers)
During the phase of definition of the Model, this type simply stores the expression
entered by the user. During @initialize(), the true data structure is constructed.
We need this, because the construction of the equation requires information from
the Model object, which may not be available at the time the equation expression
is first read.
"""
struct Equation <: AbstractEquation
    doc::String
    flags::EqnFlags
    "The original expression entered by the user"
    expr::ExtExpr      # original expression
    """
    The residual expression computed from [`expr`](@ref). 
    It is used in the evaluation functions. 
    Leads and lags of variables and shocks mentioned in [`expr`](@ref) 
    are recorded in [`vinds`](@ref) and translated to symbols in [`vsyms`](@ref).
    """
    resid::Expr     # residual expression (with variables renamed and parameters substituted)
    """
    Indexes of variables/shocks mentioned in [`expr`](@ref). Each index is in the form
    `(t, v)`, where `t` is the lag (if negative), 0, or the lead (if positive) and `v` 
    is the index of the variable/shock.
    """
    vinds::Vector{Tuple{Int64,Int64}}   # (t, v) indices of the relevant variables 
    "Symbols used in `resid` replacing mentions of variables and shocks"
    vsyms::Vector{Symbol}  # symbols representing the relevant variables in resid
    "maximum lag mentioned in `expr`"
    maxlag::Int64   # for this equation
    "maximum lead mentioned in `expr`"
    maxlead::Int64  # for this equation
    "A callable (function) evaluating the residual. Argument is a vector of Float64 same lenght as `vinds`"
    eval_resid::Function  # function evaluating the residual
    "A callable (function) evaluating the (residual, gradient) pair. Argument is a vector of Float64 same lenght as `vinds`"
    eval_RJ::Function     # Function evaluating the residual and its gradient
end

# 
# dummy constructor - just stores the expresstion without any processing
Equation(expr::ExtExpr) = Equation("", EqnFlags(), expr, Expr(:block), [], [], 0, 0, eqnnotready, eqnnotready)

# constructor that computes maxlag and maxlead on the fly
function Equation(doc, flags, expr, resid, vinds, vsyms, eval_resid, eval_RJ)
    # compute `maxlag` and `maxlead`
    maxlag, maxlead = (isempty(vinds) ? (0, 0) : extrema(v[1] for v in vinds) .* (-1, 1))
    # call the default constructor
    return Equation(doc, flags, expr, resid, vinds, vsyms, maxlag, maxlead, eval_resid, eval_RJ)
end

# Allows us to pass a Number of a Symbol or a raw Expr to calls where Equation is expected.
Base.convert(::Type{Equation}, e::ExtExpr) = Equation(e)

