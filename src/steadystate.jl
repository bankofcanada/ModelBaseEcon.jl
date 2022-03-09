##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

export SteadyStateEquation

"""
    struct SteadyStateEquation <: AbstractEquation

Data structure representing an individual steady state equation.
"""
struct SteadyStateEquation <: AbstractEquation
    type::Symbol
    vinds::Vector{Int64}
    vsyms::Vector{Symbol}
    expr::ExtExpr
    eval_resid::Function
    eval_RJ::Function
end

########################################################

"""
    struct SteadyStateVariable

Holds the steady state solution for one variable.
"""
struct SteadyStateVariable{DATA<:AbstractVector{Float64},MASK<:AbstractVector{Bool}}
    # The corresponding entry in m.allvars. Needed for its type
    name::ModelVariable
    # Its index in the m.allvars array
    index::Int
    # A view in the SteadyStateData.values location for this variable
    data::DATA
    # A view into the mask array 
    mask::MASK
end

for sym in (:lin, :log, :neglog, :steady, :exog, :shock)
    issym = Symbol("is", sym)
    eval(quote
        @inline $(issym)(s::SteadyStateVariable) = $(issym)(s.name)
    end)
end

@inline transformation(v::SteadyStateVariable) = transformation(v.name)
@inline inverse_transformation(v::SteadyStateVariable) = inverse_transformation(v.name)
@inline transform(x, v::SteadyStateVariable) = transform(x, v.name)
@inline inverse_transform(x, v::SteadyStateVariable) = inverse_transform(x, v.name)

#############################################################################
# Access to .level and .slope

function Base.getproperty(v::SteadyStateVariable, name::Symbol)
    if hasfield(typeof(v), name)
        return getfield(v, name)
    end
    data = getfield(v, :data)
    # we store transformed data, must invert to give back to user
    return name == :level ? inverse_transform(data[1], v) :
           name == :slope ? (islog(v) || isneglog(v) ? exp(data[2]) : data[2]) :
           getfield(v, name)
end

Base.setproperty!(v::SteadyStateVariable, name::Symbol, val) = begin
    data = getfield(v, :data)
    # we store transformed data, must transform user input
    if name == :level
        data[1] = transform(val, v)
    elseif name == :slope
        data[2] = (islog(v) || isneglog(v) ? log(val) : val)
    else
        setfield!(v, name, val)  # this will error (immutable)
    end
end

# use [] to get a time series of values.
# the ref= value is the t at which it equals its level
function Base.getindex(v::SteadyStateVariable, t; ref = first(t))
    if eltype(t) != eltype(ref)
        throw(ArgumentError("Must provide reference time of the same type as the time index"))
    end
    int_t = convert(Int, first(t) - ref):convert(Int, last(t) - ref)
    # we must inverse transform internal data before returning to user
    return inverse_transform(v.data[1] .+ v.data[2] .* int_t, v)
end

# pretty printing 

"""
    alignment5(io::IO, v::SteadyStateVariable)

Return a 5-tuple with the number of characters for the name, and the alignment 2-tuples for level and slope.

"""
function alignment5(io::IO, v::SteadyStateVariable)
    name = sprint(print, string(v.name.name), context = io, sizehint = 0)
    lvl_a = Base.alignment(io, v.level)
    if isshock(v) || issteady(v)
        slp_a = (0, 0)
    else
        slp_a = Base.alignment(io, v.slope)
    end
    (length(name), lvl_a..., slp_a...)
end


function alignment5(io::IO, vars::AbstractVector{SteadyStateVariable})
    a = (0, 0, 0, 0, 0)
    for v in vars
        a = max.(a, alignment5(io, v))
    end
    return a
end

function show_aligned5(io::IO, v::SteadyStateVariable, a = alignment5(io, v);
    mask = trues(2), sep1 = " = ",
    sep2 = islog(v) || isneglog(v) ? " * " : " + ",
    sep3 = islog(v) || isneglog(v) ? "^t" : "*t")
    name = sprint(print, string(v.name.name), context = io, sizehint = 0)
    if mask[1]
        lvl_a = Base.alignment(io, v.level)
        lvl = sprint(show, v.level, context = io, sizehint = 0)
    else
        lvl_a = (0, 1)
        lvl = "?"
    end
    if mask[2]
        slp_a = Base.alignment(io, v.slope)
        slp = sprint(show, v.slope, context = io, sizehint = 0)
    else
        slp_a = (0, 1)
        slp = "?"
    end
    print(io, "  ", repeat(' ', a[1] - length(name)), name,
        sep1, repeat(' ', a[2] - lvl_a[1]), lvl, repeat(' ', a[3] - lvl_a[2]))
    if (!issteady(v) && !isshock(v)) && !(v.data[2] + 1.0 ≈ 1.0)
        print(io, sep2, repeat(' ', a[4] - slp_a[1]), slp, repeat(' ', a[5] - slp_a[2]), sep3)
    end
end

@inline Base.show(io::IO, v::SteadyStateVariable) = show_aligned5(io, v)

########################################################

export SteadyStateData

"""
    SteadyStateData

Data structure that holds information about the steady state solution of the Model.
"""
struct SteadyStateData
    "List of steady state variables."
    vars::Vector{SteadyStateVariable}
    "Steady state solution vector."
    values::Vector{Float64}
    "`mask[i] == true if and only if `values[i]` holds the steady state value."
    mask::BitArray{1}
    "Steady state equations derived from the dynamic system."
    equations::Vector{SteadyStateEquation}
    "Steady state equations explicitly added with @steadystate."
    constraints::Vector{SteadyStateEquation}
    # default constructor
    SteadyStateData() = new([], [], [], [], [])
end

Base.push!(ssd::SteadyStateData, vars...) =
    for v in vars
        push!(ssd, v)
    end
Base.push!(ssd::SteadyStateData, var::Symbol) = push!(ssd, convert(ModelSymbol, var))
function Base.push!(ssd::SteadyStateData, var::ModelSymbol)
    for v in getfield(ssd, :vars)
        if v.name == var
            return v
        end
    end
    push!(getfield(ssd, :values), isexog(var) || isshock(var) ? 0.0 : 0.1, 0.0)
    push!(getfield(ssd, :mask), isexog(var) || isshock(var), isexog(var) || isshock(var) || issteady(var))
    ind = length(getfield(ssd, :vars)) + 1
    v = SteadyStateVariable(var, ind, @view(getfield(ssd, :values)[2ind.+(-1:0)]), @view(getfield(ssd, :mask)[2ind.+(-1:0)]))
    push!(getfield(ssd, :vars), v)
    return v
end

export alleqns
"""
    alleqns(ssd::SteadyStateData)

Return a list of all steady state equations.

The list contains all equations derived from the dynamic system and all explicitly added steady state constraints.
"""
@inline alleqns(ssd::SteadyStateData) = vcat(ssd.constraints, ssd.equations,)

export neqns
"""
    neqns(ssd::SteadyStateData)

Return the total number of equations in the steady state system, including the ones derived from the dynamic system and the
ones added explicitly as steady state constraints.
"""
@inline neqns(ssd::SteadyStateData) = length(ssd.equations) + length(ssd.constraints)

export geteqn
"""
    geteqn(i, ssd::SteadyStateData)

Return the i-th steady state equation. Index i is interpreted as in the output of `alleqns`.
Calling `geteqn(i, sdd)` has the same effect as `alleqn(ssd)[i]`, but it's more efficient.

### Example
```julia
# Iterate all equations like this:
for i = 1:neqns(ssd)
    eqn = geteqn(i, ssd)
    # do something awesome with `eqn` and `i`
end
```
"""
function geteqn(i::Integer, ssd::SteadyStateData)
    ci = i - length(ssd.equations)
    return ci > 0 ? ssd.constraints[ci] : ssd.equations[i]
end

Base.show(io::IO, ::MIME"text/plain", ssd::SteadyStateData) = show(io, ssd)
Base.show(io::IO, ssd::SteadyStateData) = begin
    if issssolved(ssd)
        println(io, "Steady state solved.")
    else
        println(io, "Steady state not solved.")
    end
    len = length(ssd.constraints)
    if len == 0
        println(io, "No additional constraints.")
    else
        println(io, len, " additional constraint", ifelse(len > 1, "s.", "."))
        for c in ssd.constraints
            println(io, "    ", c)
        end
    end
end

#########
# Implement access to steady state values using dot notation and index notation

function Base.propertynames(ssd::SteadyStateData, private::Bool = false)
    if private
        return ((v.name.name for v in ssd.vars)..., fieldnames(SteadyStateData)...,)
    else
        return ((v.name.name for v in ssd.vars)...,)
    end
end

function Base.getproperty(ssd::SteadyStateData, sym::Symbol)
    if sym ∈ fieldnames(SteadyStateData)
        return getfield(ssd, sym)
    else
        for v in ssd.vars
            if v.name == sym
                return v
            end
        end
        throw(ArgumentError("Unknown variable $sym."))
    end
end

Base.getindex(ssd::SteadyStateData, ind::Int) = ssd.vars[ind]
Base.getindex(ssd::SteadyStateData, sym::ModelSymbol) = getproperty(ssd, sym.name)
Base.getindex(ssd::SteadyStateData, sym::Symbol) = getproperty(ssd, sym)
Base.getindex(ssd::SteadyStateData, sym::AbstractString) = getproperty(ssd, Symbol(sym))

@inline ss_symbol(ssd::SteadyStateData, vi::Int) = Symbol("#", ssd.vars[(1+vi)÷2].name.name, "#", (vi % 2 == 1) ? :lvl : :slp, "#")

#########################
# 

export printsstate

"""
    printsstate([io::IO,] ssd::SteadyStateData)

Display steady state solution.

Steady state solution is presented in a table, where the first column is
the name of the variable, the second and third columns are the corresponding
values of the level and the slope. If the value is not determined
(as per its `mask` value) then it is displayed as "*".
"""
function printsstate(io::IO, model::AbstractModel)
    io = IOContext(io, :compact => get(io, :compact, true))
    ssd = model.sstate
    println(io, "Steady State Solution:")
    a = max.(alignment5(io, ssd.vars), (0, 0, 3, 0, 3))
    for v in ssd.vars
        show_aligned5(io, v, a, mask = v.mask)
        println(io)
    end
end
printsstate(model::AbstractModel) = printsstate(Base.stdout, model)

###########################
# Make steady state equation from dynamic equation

# Idea:
#   in the steady state equation, we assume that the variable y_ss
#   follows a linear motion expressed as y_ss[t] = y_ss#lvl + t * y_ss#slp
#   where y_ss#lvl and y_ss#slp are two unknowns we solve for.
# 
#   The dynamic equation has mentions of lags and leads. We replace those
#   with the above expression.
# 
#   Since we have two parameters to determine, we need two steady state equations
#   from each dynamic equation. We get this by writing the dynamic equation at
#   two different values of `t` - 0 and another one we call `shift`.
# 
#   Shift is a an option in the model object, which the user can set to any integer
#   other than 0. The default is 10.

"""
    SSEqnData

Internal structure used for evaluation of the residual of the steady state equation
derived from a dynamic equation.

"""
struct SSEqnData{M<:AbstractModel}
    "Whether or not to add model.shift"
    shift::Bool
    "Reference to the model object (needed for the current value of shift"
    model::Ref{M}
    "Information needed to compute the Jacobian matrix of the transformation between steady state and dynamic unknowns"
    JT::Vector
    "The dynamic equation instance"
    eqn::Equation
    SSEqnData(s, m::M, jt, e) where {M<:AbstractModel} = new{M}(s, Ref(m), jt, e)
    SSEqnData(s, m::Ref{M}, jt, e) where {M<:AbstractModel} = new{M}(s, m, jt, e)
end

##############

@inline __lag(jt, s) = s.shift ? jt.tlag + s.model[].shift : jt.tlag
@inline function __to_dyn_pt(pt, s)
    # This function applies the transformation from steady
    # state equation unknowns to dynamic equation unknowns
    buffer = fill(0.0, length(s.JT))
    for (i, jt) in enumerate(s.JT)
        if length(jt.ssinds) == 1
            pti = pt[jt.ssinds[1]]
        else
            pti = pt[jt.ssinds[1]] + __lag(jt, s) * pt[jt.ssinds[2]]
        end
        buffer[i] += pti
    end
    return buffer
end
@inline function __to_ssgrad(pt, jj, s)
    # This function inverts the transformation. jj is the gradient of the
    # dynamic equation residual with respect to the dynamic equation unknowns.
    # Here we compute the Jacobian of the transformation and use it to compute
    ss = zeros(size(pt))
    for (i, jt) in enumerate(s.JT)
        if length(jt.ssinds) == 1
            pti = pt[jt.ssinds[1]]
            ss[jt.ssinds[1]] += jj[i]
        else
            local lag_jt = __lag(jt, s)
            pti = pt[jt.ssinds[1]] + lag_jt * pt[jt.ssinds[2]]
            ss[jt.ssinds[1]] += jj[i]
            ss[jt.ssinds[2]] += jj[i] * lag_jt
        end
    end
    # NOTE regarding the above: The dynamic equation is F(x_t) = 0
    # Here we're solving F(u(l+t*s)) = 0
    # The derivative is dF/dl = F' * u' and dF/ds = F' * u' * t
    # F' is in jj[i]
    # u(x) = x, so u'(x) = 1
    return ss
end
function sseqn_resid_RJ(s::SSEqnData)
    function _resid(pt::AbstractVector{Float64})
        return s.eqn.eval_resid(__to_dyn_pt(pt, s))
    end
    function _RJ(pt::AbstractVector{Float64})
        R, jj = s.eqn.eval_RJ(__to_dyn_pt(pt, s))
        return R, __to_ssgrad(pt, jj, s)
    end
    return _resid, _RJ
end

"""
    make_sseqn(model::AbstractModel, eqn::Equation; shift::Int64=0)

Create a steady state equation from the given dynamic equation for the given model.

Internal function, do not call directly.

"""
function make_sseqn(model::AbstractModel, eqn::Equation, shift::Bool)
    local allvars = model.allvars
    tvalue(t) = shift ? t + model.shift : t
    # ssind converts the dynamic index (t, v) into
    # the corresponding indexes of steady state unknowns.
    # Returned value is a list of length 0, 1, or 2.
    function ssind((var, ti),)::Array{Int64,1}
        vi = _index_of_var(var, allvars)
        no_slope = isshock(var) || issteady(var)
        # The level unknown has index 2*vi-1.
        # The slope unknown has index 2*vi. However:
        #  * :steady and :shock variables don't have slopes
        #  * :lin and :log variables the slope is in the equation
        #    only if the effective t-index is not 0.
        if no_slope || (!shift && ti == 0)
            return [2vi - 1]
        else
            return [2vi - 1, 2vi]
        end
    end
    local ss = model.sstate
    # The steady state indexes.
    vinds = unique(vcat(map(ssind, (collect∘keys)(eqn.tsrefs))...))
    # The corresponding steady state symbols
    vsyms = Symbol[ss_symbol(ss, vi) for vi in vinds]
    # In the next loop we build the matrix JT which transforms
    # from the steady state values to the dynamic point values.
    JT = []
    for (i, (var, ti)) in enumerate(keys(eqn.tsrefs))
        val = (ssinds = indexin(ssind((var, ti)), vinds), tlag = ti)
        push!(JT, val)
    end
    type = shift == 0 ? :tzero : :tshift
    let sseqndata = SSEqnData(shift, Ref(model), JT, eqn)
        return SteadyStateEquation(type, vinds, vsyms, eqn.expr, sseqn_resid_RJ(sseqndata)...)
    end
end

###########################
# Make steady state equation from user input


"""
    setss!(model::AbstractModel, expr::Expr; type::Symbol, modelmodule::Module)

Add a steady state equation to the model. Equations added by `setss!` are in
addition to the equations generated automatically from the dynamic system.

Internal function, do not call directly. Use [`@steadystate`](@ref) instead.

"""
function setss!(model::AbstractModel, expr::Expr; type::Symbol, modelmodule::Module = moduleof(model))

    if expr.head != :(=)
        error("Expected an equation, not $(expr.head)")
    end

    @assert type ∈ (:level, :slope) "Unknown steady state equation type $type. Expected either `level` or `slope`."

    local ss = sstate(model)

    local allvars = model.allvars

    ###############################################
    #     ssprocess(val)
    # 
    # Process the given value to extract information about mentioned parameters and variables.
    # This function has the side effect of populating the vectors
    # `vinds`, `vsyms`, `val_params` and `source`
    # 
    # Algorithm is recursive over the given expression. The bottom of the recursion is the
    # processing of a `Number`, a `Symbol`, (or a `LineNumberNode`).
    # 
    # we will store indices
    local vinds = Int64[]
    local vsyms = Symbol[]
    # we will store parameters mentioned in `expr` here
    local val_params = Symbol[]
    local source = LineNumberNode[]
    # nothing to do with a number
    ssprocess(val::Number) = val
    # a symbol could be a variable (shock, auxvar), a parameter, or unknown.
    function ssprocess(val::Symbol)
        if val ∈ keys(model.parameters)
            # parameter - keep track that it's mentioned
            push!(val_params, val)
            return val
        end
        vind = indexin([val], allvars)[1]
        if vind !== nothing
            # it's a vriable of some sort: make a symbol and an index for the
            # corresponding steady state unknown
            vsym = Symbol("#", type, "#", val, "#")
            push!(vsyms, vsym)
            push!(vinds, type == :level ? 2vind - 1 : 2vind)
            if need_transform(allvars[vind])
                func = inverse_transformation(allvars[vind])
                return :($func($vsym))
            else
                return vsym
            end
        end
        # what to do with unknown symbols?
        error("unknown parameter $val")
    end
    # a sorce line information: store it and remove it from the expression
    ssprocess(val::LineNumberNode) = (push!(source, val); nothing)
    # process an expression recursively
    function ssprocess(val::Expr)
        if val.head == :(=)
            # we process the lhs and rhs seperately: there shouldn't be any equal signs
            error("unexpected equation.")
        end
        if val.head == :block
            # in a begin-end block, process each line and gather the results
            args = filter(x -> x !== nothing, map(ssprocess, val.args))
            if length(args) == 1
                # Only one thing left - no need for the begin-end anymore
                return args[1]
            else
                # reassemble the processed expressions back into a begin-end block
                return Expr(:block, args...)
            end
        elseif val.head == :call
            # in a function call, process each argument, but not the function name (args[1]) and reassemble the call
            args = filter(x -> x !== nothing, map(ssprocess, val.args[2:end]))
            return Expr(:call, val.args[1], args...)
        else
            # whatever this it, process each subexpression and reassemble it
            args = filter(x -> x !== nothing, map(ssprocess, val.args))
            return Expr(val.head, args...)
        end
    end
    # end of ssprocess() definition
    ###############################################
    # 
    lhs, rhs = expr.args
    lhs = ssprocess(lhs)
    rhs = ssprocess(rhs)
    expr.args .= MacroTools.unblock.(expr.args)
    # 
    nargs = length(vinds)
    # In case there's no source information, add a dummy one
    push!(source, LineNumberNode(0))
    # create the resid and RJ functions for the new equation
    # To do this, we use `makefuncs` from evaluation.jl
    resid, RJ = let mparams = parameters(model)
        # create a list of expressions that assign the values of model parameters to
        # variables of the same name
        param_assigments = Expr(:block)
        for p in unique(val_params)
            push!(param_assigments.args, :(local $(p) = $(mparams).$(p)))
        end
        residual = Expr(:block, source[1], :($(lhs) - $(rhs)))
        funcs_expr = makefuncs(residual, vsyms, param_assigments; mod = modelmodule)
        modelmodule.eval(funcs_expr)
    end
    # We have all the ingredients to create the instance of SteadyStateEquation
    for i = 1:2
        # remove blocks with line numbers from expr.args[i]
        a = expr.args[i]
        if Meta.isexpr(a, :block)
            args = filter(x -> !isa(x, LineNumberNode), a.args)
            if length(a.args) == 1
                expr.args[i] = args[1]
            end
        end
    end
    sscon = SteadyStateEquation(type, vinds, vsyms, expr, resid, RJ)
    if nargs == 1
        # The equation involves only one variable. See if there's already an equation
        # with just that variable and, if so, remove it.
        for (i, ssc) in enumerate(ss.constraints)
            if ssc.type == type && length(ssc.vinds) == 1 && ssc.vinds[1] == sscon.vinds[1]
                ss.constraints[i] = sscon
                return sscon
            end
        end
    end
    push!(ss.constraints, sscon)
    return sscon
end


export @steadystate

"""
    @steadystate model [type] equation

Add a steady state equation to the model.

The steady state system of the model is automatically derived from the dynamic
system. Use this macro to define additional equations for the steady state.
This is particularly useful in the case of a non-linear model that might have
multiple steady state, or the steady state might be difficult to solve for,
to help the steady state solver find the one you want to use.

  * `model` is the model instance you want to update
  * `type` (optional) is the type of constraint you want to add. This can be `level`
  or `slope`. If missing, the default is `level`
  * `equation` is the expression defining the steady state constraint. In the
  equation, use variables and shocks from the model, but without any t-references.

"""
macro steadystate(model, type::Symbol, equation::Expr)
    thismodule = @__MODULE__
    modelmodule = __module__
    return esc(:($(thismodule).setss!($(model), $(Meta.quot(equation)); type = $(QuoteNode(type)))))  # , modelmodule=$(modelmodule))))
end

macro steadystate(model, equation::Expr)
    thismodule = @__MODULE__
    return esc(:($(thismodule).setss!($(model), $(Meta.quot(equation)); type = :level))) # , modelmodule=$(modelmodule))))
end

"""
    initssdata!(m::AbstractModel)

Initialize the steady state data structure of the given model.

Do not call directly. This is an internal function, called during
[`@initialize`](@ref)

"""
function initssdata!(model::AbstractModel)
    ss = sstate(model)
    empty!(ss.vars)
    empty!(ss.values)
    empty!(ss.mask)
    for var in model.allvars
        push!(ss, var)
    end
    empty!(ss.equations)
    for eqn in alleqns(model)
        push!(ss.equations, make_sseqn(model, eqn, false))
    end
    if !model.flags.ssZeroSlope
        for eqn in alleqns(model)
            push!(ss.equations, make_sseqn(model, eqn, true))
        end
    end
    empty!(ss.constraints)
    return nothing
end

export issssolved
"""
    issssolved(sstate::SteadyStateData)

Return `true` if the steady state has been solved, or `false` otherwise.

"""
@inline issssolved(ss::SteadyStateData) = all(ss.mask)


export assign_sstate!
@inline assign_sstate!(model::AbstractModel, args) = (assign_sstate!(model.sstate, args); model)
@inline assign_sstate!(model::AbstractModel; kwargs...) = assign_sstate!(model, kwargs)
@inline assign_sstate!(ss::SteadyStateData; kwargs...) = assign_sstate!(ss, kwargs)
function assign_sstate!(ss::SteadyStateData, args)
    not_model_variables = Symbol[]
    for (key, value) in args
        sk = Symbol(key)
        if !hasproperty(ss, sk)
            push!(not_model_variables, sk)
            continue
        end
        var = getproperty(ss, sk)
        if value isa NamedTuple
            var.level = value.level
            var.slope = value.slope
        elseif value isa Union{NTuple{2,<:Number},Vector{<:Number}}
            var.level = value[1]
            var.slope = value[2]
        else
            var.level = value
            var.slope = 0
        end
        var.mask[:] .= true
    end
    if !isempty(not_model_variables)
        @warn "Model does not have the following variables: " not_model_variables
    end
    return ss
end

@inline export_sstate!(container, model::AbstractModel) = export_sstate!(container, model.sstate; ssZeroSlope = model.ssZeroSlope)
@inline export_sstate(m_or_s::Union{AbstractModel,SteadyStateData}, C::Type = Dict{Symbol,Any}; kwargs...) = export_sstate!(C(), m_or_s; kwargs...)
function export_sstate!(container, ss::SteadyStateData; ssZeroSlope::Bool = false)
    if ssZeroSlope
        for var in ss.vars
            push!(container, var.name => var.level)
        end
    else
        for var in ss.vars
            push!(container, var.name => copy(var.data))
        end
    end
    return container
end
export export_sstate, export_sstate!
