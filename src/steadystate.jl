


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

export SteadyStateData

"""
    SteadyStateData

Data structure that holds information about the steady state solution of the Model.
"""
struct SteadyStateData
    "List of steady state variables."
    vars::Vector{Symbol}
    "Steady state solution vector."
    values::Vector{Float64}
    "`mask[i] == true if and only if `values[i]` holds the steady state value."
    mask::Vector{Bool}
    "Steady state equations derived from the dynamic system."
    equations::Vector{SteadyStateEquation}
    "Steady state equations explicitly added with @steadystate."
    constraints::Vector{SteadyStateEquation}
    # default constructor
    SteadyStateData() = new([], [], [], [], [])
end

export alleqns
"""
    alleqns(ssd::SteadyStateData)

Return a list of all steady state equations.

The list contains all equations derived from the dynamic system and all explicitly added steady state constraints.
"""
@inline alleqns(ssd::SteadyStateData) = vcat(ssd.equations, ssd.constraints)

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
    if isempty(ssd.constraints)
        println(io, "No additional constraints.")
    else
        println(io, length(ssd.constraints), " additional constraints.")
        for c in ssd.constraints
            println(io, "    ", c)
        end
    end
end

#####
# These are used for indexing in the values vector

@inline makesym(T::Val, var::AbstractString) = makesym(T, Symbol(var))
@inline makesym(T::Val, var::ModelSymbol) = makesym(T, convert(Symbol, var))
@inline makesym(::Val{:level}, var::Symbol) = Symbol("$(var)#lvl")
@inline makesym(::Val{:slope}, var::Symbol) = Symbol("$(var)#slp")
@inline makeind(::Val{:level}, ind::Int64) = 2 * ind - 1
@inline makeind(::Val{:slope}, ind::Int64) = 2 * ind

#########
# Implement access to steady state values using index notation ([])

struct SSMissingVariableError <: ModelErrorBase
    var::Symbol
end
msg(me::SSMissingVariableError) = "Steady state variable $(me.var) not found."

struct SSVarData{DATA <: AbstractVector{Float64}}
    name::Symbol
    index::Int
    value::DATA
end

Base.propertynames(::SSVarData) = (:level, :slope)
Base.getproperty(vd::SSVarData, prop::Symbol) = prop == :level ? getindex(getfield(vd, :value), 1) :
                                                prop == :slope ? getindex(getfield(vd, :value), 2) :
                                                                 getfield(vd, prop)
Base.setproperty!(vd::SSVarData, prop::Symbol, val) = prop == :level ? setindex!(getfield(vd, :value), val, 1) :
                                                      prop == :slope ? setindex!(getfield(vd, :value), val, 2) :
                                                                       setfield!(vd, prop, val)

Base.getindex(vd::SSVarData, i::Int) = getindex(vd.value, i)
Base.getindex(vd::SSVarData, s) = getproperty(vd, Symbol(s))
Base.setindex!(vd::SSVarData, val, i::Int) = setindex!(vd.value, val, i)
Base.setindex!(vd::SSVarData, val, s) = setproperty!(vd, Symbol(s), val)


Base.show(io::IO, ::MIME"text/plan", vd::SSVarData) = show(io, vd)
Base.show(io::IO, vd::SSVarData) = print(io, vd.name, " : ", NamedTuple{(:level, :slope)}(vd.value))

function Base.getindex(sstate::SteadyStateData, var)
    ind = indexin([makesym(Val(:level), var)], sstate.vars)[1]
    if ind === nothing
        throw(SSMissingVariableError(var))
    else
        return SSVarData(Symbol(var), ind, view(sstate.values, ind:ind + 1))
    end
end

function Base.setindex!(sstate::SteadyStateData, val, var)
    ind = indexin([makesym(Val(:level), var)], sstate.vars)[1]
    if ind === nothing
        throw(SSMissingVariableError(v))
    elseif val isa Number
        sstate.values[ind] = val
    elseif val isa NamedTuple
        if :level in keys(val)
            sstate.values[ind] = val.level
        end
        if :slope in keys(val)
            sstate.values[ind + 1] = val.slope
        end
    else
        sstate.values[ind] = val[1]
        sstate.values[ind + 1] = val[2]
    end
    return val
end

########
# Implement access to steady state values using dot notation

function Base.getproperty(ssd::SteadyStateData, name::Symbol)
    if name ∈ fieldnames(SteadyStateData)
        return getfield(ssd, name)
    else
        return getindex(ssd, name)
    end
end

function Base.setproperty!(ssd::SteadyStateData, name::Symbol, val)
    if name ∈ fieldnames(SteadyStateData)
        return setfield!(ssd, name, val)
    else
        return setindex!(ssd, val, name)
    end
end

function Base.propertynames(ssd::SteadyStateData, private=false)
    if private
        return ((Symbol(split("$v", "#")[1]) for v in ssd.vars[1:2:end])..., fieldnames(SteadyStateData)...)
    else
        return ((Symbol(split("$v", "#")[1]) for v in ssd.vars[1:2:end])...,)
    end
end

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
function printsstate(io::IO, ssd::SteadyStateData)
    @printf io "  % 30s   %-15s \t%-15s\n" "" "level" "slope"
    printmv(m, v) = m ? @sprintf("%- 15g", v) : @sprintf("%15s", " * ")
    for i = 1:2:length(ssd.values)
        var = split(string(ssd.vars[i]), '#')[1]
        val1 = printmv(ssd.mask[i], ssd.values[i])
        val2 = printmv(ssd.mask[i + 1], ssd.values[i + 1])
        @printf io "  % 30s = %s \t%s\n" var val1 val2
    end
end
printsstate(ssd::SteadyStateData) = printsstate(Base.stdout, ssd)

# printsstate(io::IO, m::Model) = printsstate(io, m.sstate)
# printsstate(m::Model) = printsstate(m.sstate)

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
struct SSEqnData
    "Value of `shift` for the current steady state equation"
    shift::Int
    "Information needed to compute the Jacobian matrix of the transformation between steady state and dynamic unknowns"
    JT::Vector
    "The dynamic equation instance"
    eqn::Equation
end

function sseqn_resid_RJ(s::SSEqnData)
    buffer = Vector{Float64}(undef, length(s.JT))
    function to_dyn_pt(pt::AbstractVector{Float64})
        # This function applies the transformation from steady
        # state equation unknowns to dynamic equation unknowns
        fill!(buffer, 0.0)
        for (i, jt) in enumerate(s.JT)
            if length(jt.ssinds) == 1
                buffer[i] += pt[jt.ssinds[1]]
            elseif jt.type == :lin
                buffer[i] += pt[jt.ssinds[1]] + jt.tlag * pt[jt.ssinds[2]]
            elseif jt.type == :log
                buffer[i] += exp(pt[jt.ssinds[1]] + jt.tlag * pt[jt.ssinds[2]])
            else
                error("Steady or shock variable with slope!?")
            end
        end
        return buffer
    end
    function to_ssgrad(pt::AbstractVector{Float64}, jj::AbstractVector{Float64})
        # This function inverts the transformation. jj is the gradient of the
        # dynamic equation residual with respect to the dynamic equation unknowns.
        # Here we compute the Jacobian of the transformation and use it to compute
        ss = zeros(size(pt))
        for (i, jt) in enumerate(s.JT)
            if length(jt.ssinds) == 1
                ss[jt.ssinds[1]] += jj[i]
            elseif jt.type == :lin
                # transformation is dyn = ss#lvl + tlag * ss#slp
                ss[jt.ssinds[1]] += jj[i]
                ss[jt.ssinds[2]] += jj[i] * jt.tlag
            elseif jt.type == :log
                # transformation is dyn = exp(ss#lvl) * exp(tlag * ss#slp)
                A = exp(pt[jt.ssinds[1]])
                B = exp(jt.tlag * pt[jt.ssinds[2]])
                ss[jt.ssinds[1]] += jj[i] * A * B
                ss[jt.ssinds[2]] += jj[i] * A * B * jt.tlag
            end
        end
        return ss
    end
    function _resid(pt::AbstractVector{Float64})
        return s.eqn.eval_resid(to_dyn_pt(pt))
    end
    function _RJ(pt::AbstractVector{Float64})
        R, jj = s.eqn.eval_RJ(to_dyn_pt(pt))
        return R, to_ssgrad(pt, jj)
    end
    return _resid, _RJ
end

"""
    make_sseqn(model::AbstractModel, eqn::Equation; shift::Int64=0)

Create a steady state equation from the given dynamic equation for the given model.

Internal function, do not call directly.
"""
function make_sseqn(model::AbstractModel, eqn::Equation; shift::Int64=0)
    local mvars::Vector{ModelSymbol} = allvars(model)
    # local vinds = Int64[]
    # local nvars = nvariables(model)
    # local nshks = nshocks(model)
    # local nauxvars = nauxvars(model)
    # ssind converts the dynamic index (t, v) into
    # the corresponding indexes of steady state unknowns.
    # Returned value is a list of length 0, 1, or 2.
    function ssind((ti, vi), )::Array{Int64,1}
        vi_type = mvars[vi].type
        # The level unknown has index 2*vi-1.
        # The slope unknown has index 2*vi. However:
        #  * :steady and :shock variables don't have slopes
        #  * :lin and :log variables the slope is in the equation
        #    only if the slope coefficient is not 0.
        if vi_type ∈ (:steady, :shock) || ti + shift == 0
            return [2vi - 1]
        else
            return [2vi - 1, 2vi]
        end
    end
    local ss = sstate(model)
    # The steady state indexes.
    vinds = unique(vcat(map(ssind, eqn.vinds)...))
    # The corresponding steady state symbols
    vsyms = ss.vars[vinds]
    # In the next loop we build the matrix JT which transforms
    # from the steady state values to the dynamic point values.
    JT = []
    for (i, (ti, vi)) in enumerate(eqn.vinds)
        val = (ssinds = indexin(ssind((ti, vi)), vinds), tlag = ti + shift, type = mvars[vi].type)
        push!(JT, val)
    end
    type = shift == 0 ? :tzero : :tshift
    let sseqndata = SSEqnData(shift, JT, eqn)
        return SteadyStateEquation(type, vinds, vsyms, eqn.expr, sseqn_resid_RJ(sseqndata)...)
    end
end

###########################
# Make steady state equation from user input


"""
    setss!(model::AbstractModel, expr::Expr; type::Symbol, modelmodule::Module)

Add a steady state equation to the model. Equations added by `setss!` are in
addition to the equations generated automatically from the dynamic system.

!!! note
    Internal function, do not call directly. Use [`@steadystate`](@ref) instead.

"""
function setss!(model::AbstractModel, expr::Expr; type::Symbol,
    modelmodule::Module=moduleof(model))

    if expr.head != :(=)
        error("Expected an equation, not $(expr.head)")
    end

    local ss = sstate(model)
    local nvars = nvariables(model)

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
        if val ∈ model.shocks
            # shocks are replaced by their expected value
            return 0.0
        end
        vind = indexin([val], variables(model))[1]
        if vind !== nothing
            # it's a variable: make a symbol and an index for the
            # corresponding steady state unknown
            vsym = makesym(Val(type), val)
            push!(vsyms, vsym)
            push!(vinds, makeind(Val(type), vind))
            return vsym
        end
        vind = indexin([val], auxvars(model))[1]
        if vind !== nothing
            # it's an auxvar: make a symbol and an index for the
            # corresponding steady state unknown
            vsym = makesym(Val(type), val)
            push!(vsyms, vsym)
            push!(vinds, makeind(Val(type), vind + nvars))
            return vsym
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
            push!(param_assigments.args, :( local $(p) = $(mparams).$(p) ))
        end
        residual = Expr(:block, source[1], :($(lhs) - $(rhs)))
        funcs_expr = makefuncs(residual, vsyms, param_assigments; mod=modelmodule)
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
        for (i, ssc) = enumerate(ss.constraints)
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
    return esc(:($(thismodule).setss!($(model), $(Meta.quot(equation)); type=$(QuoteNode(type)))))  # , modelmodule=$(modelmodule))))
end

macro steadystate(model, equation::Expr)
    thismodule = @__MODULE__
    return esc(:($(thismodule).setss!($(model), $(Meta.quot(equation)); type=:level))) # , modelmodule=$(modelmodule))))
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
    shks = Set(shocks(model))
    for var in allvars(model)
        push!(ss.vars, makesym(Val(:level), var), makesym(Val(:slope), var))
        # default initial guess for level and slope
        if var in shks || (var isa ModelSymbol && var.type == :shock)
            push!(ss.values, 0.0, 0.0)
        else
            push!(ss.values, 1.0, 0.0)
        end
        push!(ss.mask, false, false)
    end
    empty!(ss.equations)
    for eqn in alleqns(model)
        push!(ss.equations, make_sseqn(model, eqn; shift=0))
    end
    if ! model.flags.ssZeroSlope
        shift = model.options.shift
        for eqn in alleqns(model)
            push!(ss.equations, make_sseqn(model, eqn; shift=shift))
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

