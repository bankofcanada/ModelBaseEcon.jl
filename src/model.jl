##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

export Model

const defaultoptions = Options(
    shift=10,
    substitutions=false,
    tol=1e-10,
    maxiter=20,
    verbose=false,
    variant=:default,
    warn=Options(no_t=true)
)

"""
    mutable struct ModelFlags ⋯ end

Model flags include
* `ssZeroSlope` - Set to `true` to instruct the solvers that all variables have
  zero slope in steady state and final conditions. In other words the model is
  stationary.
"""
mutable struct ModelFlags
    linear::Bool
    ssZeroSlope::Bool
    ModelFlags() = new(false, false)
end

Base.show(io::IO, ::MIME"text/plain", flags::ModelFlags) = show(io, flags)
function Base.show(io::IO, flags::ModelFlags)
    names, values = [], []
    for f in fieldnames(ModelFlags)
        push!(names, string(f))
        push!(values, getfield(flags, f))
    end
    align = maximum(length, names) + 3
    println(io, "ModelFlags")
    for (n, v) in zip(names, values)
        println(io, lpad(n, align), " = ", v)
    end
end

"""
    mutable struct Model <: AbstractModel ⋯ end

Data structure that represents a macroeconomic model.
"""
mutable struct Model <: AbstractModel
    "Options are various hyper-parameters for tuning the algorithms"
    options::Options
    "Flags contain meta information about the type of model"
    flags::ModelFlags
    sstate::SteadyStateData
    dynss::Bool
    #### Inputs from user
    # transition variables
    variables::Vector{ModelVariable}
    # shock variables
    shocks::Vector{ModelVariable}
    # transition equations
    equations::Vector{Equation}
    # parameters 
    parameters::Parameters
    # auto-exogenize mapping of variables and shocks
    autoexogenize::Dict{Symbol,Symbol}
    #### Things we compute
    maxlag::Int
    maxlead::Int
    # auxiliary variables
    auxvars::Vector{ModelVariable}
    # auxiliary equations
    auxeqns::Vector{Equation}
    # data related to evaluating residuals and Jacobian of the model equations
    evaldata::LittleDict{Symbol,AbstractModelEvaluationData}
    # data slot to be used by the solver (in StateSpaceEcon)
    solverdata::LittleDict{Symbol,Any}
    # 
    # constructor of an empty model
    Model(opts::Options) = new(merge(defaultoptions, opts),
        ModelFlags(), SteadyStateData(), false, [], [], [], Parameters(), Dict(), 0, 0, [], [],
        LittleDict{Symbol,AbstractModelEvaluationData}(), LittleDict{Symbol,Any}())
    Model() = new(deepcopy(defaultoptions),
        ModelFlags(), SteadyStateData(), false, [], [], [], Parameters(), Dict(), 0, 0, [], [],
        LittleDict{Symbol,AbstractModelEvaluationData}(), LittleDict{Symbol,Any}())
end

auxvars(model::Model) = getfield(model, :auxvars)
nauxvars(model::Model) = length(auxvars(model))

# We have to specialize allvars() nallvars() because we have auxvars here
allvars(model::Model) = vcat(variables(model), shocks(model), auxvars(model))
nallvars(model::Model) = length(variables(model)) + length(shocks(model)) + length(auxvars(model))

alleqns(model::Model) = vcat(equations(model), getfield(model, :auxeqns))
nalleqns(model::Model) = length(equations(model)) + length(getfield(model, :auxeqns))

hasevaldata(model::Model, variant::Symbol) = haskey(model.evaldata, variant)
function getevaldata(model::Model, variant::Symbol=model.options.variant, errorwhenmissing::Bool=true)
    ed = get(model.evaldata, variant, missing)
    if errorwhenmissing && ed === missing
        variant === :default && modelerror(ModelNotInitError)
        modelerror(EvalDataNotFound, variant)
    end
    return ed
end
function setevaldata!(model::Model; kwargs...)
    for (key, value) in kwargs
        push!(model.evaldata, key => value)
        model.options.variant = key
    end
    return nothing
end

hassolverdata(model::Model, solver::Symbol) = haskey(model.solverdata, solver)
function getsolverdata(model::Model, solver::Symbol, errorwhenmissing::Bool=true)
    sd = get(model.solverdata, solver, missing)
    if errorwhenmissing && sd === missing
        modelerror(SolverDataNotFound, solver)
    end
    return sd
end
setsolverdata!(model::Model; kwargs...) = push!(model.solverdata, (key => value for (key, value) in kwargs)...)

################################################################
# Specialize Options methods to the Model type

OptionsMod.getoption(model::Model; kwargs...) = getoption(model.options; kwargs...)
OptionsMod.getoption(model::Model, name::Symbol, default) = getoption(model.options, name, default)
OptionsMod.getoption(model::Model, name::AS, default) where {AS<:AbstractString} = getoption(model.options, name, default)

OptionsMod.getoption!(model::Model; kwargs...) = getoption!(model.options; kwargs...)
OptionsMod.getoption!(model::Model, name::Symbol, default) = getoption!(model.options, name, default)
OptionsMod.getoption!(model::Model, name::AS, default) where {AS<:AbstractString} = getoption!(model.options, name, default)

OptionsMod.setoption!(model::Model; kwargs...) = setoption!(model.options; kwargs...)
OptionsMod.setoption!(model::Model, name::Symbol, value) = setoption!(model.options, name, value)
OptionsMod.setoption!(model::Model, name::AS, value) where {AS<:AbstractString} = setoption!(model.options, name, value)
OptionsMod.setoption!(f::Function, model::Model) = (f(model.options); model.options)


################################################################
# Implement access to options and flags and a few other computed properties

function Base.getproperty(model::Model, name::Symbol)
    if name ∈ fieldnames(Model)
        return getfield(model, name)
    end
    if name == :nvars
        return length(getfield(model, :variables))
    elseif name == :nshks
        return length(getfield(model, :shocks))
    elseif name == :nauxs
        return length(getfield(model, :auxvars))
    elseif name == :allvars
        return vcat(getfield(model, :variables), getfield(model, :shocks), getfield(model, :auxvars))
    elseif name == :nvarshks
        return length(getfield(model, :shocks)) + length(getfield(model, :variables))
    elseif name == :varshks
        return vcat(getfield(model, :variables), getfield(model, :shocks))
    elseif name == :exogenous
        return filter(isexog, getfield(model, :variables))
    elseif name == :nexog
        return sum(isexog, getfield(model, :variables))
    elseif name == :alleqns
        return vcat(getfield(model, :equations), getfield(model, :auxeqns))
    elseif haskey(getfield(model, :parameters), name)
        return getproperty(getfield(model, :parameters), name)
    elseif name ∈ getfield(model, :options)
        return getoption(model, name, nothing)
    elseif name ∈ fieldnames(ModelFlags)
        return getfield(getfield(model, :flags), name)
    else
        ind = indexin([name], getfield(model, :variables))[1]
        if ind !== nothing
            return getindex(getfield(model, :variables), ind)
        end
        ind = indexin([name], getfield(model, :shocks))[1]
        if ind !== nothing
            return getindex(getfield(model, :shocks), ind)
        end
        ind = indexin([name], getfield(model, :auxvars))[1]
        if ind !== nothing
            return getindex(getfield(model, :auxvars), ind)
        end
    end
    return getfield(model, name)
end

function Base.propertynames(model::Model, private::Bool=false)
    return (fieldnames(Model)..., :exogenous, :nvars, :nshks, :nauxs, :nexog, :allvars, :varshks, :alleqns,
        keys(getfield(model, :options))..., fieldnames(ModelFlags)...,
        Symbol[getfield(model, :variables)...]...,
        Symbol[getfield(model, :shocks)...]...,
        keys(getfield(model, :parameters))...,)
end

function Base.setproperty!(model::Model, name::Symbol, val::Any)
    if name ∈ fieldnames(Model)
        return setfield!(model, name, val)
    elseif haskey(getfield(model, :parameters), name)
        return setproperty!(getfield(model, :parameters), name, val)
    elseif name ∈ getfield(model, :options)
        return setoption!(model, name, val)
    elseif name ∈ fieldnames(ModelFlags)
        return setfield!(getfield(model, :flags), name, val)
    else
        ind = indexin([name], getfield(model, :variables))[1]
        if ind !== nothing
            if !isa(val, Union{Symbol,ModelVariable})
                error("Cannot assign a $(typeof(val)) as a model variable. Use `m.var = update(m.var, ...)` to update a variable.")
            end
            if getindex(getfield(model, :variables), ind) != val
                error("Cannot replace a variable with a different name. Use `m.var = update(m.var, ...)` to update a variable.")
            end
            return setindex!(getfield(model, :variables), val, ind)
        end
        ind = indexin([name], getfield(model, :shocks))[1]
        if ind !== nothing
            if !isa(val, Union{Symbol,ModelVariable})
                error("Cannot assign a $(typeof(val)) as a model shock. Use `m.shk = update(m.shk, ...)` to update a shock.")
            end
            if getindex(getfield(model, :shocks), ind) != val
                error("Cannot replace a shock with a different name. Use `m.shk = update(m.shk, ...)` to update a shock.")
            end
            return setindex!(getfield(model, :shocks), val, ind)
        end
        ind = indexin([name], getfield(model, :auxvars))[1]
        if ind !== nothing
            if !isa(val, Union{Symbol,ModelVariable})
                error("Cannot assign a $(typeof(val)) as an aux variable. Use `m.aux = update(m.aux, ...)` to update an aux variable.")
            end
            if getindex(getfield(model, :auxvars), ind) != val
                error("Cannot replace an aux variable with a different name. Use `m.aux = update(m.aux, ...)` to update an aux variable.")
            end
            return setindex!(getfield(model, :auxvars), val, ind)
        end
        setfield!(model, name, val)  # will throw an error since Model doesn't have field `$name`
    end
end

################################################################
# Pretty printing the model and summary (TODO)

"""
    fullprint(model)

If a model contains more than 20 variables or more than 20 equations, its
display is truncated. In this case you can call `fullprint` to see the whole
model.
"""
function fullprint end

export fullprint
fullprint(model::Model) = fullprint(Base.stdout, model)
function fullprint(io::IO, model::Model)
    io = IOContext(io, :compact => true, :limit => false)
    nvar = length(model.variables)
    nshk = length(model.shocks)
    nprm = length(model.parameters)
    neqn = length(model.equations)
    nvarshk = nvar + nshk
    function print_things(io, things...; len=0, maxlen=40, last=false)
        s = sprint(print, things...; context=io, sizehint=0)
        print(io, s)
        len += length(s) + 2
        last && (println(io), return 0)
        (len > maxlen) ? (print(io, "\n    "); return 4) : (print(io, ", "); return len)
    end
    let len = 15
        print(io, length(model.variables), " variable(s): ")
        if nvar == 0
            println(io)
        else
            for v in model.variables[1:end-1]
                len = print_things(io, v; len=len)
            end
            print_things(io, model.variables[end]; last=true)
        end
    end
    let len = 15
        print(io, length(model.shocks), " shock(s): ")
        if nshk == 0
            println(io)
        else
            for v in model.shocks[1:end-1]
                len = print_things(io, v; len=len)
            end
            print_things(io, model.shocks[end]; last=true)
        end
    end
    let len = 15
        print(io, length(model.parameters), " parameter(s): ")
        if nprm == 0
            println(io)
        else
            params = collect(model.parameters)
            for (k, v) in params[1:end-1]
                len = print_things(io, k, " = ", v; len=len)
            end
            k, v = params[end]
            len = print_things(io, k, " = ", v; len=len, last=true)
        end
    end
    print(io, length(model.equations), " equations(s)")
    if length(model.auxeqns) > 0
        print(io, " with ", length(model.auxeqns), " auxiliary equations")
    end
    print(io, ": \n")
    function print_aux_eq(bi)
        v = model.auxeqns[bi]
        for (var, ti) in keys(v.tsrefs)
            ai = _index_of_var(var, model.allvars)
            ci = ai - nvarshk
            (1 <= ci < bi) && print_aux_eq(ci)
        end
        println(io, "   |->A$bi:   ", v)
    end
    for (i, v) in enumerate(model.equations)
        println(io, "   E$i:   ", v)
        allvars = model.allvars
        for (var, ti) in keys(v.tsrefs)
            ai = _index_of_var(var, allvars)
            if ai > nvarshk
                print_aux_eq(ai - nvarshk)
            end
        end
    end
end

function Base.show(io::IO, model::Model)
    nvar = length(model.variables)
    nshk = length(model.shocks)
    nprm = length(model.parameters)
    neqn = length(model.equations)
    nvarshk = nvar + nshk
    if nvar == nshk == nprm == neqn == 0
        print(io, "Empty model")
    elseif get(io, :compact, false) || nvar + nshk > 20 || neqn > 20
        # compact print
        print(io, nvar, " variable(s), ")
        print(io, nshk, " shock(s), ")
        print(io, nprm, " parameter(s), ")
        print(io, neqn, " equations(s)")
        if length(model.auxeqns) > 0
            print(io, " with ", length(model.auxeqns), " auxiliary equations")
        end
        print(io, ". \n")
    else
        # full print
        fullprint(io, model)
        println(io, "Maximum lag: ", model.maxlag)
        println(io, "Maximum lead: ", model.maxlead)
    end
    return nothing
end

################################################################
# The macros used in the model definition.

# Note: These macros simply store the information into the corresponding 
# arrays within the model instance. The actual processing is done in @initialize

export @variables, @logvariables, @neglogvariables, @steadyvariables, @exogenous, @shocks
export @parameters, @equations, @autoshocks, @autoexogenize

"""
    @variables model name1 name2 ...
    @variables model begin
        name1
        name2
        ...
    end

Declare the names of variables in the model. 

In the `begin-end` version the variable names can be preceeded by a description
(like a docstring) and flags like `@log`, `@steady`, `@exog`, etc. See
[`ModelVariable`](@ref) for details about this.

"""
macro variables(model, block::Expr)
    vars = filter(a -> !isa(a, LineNumberNode), block.args)
    return esc(:(unique!(append!($(model).variables, $vars)); nothing))
end
macro variables(model, vars::Symbol...)
    return esc(:(unique!(append!($(model).variables, $vars)); nothing))
end

"""
    @logvariables

Same as [`@variables`](@ref), but the variables declared with `@logvariables`
are log-transformed.
"""
macro logvariables(model, block::Expr)
    vars = filter(a -> !isa(a, LineNumberNode), block.args)
    return esc(:(unique!(append!($(model).variables, to_log.($vars))); nothing))
end
macro logvariables(model, vars::Symbol...)
    return esc(:(unique!(append!($(model).variables, to_log.($vars))); nothing))
end

"""
    @neglogvariables

Same as [`@variables`](@ref), but the variables declared with `@neglogvariables`
are negative-log-transformed.
"""
macro neglogvariables(model, block::Expr)
    vars = filter(a -> !isa(a, LineNumberNode), block.args)
    return esc(:(unique!(append!($(model).variables, to_neglog.($vars))); nothing))
end
macro neglogvariables(model, vars::Symbol...)
    return esc(:(unique!(append!($(model).variables, to_neglog.($vars))); nothing))
end

"""
    @steadyvariables

Same as [`@variables`](@ref), but the variables declared with `@steadyvariables`
have zero slope in their steady state and final conditions.

"""
macro steadyvariables(model, block::Expr)
    vars = filter(a -> !isa(a, LineNumberNode), block.args)
    return esc(:(unique!(append!($(model).variables, to_steady.($vars))); nothing))
end
macro steadyvariables(model, vars::Symbol...)
    return esc(:(unique!(append!($(model).variables, to_steady.($vars))); nothing))
end

"""
    @exogenous

Like [`@variables`](@ref), but the names declared with `@exogenous` are 
exogenous.
"""
macro exogenous(model, block::Expr)
    vars = filter(a -> !isa(a, LineNumberNode), block.args)
    return esc(:(unique!(append!($(model).variables, to_exog.($vars))); nothing))
end
macro exogenous(model, vars::Symbol...)
    return esc(:(unique!(append!($(model).variables, to_exog.($vars))); nothing))
end

"""
    @shocks

Like [`@variables`](@ref), but the names declared with `@shocks` are 
shocks.
"""
macro shocks(model, block::Expr)
    shks = filter(a -> !isa(a, LineNumberNode), block.args)
    return esc(:(unique!(append!($(model).shocks, to_shock.($shks))); nothing))
end
macro shocks(model, shks::Symbol...)
    return esc(:(unique!(append!($(model).shocks, to_shock.($shks))); nothing))
end

"""
    @autoshocks model [suffix]

Create a list of shocks that matches the list of variables. Each shock name is
created from a variable name by appending suffix. Default suffix is "_shk", but
it can be specified as the second argument too.
"""
macro autoshocks(model, suf="_shk")
    esc(quote
        $(model).shocks = ModelVariable[
            to_shock(Symbol(v.name, $(QuoteNode(suf)))) for v in $(model).variables if !isexog(v) && !isshock(v)
        ]
        push!($(model).autoexogenize, (
            v.name => Symbol(v.name, $(QuoteNode(suf))) for v in $(model).variables if !isexog(v)
        )...)
        nothing
    end)
end

"""
    @parameters model begin
        name = value
        ...
    end

Declare and define the model parameters. 

The parameters must have values. Provide the information in a series of
assignment statements wrapped inside a begin-end block. Use `@link` and `@alias`
to define dynamic links. See [`Parameters`](@ref).
"""
macro parameters(model, args::Expr...)
    if length(args) == 1 && args[1].head == :block
        args = args[1].args
    end
    ret = Expr(:block, :($(model).parameters.mod[] = $__module__))
    for a in args
        if a isa LineNumberNode
            continue
        end
        if Meta.isexpr(a, :(=), 2)
            key, value = a.args
            key = QuoteNode(key)
            # value = Meta.quot(value)
            push!(ret.args, :(push!($(model).parameters, $(key) => $(value))))
            continue
        end
        throw(ArgumentError("Parameter definitions must be assignments, not\n  $a"))
    end
    return esc(ret)
end

"""
    @autoexogenize model begin
        varname = shkname
        ...
    end

Define a mapping between variables and shocks that can be used to
conveniently  swap exogenous and endogenous variables.
"""
macro autoexogenize(model, args::Expr...)
    if length(args) == 1 && args[1].head == :block
        args = args[1].args
    end
    args = filter(a -> a isa Expr && a.head == :(=), [args...])
    autoexos = Dict{Symbol,Any}([ex.args for ex in args])
    esc(:(merge!($(model).autoexogenize, $(autoexos)); nothing))
end


"""
    @equations model begin
        lhs = rhs
        lhs = rhs
        ...
    end

Define model equations. See [`Equation`](@ref).
"""
macro equations(model, block::Expr)
    if block.head != :block
        modelerror("A list of equations must be within a begin-end block")
    end
    ret = Expr(:block)
    eqn = Expr(:block)
    for expr in block.args
        if isa(expr, LineNumberNode)
            push!(eqn.args, expr)
        else
            push!(eqn.args, expr)
            push!(ret.args, :(push!($model.equations, $(Meta.quot(eqn)))))
            eqn = Expr(:block)
        end
    end
    return esc(ret)
end

################################################################
# The processing of equations during model initialization.

export islog, islin
islog(eq::AbstractEquation) = flag(eq, :log)
islin(eq::AbstractEquation) = flag(eq, :lin)

error_process(msg, expr) = begin
    throw(ArgumentError("$msg\n  During processing of\n  $(expr)"))
end

warn_process(msg, expr) = begin
    @warn "$msg\n  During processing of\n  $(expr)"
end

"""
    process_equation(model::Model, expr; <keyword arguments>)

Process the given expression in the context of the given model and create an
Equation() instance for it.

!!! warning
    This function is for internal use only and should not be called directly.
"""
function process_equation end
# export process_equation
process_equation(model::Model, expr::String; kwargs...) = process_equation(model, Meta.parse(expr); kwargs...)
# process_equation(model::Model, val::Number; kwargs...) = process_equation(model, Expr(:block, val); kwargs...)
# process_equation(model::Model, val::Symbol; kwargs...) = process_equation(model, Expr(:block, val); kwargs...)
function process_equation(model::Model, expr::Expr;
    modelmodule::Module=moduleof(model),
    line=LineNumberNode(0),
    flags=EqnFlags(),
    doc="")

    # a list of all known time series 
    allvars = model.allvars

    # keep track of model parameters used in expression
    prefs = LittleDict{Symbol,Symbol}()
    # keep track of references to known time series in the expression
    tsrefs = LittleDict{Tuple{Symbol,Int64},Symbol}()
    # keep track of references to steady states of known time series in the expression
    ssrefs = LittleDict{Symbol,Symbol}()
    # keep track of the source code location where the equation was defined
    #  (helps with tracking the locations of errors)
    source = []

    add_tsref(var::ModelVariable, tind) = begin
        newsym = islog(var) ? Symbol("#log#", var.name, "#", tind, "#") :
                 isneglog(var) ? Symbol("#logm#", var.name, "#", tind, "#") :
                 Symbol("#", var.name, "#", tind, "#")
        push!(tsrefs, (var, tind) => newsym)
    end

    add_ssref(var::ModelVariable) = begin
        newsym = islog(var) ? Symbol("#log#", var.name, "#ss#") :
                 isneglog(var) ? Symbol("#logm#", var.name, "#ss#") :
                 Symbol("#", var.name, "#ss#")
        push!(ssrefs, var => newsym)
    end

    add_pref(par::Symbol) = begin
        newsym = par # Symbol("#", par, "#par#")
        push!(prefs, par => newsym)
    end

    ###################
    #    process(expr)
    # 
    # Process the expression, performing various tasks.
    #  + keep track of mentions of parameters and variables (including shocks)
    #  + remove line numbers from expression, but keep track so we can insert it into the residual functions
    #  + for each time-referenece of variable, create a dummy symbol that will be used in constructing the residual functions
    # 
    # leave numbers alone
    process(num::Number) = num
    # store line number and discard it from the expression
    process(line::LineNumberNode) = (push!(source, line); nothing)
    # Symbols are left alone.
    # Mentions of parameters are tracked and left in place
    # Mentions of time series throw errors (they must always have a t-reference)
    function process(sym::Symbol)
        # is this symbol a known variable?
        ind = _index_of_var(sym, allvars)
        if ind !== nothing
            if model.warn.no_t
                warn_process("Variable or shock `$(sym)` without `t` reference. Assuming `$(sym)[t]`", expr)
            end
            add_tsref(allvars[ind], 0)
            return Expr(:ref, sym, :t)
        end
        # is this symbol a known parameter
        if haskey(model.parameters, sym)
            add_pref(sym)
            return sym
        end
        # is this symbol a valid name in the model module?
        if isdefined(modelmodule, sym)
            return sym
        end
        # no idea what this is!
        error_process("Undefined `$(sym)`.", expr)
    end
    # Main version of process() - it's recursive
    function process(ex::Expr)
        # is this a docstring? 
        if ex.head == :macrocall && ex.args[1] == doc_macro
            push!(source, ex.args[2])
            doc *= ex.args[3]
            return process(ex.args[4])
        end
        # is this a macro call? if so it could be a variable flag, a meta function, or a regular macro
        if ex.head == :macrocall
            push!(source, ex.args[2])
            macroname = Symbol(lstrip(string(ex.args[1]), '@'))  # strip the leading '@'
            # check if this is a steady state mention
            if macroname ∈ (:sstate,)
                length(ex.args) == 3 || error_process("Invalid use of @(ex.args[1])", expr)
                vind = _index_of_var(ex.args[3], allvars)
                vind === nothing && error_process("Argument of @(ex.args[1]) must be a variable", expr)
                add_ssref(allvars[vind])
                return ex
            end
            # check if we have a corresponding meta function
            metafuncname = Symbol("at_", macroname) # replace @ with at_
            metafunc = isdefined(modelmodule, metafuncname) ? :($modelmodule.$metafuncname) :
                       isdefined(ModelBaseEcon, metafuncname) ? :(ModelBaseEcon.$metafuncname) : nothing
            if metafunc !== nothing
                metaargs = map(filter(!MacroTools.isline, ex.args[3:end])) do arg
                    arg = process(arg)
                    arg isa Expr ? Meta.quot(arg) :
                    arg isa Symbol ? QuoteNode(arg) :
                    arg
                end
                metaout = modelmodule.eval(Expr(:call, metafunc, metaargs...))
                return process(metaout)
            end
            error_process("Undefined meta function $(ex.args[1]).", expr)
        end
        if ex.head == :ref
            # expression is an indexing expression
            name, index = ex.args
            if haskey(model.parameters, name)
                # indexing in a parameter - leave it alone, but keep track
                add_pref(name)
                return Expr(:ref, name, modelmodule.eval(index))
            end
            vind = indexin([name], allvars)[1]  # the index of the variable
            if vind !== nothing
                # indexing in a time series
                tind = modelmodule.eval(:(
                    let t = 0
                        $index
                    end
                ))  # the lag or lead value
                add_tsref(allvars[vind], tind)
                return normal_ref(name, tind)
            end
            error_process("Undefined reference $(ex).", expr)
        end
        if ex.head == :(=)
            # expression is an equation
            # recursively process the two sides of the equation
            lhs, rhs = ex.args
            lhs = process(lhs)
            rhs = process(rhs)
            return Expr(:(=), lhs, rhs)
        end
        # if we're still here, recursively process the arguments
        args = map(process, ex.args)
        # remove `nothing`
        filter!(args) do a
            a !== nothing
        end
        if ex.head == :if
            if length(args) == 3
                return Expr(:call, :ifelse, args...)
            else
                error_process("Unable to process an `if` statement with a single branch. Use function `ifelse` instead.", expr)
            end
        end
        if ex.head == :call
            return Expr(:call, args...)
        end
        if ex.head == :block && length(args) == 1
            return args[1]
        end
        if ex.head == :incomplete
            # for incomplete expression, args[1] contains the error message
            error_process(ex.args[1], expr)
        end
        error_process("Can't process $(ex).", expr)
    end

    ##################
    #    make_residual_expression(expr)
    # 
    # Convert a processed equation into an expression that evaluates the residual.
    # 
    #  + each mention of a time-reference is replaced with its symbol
    make_residual_expression(any) = any
    make_residual_expression(name::Symbol) = haskey(model.parameters, name) ? prefs[name] : name
    make_residual_expression(var::ModelVariable, newsym::Symbol) = need_transform(var) ? :($(inverse_transformation(var))($newsym)) : newsym
    function make_residual_expression(ex::Expr)
        if ex.head == :ref
            varname, tindex = ex.args
            vind = _index_of_var(varname, allvars)
            if vind !== nothing
                # The index expression is either t, or t+n or t-n. We made sure of that in process() above.
                if isa(tindex, Symbol) && tindex == :t
                    tind = 0
                elseif isa(tindex, Expr) && tindex.head == :call && tindex.args[1] == :- && tindex.args[2] == :t
                    tind = -tindex.args[3]
                elseif isa(tindex, Expr) && tindex.head == :call && tindex.args[1] == :+ && tindex.args[2] == :t
                    tind = +tindex.args[3]
                else
                    error_process("Unrecognized t-reference expression $tindex.", expr)
                end
                var = allvars[vind]
                newsym = tsrefs[(var, tind)]
                return make_residual_expression(var, newsym)
            end
        elseif ex.head === :macrocall
            macroname, _, varname = ex.args
            macroname === Symbol("@sstate") || error_process("Unexpected macro call.", expr)
            vind = _index_of_var(varname, allvars)
            vind === nothing && error_process("Not a variable name in steady state reference $(ex)", expr)
            var = allvars[vind]
            newsym = ssrefs[var]
            return make_residual_expression(var, newsym)
        elseif ex.head == :(=)
            lhs, rhs = map(make_residual_expression, ex.args)
            if flags.log
                return Expr(:call, :log, Expr(:call, :/, lhs, rhs))
            else
                return Expr(:call, :-, lhs, rhs)
            end
        end
        return Expr(ex.head, map(make_residual_expression, ex.args)...)
    end

    # call process() to gather information
    new_expr = process(expr)
    MacroTools.isexpr(new_expr, :(=)) || error_process("Expected equation.", expr)
    # if source information missing, set from argument
    filter!(l -> l !== nothing, source)
    push!(source, line)
    # make a residual expressoin for the eval function
    residual = make_residual_expression(new_expr)
    # add the source information to residual expression
    residual = Expr(:block, source[1], residual)
    tssyms = values(tsrefs)
    sssyms = values(ssrefs)
    psyms = values(prefs)
    funcs_expr = makefuncs(residual, tssyms, sssyms, psyms, modelmodule)
    resid, RJ = modelmodule.eval(funcs_expr)
    _update_eqn_params!(resid, model.parameters)
    return Equation(doc, flags, expr, residual, tsrefs, ssrefs, prefs, resid, RJ)
end


# we must export this because we call it in the module where the model is being defined
export add_equation!

"""
    add_equation!(model::Model, expr::Expr; modelmodule::Module)

Process the given expression in the context of the given module, create the
Equation() instance for it, and add it to the model instance.

Usually there's no need to call this function directly. It is called during
[`@initialize`](@ref).
"""
function add_equation!(model::Model, expr::Expr; modelmodule::Module=moduleof(model))
    source = LineNumberNode[]
    auxeqns = Expr[]
    flags = EqnFlags()
    doc = ""

    # keep track if we've processed the "=" yet. (eqn flags are only valid before)
    done_equalsign = Ref(false)

    ##################################
    # We preprocess() the expression looking for substitutions. 
    # If we find one, we create an auxiliary variable and equation.
    # We also keep track of line number, so we can label the aux equation as
    # defined on the same line.
    # We also look for doc string and flags (@log, @lin)
    # 
    # We make sure to make a copy of the expression and not to overwrite it. 
    # 
    preprocess(any) = any
    function preprocess(line::LineNumberNode)
        push!(source, line)
        return line
    end
    function preprocess(ex::Expr)
        if ex.head === :block && ex.args[1] isa LineNumberNode && length(ex.args) == 2
            push!(source, ex.args[1])
            return preprocess(ex.args[2])
        end
        if ex.head === :macrocall
            mname, mline = ex.args[1:2]
            margs = ex.args[3:end]
            push!(source, mline)
            if mname == doc_macro
                doc = margs[1]
                return preprocess(margs[2])
            end
            if !done_equalsign[]
                fname = Symbol(lstrip(string(mname), '@'))
                if hasfield(EqnFlags, fname) && length(margs) == 1
                    setfield!(flags, fname, true)
                    return preprocess(margs[1])
                end
            end
            return Expr(:macrocall, mname, nothing, (preprocess(a) for a in margs)...)
        end
        if ex.head === :(=)
            # expression is an equation
            done_equalsign[] && error_process("Multiple equali signs.", expr)
            done_equalsign[] = true
            # recursively process the two sides of the equation
            lhs, rhs = ex.args
            lhs = preprocess(lhs)
            rhs = preprocess(rhs)
            return Expr(:(=), lhs, rhs)
        end
        # recursively preprocess all arguments
        ret = Expr(ex.head)
        for i in eachindex(ex.args)
            push!(ret.args, preprocess(ex.args[i]))
        end
        if getoption!(model; substitutions=true)
            local arg
            matched = @capture(ret, log(arg_))
            # is it log(arg) 
            if matched && isa(arg, Expr)
                local var1, var2, ind1, ind2
                # is it log(x[t]) ? 
                matched = @capture(arg, var1_[ind1_])
                if matched
                    mv = model.:($var1)
                    if mv isa ModelVariable
                        if islog(mv)
                            # log variable is always positive, no need for substitution
                            @goto skip_substitution
                        elseif isshock(mv) || isexog(mv)
                            if model.verbose
                                @info "Found log($var1), which is a shock or exogenous variable. Make sure $var1 data is positive."
                            end
                            @goto skip_substitution
                        elseif islin(mv) && model.verbose
                            @info "Found log($var1). Consider making $var1 a log variable."
                        end
                    end
                else
                    # is it log(x[t]/x[t-1]) ?
                    matched2 = @capture(arg, op_(var1_[ind1_], var2_[ind2_]))
                    if matched2 && op ∈ (:/, :+, :*) && has_t(ind1) && has_t(ind2) && islog(model.:($var1)) && islog(model.:($var2))
                        @goto skip_substitution
                    end
                end
                aux_expr = process_equation(model, Expr(:(=), arg, 0); modelmodule=modelmodule)
                if isempty(aux_expr.tsrefs)
                    # arg doesn't contain any variables, no need for substitution
                    @goto skip_substitution
                end
                # substitute log(something) with auxN and add equation exp(auxN) = something
                push!(model.auxvars, :new)
                model.auxvars[end] = auxs = Symbol("aux", model.nauxs)
                push!(auxeqns, Expr(:(=), Expr(:call, :exp, Expr(:ref, auxs, :t)), arg))
                return Expr(:ref, auxs, :t)
                @label skip_substitution
                nothing
            end
        end
        return ret
    end

    new_expr = preprocess(expr)
    if isempty(source)
        push!(source, LineNumberNode(0))
    end
    eqn = process_equation(model, new_expr; modelmodule=modelmodule, line=source[1], flags=flags, doc=doc)
    push!(model.equations, eqn)
    model.maxlag = max(model.maxlag, eqn.maxlag)
    model.maxlead = max(model.maxlead, eqn.maxlead)
    model.dynss = model.dynss || !isempty(eqn.ssrefs)
    for i ∈ eachindex(auxeqns)
        eqn = process_equation(model, auxeqns[i]; modelmodule=modelmodule, line=source[1])
        push!(model.auxeqns, eqn)
        model.maxlag = max(model.maxlag, eqn.maxlag)
        model.maxlead = max(model.maxlead, eqn.maxlead)
    end
    empty!(model.evaldata)
    return model
end
@assert precompile(add_equation!, (Model, Expr))


############################
### Initialization routines

export @initialize

"""
    initialize!(model, modelmodule)

In the model file, after all declarations of flags, parameters, variables, and
equations are done, it is necessary to initialize the model instance. Usually it
is easier to call [`@initialize`](@ref), which automatically sets the
`modelmodule` value. When it is necessary to set the `modelmodule` argument to
some other module, then this can be done by calling this function instead of the
macro.
"""
function initialize!(model::Model, modelmodule::Module)
    # Note: we cannot use moduleof here, because the equations are not initialized yet.
    if !isempty(model.evaldata)
        modelerror("Model already initialized.")
    end
    initfuncs(modelmodule)
    samename = Symbol[intersect(model.allvars, keys(model.parameters))...]
    if !isempty(samename)
        modelerror("Found $(length(samename)) names that are both variables and parameters: $(join(samename, ", "))")
    end
    model.parameters.mod[] = modelmodule
    varshks = model.varshks
    model.variables = varshks[.!isshock.(varshks)]
    model.shocks = varshks[isshock.(varshks)]
    empty!(model.auxvars)
    eqns = [e.expr for e in model.equations]
    empty!(model.equations)
    empty!(model.auxeqns)
    model.dynss = false
    for e in eqns
        add_equation!(model, e; modelmodule=modelmodule)
    end
    initssdata!(model)
    update_links!(model.parameters)
    if !model.dynss
        # Note: we cannot set any other evaluation method yet - they require steady
        # state solution and we don't have that yet.
        setevaldata!(model; default=ModelEvaluationData(model))
    else
        # if dynss is true, then we need the steady state even for the standard MED
        nothing
    end
    return nothing
end

"""
    @initialize model

Prepare a model instance for analysis. Call this macro after all parameters,
variable names, shock names and equations have been declared and defined.
"""
macro initialize(model::Symbol)
    # @__MODULE__ is this module (ModelBaseEcon)
    # __module__ is the module where this macro is called (the module where the model exists)
    return quote
        $(@__MODULE__).initialize!($(model), $(__module__))
    end |> esc
end

##########################

eval_RJ(point::AbstractMatrix{Float64}, model::Model, variant::Symbol=model.options.variant) = eval_RJ(point, getevaldata(model, variant))
eval_R!(res::AbstractVector{Float64}, point::AbstractMatrix{Float64}, model::Model, variant::Symbol=model.options.variant) = eval_R!(res, point, getevaldata(model, variant))
@inline issssolved(model::Model) = issssolved(model.sstate)

##########################

# export update_auxvars
"""
    update_auxvars(point, model; tol=model.tol, default=0.0)

Calculate the values of auxiliary variables from the given values of regular
variables and shocks.

Auxiliary variables were introduced as substitutions, e.g. log(expression) was
replaced by aux1 and equation was added exp(aux1) = expression, where expression
contains regular variables and shocks.

This function uses the auxiliary equation to compute the value of the auxiliary
variable for the given values of other variables. Note that the given values of
other variables might be inadmissible, in the sense that expression is negative.
If that happens, the auxiliary variable is set to the given `default` value.

If the `point` array does not contain space for the auxiliary variables, it is
extended appropriately.

If there are no auxiliary variables/equations in the model, return *a copy* of
`point`.

!!! note
    The current implementation is specialized only to log substitutions. TODO:
    implement a general approach that would work for any substitution.
"""
function update_auxvars(data::AbstractArray{Float64,2}, model::Model;
    tol::Float64=model.options.tol, default::Float64=0.0
)
    nauxs = length(model.auxvars)
    if nauxs == 0
        return copy(data)
    end
    (nt, nv) = size(data)
    nvarshk = length(model.variables) + length(model.shocks)
    if nv ∉ (nvarshk, nvarshk + nauxs)
        modelerror("Incorrect number of columns $nv. Expected $nvarshk or $(nvarshk + nauxs).")
    end
    mintimes = 1 + model.maxlag + model.maxlead
    if nt < mintimes
        modelerror("Insufficient time periods $nt. Expected $mintimes or more.")
    end
    allvars = model.allvars
    result = [data[:, 1:nvarshk] zeros(nt, nauxs)]
    for (i, eqn) in enumerate(model.auxeqns)
        for t in (eqn.maxlag+1):(nt-eqn.maxlead)
            idx = [CartesianIndex((t + ti, _index_of_var(var, allvars))) for (var, ti) in keys(eqn.tsrefs)]
            res = eqn.eval_resid(result[idx])
            if res < 1.0
                result[t, nvarshk+i] = log(1.0 - res)
            else
                result[t, nvarshk+i] = default
            end
        end
    end
    return result
end
