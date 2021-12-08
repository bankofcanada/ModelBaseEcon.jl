##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020, Bank of Canada
# All rights reserved.
##################################################################################

export Model

const defaultoptions = Options(
    shift=10,
    substitutions=false, 
    tol=1e-10, 
    maxiter=20,
    verbose=false,
    warn=Options(no_t=true)
)

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
    Model <: AbstractModel

Data structure that represents a macroeconomic state space model.

"""
mutable struct Model <: AbstractModel
    "Options are various hyper-parameters for tuning the algorithms"
    options::Options
    "Flags contain meta information about the type of model"
    flags::ModelFlags
    sstate::SteadyStateData
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
    maxlag::Int64
    maxlead::Int64
    # auxiliary variables
    auxvars::Vector{ModelVariable}
    # auxiliary equations
    auxeqns::Vector{Equation}
    # ssdata::SteadyStateData
    evaldata::AbstractModelEvaluationData
    # 
    # constructor of an empty model
    Model(opts::Options) = new(merge(defaultoptions, opts), 
        ModelFlags(), SteadyStateData(), [], [], [], Parameters(), Dict(), 0, 0, [], [], NoMED)
    Model() = new(deepcopy(defaultoptions),
        ModelFlags(), SteadyStateData(), [], [], [], Parameters(), Dict(), 0, 0, [], [], NoMED)
end


@inline auxvars(m::Model) = getfield(m, :auxvars)
@inline nauxvars(m::Model) = length(auxvars(m))

# We have to specialize allvars() nallvars() because we have auxvars here
@inline allvars(m::Model) = vcat(variables(m), shocks(m), auxvars(m))
@inline nallvars(m::Model) = length(variables(m)) + length(shocks(m)) + length(auxvars(m))

@inline alleqns(m::Model) = vcat(equations(m), getfield(m, :auxeqns))
@inline nalleqns(m::Model) = length(equations(m)) + length(getfield(m, :auxeqns))

################################################################
# Specialize Options methods to the Model type

OptionsMod.getoption(model::Model; kwargs...) = getoption(model.options; kwargs...)
OptionsMod.getoption(model::Model, name::Symbol, default) = getoption(model.options, name, default)
OptionsMod.getoption(model::Model, name::AS, default) where AS <: AbstractString = getoption(model.options, name, default)

OptionsMod.getoption!(model::Model; kwargs...) = getoption!(model.options; kwargs...)
OptionsMod.getoption!(model::Model, name::Symbol, default) = getoption!(model.options, name, default)
OptionsMod.getoption!(model::Model, name::AS, default) where AS <: AbstractString = getoption!(model.options, name, default)

OptionsMod.setoption!(model::Model; kwargs...) = setoption!(model.options; kwargs...)
OptionsMod.setoption!(model::Model, name::Symbol, value) = setoption!(model.options, name, value)
OptionsMod.setoption!(model::Model, name::AS, value) where AS <: AbstractString = setoption!(model.options, name, value)
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
    elseif name == :varshks
        return vcat(getfield(model, :variables), getfield(model, :shocks))
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
        error("This Model doesn't have property $name")
    end
end

function Base.propertynames(model::Model, private::Bool=false)
    return (fieldnames(Model)..., :nvars, :nshks, :nauxs, :allvars, :varshks, :alleqns, 
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
            if getindex(getfield(model, :variables), ind) != val
                throw(ArgumentError("Cannot replace variable with a different name. Use `m.var = update(m.var, ...)` to update variable."))
            end
            return setindex!(getfield(model, :variables), val, ind)
        end
        ind = indexin([name], getfield(model, :shocks))[1]
        if ind !== nothing
            if getindex(getfield(model, :shocks), ind) != val
                throw(ArgumentError("Cannot replace shock with a different name. Use `m.shk = update(m.shk, ...)` to update shock."))
            end
            return setindex!(getfield(model, :shocks), val, ind)
        end
        ind = indexin([name], getfield(model, :auxvars))[1]
        if ind !== nothing
            if getindex(getfield(model, :auxvars), ind) != val
                throw(ArgumentError("Cannot replace aux variable with a different name. Use `m.aux = update(m.aux, ...)` to update aux variable."))
            end
            return setindex!(getfield(model, :auxvars), val, ind)
        end
        setfield!(model, name, val)  # will throw an error since Model doesn't have field `$name`
    end
end

################################################################
# Pretty printing the model and summary (TODO)

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
            for v in model.variables[1:end - 1]
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
            for v in model.shocks[1:end - 1]
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
            for (k, v) in params[1:end - 1]
                len = print_things(io, k, " = ", v; len=len)
            end
            k, v = params[end]
            len = print_things(io, k, " = ", v; len=len, last=true)
        end
    end
    print(io, length(model.equations), " equations(s) with ", length(model.auxeqns), " auxiliary equations: \n")
    function print_aux_eq(bi)
        v = model.auxeqns[bi]
        for (_, ai) in filter(tv -> tv[2] > nvarshk, v.vinds)
            ci = ai - nvarshk
            ci < bi && print_aux_eq(ci)
        end
        println(io, "   |->A$bi:   ", v)
    end
    for (i, v) in enumerate(model.equations)
        println(io, "   E$i:   ", v)
        for (_, ai) in filter(tv -> tv[2] > nvarshk, v.vinds)
            print_aux_eq(ai - nvarshk)
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
        print(io, neqn, " equations(s) with ", length(model.auxeqns), " auxiliary equations.")
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
    @variables model names...
    @variables model begin
        names...
    end

Define the names of transition variables in the model.

### Example
```jldoctest
@variables model a b c

# If the list is long, use a begin-end block separating names with newline or semicolon
@variables model begin
    a; b
    c
end
````
"""
macro variables(model, block::Expr)
    vars = filter(a -> !isa(a, LineNumberNode), block.args)
    return esc(:(unique!(append!($(model).variables, $vars)); nothing ))
end
macro variables(model, vars::Symbol...)
    return esc(:( unique!(append!($(model).variables, $vars)); nothing ))
end

macro logvariables(model, block::Expr)
    vars = filter(a -> !isa(a, LineNumberNode), block.args)
    return esc(:(unique!(append!($(model).variables, to_log.($vars))); nothing ))
end
macro logvariables(model, vars::Symbol...)
    return esc(:( unique!(append!($(model).variables, to_log.($vars))); nothing ))
end

macro neglogvariables(model, block::Expr)
    vars = filter(a -> !isa(a, LineNumberNode), block.args)
    return esc(:(unique!(append!($(model).variables, to_neglog.($vars))); nothing ))
end
macro neglogvariables(model, vars::Symbol...)
    return esc(:( unique!(append!($(model).variables, to_neglog.($vars))); nothing ))
end

macro steadyvariables(model, block::Expr)
    vars = filter(a -> !isa(a, LineNumberNode), block.args)
    return esc(:(unique!(append!($(model).variables, to_steady.($vars))); nothing ))
end
macro steadyvariables(model, vars::Symbol...)
    return esc(:( unique!(append!($(model).variables, to_steady.($vars))); nothing ))
end

macro exogenous(model, block::Expr)
    vars = filter(a -> !isa(a, LineNumberNode), block.args)
    return esc(:(unique!(append!($(model).variables, to_exog.($vars))); nothing ))
end
macro exogenous(model, vars::Symbol...)
    return esc(:( unique!(append!($(model).variables, to_exog.($vars))); nothing ))
end

"""
    @shocks model names...
    @shocks model begin
        names...
    end

Define the names of transition shocks in the model.

### Example
```jldoctest
@shocks model a_shk b_shk c_shk

# If the list is long, use a begin-end block separating names with newline or semicolon
@shocks model begin
    a_shk; b_shk
    c_shk
end
````
"""
macro shocks(model, block::Expr)
    shks = filter(a -> !isa(a, LineNumberNode), block.args)
    return esc(:( unique!(append!($(model).shocks, to_shock.($shks))); nothing ))
end
macro shocks(model, shks::Symbol...)
    return esc(:( unique!(append!($(model).shocks, to_shock.($shks))); nothing ))
end

"""
    @autoshocks model

Create a list of shocks that matches the list of variables.  Each shock name is
created from a variable name by appending "_shk".
"""
macro autoshocks(model, suf="_shk")
    esc( quote 
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

The parameters must have values. Provide the information in a series of assignment
statements wrapped inside a begin-end block. The names can be used in equations
as if they were regular variables.
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
    esc(:( merge!($(model).autoexogenize, $(autoexos)); nothing ))
end


"""
Usage example:
```
@equations model begin
    y[t] = a * y[t-1] + b * y[t+1] + y_shk[t]
```
"""
macro equations(model, block::Expr)
    if block.head != :block
        error("A list of equations mush be within a begin-end block")
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

const log_eqn_type = Symbol("@", :log)

export islog
@inline islog(eq::AbstractEquation) = eq.type === log_eqn_type

error_process(msg, expr) = begin
    throw(ArgumentError("$msg\n  During processing of\n  $(expr)"))
end

warn_process(msg, expr) = begin
    @warn "$msg\n  During processing of\n  $(expr)"
end

"""
    process_equation(model::Model, expr; <keyword arguments>)

Process the given expression in the context of the given model and create 
an Equation() instance for it.

Internal function. There should be no need to call directly.

"""
function process_equation end
# export process_equation
process_equation(model::Model, expr::String; kwargs...) = process_equation(model, Meta.parse(expr); kwargs...)
# process_equation(model::Model, val::Number; kwargs...) = process_equation(model, Expr(:block, val); kwargs...)
# process_equation(model::Model, val::Symbol; kwargs...) = process_equation(model, Expr(:block, val); kwargs...)
function process_equation(model::Model, expr::Expr; 
    modelmodule::Module=moduleof(model), 
    line=LineNumberNode(0),
    type=default_eqn_type,
    doc="")

    # a list of all known time series 
    allvars = [model.variables; model.shocks; model.auxvars]

    # keep track of model parameters used in expression
    parameters = Set{Symbol}()
    # keep track of references to known time series in the expression
    references = Dict{Tuple{Int64,Int64},Symbol}()
    # keep track of the source code location where the equation was defined
    #  (helps with tracking the locations of errors)
    source = []

    add_reference(sym, tind) = add_reference(sym, tind, indexin([sym], allvars)[1])
    add_reference(sym::Symbol, tind::Int, vind::Int) = begin
        if islog(allvars[vind])
            vsym = Symbol("#log", sym, "#", tind, "#")
        elseif isneglog(allvars[vind])
            vsym = Symbol("#logm", sym, "#", tind, "#")
        else
            vsym = Symbol("#", sym, "#", tind, "#")
        end
        push!(references, (tind, vind) => vsym) # keep track of indexes and dummy symbol
    end

    ###################
    #    process(expr)
    # 
    # Process the expression, performing various tasks.
    #  * keep track of mentions of parameters and variables (including shocks)
    #  * remove line numbers from expression, but keep track so we can insert it into the residual functions
    #  * for each time-referenece of variable, create a dummy symbol that will be used in constructing the residual functions
    # 
    # leave numbers alone
    process(num::Number) = num
    # store line number and discard it from the expression
    function process(line::LineNumberNode)
        push!(source, line)
        return nothing
    end
    # Symbols are left alone.
    # Mentions of parameters are tracked and left in place
    # Mentions of time series throw errors (they must always have a t-reference)
    function process(sym::Symbol)
        if sym ∈ model.variables
            if model.warn.no_t
                warn_process("Variable `$(sym)` without `t` reference. Assuming `$(sym)[t]`", expr)
            end
            add_reference(sym, 0)
            return Expr(:ref, sym, :t)
        elseif sym ∈ model.shocks
            if model.warn.no_t
                warn_process("Shock `$(sym)` without `t` reference. Assuming `$(sym)[t]`", expr)
            end
            add_reference(sym, 0)
            return Expr(:ref, sym, :t)
        elseif sym ∈ model.auxvars
            error_process("Auxiliary `$(sym)` without `t` reference.", expr)
        elseif haskey(model.parameters, sym)
            push!(parameters, sym)
        else
            # is this symbol valid in the model module?
            try
                modelmodule.eval(sym)
            catch
                error_process("Unknown symbol `$(sym)`.", expr)
            end
        end
        return sym
    end
    # Main version of process() - it's recursive
    function process(ex::Expr)
        if ex.head == :macrocall && ex.args[1] == doc_macro
            push!(source, ex.args[2])
            doc *= ex.args[3]
            return process(ex.args[4])
        end
        if ex.head == :macrocall
            push!(source, ex.args[2])
            if length(ex.args) == 3 && MacroTools.isexpr(ex.args[3], :(=))
                type = ex.args[1]
                return process(ex.args[3])
            end
            mfunc = Symbol("at_", string(ex.args[1])[2:end]) # replace leading @ with at_
            if isdefined(modelmodule, mfunc)
                mfunc = :( $modelmodule.$mfunc )
            elseif isdefined(ModelBaseEcon, mfunc)
                mfunc = :( ModelBaseEcon.$mfunc)
            else
                error_process("Unknown meta function $(ex.args[1]).", ex)
            end
            margs = map(filter(x -> !isa(x, LineNumberNode), ex.args[3:end])) do arg
                arg = process(arg)
                arg isa Expr ? Meta.quot(arg) : 
                arg isa Symbol ? QuoteNode(arg) : 
                arg 
            end
            ex = modelmodule.eval(Expr(:call, mfunc, margs...))
        end
        if ex.head == :ref
            # expression is an indexing expression
            name, index = ex.args
            if haskey(model.parameters, name)
                # indexing in a parameter - leave it alone, but keep track
                push!(parameters, name)
                return Expr(:ref, name, modelmodule.eval(index))
            end
            vind = indexin([name], allvars)[1]  # the index of the variable
            if vind !== nothing
                # indexing in a time series
                tind = modelmodule.eval(:(let t = 0; $index end))  # the lag or lead value
                add_reference(name, tind, vind)
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
    #  * each mention of a time-reference is replaced with its symbol
    make_residual_expression(any) = any
    function make_residual_expression(ex::Expr)
        if ex.head == :ref
            name, index = ex.args
            vind = indexin([name], allvars)[1]
            if vind !== nothing
                # The index expression is either t, or t+n or t-n. We made sure of that in process() above.
                if isa(index, Symbol) && index == :t
                    tind = 0
                elseif isa(index, Expr) && index.head == :call && index.args[1] == :- && index.args[2] == :t
                    tind = -index.args[3]
                elseif isa(index, Expr) && index.head == :call && index.args[1] == :+ && index.args[2] == :t
                    tind = +index.args[3]
                else
                    error_process("Unrecognized t-reference expression $index.", expr)
                end
                if need_transform(allvars[vind])
                    func = inverse_transformation(allvars[vind])
                    arg = references[(tind, vind)]
                    return :($func($arg))
                else
                    return references[(tind, vind)]
                end
            end
        elseif ex.head == :(=)
            if type == log_eqn_type
                return Expr(:call, :log, Expr(:call, :/, map(make_residual_expression, ex.args)...))
            else
                return Expr(:call, :-, map(make_residual_expression, ex.args)...)
            end
        end
        return Expr(ex.head, map(make_residual_expression, ex.args)...)
    end

    # call process() to gather information
    new_expr = process(expr)
    MacroTools.isexpr(new_expr, :(=)) || error_process("Expected equation.", expr)
    type ∈ (log_eqn_type, default_eqn_type) || error_process("Unknown equation type $(type).", expr)
    # if source information missing, set from argument
    push!(source, line)
    # collect the indices and dummy symbols of the mentioned variables
    # NOTE: Julia documentation assures us that keys() and values() iterate elements of the Dict in the same order!
    vinds = collect(keys(references))
    vsyms = collect(values(references))
    # make a residual expressoin for the eval function
    residual = make_residual_expression(new_expr)
    # add the source information to residual expression
    residual = Expr(:block, source[1], residual)
    resid, RJ = let mparams = model.parameters
        # create a list of expressions that assign the values of model parameters to 
        # variables of the same name
        param_assigments = Expr(:block)
        for p in parameters
            push!(param_assigments.args, :( local $(p) = $(mparams).$(p) ))
        end
        funcs_expr = makefuncs(residual, vsyms, param_assigments; mod=modelmodule)
        modelmodule.eval(funcs_expr)
    end
    return Equation(doc, type, expr, residual, vinds, vsyms, resid, RJ)
end


# we must export this because we call it in the module where the model is being defined
export add_equation!

"""
    add_equation!(model::Model, expr::Expr; modelmodule::Module)

Process the given expression in the context of the given module, create
the Equation() instance for it and add it to the model instance. 
"""

function add_equation!(model::Model, expr::Expr; modelmodule::Module=moduleof(model))
    source = LineNumberNode[]
    auxeqns = Expr[]
    type = default_eqn_type
    doc = ""

    ##################################
    # We process() the expression looking for substitutions. 
    # If we find one, we create an auxiliary variable and equation.
    # We also keep track of line number, so we can label the aux equation as
    # defined on the same line.
    # 
    # We make sure to make a copy of the expression and not to overwrite it. 
    # 
    process(any) = any
    function process(line::LineNumberNode)
        push!(source, line)
        return line
    end
    function process(expr::Expr)
        if expr.head == :block && expr.args[1] isa LineNumberNode && length(expr.args) == 2
            push!(source, expr.args[1])
            return process(expr.args[2])
        end
        if expr.head == :macrocall
            mname, mline = expr.args[1:2]
            margs = expr.args[3:end]
            push!(source, mline)
            if mname == doc_macro
                doc = margs[1]
                return process(margs[2])
            end
            if mname == log_eqn_type && length(margs) == 1
                type = mname
                return process(margs[1])
            end
            return Expr(:macrocall, mname, nothing, process.(margs)...)
        end
        # recursively process all arguments
        ret = Expr(expr.head)
        for i in eachindex(expr.args)
            push!(ret.args, process(expr.args[i]))
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
                else
                    # is it log(x[t]/x[t-1]) ?
                    matched2 = @capture(arg, op_(var1_[ind1_], var2_[ind2_]))
                    if matched2 && op ∈ (:/, :+, :*) && has_t(ind1) && has_t(ind2) && islog(model.:($var1)) && islog(model.:($var2))
                        @goto skip_substitution
                    end
                end
                aux_expr = process_equation(model, Expr(:(=), arg, 0); modelmodule=modelmodule)
                if length(aux_expr.vinds) == 0
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

    new_expr = process(expr)
    if isempty(source)
        push!(source, LineNumberNode(0))
    end
    eqn = process_equation(model, new_expr; modelmodule=modelmodule, line=source[1], type=type, doc=doc)
    push!(model.equations, eqn)
    model.maxlag = max(model.maxlag, eqn.maxlag)
    model.maxlead = max(model.maxlead, eqn.maxlead)
    for i ∈ eachindex(auxeqns)
        eqn = process_equation(model, auxeqns[i]; modelmodule=modelmodule, line=source[1])
        push!(model.auxeqns, eqn)
        model.maxlag = max(model.maxlag, eqn.maxlag)
        model.maxlead = max(model.maxlead, eqn.maxlead)
    end
    model.evaldata = NoMED
    return model
end
@assert precompile(add_equation!, (Model, Expr))


############################
### Initialization routines

export @initialize

function initialize!(model::Model, modelmodule::Module)
    # Note: we cannot use moduleof here, because the equations are not initialized yet.
    if model.evaldata !== NoMED
        error("Model already initialized.")
    end
    initfuncs(modelmodule)
    model.parameters.mod[] = modelmodule
    varshks = model.varshks
    model.variables = varshks[.!isshock.(varshks)]
    model.shocks = varshks[isshock.(varshks)]
    empty!(model.auxvars)
    eqns = [e.expr for e in model.equations]
    empty!(model.equations)
    empty!(model.auxeqns)
    for e in eqns
        add_equation!(model, e; modelmodule=modelmodule)
    end
    for (i, v) in enumerate(model.allvars)
        model.:($(v.name)) = update(v, index=i)
    end
    model.evaldata = ModelEvaluationData(model)
    initssdata!(model)
    update_links!(model.parameters)
    return nothing
end

"""
    @initialize model

Prepare a model instance for analysis. Call this macro after all
variable names, shock names and equations have been defined.
"""
macro initialize(model::Symbol)
    # thismodule = @__MODULE__
    # modelmodule = __module__
    return quote
        $(@__MODULE__).initialize!($(model), $(__module__))
    end |> esc
end

##########################

eval_RJ(x::AbstractMatrix{Float64}, m::Model) = eval_RJ(x, m.evaldata)
eval_R!(r::AbstractVector{Float64}, x::AbstractMatrix{Float64}, m::Model) = eval_R!(r, x, m.evaldata)
# @inline printsstate(io::IO, m::Model) = printsstate(io, m.sstate)
# @inline printsstate(m::Model) = printsstate(m.sstate)
@inline issssolved(m::Model) = issssolved(m.sstate)

##########################

# export update_auxvars
"""
    update_auxvars(point, model; tol=model.tol, default=0.0)

Calculate the values of auxiliary variables from the given values of regular variables and shocks.

Auxiliary variables were introduced as substitutions, e.g. log(expression) was replaced by aux1 and
equation was added exp(aux1) = expression, where expression contains regular variables and shocks.

This function uses the auxiliary equation to compute the value of the auxiliary variable for the
given values of other variables. Note that the given values of other variables might be inadmissible, 
in the sense that expression is negative. If that happens, the auxiliary variable is set to the given 
`default` value.

If the `point` array does not contain space for the auxiliary variables, it is extended appropriately.

If there are no auxiliary variables/equations in the model, return *a copy* of `point`.

!!! note
    The current implementation is specialized only to log substitutions.
    TODO: implement a general approach that would work for any substitution.
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
        error("Incorrect number of columns $nv. Expected $nvarshk or $(nvarshk + nauxs).")
    end
    mintimes = 1 + model.maxlag + model.maxlead
    if nt < mintimes
        error("Insufficient time periods $nt. Expected $mintimes or more.")
    end
    result = [data[:,1:nvarshk] zeros(nt, nauxs)]
    for (i, eqn) in enumerate(model.auxeqns)
        for t in (eqn.maxlag + 1):(nt - eqn.maxlead)
            idx = [CartesianIndex((t + ti, vi)) for (ti, vi) in eqn.vinds]
            res = eqn.eval_resid(result[idx])
            if res < 1.0
                result[t, nvarshk + i] = log(1.0 - res)
            else
                result[t, nvarshk + i] = default
            end
        end
    end
    return result
end