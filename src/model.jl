
export Model

const defaultoptions = Options(
    shift = 10,
    substitutions = true, 
    tol = 1e-10, 
    maxiter = 20,
    verbose = false
)

mutable struct ModelFlags
    linear::Bool
    ssZeroSlope::Bool
    ModelFlags() = new(false, false)
end

Base.show(io::IO, ::MIME"text/plain", flags::ModelFlags) = begin
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
    variables::Vector{Symbol}
    # shock variables
    shocks::Vector{Symbol}
    # transition equations
    equations::Vector{Equation}
    # parameters 
    parameters::Dict{Symbol,Any}
    # auto-exogenize mapping of variables and shocks
    autoexogenize::Dict{Symbol,Symbol}
    #### Things we compute
    maxlag::Int64
    maxlead::Int64
    # auxiliary variables
    auxvars::Vector{Symbol}
    # auxiliary equations
    auxeqns::Vector{Equation}
    # ssdata::SteadyStateData
    evaldata::AbstractModelEvaluationData
    # 
    # constructor of an empty model
    Model(opts::Options = defaultoptions) = new(merge(defaultoptions, opts), 
        ModelFlags(), SteadyStateData(), [], [], [], Dict(), Dict(), 0, 0, [], [], NoMED)
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
        return getindex(getfield(model, :parameters), name)
    elseif name ∈ getfield(model, :options)
        return getoption(model, name, nothing)
    elseif name ∈ fieldnames(ModelFlags)
        return getfield(getfield(model, :flags), name)
    else
        error("type Model has no property $name")
    end
end

function Base.propertynames(model::Model, private = false)
    return (fieldnames(Model)..., :nvars, :nshks, :nauxs, :allvars, :varshks, :alleqns, 
    keys(getfield(model, :options))..., fieldnames(ModelFlags)...,)
end

function Base.setproperty!(model::Model, name::Symbol, val::Any)
    if name ∈ fieldnames(Model)
        return setfield!(model, name, val)
    elseif haskey(getfield(model, :parameters), name)
        return setindex!(getfield(model, :parameters), val, name)
    elseif name ∈ getfield(model, :options)
        return setoption!(model, name, val)
    elseif name ∈ fieldnames(ModelFlags)
        return setfield!(getfield(model, :flags), name, val)
    else
        error("type Model cannot set property $name")
    end
end

################################################################
# Pretty printing the model and summary (TODO)

export fullprint
fullprint(model::Model) = fullprint(Base.stdout, model)
function fullprint(io::IO, model::Model)
    nvar = length(model.variables)
    nshk = length(model.shocks)
    nprm = length(model.parameters)  
    neqn = length(model.equations)  
    nvarshk = nvar + nshk
    function print_thing(io, thing; len = 0, maxlen = 40, last = false) 
        s = string(thing); print(io, s)
        len += length(s) + 2
        last && (println(io), return 0)
        (len > maxlen) ? (print(io, "\n    "); return 4) : (print(io, ", "); return len)
    end
    let len = 15
        print(io, length(model.variables), " variable(s): ")
        for v in model.variables[1:end - 1]
            len = print_thing(io, v; len = len)
        end
        nvar > 0 && print_thing(io, model.variables[end]; last = true)
    end
    let len = 15
        print(io, length(model.shocks), " shock(s): ")
        for v in model.shocks[1:end - 1]
            len = print_thing(io, v; len = len)
        end
        nshk > 0 && print_thing(io, model.shocks[end]; last = true)
    end
    let len = 15
        print(io, length(model.parameters), " parameter(s): ")
        params = collect(keys(model.parameters))
        for k in params[1:end - 1]
            v = model.parameters[k]
            len = print_thing(io, "$(k) = $(v)"; len = len)
        end
        if nprm > 0 
            k = params[end]
            v = model.parameters[k]
            len = print_thing(io, "$(k) = $(v)"; len = len, last = true)
        end
    end
    print(io, length(model.equations), " equations(s) with ", length(model.auxeqns), " auxiliary equations: \n")
    function print_aux_eq(bi)
        v = model.auxeqns[bi]
        for (_, ai) in filter(tv->tv[2] > nvarshk, v.vinds)
            ci = ai - nvarshk
            ci < bi && print_aux_eq(ci)
        end
        println(io, "   |->A$bi:   ", v)
    end
    for (i, v) in enumerate(model.equations)
        println(io, "   E$i:   ", v)
        for (_, ai) in filter(tv->tv[2] > nvarshk, v.vinds)
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

issymbol(::Symbol) = true
issymbol(::Any) = false
isequation(::Any) = false
isequation(expr::Expr) = expr.head == :(=)

# Note: These macros simply store the information into the corresponding 
# arrays within the model instance. The actual processing is done in @initialize

export @variables, @shocks, @parameters, @equations, @autoshocks, @autoexogenize

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
    vars = filter(issymbol, block.args)
    return esc(:(unique!(append!($(model).variables, $vars)), nothing))
end
macro variables(model, vars::Symbol...)
    return esc(:( unique!(append!($(model).variables, $vars)); nothing ))
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
    shks = filter(issymbol, block.args)
    return esc(:( unique!(append!($(model).shocks, $shks)); nothing ))
end
macro shocks(model, shks::Symbol...)
    return esc(:( unique!(append!($(model).shocks, $shks)); nothing ))
end

"""
    @autoshocks model

Create a list of shocks that matches the list of variables.  Each shock name is
created from a variable name by appending "_shk".
"""
macro autoshocks(model)
    esc(:(
        $(model).shocks = map(x->Meta.parse("$(x)_shk"), $(model).variables);
        nothing
    ))
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
    mevalparam((sym, val)) = (sym, __module__.eval(val))
    mevalparam(ex::Expr) = mevalparam(ex.args)
    if length(args) == 1 && args[1].head == :block
        args = args[1].args
    end
    args = filter(isequation, [args...])
    params = Dict{Symbol,Any}(map(mevalparam, args))
    return esc(:( merge!($(model).parameters, $(params)); nothing ))
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
    args = filter(isequation, [args...])
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
        error("list of equations mush be within a begin-end block")
    end
    eqns = Vector{Expr}()
    eqn = Expr(:block)
    for expr in block.args
        if isa(expr, LineNumberNode)
            push!(eqn.args, expr)
        elseif isa(expr, Expr) && expr.head == :(=)
            push!(eqn.args, expr)
            push!(eqns, eqn)            
            eqn = Expr(:block)
        else
            eqn = Expr(:block)
        end
    end
    esc(:( push!($(model).equations, eval(Meta.quot($(eqns)))...); nothing ))
end

################################################################
# The processing of equations during model initialization.


"""
    process_equation(model::Model, expr; <keyword arguments>)

Process the given expression in the context of the given model and create 
an Equation() instance for it.

!!! note
    Internal function. There should be no need to call directly.

### Implementation (for developers)

"""
function process_equation end
# export process_equation
process_equation(model::Model, expr::String; kwargs...) = process_equation(model, Meta.parse(expr); kwargs...)
# process_equation(model::Model, val::Number; kwargs...) = process_equation(model, Expr(:block, val); kwargs...)
# process_equation(model::Model, val::Symbol; kwargs...) = process_equation(model, Expr(:block, val); kwargs...)
function process_equation(model::Model, expr::Expr; 
    modelmodule::Module = moduleof(model), 
    line = LineNumberNode(0))

    # a list of all known time series 
    allvars = [model.variables; model.shocks; model.auxvars]

    # keep track of model parameters used in expression
    parameters = Set{Symbol}()
    # keep track of references to known time series in the expression
    references = Dict{Tuple{Int64,Int64},Symbol}()
    # keep track of the source code location where the equation was defined
    #  (helps with tracking the locations of errors)
    source = []

    ###################
    #    process(expr)
    # 
    # Process the expression, performing various tasks.
    #  * keep track of mentions of parameters and variables (including shock)
    #  * remove line numbers from expression, but keep track so we can insert it into the residual functions
    #  * for each time-referenece of variable, create a dummy symbol that will be used in constructing the residual functions.
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
            error("Variable `$(sym)` without `t` reference.")
        elseif sym ∈ model.shocks            
            error("Shock `$(sym)` without `t` reference.")
        elseif sym ∈ model.auxvars
            error("Auxiliary `$(sym)` without `t` reference.")
        elseif haskey(model.parameters, sym)
            push!(parameters, sym)
        end
        return sym
    end
    # Main version of process() - it's recursive
    function process(ex::Expr)
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
                vsym = Symbol("#$name#$tind#")      # replace with a dummy symbol
                push!(references, (tind, vind) => vsym) # keep track of indexes and dummy symbol
                if tind > 0  # this isn't really necessary, it's just for pretty printing
                    return Expr(:ref, name, :(t + $tind))
                elseif tind < 0
                    return Expr(:ref, name, :(t - $(-tind)))
                else
                    return Expr(:ref, name, :t)
                end
            end
            error("Undefined reference $(ex).")
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
        if ex.head == :call
            return Expr(:call, args...)
        end
        if ex.head == :block && length(args) == 1
            return args[1]
        end
        if ex.head == :incomplete
            # for incomplete expression, args[1] contains the error message
            error(ex.args[1])
        end
        error("Can't process $(ex).")
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
                    error("Unrecognized t-reference expression $index.")
                end
                return references[(tind, vind)]
            end
        elseif ex.head == :(=)
            return Expr(:call, :-, map(make_residual_expression, ex.args)...)
        end
        return Expr(ex.head, map(make_residual_expression, ex.args)...)
    end

    # call process() to gather information
    expr = process(expr)
    # if source information missing, set from argument
    if isempty(source)
        push!(source, line)
    end
    # collect the indices and dummy symbols of the mentioned variables
    # NOTE: Julia documentation assures us that keys() and values() iterate elements of the Dict in the same order!
    vinds = collect(keys(references))
    vsyms = collect(values(references))
    # make a residual expressoin for the eval function
    residual = make_residual_expression(expr)
    # add the source information to residual expression
    residual = Expr(:block, source[1], residual)
    resid, RJ = let mparams = model.parameters
        # create a list of expressions that assign the values of model parameters to 
        # variables of the same name
        param_assigments = Expr(:block)
        for p in parameters
            # pval = mparams[p]
            # ptype = typeof(pval)
            pa = Expr(:(=), p, Expr(:ref, :( $(mparams) ), QuoteNode(p)))
            # pa = :( $(p) = $(mparams[p]) )
            push!(param_assigments.args, Expr(:local, pa))
        end
        funcs_expr = makefuncs(residual, vsyms, param_assigments; mod = modelmodule)
        modelmodule.eval(funcs_expr)
    end
    return Equation(expr, residual, vinds, vsyms, resid, RJ)
end


# we must export this because we call it in the module where the model is being defined
export add_equation!

"""
    add_equation!(model::Model, expr::Expr; modelmodule::Module)

Process the given expression in the context of the given module, create
the Equation() instance for it and add it to the model instance. 
"""

function add_equation!(model::Model, expr::Expr; modelmodule::Module = moduleof(model))
    source = LineNumberNode[]
    auxeqns = Expr[]

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
        # recursively process all arguments
        args = []
        for i in eachindex(expr.args)
            push!(args, process(expr.args[i]))
        end
        if getoption!(model; substitutions = true)
            if expr.head == :call && args[1] == :log && isa(args[2], Expr)
                # log(something)
                aux_expr = process_equation(model, args[2]; modelmodule = modelmodule)
                if length(aux_expr.vinds) == 0
                    # something doesn't contain any variables, no need for substitution
                    return Expr(:call, :log, args[2])
                end
                # substitute log(something) with auxN and add equation exp(auxN) = something
                naux = 1 + length(model.auxvars)
                auxs = Symbol("aux$(naux)")
                push!(model.auxvars, auxs)
                push!(auxeqns, Expr(:(=), Expr(:call, :exp, Expr(:ref, auxs, :t)), args[2]))
                return Expr(:ref, auxs, :t)
            end
        end
        return Expr(expr.head, args...)
    end

    new_expr = process(expr)
    if isempty(source)
        push!(source, LineNumberNode(0))
    end
    eqn = process_equation(model, new_expr; modelmodule = modelmodule, line = source[1])
    push!(model.equations, eqn)
    model.maxlag = max(model.maxlag, eqn.maxlag)
    model.maxlead = max(model.maxlead, eqn.maxlead)
    for i ∈ eachindex(auxeqns)
        eqn = process_equation(model, auxeqns[i]; modelmodule = modelmodule, line = source[1])
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
    eqns = [e.expr for e in model.equations]
    empty!(model.equations)
    empty!(model.auxvars)
    empty!(model.auxeqns)
    for e in eqns
        add_equation!(model, e; modelmodule = modelmodule)
    end
    model.evaldata = ModelEvaluationData(model)
    initssdata!(model)
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
@inline printsstate(io::IO, m::Model) = printsstate(io, m.sstate)
@inline printsstate(m::Model) = printsstate(m.sstate)
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
                tol::Float64 = model.options.tol, default::Float64 = 0.0
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
    result = [data[:,1:nvarshk] zeros(nt, length(model.auxvars))]
    for (i, eqn) in enumerate(model.auxeqns)
        for t in (model.maxlag + 1):(nt - model.maxlead)
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