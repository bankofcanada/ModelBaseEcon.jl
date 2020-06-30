
export Model

const defaultoptions = Options(
    shift = 10,
    substitutions = true, 
    tol = 1e-10, 
    maxiter = 20,
    verbose = false
)

mutable struct ModelFlags
    ssZeroSlope::Bool
    ModelFlags() = new(false)
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
    elseif name ∈ keys(getfield(model, :parameters))
        return getindex(getfield(model, :parameters), name)
    elseif name ∈ getfield(model, :options)
        return getoption(model, name, nothing)
    elseif name ∈ fieldnames(getfield(model, :flags))
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
    elseif name ∈ keys(getfield(model, :parameters))
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
    elseif !get(io, :compact, true) && nvar < 20 && nshk < 20 && neqn < 20
        # full print
        fullprint(io, model)
        println(io, "Maximum lag: ", model.maxlag)
        println(io, "Maximum lead: ", model.maxlead)
    else  # compact print
        print(io, nvar, " variable(s), ")
        print(io, nshk, " shock(s), ")
        print(io, nprm, " parameter(s), ")
        print(io, neqn, " equations(s) with ", length(model.auxeqns), " auxiliary equations.")
    end
    return nothing
end

################################################################
# The macros used in the model definition.

issymbol(::Symbol) = true
issymbol(::Any) = false
isequation(::Any) = false
isequation(expr::Expr) = expr.head==:(=)

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
    return esc(:( unique!(append!($(model).variables, $vars)), nothing ))
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
    params = Dict{Symbol, Any}(map(mevalparam, args))
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
    autoexos = Dict{Symbol, Any}([ex.args for ex in args])
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
    esc(:( push!($(model).equations, eval(Meta.quot($(eqns)))... ); nothing ))
end
