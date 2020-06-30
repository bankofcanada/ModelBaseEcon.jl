
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

