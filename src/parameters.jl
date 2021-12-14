##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020, Bank of Canada
# All rights reserved.
##################################################################################

using Lazy: @forward

# export Parameters, ParamAlias, ParamLink, peval, @alias, @link
export Parameters, ModelParam, peval, @alias, @link

abstract type AbstractParam end


"""
    struct Parameters <: AbstractDict{Symbol, Any}

Container for model parameters. It functions as a `Dict` where the keys are the
parameter names. Simple parameter values are stored directly. Special parameters
depend on other parameters are are wrapped in the appropriate data structures to
keep track of such dependencies. There are two types of special parameters -
aliases and links.

Individual parameters can be accessed in two different ways - dot and bracket
notation.

Read access by dot notation calls [`peval`](@ref) while bracket notation
doesn't. This makes no difference for simple parameters. For special parameters,
access by bracket notation returns its internal structure, while access by dot
notation returns its current value depending on other parameters.

Write access is the same in both dot and bracket notation. The new parameter
value is assigned directly in the case of simple parameter. To create an alias
parameter, use the [`@alias`](@ref) macro. To create a link parameter use the
[`@link`](@ref) macro.


See also: [`ModelParam`](@ref), [`peval`](@ref), [`@alias`](@ref),
[`@link`](@ref), [`update_links!`](@ref).
"""
struct Parameters{P<:AbstractParam} <: AbstractDict{Symbol,P}
    mod::Ref{Module}
    contents::Dict{Symbol,P}
end

"""
    mutable struct ModelParam

Contains a model parameter. For a simple parameter it simply stores its value.
For a link or an alias, it stores the link information and also caches the
current value for speed.
"""
mutable struct ModelParam <: AbstractParam
    depends::Set{Symbol}  # stores the names of parameter that depend on this one
    link::Union{Nothing,Symbol,Expr}
    value
end
ModelParam() = ModelParam(Set{Symbol}(), nothing, nothing)
ModelParam(value) = ModelParam(Set{Symbol}(), nothing, value)
ModelParam(value::Union{Symbol,Expr}) = ModelParam(Set{Symbol}(), value, nothing)

"""
    Parameters([mod::Module])

When creating an instance of `Parameters`, optionally one can specify the module
in which parameter expressions will be evaluated. This only matters if there are
any link parameters that depend on custom functions or global
variables/constants. In this case, the `mod` argument should be the module in
which these definitions exist.
"""
Parameters(mod::Module=@__MODULE__) = Parameters(Ref(mod), Dict{Symbol,ModelParam}())

"""
    params = @parameters

When called without any arguments, return an empty [`Parameters`](@ref)
container, with its evaluation module set to the module in which the macro is
being called.
"""
macro parameters()
    return :( Parameters($__module__) )
end

# To deepcopy() Parameters, we make a new Ref to the same module and a deepcopy of contents.
Base.deepcopy_internal(p::Parameters, stackdict::IdDict) = Parameters(Ref(p.mod[]), Base.deepcopy_internal(p.contents, stackdict))

# The following functionality is forwarded to the contents
# iteration
@forward Parameters.contents Base.keys, Base.values, Base.pairs
@forward Parameters.contents Base.iterate, Base.length
# bracket notation read access
@forward Parameters.contents Base.getindex
# dict access
@forward Parameters.contents Base.get, Base.get!

"""
    @alias name

    Create a parameter alias. Use `@alias` in the [`@parameters`](@ref) section of your
model definition.
```
@parameters model begin
    a = 5
    b = @alias a
end
```
"""
macro alias(arg)
    return arg isa Symbol ? ModelParam(arg) : :( throw(ArgumentError("`@alias` requires a symbol. Use `@link` with expressions.")) )
end

"""
    @link expr

Create a parameter link. Use `@link` in the [`@parameters`](@ref) section of
your model definition.

If your parameter depends on other parameters, then you use `@link` to declare
that. The expression can be any valid Julia code.
```
@parameters model begin
    a = 5
    b = @link a + 1
end
```

When a parameter the link depends on is assigned a new value, the link that
depends on it gets updated automatically.

!!! note "Important note"
    There are two cases in which the value of a link does not get updated automatically.
    If the parameter it depends on is mutable, e.g. a `Vector`, it is possible for it to get
    updated in place. The other case is when the link contains global variable or custom function.

    In such case, it is necessary to call [`update_links!`](@ref).
"""
macro link(arg)
    return arg isa Union{Symbol,Expr} ? ModelParam(arg) : :( throw(ArgumentError("`@link` requires an expression.")) )
end

Base.show(io::IO, p::ModelParam) = begin
    p.link === nothing ? print(io, p.value) :
    p.link isa Symbol ? print(io, "@alias ", p.link) :
                        print(io, "@link ", p.link)
end

Base.:(==)(l::ModelParam, r::ModelParam) = l.link === nothing && l.value == r.value || l.link == r.link


###############################
# setindex!

_value(val) = val isa ModelParam ? val.value : val
_link(val) = val isa ModelParam ? val.link : nothing

function _rmlink(params, p, key)
    # we have to remove `key` from the `.depends` of all parameters we depend on
    MacroTools.postwalk(p.link) do e
        if e isa Symbol
            ep = get(params.contents, e, nothing)
            ep !== nothing && delete!(ep.depends, key)
        end
        return e
    end
    return
end

function _check_circular(params, val, key)
    # we have to check for circular dependencies
    deps = Set{Symbol}()
    while true
        MacroTools.postwalk(_link(val)) do e
            if e isa Symbol
                if e == key
                    throw(ArgumentError("Circular dependency of $(key) and $(e) in redefinition of $(key)."))
                end
                if e in keys(params.contents)
                    push!(deps, e)
                end
            end
            return e
        end
        if isempty(deps)
            return
        end
        k = pop!(deps)
        val = params[k]
    end
end

function _addlink(params, val, key)
    if !isa(val, ModelParam) || val.link === nothing
        return
    end
    # we have to add `key` to the `.depends` of all parameters we depend on
    MacroTools.postwalk(val.link) do e
        if e isa Symbol
            ep = get(params.contents, e, nothing)
            ep !== nothing && push!(ep.depends, key)
        end
        return e
    end
    return
end

"""
    peval(params, what)

Evaluate the given expression in the context of the given parameters.

If `what` is a `ModelParam`, its current value is returned. If there's a chance
it might be out of date, call [`update_links!`](@ref).

If `what` is a Symbol or an Expr, all mentions of parameter names are
substituted by their values and the the expression is evaluated.

If `what is any other value, it is returned unchanged.`

See also: [`Parameters`](@ref), [`@alias`](@ref), [`@link`](@ref),
[`ModelParam`](@ref), [`update_links!`](@ref).

"""
peval(params, val) = val
peval(params, par::ModelParam) = par.value
peval(params, sym::Symbol) = sym ∈ keys(params) ? peval(params, params[sym]) : sym
function peval(params, expr::Expr)
    ret = Expr(expr.head)
    ret.args = [peval(params, a) for a in expr.args]
    params.mod[].eval(ret)
end

function _update_values(params, p, key)
    # update my own value
    if p.link !== nothing
        p.value = peval(params, p.link)
    end
    # update values that depend on me
    deps = copy(p.depends)
    while !isempty(deps)
        pk = params.contents[pop!(deps)]
        pk.value = peval(params, pk.link)
        if !isempty(pk.depends)
            push!(deps, pk.depends...)
        end
    end
end

"""
    iterate(params::Parameters)

Iterates the given Parameters collection in the order of dependency.
Specifically, each parameter comes up only after all parameters it depends on
have already been visited. The order within that is alphabetical.
"""
function Base.iterate(params::Parameters, done=Set{Symbol}())
    if length(done) == length(params.contents)
        return nothing
    end
    for k in sort(collect(keys(params.contents)))
        if k in done
            continue
        end
        v = params.contents[k]
        if v.link === nothing
            return k => v, push!(done, k)
        end
        ready = true
        MacroTools.postwalk(v.link) do e
            if e ∈ keys(params) && e ∉ done
                ready = false
            end
            e
        end
        if ready
            return k => v, push!(done, k)
        end
    end
end

export update_links!

"""
    update_links!(model)
    update_links!(params)

Recompute the current values of all parameters.

Typically when a new value of a parameter is assigned, all parameter links and
aliases that depend on it are updated recursively. If a parameter is mutable, e.g.
a Vector or another collection, its value can be updated in place without
re-assigning it, thus the automatic update does not happen. In this case, it is
necessary to call `update_links!`.
"""
update_links!(m::AbstractModel) = update_links!(parameters(m))
function update_links!(params::Parameters)
    for (k, v) in params
        if v.link !== nothing
            v.value = peval(params, v.link)
        end
    end
end

function Base.setindex!(params::Parameters, val, key)
    if key in fieldnames(typeof(params))
        throw(ArgumentError("Invalid parameter name: $key."))
    end
    # If param[key] is not a link, then key doesn't appear in anyone's depends
    # Invariant: my.depends always contains the names of parameters that depend on me.
    _check_circular(params, val, key)
    p = get!(params.contents, key, ModelParam())
    _rmlink(params, p, key)
    _addlink(params, val, key)
    p.link = _link(val)
    p.value = _value(val)
    _update_values(params, p, key)
    params
end

Base.propertynames(params::Parameters) = tuple(keys(params)...)

function Base.setproperty!(params::Parameters, key::Symbol, val)
    if key ∈ fieldnames(typeof(params))
        return setfield!(params, key, val)
    else
        return setindex!(params, val, key)
    end
end

function Base.getproperty(params::Parameters, key::Symbol)
    if key ∈ fieldnames(typeof(params))
        return getfield(params, key)
    end
    par = get(params, key, nothing)
    if par === nothing
        throw(ArgumentError("Unknown parameter $key."))
    else
        return peval(params, par)
    end
end


"""
    assign_parameters!(model, collection; [options])
    assign_parameters!(model; [options], param=value, ...)

Assign values to model parameters. New parameters can be given as key-value pairs
in the function call, or in a collection, such as a `Dict` or a `NamedTuple`.
Individual parameters can be assigned directly to the `model` using
dot-notation. This function should be more convenient when all parameters values
are loaded from a file and available in a dictionary or some other key-value
collection.

There are two options that control the behaviour.
  * `preserve_links=true` - if set to `true` new values for link-parameters are
    ignored and the link is updated automatically from the new values of
    parameters it depends on. If set to `false` any link parameters are
    overwritten and become non-link parameters set to the given new values.
  * `check=true` - if a parameter with the given name does not exist we ignore
    it. When `check` is set to `true` we issue a warning, when set to `false` we
    ignore it silently.

Example
```
julia> @using_example E1
julia> assign_parameters(E1.model; α=0.3, β=0.7)
```

"""
@inline assign_parameters!(mp::Union{AbstractModel, Parameters};  preserve_links=true, check=true, args...) =
            assign_parameters!(mp, args; preserve_links, check)

@inline assign_parameters!(model::AbstractModel, args; kwargs...) =
    (assign_parameters!(model.parameters, args; kwargs...); model)
    
function assign_parameters!(params::Parameters, args; 
            preserve_links=true, 
            check=true)
    for (skey, value) in args
        key = Symbol(skey)
        p = get(params.contents, key, nothing)
        # if not a parameter, do nothing
        if p === nothing 
            check && @warn "Unknown parameter: $key"
            continue
        end
        # if a link and preserve_links is false, do nothing
        preserve_links && p.link !== nothing && continue
        # assign new value. Note that if the given value is a parameter, it might contain new link.
        _rmlink(params, p, key)
        _addlink(params, value, key)
        p.link = _link(value)
        p.value = _value(value)
        _update_values(params, p, key)
    end
    return params
end
export assign_parameters!

@inline export_parameters(model::AbstractModel; kwargs...) = export_parameters!(Dict{Symbol, Any}(), model.parameters; kwargs...)
@inline export_parameters(params::Parameters; kwargs...) = export_parameters!(Dict{Symbol, Any}(), params; kwargs...)
@inline export_parameters!(container, model::AbstractModel; kwargs...) = export_parameters!(container, model.parameters; kwargs...)
function export_parameters!(container, params::Parameters; include_links=true)
    for (key, value) in params
        if include_links || value.link === nothing
            push!(container, key => peval(params, value))
        end
    end
    return container
end
export export_parameters, export_parameters!
