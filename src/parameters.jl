#

using Lazy: @forward

export Parameters, ParamAlias, ParamLink, peval, @alias, @link

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


See also: [`ParamAlias`](@ref), [`ParamLink`](@ref), [`peval`](@ref),
[`@alias`](@ref), [`@link`](@ref)
"""
struct Parameters <: AbstractDict{Symbol,Any}
    mod::Ref{Module}
    contents::Dict{Symbol,Any}
end

"""
    Parameters([mod::Module])

When creating an instance of `Parameters`, optionally one can specify the module
in which parameter expressions will be evaluated. This only matters if there are
any link parameters that depend on custom functions or global
variables/constants. In this case, the `mod` argument should be the module in
which these definitions exist.



"""
Parameters(mod::Module=@__MODULE__) = Parameters(Ref(mod), Dict{Symbol,Any}())

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
Base.get(params::Parameters, key, default) = get(params.contents, key, default)
# Base.get!(params::Parameters, key, default) = get!(params.contents, key, default)

@forward Parameters.constants Base.get, Base.get!

"""
    struct ParamAlias

Represents a parameter that is an alias for another parameter.

See also: [`ParamAlias`](@ref), [`ParamLink`](@ref), [`peval`](@ref),
[`@alias`](@ref), [`@link`](@ref), [`Parameters`](@ref)
"""
struct ParamAlias
    name::Symbol
end

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
    return arg isa Symbol ? ParamAlias(arg) : :( throw(ArgumentError("`@alias` requires a symbol. Use `@link` with expressions.")) )
end

Base.show(io::IO, p::ParamAlias) = show(io, MIME"text/plain"(), p)
Base.show(io::IO, ::MIME"text/plain", p::ParamAlias) = print(io, "@alias ", p.name)

"""
    struct ParamLink

Parameters that are expressions, possibly depending on other parameters, are
stored in instances of this type.

See also: [`ParamAlias`](@ref), [`ParamLink`](@ref), [`peval`](@ref),
[`@alias`](@ref), [`@link`](@ref), [`Parameters`](@ref)
"""
struct ParamLink
    link::Expr
end

Base.show(io::IO, p::ParamLink) = show(io, MIME"text/plain"(), p)
Base.show(io::IO, ::MIME"text/plain", p::ParamLink) = print(io, "@link ", p.link)

Base.:(==)(l::ParamLink, r::ParamLink) = l.link == r.link

"""
    @link expr

Create a parameter link. Use `@link` in the [`@parameters`](@ref) section of your model definition.

If your parameter depends on other parameters, then you use `@link` to declare
that. The expression can be any valid Julia code.
```
@parameters model begin
    a = 5
    b = @link a + 1
end
```

You can declare a parameter link even if does not depend on other parameters.
The difference between this and a simple parameter defined with the same
expression is when the expression gets evaluated. With a simple expression, it
gets evaluated immediately and the parameter value is the value of the
expression at the time the `@parameters` block is evaluated. With `@link`, the
expression is stored as such and gets evaluated every time the parameter is
accessed. This may have a performance penalty, but allows more flexibility. For
example, a parameter may depend on a custom function or a global variable. If
those change, the link parameter value immediately reflects that, while a simple
parameter value has to be re-assigned.

```
myfunc() = 1.0
@parameters model begin
    a = myfunc()
    b = @link myfunc()
    c = @link a + b
end

julia> model.a
1.0

julia> model.b
1.0

julia> model.c
2.0

julia> myfunc() = 5.0;  # redefine myfunc()

julia> model.a    # no change
1.0

julia> model.b    # new value
5.0

julia> model.c    # uses the new value of b
6.0
```

"""
macro link(arg)
    return arg isa Expr ? ParamLink(arg) :
           arg isa Symbol ? ParamAlias(sym) :
           :( throw(ArgumentError("`@link` requires an expression.")) )
end

"""
    build_deps(params, val)

Internal function.

Scan the expression `val` and build a list of dependencies. Valid dependency names are the
keys of params. Return value is a Vector{Symbol}.
"""
build_deps(params::Parameters, e) = build_deps(Set{Symbol}(), keys(params), e)
build_deps(deps, pkeys, e) = deps
build_deps(deps, pkeys, e::ParamAlias) = build_deps(deps, pkeys, e.name)
build_deps(deps, pkeys, e::ParamLink) = build_deps(deps, pkeys, e.link)
build_deps(deps, pkeys, e::Symbol) = e ∈ pkeys ? push!(deps, e) : deps
function build_deps(deps, pkeys, e::Expr)
    foreach(e.args) do expr
        build_deps(deps, pkeys, expr)
    end
    deps
end

# bracket notation write access
function Base.setindex!(params::Parameters, val, key)
    if val isa Union{ParamLink,ParamAlias}
        val = val isa ParamLink ? val.link : val.name
        deps = build_deps(params, val)
        while length(deps) > 0
            d = pop!(deps)
            ddeps = build_deps(params, params[d])
            if isempty(ddeps)
                continue
            elseif key ∈ ddeps
                throw(ArgumentError("Circular dependency of $(key) and $(d) in redefinition of $(key)."))
            else
                push!(deps, ddeps...)
            end
        end
        setindex!(params.contents, val isa Expr ? ParamLink(val) : ParamAlias(val), key)
    else
        setindex!(params.contents, params.mod[].eval(val), key)
    end
    return params
end

"""
    peval(params, expr)

Evaluate the given expression. If the expression contains any parameter names,
their values are evaluated and substituted as necessary. The evaluations are
carried out in the module of params.

See also: [`Parameters`](@ref), [`@alias`](@ref), [`@link`](@ref)
"""
peval(params::Parameters, e) = e
peval(params::Parameters, e::ParamAlias) = peval(params, params[e.name])
peval(params::Parameters, e::ParamLink) = peval(params, e.link)
peval(params::Parameters, e::Symbol) = e ∈ keys(params) ? peval(params, params[e]) : e
function peval(params::Parameters, e::Expr)
    r = Expr(e.head)
    for i in 1:length(e.args)
        push!(r.args, peval(params, e.args[i]))
    end
    return params.mod[].eval(r)
end

# dot notation access
Base.propertynames(params::Parameters) = tuple(keys(params)...)
Base.getproperty(params::Parameters, s::Symbol) = s ∈ fieldnames(typeof(params)) ? getfield(params, s) : peval(params, s)
Base.setproperty!(params::Parameters, s::Symbol, val) = s ∈ fieldnames(typeof(params)) ? setfield!(params, s, val) : setindex!(params, val, s)
