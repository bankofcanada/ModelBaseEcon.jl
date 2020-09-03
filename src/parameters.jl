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

Write access is the same in both dot and bracket notation. A special parameter
is created when the value is a `Symbol` or and `Expr`. If `Symbol`, and the
given symbol is the name of an existing parameters, the new parameter is an
alias. If `Expr`, the new parameter is a link. Otherwise the new parameter is a
simple parameters.

See also: [`ParamAlias`](@ref), [`ParamLink`](@ref), [`peval`](@ref)
"""
struct Parameters <: AbstractDict{Symbol,Any}
    mod::Ref{Module}
    contents::Dict{Symbol,Any}
end

"""
    Parameters([mod::Module])

When creating an instance of `Parameters`, optionally one can specify the module
in which parameter expressions will be evaluated. This only matters if there are
any link parameters that depend on custom functions or constants. In this case,
the `mod` argument should be the module in which these definitions exist.

"""
Parameters(mod::Module=@__MODULE__) = Parameters(Ref(mod), Dict{Symbol,Any}())

# To deepcopy() Parameters, we make a new Ref to the same module and a deepcopy of contents.
Base.deepcopy_internal(p::Parameters, stackdict::IdDict) = Parameters(Ref(p.mod[]), Base.deepcopy_internal(p.contents, stackdict))

# The following functionality is forwarded to the contents
# iteration
@forward Parameters.contents Base.keys, Base.values, Base.pairs
@forward Parameters.contents Base.iterate, Base.length
# bracket notation read access
@forward Parameters.contents Base.getindex
# dict access
Base.get(pars::Parameters, key, default) = get(pars.contents, key, default)
# Base.get!(pars::Parameters, key, default) = get!(pars.contents, key, default)

@forward Parameters.constants Base.get, Base.get!

# """
#     getdepends(p)

# Return a tuple of names of parameters on which the given parameter, `p`,
# depends. 
# """
# getdepends(p) = tuple()   # fall-back for simple parameters. Depend on nothing

"""
    struct ParamAlias

Represents a parameter that is an alias for another parameter.

See also: [`Parameters`](@ref), [`ParamLink`](@ref), [`peval`](@ref)
"""
struct ParamAlias
    name::Symbol
end
# getdepends(p::ParamAlias) = (p.name,)    # alias depends on its target

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
    if arg isa Expr
        throw(ArgumentError("`@alias` requires a symbol. Use `@link` with expressions."))
    end
    ParamAlias(arg)
end

Base.show(io::IO, p::ParamAlias) = show(io, MIME"text/plain"(), p)
Base.show(io::IO, ::MIME"text/plain", p::ParamAlias) = print(io, "@alias ", p.name)

"""
    struct ParamLink

Parameters that are expressions, possibly depending on other parameters, are
stored in instances of this type.

See also: [`Parameters`](@ref), [`ParamAlias`](@ref), [`peval`](@ref)
"""
struct ParamLink
    link::Expr
end
# ParamLink(deps::Vector, expr::Expr) = ParamLink(tuple(Symbol.(deps)...), expr)
# getdepends(p::ParamLink) = p.depends   

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
The difference between this and a simple parameter with the same expression is
in when the expression gets evaluated. With a simple expression, it gets
evaluated immediately and the parameter value is the value of the expression at
the time the `@parameters` block is evaluated. With a parameter link, the
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
           throw(ArgumentError("`@link` requires an expression."))
end

"""
    build_deps(pars, val)

Internal function.

Scan the expression `val` and build a list of dependencies. Valid names are the
keys of pars. Return value is a Vector{Symbol}.
"""
build_deps(pars::Parameters, e) = build_deps(Set{Symbol}(), keys(pars), e)
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
function Base.setindex!(pars::Parameters, val, key)
    if val isa Union{ParamLink, ParamAlias}
        val = val isa ParamLink ? val.link : val.name
        deps = build_deps(pars, val)
        while length(deps) > 0
            d = pop!(deps)
            ddeps = build_deps(pars, pars[d])
            if isempty(ddeps)
                continue
            elseif key ∈ ddeps
                throw(ArgumentError("Circular dependency of $(key) and $(d) in redefinition of $(key)."))
            else
                push!(deps, ddeps...)
            end
        end
        setindex!(pars.contents, val isa Expr ? ParamLink(val) : ParamAlias(val), key)
    else
        setindex!(pars.contents, pars.mod[].eval(val), key)
    end
    return pars
end

"""
    peval(params, expr)

Evaluate the given expression. If the expression contains any parameter names,
their values are evaluated and substituted as necessary. The evaluations are
carried out in the module of params.

See also: [`Parameters`](@ref), [`ParamAlias`](@ref), [`ParamLink`](@ref)
"""
peval(pars::Parameters, e) = e
peval(pars::Parameters, e::ParamAlias) = peval(pars, pars[e.name])
peval(pars::Parameters, e::ParamLink) = peval(pars, e.link)
peval(pars::Parameters, e::Symbol) = e ∈ keys(pars) ? peval(pars, pars[e]) : e
function peval(pars::Parameters, e::Expr)
    r = Expr(e.head)
    for i in 1:length(e.args)
        push!(r.args, peval(pars, e.args[i]))
    end
    return pars.mod[].eval(r)
end

# dot notation access
Base.propertynames(pars::Parameters) = tuple(keys(pars)...)
Base.getproperty(pars::Parameters, s::Symbol) = s ∈ fieldnames(typeof(pars)) ? getfield(pars, s) : peval(pars, s)
Base.setproperty!(pars::Parameters, s::Symbol, val) = s ∈ fieldnames(typeof(pars)) ? setfield!(pars, s, val) : setindex!(pars, val, s)
