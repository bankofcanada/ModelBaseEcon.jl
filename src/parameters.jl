# 

using Lazy: @forward

export Parameters, ParamAlias, ParamLink, peval

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
@forward Parameters.constants Base.get, Base.get!

"""
    getdepends(p)

Return a tuple of names of parameters on which the given parameter, `p`,
depends. 
"""
getdepends(p) = tuple()   # fall-back for simple parameters. Depend on nothing

"""
    struct ParamAlias

Represents a parameter that is an alias for another parameter.

See also: [`Parameters`](@ref), [`ParamLink`](@ref), [`peval`](@ref)
"""
struct ParamAlias
    name::Symbol
end
getdepends(p::ParamAlias) = (p.name,)    # alias depends on its target

Base.show(io::IO, p::ParamAlias) = show(io, MIME"text/plain"(), p)
Base.show(io::IO, ::MIME"text/plain", p::ParamAlias) = print(io, p.name)

"""
    struct ParamLink

Parameters that are expressions, possibly depending on other parameters, are
stored in instances of this type.

See also: [`Parameters`](@ref), [`ParamAlias`](@ref), [`peval`](@ref)
"""
struct ParamLink{N}
    # link builds and stores its list of dependencies
    depends::NTuple{N,Symbol}
    link::Expr
end
ParamLink(deps::Vector{Symbol}, expr::Expr) = ParamLink(tuple(deps...), expr)
getdepends(p::ParamLink) = p.depends   

Base.show(io::IO, p::ParamLink) = show(io, MIME"text/plain"(), p)
Base.show(io::IO, ::MIME"text/plain", p::ParamLink) = print(io, p.link)

"""
    build_deps!(deps, pkeys, val)

Internal function.

Scan the expression `val` and build a list of dependencies. All valid names are
provided in `pkeys`. The resulting list is updated in place in vector `deps`.
"""
build_deps!(deps, pkeys, e) = deps
build_deps!(deps, pkeys, e::Symbol) = e ∈ pkeys ? unique!(push!(deps, e)) : deps
function build_deps!(deps, pkeys, e::Expr)
    foreach(e.args) do expr
        build_deps!(deps, pkeys, expr)
    end
    deps
end

# bracket notation write access
function Base.setindex!(pars::Parameters, val, key)
    if val isa Expr
        deps = build_deps!(Symbol[], keys(pars), val)
        alldeps = copy(deps)
        while length(alldeps) > 0
            d = pop!(alldeps)
            ddeps = getdepends(pars[d])
            if key ∈ ddeps
                throw(ArgumentError("Circular dependency of $(key) and $(d) in redefinition of $(key)."))
            end
            append!(alldeps, ddeps)
        end
        val = ParamLink(deps, val)
    elseif val isa Symbol && val in keys(pars)
        val = ParamAlias(val)
    end
    setindex!(pars.contents, val, key)
    return pars
end

"""
    peval(params, expr)

Evaluate the given expression. If the expression contains any parameter names,
their values are evaluated and substituted as necessary. The evaluations are
carried out in the module of params.

See also: [`Parameters`](@ref), [`ParamAlias`](@ref), [`ParamLink`](@ref)
"""
peval(p::Parameters, e) = e
peval(p::Parameters, e::ParamAlias) = peval(p, p[e.name])
peval(p::Parameters, e::ParamLink) = peval(p, e.link)
peval(p::Parameters, e::Symbol) = e ∈ keys(p) ? peval(p, p[e]) : e
function peval(p::Parameters, e::Expr)
    r = Expr(e.head)
    for i in 1:length(e.args)
        push!(r.args, peval(p, e.args[i]))
    end
    return p.mod[].eval(r)
end

# dot notation access
Base.propertynames(p::Parameters) = tuple(keys(p)...)
Base.getproperty(p::Parameters, s::Symbol) = s ∈ fieldnames(typeof(p)) ? getfield(p, s) : peval(p, s)
Base.setproperty!(p::Parameters, s::Symbol, val) = s ∈ fieldnames(typeof(p)) ? setfield!(p, s, val) : setindex!(p, val, s)
