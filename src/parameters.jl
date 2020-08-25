# 

using Lazy: @forward

export Parameters, ParamAlias, ParamLink, peval

"""
    struct Parameters

Container for model parameters. It functions as a `Dict` where the keys are the
parameter names. Simple parameter values are stored directly. Parameters that
depend on other parameters are wrapped in the appropriate data structures to
keep track of such dependencies.

See also: [`ParamAlias`](@ref), [`ParamLink`](@ref), [`getdepends`](@ref)
"""
struct Parameters <: AbstractDict{Symbol, Any}
    mod::Ref{Module}
    contents::Dict{Symbol, Any}
end

# Instance constructors is the same as for Dict
Parameters(mod::Module=@__MODULE__) = Parameters(Ref(mod), Dict{Symbol, Any}())

# To deepcopy() Parameters, we make a new Ref to the same module and a deepcopy of contents.
Base.deepcopy_internal(p::Parameters, stackdict::IdDict) = Parameters(Ref(p.mod[]), Base.deepcopy_internal(p.contents, stackdict))

# The following functionality is forwarded to the contents 
@forward Parameters.contents Base.getindex
@forward Parameters.contents Base.keys, Base.values, Base.pairs
@forward Parameters.contents Base.iterate, Base.length

"""
    getdepends(p)

Return a tuple of names of parameters on which the given parameter, `p`,
depends. 
"""
getdepends(p) = tuple()

"""
    struct ParamAlias

Represents a parameter that is an alias for another parameter.
See also: [`Parameters`](@ref), [`ParamLink`](@ref), [`getdepends`](@ref)
"""
struct ParamAlias
    name::Symbol
end
getdepends(p::ParamAlias) = (p.name, )

Base.show(io::IO, p::ParamAlias) = show(io, MIME"text/plain"(), p)
Base.show(io::IO, ::MIME"text/plain", p::ParamAlias) = print(io, "Alias of: ", p.name)

"""
    struct ParamLink

Parameters that depend on other parameters are stored in instances of this type.
It keeps track of the parameter names it depends on. See also [`getdepends`](@ref)
"""
struct ParamLink{N}
    depends::NTuple{N, Symbol}
    link::Expr
end
ParamLink(deps::Vector{Symbol}, expr) = ParamLink(tuple(deps...), expr)
getdepends(p::ParamLink) = p.depends

Base.show(io::IO, p::ParamLink) = show(io, MIME"text/plain"(), p)
Base.show(io::IO, ::MIME"text/plain", p::ParamLink) = begin
    length(p.depends) == 0 && (print(io, "External link: ", p.link); return)
    length(p.depends) == 1 && (print(io, "Link to ", p.depends[1], ": ", p.link); return)
    print(io, "Link to (", join(p.depends, ", "), "): ", p.link)
end

"""
    build_deps!(deps, pkeys, val)

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

# except the special cases
function Base.setindex!(pars::Parameters, val, key)
    if val isa Expr
        deps = build_deps!(Symbol[], keys(pars), val)
        # recursively scan the dependencies of the dependencies
        # we must not allow circular dependencies.
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

Base.propertynames(p::Parameters) = tuple(keys(p)...)
Base.getproperty(p::Parameters, s::Symbol) = s ∈ fieldnames(typeof(p)) ? getfield(p, s) : peval(p, s)
Base.setproperty!(p::Parameters, s::Symbol, val) = s ∈ fieldnames(typeof(p)) ? setfield!(p, s, val) : setindex!(p, val, s)

