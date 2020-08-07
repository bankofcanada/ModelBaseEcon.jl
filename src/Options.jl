"""
    OptionsMod

Sub-module of ModelBaseEcon, although it can be used independently.
Implements the [`Options`](@ref) data structure.

## Contents
  * [`Options`](@ref)
  * [`getoption`](@ref) - read the value of an option
  * [`getoption!`](@ref) - if not present, also create an option
  * [`setoption!`](@ref) - create or update the value of an option 

"""
module OptionsMod

export Options, getoption, getoption!, setoption!

"""
    Options

A collection of key-value pairs representing the options controlling the
behaviour or the definition of a Model object. The key is the option name and is
always a Symbol, or converted to Symbol, while the value can be anything.

The options can be accessed using dot notation. Functions [`getoption`](@ref)
and [`setoption!`](@ref) are also provided. They can be used for programmatic
processing of options as well as when the option name is not a valid Julia
identifier.

See also: [`Options`](@ref), [`getoption`](@ref), [`getoption!`](@ref),
[`setoption!`](@ref)

# Examples
```jldoctest
julia> o = Options(maxiter=20, tol=1e-7)
Options:
    maxiter=20
    tol=1.0e-7

julia> o.maxiter = 25
25

julia> o
Options:
    maxiter=25
    tol=1.0e-7

```
"""
struct Options 
    contents::Dict{Symbol,Any}
    # Options() = new(Dict())
    Options(c::Dict{Symbol,<:Any}) = new(c)
end

############
# Constructors

"""
    Options(key=value, ...)
    Options(:key=>value, ...)

Construct an Options instance with key-value pairs given as keyword arguments or
as a list of pairs. If the latter is used, each key must be a `Symbol`.
"""
Options(; kwargs...) = Options(Dict{Symbol,Any}(kwargs))
# Options(pair::Pair{Symbol, T}) where T = Options(Dict(pair))
Options(pairs::Pair{Symbol,<:Any}...) = Options(Dict(pairs...))
Options(pairs::Pair{<:AbstractString,<:Any}...) = Options(Dict(Symbol(k) => v for (k, v) in pairs))

"""
    Options(::Options)

Construct an Options instance as an exact copy of an existing instance.
"""
Options(opts::Options) = Options(deepcopy(Dict(opts.contents)))
    
############
# merge

"""
    merge(o1::Options, o2::Options, ...)
    
Merge the given Options instances into a new Options instance.
If the same option key exists in more than one instance, keep the value from
the last one.
"""
Base.merge(o1::Options, o2::Options...) = Options(merge(o1.contents, (o.contents for o in o2)...))

"""
    merge!(o1::Options, o2::Options...)

Update the first argument, adding all options from the remaining arguments. If the same
option exists in multiple places, use the last one.
"""
Base.merge!(o1::Options, o2::Options...) = (merge!(o1.contents, (o.contents for o in o2)...); o1)


############
# Access by dot notation

Base.propertynames(opts::Options) = tuple(keys(opts.contents)...)

Base.setproperty!(opts::Options, name::Symbol, val) = opts.contents[name] = val

Base.getproperty(opts::Options, name::Symbol) = 
    name ∈ fieldnames(Options) ? getfield(opts, name) :
    name ∈ keys(opts.contents) ? opts.contents[name]  :
                                 error("option $name not set.");

############
# Pretty printing

function Base.show(io::IO, opts::Options) 
    recur_io = IOContext(io, :SHOWN_SET => opts.contents,
                             :typeinfo => eltype(opts.contents),
                             :compact => get(io, :compact, true))
    print(io, length(opts.contents), " Options:")
    if !isempty(opts)
        for (key, value) in opts.contents
            print(io, "\n    ", key, " = ")
            show(recur_io, value)
        end
    end
    # print(io, "\n")
end
export getoption!, getoption, setoption!

############
# Iteration

Base.iterate(opts::Options) = iterate(opts.contents)
Base.iterate(opts::Options, state) = iterate(opts.contents, state)

############
# getoption, and setoption

"""
    getoption(o::Options; name=default [, name=default, ...])
    getoption(o::Options, name, default)

Retrieve the value of an option or a set of options.  The provided defaults
are used when the option doesn't exit.

The return value is the value of the option requested or, if the option doesn't
exist, the default. In the first version of the function, if there are more than
one options requested, the return value is a tuple.

In the second version, the name could be a symbol or a string, which can be helpful
if the name of the option is not a valid identifier.
"""
function getoption end
function getoption(opts::Options; kwargs...)
    if length(kwargs) == 1 
        return get(opts.contents, first(kwargs)...)
    else
        return tuple((get(opts.contents, kv...) for kv in kwargs)...)
    end
end
getoption(opts::Options, name::Symbol, default) = get(opts.contents, name, default)
getoption(opts::Options, name::S where S <: AbstractString, default) = get(opts.contents, Symbol(name), default)

"""
    getoption!(o::Options; name=default [, name=default, ...])
    getoption!(o::Options, name, default)

Retrieve the value of an option or a set of options. If the name does not match
an existing option, the Options instance is updated by inserting the given name
and default value.

The return value is the value of the option requested (or the default). In the
first version of the function, if there are more than one options requested, the
return value is a tuple.

In the second version, the name could be a symbol or a string, which can be
helpful if the name of the option is not a valid identifier.
"""
function getoption! end
function getoption!(opts::Options; kwargs...)
    if length(kwargs) == 1 
        return get!(opts.contents, first(kwargs)...)
    else
        return tuple((get!(opts.contents, kv...) for kv in kwargs)...)
    end
end
getoption!(opts::Options, name::Symbol, default) = get!(opts.contents, name, default)
getoption!(opts::Options, name::S where S <: AbstractString, default) = get!(opts.contents, Symbol(name), default)

"""
    setoption!(o::Options; name=default [, name=default, ...])
    setoption!(o::Options, name, default)

Retrieve the value of an option or a set of options. If the name does not match
an existing option, the Options instance is updated by inserting the given name
and default value.

The return value is the value of the option requested (or the default). In the
first version of the function, if there are more than one options requested, the
return value is a tuple.

In the second version, the name could be a symbol or a string, which can be
helpful if the name of the option is not a valid identifier.
"""
function setoption! end
setoption!(opts::Options; kwargs...) = (push!(opts.contents, kwargs...); opts)
setoption!(opts::Options, name::S where S <: AbstractString, value) = (push!(opts.contents, Symbol(name) => value); opts)
setoption!(opts::Options, name::Symbol, value) = (push!(opts.contents, name => value); opts)

############

Base.in(name, o::Options) = Symbol(name) ∈ keys(o.contents)

Base.keys(o::Options) = keys(o.contents)
Base.values(o::Options) = values(o.contents)
end # module

using .OptionsMod
export Options, getoption, getoption!, setoption!
