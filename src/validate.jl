##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

module Validate

using ..IR

export validate, ValidationError, LinkCycleError, REGISTERED_FUNCTIONS

# Functions allowed inside equations and @link expressions.
# Anything in this set passes through symbol resolution untouched; anything
# not in this set and not a declared name raises an error.
const REGISTERED_FUNCTIONS = Set{Symbol}([
    # Arithmetic
    :+, :-, :*, :/, :^, :%, :÷,
    # Math
    :log, :log2, :log10, :exp, :sqrt, :cbrt, :abs,
    :sin, :cos, :tan, :asin, :acos, :atan,
    :sinh, :cosh, :tanh, :asinh, :acosh, :atanh,
    :floor, :ceil, :round, :sign,
    # Conditionals (CTarget-compatible)
    :ifelse, :min, :max,
    # Smooth helpers
    :heaviside,
    # Comparison (used inside ifelse)
    :(==), :(!=), :<, :>, :<=, :>=,
    # Boolean (used inside ifelse)
    :&, :|, :!,
])

struct ValidationError <: Exception
    msg::String
end
Base.showerror(io::IO, e::ValidationError) = print(io, "ValidationError: ", e.msg)

"""
    LinkCycleError(cycle::Vector{Symbol})

Raised by `link_topology` when the `@link` parameter graph contains a
cycle. `cycle` is the path of parameter names forming the loop, with the
entry parameter repeated at the end - e.g. `[:a, :b, :a]` for
`a = @link b + 1`, `b = @link a - 1`.
"""
struct LinkCycleError <: Exception
    cycle::Vector{Symbol}
end
function Base.showerror(io::IO, e::LinkCycleError)
    print(io, "LinkCycleError: circular @link chain: ",
          join(string.(e.cycle), " → "))
end

# ----------------------------------------------------------------------
# @link cycle detection + topological order
# ----------------------------------------------------------------------

"""
Return `(order::Vector{Int}, deps::Dict{Symbol,Vector{Symbol}})`:

- `order` is a topologically-sorted list of indices into `model.params`,
  with linked params placed after the names they depend on.
- `deps` maps each linked-param name to the names it directly references.

Raises `LinkCycleError` if a cycle is detected.
"""
function link_topology(model::IR.ModelDef)
    deps = Dict{Symbol, Vector{Symbol}}()
    paramset = Set(p.name for p in model.params)
    for p in model.params
        if p.kind === IR.PARAM_LINKED
            refs = Symbol[]
            _collect_symbol_refs!(refs, p.value)
            # Restrict to references that name other parameters.
            deps[p.name] = filter(r -> r in paramset, refs)
        end
    end
    # Standard iterative DFS with white/grey/black coloring for cycle detection.
    WHITE, GREY, BLACK = 0, 1, 2
    color = Dict{Symbol, Int}(p.name => WHITE for p in model.params)
    order_names = Symbol[]
    function visit(name, path)
        c = get(color, name, WHITE)
        c == BLACK && return
        if c == GREY
            # `path` is the DFS stack; the cycle is the suffix from the
            # first occurrence of `name` onward, with `name` repeated to
            # close the loop - e.g. [:a, :b, :a].
            start = findfirst(==(name), path)
            cycle = vcat(path[start:end], name)
            throw(LinkCycleError(cycle))
        end
        color[name] = GREY
        for d in get(deps, name, Symbol[])
            visit(d, vcat(path, name))
        end
        color[name] = BLACK
        push!(order_names, name)
    end
    for p in model.params
        visit(p.name, Symbol[])
    end
    name_to_idx = Dict(p.name => i for (i, p) in pairs(model.params))
    order = [name_to_idx[n] for n in order_names]
    return order, deps
end

function _collect_symbol_refs!(out::Vector{Symbol}, ex)
    if ex isa Symbol
        push!(out, ex)
    elseif ex isa Expr
        # Skip the function head for `f(args...)` calls so we don't
        # treat `log` as a ref to a parameter named `log`.
        if ex.head === :call
            for a in ex.args[2:end]
                _collect_symbol_refs!(out, a)
            end
        else
            for a in ex.args
                _collect_symbol_refs!(out, a)
            end
        end
    end
    return out
end

# ----------------------------------------------------------------------
# Equation symbol resolution
# ----------------------------------------------------------------------

"""
Return the `Set{Symbol}` of symbols that appear in equations as names
(not function-position calls), excluding the symbol `t` (the time index).
Variables and shocks are referenced via `name[t+/-k]`, so we look inside
`:ref` expressions.
"""
function _equation_referenced_names(eq::IR.EquationAST)
    refs = Set{Symbol}()
    function walk(ex)
        if ex isa Symbol
            ex === :t && return
            push!(refs, ex)
        elseif ex isa Expr
            if ex.head === :call
                # Function name in args[1] is checked separately.
                fn = ex.args[1]
                if fn isa Symbol && !(fn in REGISTERED_FUNCTIONS)
                    throw(ValidationError(
                        "unknown function `$fn` in equation at $(eq.src)"))
                end
                for a in ex.args[2:end]
                    walk(a)
                end
            elseif ex.head === :ref
                # x[t], x[t-1], x[t+k]
                head = ex.args[1]
                head isa Symbol || throw(ValidationError(
                    "unsupported indexed expression at $(eq.src): $ex"))
                push!(refs, head)
                for a in ex.args[2:end]
                    walk(a)
                end
            else
                for a in ex.args
                    walk(a)
                end
            end
        end
        # Numbers, LineNumberNodes, etc. - ignore
    end
    walk(eq.residual)
    return refs
end

"""
Validate a `ModelDef`. Performs:

1. `@link` cycle detection (and returns a topological order via the closure).
2. Equation symbol resolution: every name referenced must be a declared
   variable, shock, or parameter.

Returns `(link_order::Vector{Int}, link_deps::Dict)` for downstream use.
Raises `ValidationError` on the first failure.
"""
function validate(model::IR.ModelDef)
    link_order, link_deps = link_topology(model)
    declared = Set{Symbol}()
    for v in model.vars;   push!(declared, v.name); end
    for s in model.shocks; push!(declared, s.name); end
    for p in model.params; push!(declared, p.name); end
    for eq in model.equations
        refs = _equation_referenced_names(eq)
        for r in refs
            r in declared && continue
            r in REGISTERED_FUNCTIONS && continue
            throw(ValidationError(
                "unknown symbol `$r` in equation at $(eq.src) " *
                "(not a variable, shock, parameter, or registered function)"))
        end
    end
    return link_order, link_deps
end

end # module Validate
