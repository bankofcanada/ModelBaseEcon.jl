##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

"""
    Compat

The v0.8.0 backward-compatibility layer for the Symbolics-based rewrite.

The rewrite split the legacy mutable `Model` into two types: `ModelDef`
(the DSL-stage builder all macros append to) and `CompiledModel` (the
frozen, solver-facing record produced by `@initialize`). This module
re-surfaces the *legacy* public API on top of that split so existing BoC
code — `m = Model()`, mutate, `@initialize m`, then `m.maxlag` /
`m.sstate` / `parameters(m)` / `islog(m, :v)` — keeps working unchanged.

Three pieces:

1. `Model(...)` builds a `ModelDef` (legacy public name = the builder, A1).
2. `getproperty(::ModelDef, sym)` forwards the legacy property surface to
   the declarations (pre-`@initialize`) or to the cached `CompiledModel`
   (post-`@initialize`), falling through to `getfield` for the real
   `ModelDef` fields the rewrite's own code uses.
3. Accessor (A2) / predicate (A3) functions and the `@initialize` /
   `@reinitialize` / example-loader macros, in their legacy spellings.
"""
module Compat

using ..IR
using ..Compile
using ..Linearize
# The DFM subsystem already owns the generic names `shocks`/`nshocks`/
# `isshock`/`allvars`/`nallvars` (defined on DFM block/model types). We
# extend those same generics with equation-model methods so there is one
# binding per name — no export clash at the top-level module entry.
import ..DFMModels: shocks, nshocks, isshock, allvars, nallvars

# A1 — the public legacy type name `Model` builds a ModelDef.
# `Model()` / `Model(:name)` is the legacy entry point; the rewrite's own
# code constructs `ModelDef` directly.
export Model
"""
    Model([name::Symbol]) -> ModelDef

Construct an empty model. In the Symbolics rewrite the user-facing model
object is a `ModelDef` (the DSL-stage builder); `@initialize` freezes it
into a `CompiledModel`. `Model` is kept as the legacy public spelling of
the builder constructor (A1).
"""
Model(name::Symbol = :model) = IR.ModelDef(name)

# ----------------------------------------------------------------------
# A2 — generic accessors over a ModelDef.
#
# These mirror the legacy free-function accessors. They read the
# declaration vectors directly so they work both before and after
# `@initialize`.
# ----------------------------------------------------------------------

# Note: `shocks`/`nshocks`/`allvars`/`nallvars` are exported at the module
# entry via the DFM subsystem (we extend those generics here, below).
export parameters, variables, equations
export nvariables, nparameters, nequations, alleqns

"Parameter declarations of the model (A2)."
parameters(m::IR.ModelDef) = m.params
"Variable declarations of the model (A2)."
variables(m::IR.ModelDef) = m.vars
"Equation ASTs of the model (A2)."
equations(m::IR.ModelDef) = m.equations
"Shock declarations of the model (A2)."
shocks(m::IR.ModelDef) = m.shocks
"Variables followed by shocks — the full solver column order (A2)."
allvars(m::IR.ModelDef) = vcat(m.vars, m.shocks)

# Legacy `length(model)` / `model[i]` count and index equations. On a
# ModelDef they read the AST list (post-init they could forward to the
# compiled model, but the counts match).
Base.length(m::IR.ModelDef) = length(getfield(m, :equations))
Base.getindex(m::IR.ModelDef, i::Int) =
    getfield(m, :compiled) === nothing ? getfield(m, :equations)[i] :
                                         getfield(m, :compiled)[i]

nvariables(m::IR.ModelDef) = length(m.vars)
nshocks(m::IR.ModelDef) = length(m.shocks)
nparameters(m::IR.ModelDef) = length(m.params)
nequations(m::IR.ModelDef) = length(m.equations)
nallvars(m::IR.ModelDef) = length(m.vars) + length(m.shocks)
"Equation ASTs of the model (legacy `alleqns` spelling, A2)."
alleqns(m::IR.ModelDef) = m.equations

# ----------------------------------------------------------------------
# A3 — predicates over a variable's declared kind.
#
# The rewrite stores kind as a `VarKind` enum (the improvement, preserve
# C2); these predicates are the legacy compat surface. They accept either
# a `VarDecl`/`ShockDecl` directly or a `(model, name)` pair.
# ----------------------------------------------------------------------

# `isshock` is exported via DFM; we extend that generic here.
export islog, islin

_finddecl(m::IR.ModelDef, name::Symbol) = begin
    for v in m.vars;   v.name === name && return v; end
    for s in m.shocks; s.name === name && return s; end
    error("no variable or shock named $name")
end

"True iff the variable is a `@log` variable (A3)."
islog(v::IR.VarDecl) = v.kind === IR.VAR_LOG
islog(::IR.ShockDecl) = false
islog(m::IR.ModelDef, name::Symbol) = islog(_finddecl(m, name))

"True iff the variable is linear/normal (not `@log`); legacy predicate (A3)."
islin(v::IR.VarDecl) = v.kind === IR.VAR_NORMAL
islin(::IR.ShockDecl) = false
islin(m::IR.ModelDef, name::Symbol) = islin(_finddecl(m, name))

"True iff the name is a shock (A3)."
isshock(::IR.VarDecl) = false
isshock(::IR.ShockDecl) = true
isshock(m::IR.ModelDef, name::Symbol) = isshock(_finddecl(m, name))

# ----------------------------------------------------------------------
# Model flags shim.
#
# Legacy models carry a mutable `flags` object with a `linear` field (and
# the model exposes `m.linear` directly). The rewrite treats linearity
# per-equation (`@lin`), so the model-level flag is purely informational —
# but legacy code and the E1–E7 example models set and read it, so we keep
# a real mutable holder. `substitutions` (the removed aux-substitution
# engine) is accepted and ignored.
# ----------------------------------------------------------------------
export ModelFlags
mutable struct ModelFlags
    linear::Bool
    ssZeroSlope::Bool
end
ModelFlags() = ModelFlags(false, false)
Base.show(io::IO, f::ModelFlags) = print(io, "ModelFlags(linear=", f.linear,
                                          ", ssZeroSlope=", f.ssZeroSlope, ")")
Base.show(io::IO, ::MIME"text/plain", f::ModelFlags) = show(io, f)

# ----------------------------------------------------------------------
# A5 — `update_links!` is a documented no-op.
#
# The eager core resolves @link parameters at `@initialize`, so there is
# nothing to refresh at runtime. Kept so legacy callers are unbroken.
# ----------------------------------------------------------------------
export update_links!
"""
    update_links!(m) -> m

No-op in the Symbolics rewrite: `@link` parameters are resolved eagerly at
`@initialize`, so there is no lazy link table to refresh (A5). Returns `m`.
"""
update_links!(m) = m

# ----------------------------------------------------------------------
# getproperty forwarding shim on ModelDef.
#
# Legacy code accesses model properties (`m.variables`, `m.maxlag`,
# `m.sstate`, …) that are NOT raw fields of `ModelDef`. We forward those
# names to the accessors (pre-init) or to the cached `CompiledModel`
# (post-init), and fall through to `getfield` for the real fields the
# rewrite's own code uses (`m.vars`, `m.equations`, `m.compiled`, …).
# ----------------------------------------------------------------------

# Real ModelDef fields — always served by getfield, never intercepted.
const _MODELDEF_FIELDS = fieldnames(IR.ModelDef)

# Lazily materialize the legacy `flags` holder on the ModelDef's `flags`
# field (a real field so it survives `deepcopy`). `nothing` until touched.
function _flags(m::IR.ModelDef)
    f = getfield(m, :flags)
    if f === nothing
        f = ModelFlags()
        setfield!(m, :flags, f)
    end
    return f::ModelFlags
end

# Legacy property names that map to accessor values on the def itself.
function _def_property(m::IR.ModelDef, s::Symbol)
    s === :variables && return variables(m)
    s === :parameters && return parameters(m)
    s === :shocks && return shocks(m)
    s === :allvars && return allvars(m)
    s === :autoexogenize && return getfield(m, :autoexog)
    s === :maxlag && return _maxlag(m)
    s === :maxlead && return _maxlead(m)
    s === :linear && return _flags(m).linear
    return nothing  # signal: not a def-level legacy property
end

function Base.getproperty(m::IR.ModelDef, s::Symbol)
    # `flags` is a real field but must be lazily materialized into a
    # ModelFlags holder (raw field is `nothing` until first touched).
    s === :flags && return _flags(m)
    s in _MODELDEF_FIELDS && return getfield(m, s)
    # `equations` collides with a real field name, so it is served by getfield
    # above; the legacy `m.equations` therefore already returns the AST vector.
    val = _def_property(m, s)
    val === nothing || return val
    # Post-init legacy properties live on the cached CompiledModel.
    compiled = getfield(m, :compiled)
    if compiled !== nothing
        return getproperty(compiled, s)
    end
    error("type ModelDef has no property $s (model not yet @initialize'd?)")
end

function Base.propertynames(m::IR.ModelDef, private::Bool = false)
    return (_MODELDEF_FIELDS...,
            :variables, :parameters, :shocks, :allvars, :autoexogenize,
            :maxlag, :maxlead, :linear)
end

function Base.setproperty!(m::IR.ModelDef, s::Symbol, v)
    s in _MODELDEF_FIELDS && return setfield!(m, s, v)
    if s === :linear
        _flags(m).linear = v
        return v
    elseif s === :substitutions
        # Legacy aux-substitution toggle; the engine was removed (the
        # Symbolics core differentiates subexpressions directly). Accepted
        # and ignored so legacy fixtures (E7) still load.
        return v
    end
    error("type ModelDef has no settable property $s")
end

# maxlag / maxlead are derivable from the declared equations' time refs even
# before compile, but the authoritative values come from the compiled model;
# pre-init we scan the AST for `[t ± k]` offsets.
function _lagleads(m::IR.ModelDef)
    maxlag = 0
    maxlead = 0
    for eq in m.equations
        for off in _offsets(eq.residual)
            off < 0 && (maxlag = max(maxlag, -off))
            off > 0 && (maxlead = max(maxlead, off))
        end
    end
    return maxlag, maxlead
end
_maxlag(m::IR.ModelDef) = _lagleads(m)[1]
_maxlead(m::IR.ModelDef) = _lagleads(m)[2]

# Collect integer time offsets from `name[t]`, `name[t-1]`, `name[t+2]` refs.
function _offsets(ex)
    offs = Int[]
    _collect_offsets!(offs, ex)
    return offs
end
_collect_offsets!(offs, ::Any) = nothing
function _collect_offsets!(offs, ex::Expr)
    if ex.head === :ref && length(ex.args) == 2
        idx = ex.args[2]
        if idx === :t
            push!(offs, 0)
        elseif idx isa Expr && idx.head === :call && length(idx.args) == 3 &&
               idx.args[2] === :t && idx.args[3] isa Integer
            push!(offs, idx.args[1] === :+ ? idx.args[3] : -idx.args[3])
        end
    end
    for a in ex.args
        _collect_offsets!(offs, a)
    end
    return nothing
end

# ----------------------------------------------------------------------
# getproperty forwarding on CompiledModel.
#
# Legacy code reads `m.eqns`/`m.maxlag`/`m.name`/… off the model. The
# CompiledModel struct stores `name`/`eqns`/`param_layout`/`defs`/`ss_eqns`
# as real fields; the remaining legacy names are derived here.
# ----------------------------------------------------------------------

const _COMPILED_FIELDS = fieldnames(Compile.CompiledModel)

function Base.getproperty(m::Compile.CompiledModel, s::Symbol)
    s in _COMPILED_FIELDS && return getfield(m, s)
    defs = getfield(m, :defs)
    s === :variables && return variables(defs)
    s === :parameters && return parameters(defs)
    s === :shocks && return shocks(defs)
    s === :allvars && return allvars(defs)
    s === :equations && return getfield(m, :eqns)
    s === :maxlag && return _maxlag(defs)
    s === :maxlead && return _maxlead(defs)
    s === :flags && return _flags(defs)
    s === :linear && return _flags(defs).linear
    error("type CompiledModel has no property $s")
end

function Base.setproperty!(m::Compile.CompiledModel, s::Symbol, v)
    s in _COMPILED_FIELDS && return setfield!(m, s, v)
    defs = getfield(m, :defs)
    if s === :linear
        _flags(defs).linear = v
        return v
    elseif s === :flags
        setfield!(defs, :flags, v)
        return v
    end
    error("type CompiledModel has no settable property $s")
end

function Base.propertynames(m::Compile.CompiledModel, private::Bool = false)
    return (_COMPILED_FIELDS...,
            :variables, :parameters, :shocks, :allvars, :equations,
            :maxlag, :maxlead, :flags, :linear)
end

# ----------------------------------------------------------------------
# Compat @initialize / @reinitialize — cache the compiled model on the def.
#
# Legacy idiom is `@initialize model` (statement form, model mutated in
# place) AND `m = @initialize m`. We make both work: the macro builds the
# frozen CompiledModel, stores it in `def.compiled`, and returns the def.
# ----------------------------------------------------------------------

export var"@initialize", var"@reinitialize"

"""
    @initialize model

Freeze `model` (a `ModelDef`) into a `CompiledModel`, cache it on
`model.compiled`, and return `model`. Supports both the legacy in-place
statement form (`@initialize model`) and the binding form
(`m = @initialize m`) — the return value is the same `ModelDef` (A1, the
v0.8.0 lifecycle decision).
"""
macro initialize(def)
    return esc(quote
        local _d = $def
        _d.compiled = $(Compile.initialize_model)(_d)
        _d
    end)
end

"""
    @reinitialize model

Rebuild `model.compiled` after `model` was edited, reusing RGFs for
unchanged equations. Returns `model`.
"""
macro reinitialize(def)
    return esc(quote
        local _d = $def
        local _prev = _d.compiled
        if _prev === nothing
            _d.compiled = $(Compile.initialize_model)(_d)
        else
            _d.compiled = first($(Compile.reinitialize_model)(_prev, _d))
        end
        _d
    end)
end

# ----------------------------------------------------------------------
# Example-loader macros (ported from the legacy module entry, cf06e17).
# Used by the surviving E1–E7 testsets to load examples/<name>.jl.
# ----------------------------------------------------------------------

export var"@using_example", var"@include_example"

"""
    @using_example name

Load `examples/<name>.jl` as a module and bring it into scope. Mirrors the
legacy loader; the example models ship in the package `examples/` folder.
"""
macro using_example(name)
    examples_path = abspath(joinpath(dirname(pathof(@__MODULE__)), "..", "examples"))
    return esc(quote
        push!(LOAD_PATH, $(examples_path))
        using $(name)
        pop!(LOAD_PATH)
        $(name)
    end)
end

"""
    @include_example name

`include` the example source `examples/<name>.jl` into the calling module,
defining its module. Idempotent unless `force` is passed.
"""
macro include_example(name, args...)
    force = false
    verbose = true
    for a in args
        if a === :force
            force = true
        elseif a isa Expr && a.head === :(=) && a.args[1] === :force
            force = a.args[2] === true
        elseif a === :quiet
            verbose = false
        end
    end
    example_path = joinpath(dirname(@__DIR__), "examples", string(name, ".jl"))
    return esc(quote
        if isdefined(@__MODULE__, $(QuoteNode(name))) && !$force
            $(verbose ? :(@info $("Example $name already loaded.")) : nothing)
            $(name)
        else
            $(verbose ? :(@info $("Including \"$example_path\"")) : nothing)
            include($example_path)
            $(name)
        end
    end)
end

# ----------------------------------------------------------------------
# moduleof — legacy introspection helper used by a few surviving testsets.
# ----------------------------------------------------------------------
export moduleof
"Module in which the model's generated code lives (legacy introspection)."
moduleof(::IR.ModelDef) = ModelBaseEcon
moduleof(::Compile.CompiledModel) = ModelBaseEcon

end # module Compat
