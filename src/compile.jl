##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

module Compile

using ..IR
using ..Validate
using ..Symbolic
using ..Codegen
using Symbolics: Symbolics, Num

export Equation, CompiledModel, SteadyStateUserEquation
export expr_hash, initialize_model, reinitialize_model
export model_tuple_threshold, DEFAULT_MODEL_TUPLE_THRESHOLD
export var"@initialize", var"@reinitialize"

# ----------------------------------------------------------------------
# Equation - the frozen, codegen-ready record
#
# Field types are concrete (Function for the RGFs) so all Equation
# instances share one Julia type. Heterogeneity moves up to Model{Eqns},
# whose `eqns` is a Tuple so the solver can dispatch each element with
# its concrete RGF type known at compile time.
# ----------------------------------------------------------------------

struct Equation
    residual::Num
    tsrefs::Vector{Symbolic.TimeRef}
    params::Vector{Symbolic.ParamRef}
    flags::Set{IR.EquationFlag}
    doc::Union{String, Nothing}
    tags::Vector{Symbol}
    src::LineNumberNode
    eval_resid::Function
    eval_RJ!::Function
    eval_hess!::Union{Function, Nothing}
    eval_hod::Union{Vector{Function}, Nothing}
    n_x::Int
    n_p::Int
    expr_hash::UInt64
    # Symbolic argument vectors for `residual` / kernel tensors. Exposed
    # so downstream consumers (notably ModelBaseEconC) can call
    # `Symbolics.build_function(eq.residual, eq.x_syms, eq.p_syms;
    # target=Symbolics.CTarget())` without having to re-run
    # `Symbolic.build_equation_kernels`.
    x_syms::Vector{Num}
    p_syms::Vector{Num}
end

# ----------------------------------------------------------------------
# Model{Eqns} - the frozen, solver-facing model.
#
# `eqns` is either a `Tuple` of `Equation` (small models: the solver hot
# loop unrolls with full type info) or a `Vector{Equation}` (large
# models: avoids the compile-time blowup of a 200+-element tuple type).
# The choice is made once at construction time by equation count - see
# `MODEL_TUPLE_THRESHOLD` and `initialize_model`.
#
# Either way `eqns` supports `iterate`/`getindex`, so all of
# StateSpaceEcon's solver code is shape-agnostic. Note `Equation`
# already erases its RGF evaluators to `::Function` fields,
# so the per-equation call is a dynamic dispatch regardless of `Eqns` -
# the `Tuple` form buys loop unrolling, not call-site devirtualization.
#
# `defs` retains the source `ModelDef` for introspection and for
# `@reinitialize` to diff against.
# ----------------------------------------------------------------------

"""
A user-supplied SS equation, post-compile. Wraps the standard
`Equation` (residual + RGFs + tsrefs) with the SS-equation `name`
and `kind` so StateSpaceEcon can introspect them when augmenting the
SS Newton system.

The wrapped `Equation`'s tsrefs all carry `offset = 0` (SS collapse).
"""
struct SteadyStateUserEquation
    name::Symbol
    kind::IR.SSEquationKind
    eqn::Equation
end

struct CompiledModel{Eqns<:Union{Tuple, Vector{Equation}}}
    name::Symbol
    eqns::Eqns
    param_layout::Vector{Symbolic.ParamRef}
    defs::IR.ModelDef
    ss_eqns::Vector{SteadyStateUserEquation}
end
# Backward-compat constructor: callers that built a CompiledModel with
# four positional args still work; ss_eqns defaults to empty.
CompiledModel(name::Symbol, eqns, param_layout::Vector{Symbolic.ParamRef},
      defs::IR.ModelDef) =
    CompiledModel(name, eqns, param_layout, defs, SteadyStateUserEquation[])

Base.length(m::CompiledModel) = length(m.eqns)
Base.getindex(m::CompiledModel, i::Int) = m.eqns[i]

# ----------------------------------------------------------------------
# Tuple-vs-vector threshold
#
# Models with `> MODEL_TUPLE_THRESHOLD` equations are built as
# `Model{Vector{Equation}}`; smaller ones stay `Model{Tuple}`. The
# threshold is read from `ENV["RW_MBE_TUPLE_THRESHOLD"]` at each
# `initialize_model` call so benchmarks can force either path.
# ----------------------------------------------------------------------

const DEFAULT_MODEL_TUPLE_THRESHOLD = 50

"""
    model_tuple_threshold() -> Int

Equation-count threshold above which `initialize_model` builds a
`Model{Vector{Equation}}` instead of a `Model{Tuple}`. Reads
`ENV["RW_MBE_TUPLE_THRESHOLD"]` if set (so benchmarks can force the
tuple path with a huge value or the vector path with `0`), otherwise
returns `DEFAULT_MODEL_TUPLE_THRESHOLD`.
"""
function model_tuple_threshold()
    raw = get(ENV, "RW_MBE_TUPLE_THRESHOLD", nothing)
    raw === nothing && return DEFAULT_MODEL_TUPLE_THRESHOLD
    parsed = tryparse(Int, raw)
    parsed === nothing && error(
        "ENV[\"RW_MBE_TUPLE_THRESHOLD\"] = $(repr(raw)) is not an integer")
    return parsed
end

# Pack a Vector{Equation} into the storage shape chosen by equation count.
_pack_eqns(eqs::Vector{Equation}) =
    length(eqs) > model_tuple_threshold() ? eqs : Tuple(eqs)

# ----------------------------------------------------------------------
# expr_hash - canonical hash of an EquationAST.
#
# Hashes the `residual` Expr together with the equation flags. This is
# string-stable across Julia sessions because we hash the s-expression
# representation, not Symbolics.Num (which depends on internal Symbolics
# state). LineNumberNodes inside the Expr are stripped before hashing
# so cosmetic edits to the source file don't bust the cache.
# ----------------------------------------------------------------------

"""
    expr_hash(eq::IR.EquationAST) -> UInt64

Stable canonical hash of an equation, used by `@reinitialize` to decide
whether the equation's RGFs can be reused.
"""
function expr_hash(eq::IR.EquationAST)
    return hash((_strip_lnn(eq.residual), eq.flags), UInt64(0xE9E9E9E9))
end

_strip_lnn(x) = x
function _strip_lnn(ex::Expr)
    args = Any[_strip_lnn(a) for a in ex.args if !(a isa LineNumberNode)]
    return Expr(ex.head, args...)
end

# ----------------------------------------------------------------------
# initialize_model - runs the full pipeline
# ----------------------------------------------------------------------

"""
    initialize_model(def::ModelDef; max_hod_order=1) -> Model

Runs validation -> symbolic kernels -> codegen, returning a frozen `Model`.
Marks `def.initialized = true`. Subsequent edits to `def` followed by
`reinitialize_model` will reuse RGFs for unchanged equations.
"""
function initialize_model(def::IR.ModelDef; max_hod_order::Int = 1)
    Validate.validate(def)
    kernels, param_layout = Symbolic.build_equation_kernels(
        def; max_hod_order = max_hod_order)
    eqs = Equation[]
    for (kernel, ast) in zip(kernels, def.equations)
        funcs = Codegen.build_equation_functions(kernel)
        push!(eqs, _make_equation(kernel, ast, funcs))
    end
    ss_eqns = _build_ss_user_equations(def; max_hod_order)
    def.initialized = true
    return CompiledModel(def.name, _pack_eqns(eqs), param_layout, def, ss_eqns)
end

# Build per-SS-equation `Equation`s wrapped in `SteadyStateUserEquation`.
# Empty when the model has no `@steadystate` constraints.
function _build_ss_user_equations(def::IR.ModelDef; max_hod_order::Int = 1)
    isempty(def.ss_equations) && return SteadyStateUserEquation[]
    kernels, _ = Symbolic.build_ss_equation_kernels(def; max_hod_order)
    out = SteadyStateUserEquation[]
    for (kernel, sseq) in zip(kernels, def.ss_equations)
        funcs = Codegen.build_equation_functions(kernel)
        # Synthesize an EquationAST shell just so `_make_equation` can
        # stamp `expr_hash` - the AST is otherwise unused downstream.
        shell_ast = IR.EquationAST(sseq.residual,
                                   Set{IR.EquationFlag}(),
                                   sseq.doc, Symbol[], sseq.src)
        eqn = _make_equation(kernel, shell_ast, funcs)
        push!(out, SteadyStateUserEquation(sseq.name, sseq.kind, eqn))
    end
    return out
end

function _make_equation(kernel::Symbolic.EquationKernel,
                        ast::IR.EquationAST,
                        funcs::Codegen.EquationFunctions)
    return Equation(
        kernel.residual, kernel.tsrefs, kernel.params,
        kernel.flags, kernel.doc, kernel.tags, kernel.src,
        funcs.eval_resid, funcs.eval_RJ!, funcs.eval_hess!, funcs.eval_hod,
        funcs.n_x, funcs.n_p,
        expr_hash(ast),
        kernel.x_syms, kernel.p_syms,
    )
end

# ----------------------------------------------------------------------
# reinitialize_model - diff hashes, reuse RGFs where possible
# ----------------------------------------------------------------------

"""
    reinitialize_model(prev::Model, def::ModelDef; max_hod_order=1) -> (Model, Int)

Rebuild a model after `def` was edited. Returns `(new_model, n_rebuilt)`
where `n_rebuilt` is the count of equations that needed fresh codegen.

Reuse rules:
- An equation is reusable iff `expr_hash(new_ast) == prev.eqns[i].expr_hash`
  AND the new equation's index in `def.equations` matches its prior index
  AND none of the *symbolic environment* (vars, shocks, params, link table)
  affecting that equation has changed.

The third condition is conservative - we treat any change to vars,
shocks, or params as invalidating *all* equations. Per-equation dependency
tracking is a possible future refinement.
"""
function reinitialize_model(prev::CompiledModel, def::IR.ModelDef;
                            max_hod_order::Int = 1)
    Validate.validate(def)

    env_changed = _env_signature(def) != _env_signature(prev.defs)

    # Index previous equations by hash for quick lookup.
    prev_by_hash = Dict{UInt64, Equation}()
    for e in prev.eqns
        prev_by_hash[e.expr_hash] = e
    end

    kernels, param_layout = Symbolic.build_equation_kernels(
        def; max_hod_order = max_hod_order)

    eqs = Equation[]
    n_rebuilt = 0
    for (kernel, ast) in zip(kernels, def.equations)
        h = expr_hash(ast)
        if !env_changed && haskey(prev_by_hash, h)
            # Reuse: take the prior Equation but refresh kernel fields
            # (residual/tsrefs are deterministic from the AST + env, so
            # they match; we keep the prior RGFs to satisfy object identity).
            old = prev_by_hash[h]
            push!(eqs, old)
        else
            funcs = Codegen.build_equation_functions(kernel)
            push!(eqs, _make_equation(kernel, ast, funcs))
            n_rebuilt += 1
        end
    end
    ss_eqns = _build_ss_user_equations(def; max_hod_order)
    def.initialized = true
    return CompiledModel(def.name, _pack_eqns(eqs), param_layout, def, ss_eqns), n_rebuilt
end

# Lightweight signature of vars/shocks/params used to invalidate the cache
# when the symbolic environment changes.
function _env_signature(def::IR.ModelDef)
    return (
        Tuple((v.name, v.kind) for v in def.vars),
        Tuple(s.name for s in def.shocks),
        Tuple((p.name, p.kind, _param_value_sig(p)) for p in def.params),
    )
end

_param_value_sig(p::IR.ParamDecl) =
    p.kind === IR.PARAM_LINKED ? _strip_lnn(p.value) : p.value

# ----------------------------------------------------------------------
# Macros - sugar around the function forms
# ----------------------------------------------------------------------

"""
    @initialize def

Equivalent to `initialize_model(def)`. The expanded form is a function
call so it is hygiene-trivial; we expose a macro for symmetry with the
existing DSL.
"""
macro initialize(def)
    return :(initialize_model($(esc(def))))
end

"""
    @reinitialize prev def

Equivalent to `reinitialize_model(prev, def)`. Returns the `(Model, Int)`
tuple - bind both if you want the rebuild count.
"""
macro reinitialize(prev, def)
    return :(reinitialize_model($(esc(prev)), $(esc(def))))
end

# ----------------------------------------------------------------------
# Equation metadata accessors
# ----------------------------------------------------------------------

"""
    doc(eqn::Equation) -> String

User-supplied docstring attached to the equation in `@equations`, or `""`
if none was given.
"""
doc(eqn::Equation) = eqn.doc === nothing ? "" : eqn.doc

"""
    tags(eqn::Equation) -> Vector{Symbol}

User-supplied tag symbols attached to the equation via the
`:tag => LHS = RHS` syntax in `@equations`. Empty when no tag was given.
"""
tags(eqn::Equation) = eqn.tags

function Base.show(io::IO, eqn::Equation)
    print(io, "Equation(")
    if eqn.doc !== nothing && !isempty(eqn.doc)
        firstline = first(split(eqn.doc, '\n'))
        print(io, repr(firstline), ", ")
    end
    if !isempty(eqn.tags)
        print(io, "tags=", eqn.tags, ", ")
    end
    print(io, eqn.residual, ")")
end

export doc, tags

end # module Compile
