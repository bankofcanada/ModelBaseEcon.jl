module IR

export VarKind, VAR_NORMAL, VAR_LOG, VAR_SHOCK
export ParamKind, PARAM_SCALAR, PARAM_ARRAY, PARAM_LINKED
export EquationFlag, EQ_LIN, EQ_LOG
export VarDecl, ShockDecl, ParamDecl, EquationAST, AutoexogPair
export SSEquationAST, SSEquationKind, SS_LEVEL, SS_SLOPE
export param_array_length
export ModelDef
export add_var!, add_shock!, add_param!, add_equation!, add_autoexog!, add_exog!
export add_ss_equation!, delete_ss_equations!, next_ss_eqn_name
export is_exogenous, exogenous_names
export find_decl

@enum VarKind VAR_NORMAL VAR_LOG VAR_SHOCK
@enum ParamKind PARAM_SCALAR PARAM_ARRAY PARAM_LINKED
@enum EquationFlag EQ_LIN EQ_LOG

struct VarDecl
    name::Symbol
    kind::VarKind
    doc::Union{String, Nothing}
end
VarDecl(name::Symbol; kind::VarKind=VAR_NORMAL, doc=nothing) = VarDecl(name, kind, doc)

"""
A shock (or, with `exogenous=true`, an exogenous variable).

For the RW stacked-time solver an exogenous variable and a shock are the
same thing - a data column in `e_full`, read by equations, never solved
for. They are stored in the single `ModelDef.shocks` list so the solver,
symbolic core, and validator need no exogenous-specific code path. The
`exogenous` flag preserves the source-level distinction for
introspection and `@autoexogenize`. See the `@exogenous` macro.
"""
struct ShockDecl
    name::Symbol
    doc::Union{String, Nothing}
    exogenous::Bool
end
ShockDecl(name::Symbol; doc=nothing, exogenous::Bool=false) =
    ShockDecl(name, doc, exogenous)

"""
A parameter declaration.

- `kind == PARAM_SCALAR`: `value` is a `Float64` (or convertible).
- `kind == PARAM_ARRAY`:  `value` is a `Vector{Float64}`.
- `kind == PARAM_LINKED`: `value` is the defining `Expr` (resolved later at compile time).
"""
struct ParamDecl
    name::Symbol
    kind::ParamKind
    value::Any
    doc::Union{String, Nothing}
end
ParamDecl(name::Symbol, value; kind::ParamKind=PARAM_SCALAR, doc=nothing) =
    ParamDecl(name, kind, value, doc)

"""
    param_array_length(p::ParamDecl) -> Int

Number of flat scalar slots a parameter contributes to the flat
parameter vector: the array length for `PARAM_ARRAY`, and `1` for
`PARAM_SCALAR`. `PARAM_LINKED` params contribute no root slots, so this
is an error for them - callers should filter linked params out first.
"""
function param_array_length(p::ParamDecl)
    if p.kind === PARAM_ARRAY
        return length(p.value)::Int
    elseif p.kind === PARAM_SCALAR
        return 1
    else
        error("param_array_length: $(p.name) is $(p.kind), not a root parameter")
    end
end

"""
A single equation as parsed from the user's @equations block.

`expr` is the raw LHS = RHS expression with `=` rewritten to `-` so it
is ready to be passed to symbolic conversion as a residual `F` such that `F = 0`.
`flags` records per-equation annotations (`@lin`, `@log`).
`tsrefs` is populated during symbol resolution, not at macro expansion.
"""
struct EquationAST
    residual::Expr           # LHS - RHS
    flags::Set{EquationFlag}
    doc::Union{String, Nothing}
    tags::Vector{Symbol}     # user-supplied :tag prefixes, in declaration order
    src::LineNumberNode      # for error messages
end
EquationAST(residual::Expr, flags::Set{EquationFlag},
            doc::Union{String, Nothing}, src::LineNumberNode) =
    EquationAST(residual, flags, doc, Symbol[], src)

struct AutoexogPair
    var::Symbol
    shock::Symbol
end

# ----------------------------------------------------------------------
# Steady-state user equations
#
# `@steadystate model lhs = rhs` adds an extra equation to the SS system.
# Only `@level` constraints are supported; `@slope` is not yet implemented.
#
# The residual Expr stores the variable references without time
# indexing (`c`, not `c[t]`) - legacy syntax. The symbolic-kernel build
# rewrites bare names to `name[t]` so the existing dynamic-equation
# kernel pipeline accepts them unchanged.
# ----------------------------------------------------------------------

@enum SSEquationKind SS_LEVEL SS_SLOPE

"""
A user-supplied steady-state equation. `name` is the auto-assigned tag
(`_SSEQ1`, `_SSEQ2`, ...) unless the user supplied one. `residual` is the
already-rewritten `lhs - rhs` `Expr` over un-time-indexed variable
names. `kind` is `SS_LEVEL` (the only currently supported kind).
"""
struct SSEquationAST
    name::Symbol
    residual::Expr
    kind::SSEquationKind
    doc::Union{String, Nothing}
    src::LineNumberNode
end
SSEquationAST(name::Symbol, residual::Expr;
              kind::SSEquationKind = SS_LEVEL,
              doc = nothing,
              src::LineNumberNode = LineNumberNode(0, :ss)) =
    SSEquationAST(name, residual, kind, doc, src)

"""
The mutable container that all DSL macros append to. One per user model.
"""
mutable struct ModelDef
    name::Symbol
    vars::Vector{VarDecl}
    shocks::Vector{ShockDecl}
    params::Vector{ParamDecl}
    equations::Vector{EquationAST}
    autoexog::Vector{AutoexogPair}
    ss_equations::Vector{SSEquationAST}
    initialized::Bool
    # Compatibility slot (v0.8.0 alias layer): when the model is initialized
    # in-place via the legacy `@initialize model` idiom, the frozen compiled
    # `Model` is cached here so legacy property access (`model.maxlag`,
    # `model.eqns`, residual evaluation, ...) keeps working on the same object.
    # Typed `Any` because `Model` is defined later (compile.jl). `nothing`
    # until initialized. Not part of the rewrite's own pipeline, which uses
    # `initialize_model(def) -> Model` directly.
    compiled::Any
    # Compatibility slot for the legacy model-level `flags` holder
    # (`flags.linear`, ...). Typed `Any` because the `ModelFlags` type is
    # defined later (compat.jl). Travels with the object through `deepcopy`,
    # unlike an identity-keyed side table. `nothing` until first touched.
    flags::Any
end
ModelDef(name::Symbol=:model) =
    ModelDef(name, VarDecl[], ShockDecl[], ParamDecl[],
             EquationAST[], AutoexogPair[], SSEquationAST[], false, nothing, nothing)

# ----------------------------------------------------------------------
# Mutation API used by macros
# ----------------------------------------------------------------------

function _check_unique(model::ModelDef, name::Symbol)
    for v in model.vars;   v.name === name && error("duplicate name: $name (already a variable)"); end
    for s in model.shocks; s.name === name && error("duplicate name: $name (already a shock)"); end
    for p in model.params; p.name === name && error("duplicate name: $name (already a parameter)"); end
    return nothing
end

function add_var!(model::ModelDef, decl::VarDecl)
    _check_unique(model, decl.name)
    push!(model.vars, decl)
    return decl
end

function add_shock!(model::ModelDef, decl::ShockDecl)
    _check_unique(model, decl.name)
    push!(model.shocks, decl)
    return decl
end

function add_param!(model::ModelDef, decl::ParamDecl)
    _check_unique(model, decl.name)
    push!(model.params, decl)
    return decl
end

"""
    add_exog!(model, decl::ShockDecl) -> ShockDecl

Register an exogenous variable. Exogenous variables are stored in the
shock list (the solver treats them identically) with `exogenous=true`.
`decl` must already carry `exogenous=true`.
"""
function add_exog!(model::ModelDef, decl::ShockDecl)
    decl.exogenous || error("add_exog!: ShockDecl for $(decl.name) must have exogenous=true")
    _check_unique(model, decl.name)
    push!(model.shocks, decl)
    return decl
end

"""
    is_exogenous(model, name::Symbol) -> Bool

True iff `name` was declared via `@exogenous` (a shock-list entry tagged
`exogenous`). False for plain shocks and for non-shock names.
"""
function is_exogenous(model::ModelDef, name::Symbol)
    for s in model.shocks
        s.name === name && return s.exogenous
    end
    return false
end

"""
    exogenous_names(model) -> Vector{Symbol}

Names declared via `@exogenous`, in declaration order.
"""
exogenous_names(model::ModelDef) =
    Symbol[s.name for s in model.shocks if s.exogenous]

function add_equation!(model::ModelDef, eq::EquationAST)
    push!(model.equations, eq)
    return eq
end

function add_autoexog!(model::ModelDef, pair::AutoexogPair)
    push!(model.autoexog, pair)
    return pair
end

"""
    next_ss_eqn_name(model) -> Symbol

Auto-generated tag for the next unnamed `@steadystate` equation:
`_SSEQ1`, `_SSEQ2`, ... picking the lowest free index.
"""
function next_ss_eqn_name(model::ModelDef)
    used = Set(e.name for e in model.ss_equations)
    i = 1
    while Symbol("_SSEQ", i) in used
        i += 1
    end
    return Symbol("_SSEQ", i)
end

"""
    add_ss_equation!(model, eq::SSEquationAST) -> SSEquationAST

Append a user-supplied SS equation. If `eq.name` matches an existing
entry, the prior one is replaced (legacy semantics - a re-issued
constraint on the same key overrides the earlier value).
"""
function add_ss_equation!(model::ModelDef, eq::SSEquationAST)
    for (i, e) in pairs(model.ss_equations)
        if e.name === eq.name
            model.ss_equations[i] = eq
            return eq
        end
    end
    push!(model.ss_equations, eq)
    return eq
end

"""
    delete_ss_equations!(model, names) -> Vector{Symbol}

Remove SS equations whose `name` appears in `names`. Returns the
actually-removed names (silently ignores unknown tags, matching the
legacy `delete_sstate_equations!` no-op-on-missing behaviour).
"""
function delete_ss_equations!(model::ModelDef, names)
    name_set = Set(Symbol[n for n in names])
    removed = Symbol[]
    keep = SSEquationAST[]
    for e in model.ss_equations
        if e.name in name_set
            push!(removed, e.name)
        else
            push!(keep, e)
        end
    end
    model.ss_equations = keep
    return removed
end

"""
Look up a declared name. Returns `(:var, idx)`, `(:shock, idx)`,
`(:param, idx)`, or `nothing` if undeclared.
"""
function find_decl(model::ModelDef, name::Symbol)
    for (i, v) in pairs(model.vars);   v.name === name && return (:var,   i); end
    for (i, s) in pairs(model.shocks); s.name === name && return (:shock, i); end
    for (i, p) in pairs(model.params); p.name === name && return (:param, i); end
    return nothing
end

end # module IR
