module Export

using ..IR
using ..Compile

export export_model

# ----------------------------------------------------------------------
# export_model - write a Model out as a runnable .jl file (G8)
#
# The emitted file defines `build_<name>()` which reconstructs the model
# via the RW-MBE DSL (`@parameters`, `@variables`, `@logvariables`,
# `@shocks`, `@exogenous`, `@autoexogenize`, `@equations`, `@initialize`).
# Round-trip target: per-equation `expr_hash` equal to the original, and
# the flat-parameter vector byte-identical. See PLAN_v2.1 §5.
#
# `@steadystate` user constraints (G4) and array-param emission with
# G9 tag/doc metadata are emitted to the extent they exist on the model.
# ----------------------------------------------------------------------

_check_name(name::AbstractString) =
    Base.isidentifier(name) || throw(ArgumentError(
        "export_model: model name must be a valid Julia identifier, got $(repr(name))"))

# Float64 round-trip-correct formatting. Julia's `repr(::Float64)` uses
# Ryu, which guarantees parse(Float64, repr(x)) === x for every Float64.
# That's equivalent to %.17g without the Printf dependency.
_fmt(x::Float64) = repr(x)
_fmt(x::Real)    = _fmt(Float64(x))

_doc_string(d::Union{String,Nothing}) = d === nothing ? "" : d

# Quoted Julia-source literal of a String. Uses `repr` so embedded
# newlines, backslashes, and quotes survive.
_quote_str(s::AbstractString) = repr(s)

# Indented `println` helper.
_pl(io::IO, indent::Int, args...) = (print(io, " "^indent); println(io, args...))

# ----------------------------------------------------------------------
# Emit `@parameters` block. Scalars use %.17g, arrays emit a literal
# Vector, linked params emit `@link <expr>`.
# ----------------------------------------------------------------------
function _emit_parameters(io::IO, def::IR.ModelDef)
    isempty(def.params) && return
    println(io, "    @parameters m begin")
    for p in def.params
        d = _doc_string(p.doc)
        if !isempty(d)
            _pl(io, 8, _quote_str(d))
        end
        if p.kind === IR.PARAM_SCALAR
            _pl(io, 8, p.name, " = ", _fmt(p.value))
        elseif p.kind === IR.PARAM_ARRAY
            arr = p.value::Vector{Float64}
            _pl(io, 8, p.name, " = [", join((_fmt(v) for v in arr), ", "), "]")
        elseif p.kind === IR.PARAM_LINKED
            # p.value is the defining Expr (e.g. :(1/(1+ρ))).
            _pl(io, 8, p.name, " = @link ", p.value)
        else
            error("export_model: unknown ParamKind $(p.kind) for $(p.name)")
        end
    end
    println(io, "    end")
    println(io)
end

# ----------------------------------------------------------------------
# Emit `@variables` / `@logvariables`. Each variable lives in exactly one
# block; the kind field decides which.
# ----------------------------------------------------------------------
function _emit_variables(io::IO, def::IR.ModelDef)
    normals = [v for v in def.vars if v.kind === IR.VAR_NORMAL]
    logs    = [v for v in def.vars if v.kind === IR.VAR_LOG]
    if !isempty(normals)
        println(io, "    @variables m begin")
        for v in normals
            d = _doc_string(v.doc)
            isempty(d) || _pl(io, 8, _quote_str(d))
            _pl(io, 8, v.name)
        end
        println(io, "    end")
        println(io)
    end
    if !isempty(logs)
        println(io, "    @logvariables m begin")
        for v in logs
            d = _doc_string(v.doc)
            isempty(d) || _pl(io, 8, _quote_str(d))
            _pl(io, 8, v.name)
        end
        println(io, "    end")
        println(io)
    end
end

# ----------------------------------------------------------------------
# Emit `@shocks` and `@exogenous`. RW stores both in `defs.shocks`; the
# `exogenous::Bool` discriminator picks the block.
# ----------------------------------------------------------------------
function _emit_shocks(io::IO, def::IR.ModelDef)
    shocks = [s for s in def.shocks if !s.exogenous]
    exogs  = [s for s in def.shocks if  s.exogenous]
    if !isempty(shocks)
        println(io, "    @shocks m begin")
        for s in shocks
            d = _doc_string(s.doc)
            isempty(d) || _pl(io, 8, _quote_str(d))
            _pl(io, 8, s.name)
        end
        println(io, "    end")
        println(io)
    end
    if !isempty(exogs)
        println(io, "    @exogenous m begin")
        for s in exogs
            d = _doc_string(s.doc)
            isempty(d) || _pl(io, 8, _quote_str(d))
            _pl(io, 8, s.name)
        end
        println(io, "    end")
        println(io)
    end
end

# ----------------------------------------------------------------------
# Emit `@autoexogenize`.
# ----------------------------------------------------------------------
function _emit_autoexog(io::IO, def::IR.ModelDef)
    isempty(def.autoexog) && return
    println(io, "    @autoexogenize m begin")
    for p in def.autoexog
        _pl(io, 8, p.var, " = ", p.shock)
    end
    println(io, "    end")
    println(io)
end

# ----------------------------------------------------------------------
# Emit `@equations`. The stored residual is `Expr(:call, :-, lhs, rhs)`;
# we split it back into `lhs = rhs`. Flags become `@lin`/`@log` prefixes
# (legacy ordering: `@lin` outside `@log`). Tags become a chained
# `:tag => …` prefix on the LHS. Docs become a preceding string literal.
# ----------------------------------------------------------------------
function _equation_lhs_rhs(eq::IR.EquationAST)
    res = eq.residual
    if !(res isa Expr && res.head === :call && length(res.args) == 3 &&
         res.args[1] === :-)
        error("export_model: malformed residual (expected `lhs - rhs`): $res")
    end
    return res.args[2], res.args[3]
end

# Wrap `inner` in flag macrocalls. Order matches `_peel_eq_flags`: the
# outermost macro is applied last when re-parsing, so to round-trip the
# original parse we wrap @log first then @lin.
function _wrap_flags(inner::Expr, flags::Set{IR.EquationFlag})
    out = inner
    if IR.EQ_LOG in flags
        out = Expr(:macrocall, Symbol("@log"), LineNumberNode(0), out)
    end
    if IR.EQ_LIN in flags
        out = Expr(:macrocall, Symbol("@lin"), LineNumberNode(0), out)
    end
    return out
end

# Strip LineNumberNodes from a macrocall expression for compact printing.
_strip_macrocall_lnn(ex) = ex
function _strip_macrocall_lnn(ex::Expr)
    if ex.head === :macrocall
        args = Any[a for a in ex.args if !(a isa LineNumberNode)]
        return Expr(:macrocall, args[1], LineNumberNode(0),
                    map(_strip_macrocall_lnn, args[2:end])...)
    end
    return Expr(ex.head, map(_strip_macrocall_lnn, ex.args)...)
end

function _emit_equations(io::IO, def::IR.ModelDef)
    isempty(def.equations) && return
    println(io, "    @equations m begin")
    for eq in def.equations
        d = _doc_string(eq.doc)
        isempty(d) || _pl(io, 8, _quote_str(d))
        lhs, rhs = _equation_lhs_rhs(eq)
        body = Expr(:(=), lhs, rhs)
        wrapped = _wrap_flags(body, eq.flags)
        # Chained `:t1 => :t2 => …` tags applied to LHS when no flags
        # absorbed the assignment; when flags wrap, the tag prefix lands
        # at the top of the resulting Expr. We always print the tag
        # prefix textually, matching the parse shape `_peel_eq_tags`
        # handles.
        stripped = _strip_macrocall_lnn(wrapped)
        printed = sprint(show, stripped; context = :compact => false)
        # `show` on a `:macrocall` Expr emits `#= ... =# @lin ...`; trim
        # the source-location comment for tidiness. The leading `:(` is
        # only present for top-level expressions of certain heads - drop
        # the surrounding `:(` / `)` if Julia's printer added them.
        printed = _strip_quote_wrap(printed)
        if isempty(eq.tags)
            _pl(io, 8, printed)
        else
            tag_prefix = join((string(":", t, " => ") for t in eq.tags), "")
            _pl(io, 8, tag_prefix, printed)
        end
    end
    println(io, "    end")
    println(io)
end

# v2.1 G4 - emit `@steadystate m begin … end` so the round-trip
# `build_<name>()` reconstructs user-supplied SS constraints.
function _emit_ss_equations(io::IO, def::IR.ModelDef)
    isempty(def.ss_equations) && return
    println(io, "    @steadystate m begin")
    for sseq in def.ss_equations
        # `residual` is `lhs - rhs`; re-split for printing.
        if sseq.residual.head === :call && sseq.residual.args[1] === :- &&
           length(sseq.residual.args) == 3
            lhs = sseq.residual.args[2]
            rhs = sseq.residual.args[3]
        else
            # Defensive: emit `residual = 0` if we can't peel the minus.
            lhs = sseq.residual
            rhs = 0
        end
        body = Expr(:(=), lhs, rhs)
        printed = _strip_quote_wrap(sprint(show, body))
        qualifier = sseq.kind === IR.SS_LEVEL ? "@level " : "@slope "
        _pl(io, 8, qualifier, printed)
    end
    println(io, "    end")
    println(io)
end

# `show(::IO, ::Expr)` wraps with `:( ... )` for non-block exprs. Strip
# that wrapper if present so the emitted text is a direct expression.
function _strip_quote_wrap(s::AbstractString)
    s2 = strip(s)
    if startswith(s2, ":(") && endswith(s2, ")")
        return s2[3:end-1]
    elseif startswith(s2, "quote") && endswith(s2, "end")
        # Multiline block - fall back to raw form.
        return s2
    end
    return s2
end

# ----------------------------------------------------------------------
# Top-level export_model
# ----------------------------------------------------------------------

"""
    export_model(model::Compile.Model, name::Symbol, dir::AbstractString;
                 build_fn::Symbol = Symbol("build_", name)) -> String

Write `model` out as a runnable Julia source file at `joinpath(dir, "<name>.jl")`.
Returns the absolute path written.

The emitted file defines `<build_fn>()` returning a fresh `Model` whose
per-equation `expr_hash` matches the original to the bit, and whose
flat-parameter vector is byte-identical.

Caller is responsible for `using RWModelBaseEcon` in the consumer file
before `include`ing the emitted source - the emitted source assumes the
DSL macros are in scope.

Limitations at v2.1:
- `@steadystate` user constraints (G4) - emitted when G4 lands; until then,
  no SS constraints will exist on the model and nothing is written.
- Array-parameter element documentation (sub-tag granularity) is not
  preserved; the array as a whole is.
"""
function export_model(model::Compile.Model, name::Symbol, dir::AbstractString;
                      build_fn::Symbol = Symbol(:build_, name))
    _check_name(string(name))
    _check_name(string(build_fn))
    isdir(dir) || mkpath(dir)
    path = abspath(joinpath(dir, string(name) * ".jl"))
    open(path, "w") do io
        _emit_file(io, model.defs, name, build_fn)
    end
    return path
end

function _emit_file(io::IO, def::IR.ModelDef, name::Symbol, build_fn::Symbol)
    println(io, "# Generated by RWModelBaseEcon.Export.export_model - do not edit by hand.")
    println(io, "# Round-trip target: per-equation expr_hash equality with the source model.")
    println(io)
    println(io, "function ", build_fn, "()")
    println(io, "    m = ModelDef(", QuoteNode(name), ")")
    println(io)
    _emit_parameters(io, def)
    _emit_variables(io, def)
    _emit_shocks(io, def)
    _emit_autoexog(io, def)
    _emit_equations(io, def)
    _emit_ss_equations(io, def)
    println(io, "    return initialize_model(m)")
    println(io, "end")
    return nothing
end

end # module Export
