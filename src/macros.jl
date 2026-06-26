##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

module Macros

using ..IR
using ..MetaFuncs: expand_metafuncs
using MacroTools: MacroTools, @capture, postwalk, prewalk, rmlines, isexpr

export var"@variables", var"@logvariables", var"@shocks", var"@parameters",
       var"@equations", var"@autoexogenize", var"@exogenous", var"@autoshocks",
       var"@steadystate"

# ----------------------------------------------------------------------
# Block walking helpers
# ----------------------------------------------------------------------
# A "decl block" is what's inside `begin ... end` after a macro like
# @variables. We walk it linearly, tracking pending docstrings.
#
# Patterns we accept inside a decl block:
#   "doc"; name              -> name with doc
#   name                     -> bare name
#   name = value             -> name with value (parameters)
#   name = @link expr        -> linked parameter
#   name; name; name         -> multiple bare names on one line (Expr(:block) of Symbols)
#   var = shock              -> autoexog pair
# Comments and LineNumberNodes are skipped. Trailing `;` is accommodated by
# Julia's parser collapsing them to nothing.
# ----------------------------------------------------------------------

"""
Strip line-number nodes and unwrap a single-statement `begin` block.
Returns a `Vector{Any}` of statements.

`Core.@doc` macrocalls (which Julia's parser produces when a string
literal precedes a binding inside a `begin` block) are flattened into
their constituent `(doc_string, body)` parts so downstream walkers can
re-pair them naturally.
"""
function _block_statements(blk)
    raw = if blk isa Expr && blk.head === :block
        Any[s for s in blk.args if !(s isa LineNumberNode)]
    elseif blk isa Expr && blk.head === :tuple
        collect(blk.args)
    else
        Any[blk]
    end
    out = Any[]
    for s in raw
        if _is_core_doc(s)
            # s = :(Core.@doc "string" body)
            doc = s.args[3]
            body = s.args[4]
            push!(out, doc)
            push!(out, body)
        else
            push!(out, s)
        end
    end
    return out
end

"""
True iff `ex` is the `Core.@doc "string" body` form Julia's parser emits
when a string literal precedes a binding inside `begin ... end`.
"""
function _is_core_doc(ex)
    ex isa Expr || return false
    ex.head === :macrocall || return false
    head = ex.args[1]
    head === GlobalRef(Core, Symbol("@doc")) && return true
    head isa Expr && head.head === :(.) && return false  # not a Core.@doc
    head === Symbol("@doc") && return true
    return false
end

"""
Walk a vector of statements pairing leading docstrings with the next non-doc
statement. Yields `(doc::Union{String,Nothing}, stmt)` pairs.
"""
function _with_docs(stmts)
    pairs = Tuple{Union{String,Nothing}, Any}[]
    pending_doc = nothing
    for s in stmts
        if s isa String
            pending_doc = s
        else
            push!(pairs, (pending_doc, s))
            pending_doc = nothing
        end
    end
    return pairs
end

"""
Split a statement that might be `a; b; c` (a `Expr(:block, ...)`) or a single
name into a vector of bare-name expressions. Handles the `a; b; c` shorthand
in @variables blocks.
"""
function _split_names(stmt)
    if stmt isa Symbol
        return Symbol[stmt]
    elseif stmt isa Expr && stmt.head === :block
        out = Symbol[]
        for s in stmt.args
            s isa LineNumberNode && continue
            s isa Symbol         && (push!(out, s); continue)
            error("unexpected element in name block: $s")
        end
        return out
    elseif stmt isa Expr && stmt.head === :tuple
        # `a, b, c`
        out = Symbol[]
        for s in stmt.args
            s isa Symbol || error("expected name, got $s")
            push!(out, s)
        end
        return out
    else
        error("expected variable name(s), got $(stmt)")
    end
end

# ----------------------------------------------------------------------
# @variables / @logvariables / @shocks
# ----------------------------------------------------------------------

"""
True iff `stmt` is an inline `@log name` annotation inside a `@variables`
block. Returns `(is_log, inner_stmt)` - `inner_stmt` is `stmt` with the
`@log` peeled, or `stmt` unchanged when no annotation is present.
"""
function _peel_inline_log(stmt)
    if stmt isa Expr && stmt.head === :macrocall
        head = stmt.args[1]
        sym = head isa GlobalRef ? head.name : head
        if sym === Symbol("@log")
            return true, stmt.args[end]
        end
    end
    return false, stmt
end

function _emit_var_decls(modelvar, blk, kind::IR.VarKind)
    stmts = _block_statements(blk)
    pairs = _with_docs(stmts)
    out = Expr(:block)
    for (doc, stmt) in pairs
        # Inline `@log name` overrides the block-level kind for one entry
        # (FRBUS-style mixed log / non-log @variables blocks).
        is_log, inner = _peel_inline_log(stmt)
        entry_kind = is_log ? IR.VAR_LOG : kind
        for name in _split_names(inner)
            push!(out.args, :(IR.add_var!($(esc(modelvar)),
                IR.VarDecl($(QuoteNode(name)), $entry_kind, $doc))))
        end
    end
    return out
end

function _emit_shock_decls(modelvar, blk_or_name)
    out = Expr(:block)
    if blk_or_name isa Symbol
        push!(out.args, :(IR.add_shock!($(esc(modelvar)),
            IR.ShockDecl($(QuoteNode(blk_or_name)), nothing, false))))
        return out
    end
    stmts = _block_statements(blk_or_name)
    pairs = _with_docs(stmts)
    for (doc, stmt) in pairs
        for name in _split_names(stmt)
            push!(out.args, :(IR.add_shock!($(esc(modelvar)),
                IR.ShockDecl($(QuoteNode(name)), $doc, false))))
        end
    end
    return out
end

"""
Emit `add_exog!` calls for an `@exogenous` block. Accepts the same
`begin ... end` decl-block grammar as `@variables` / `@shocks`:
docstrings, bare names, and inline `@log` annotations are all parsed -
but exogenous variables carry no equation and no log-codegen, so a
`@log` flag on them is silently dropped (it only documents intent).
"""
function _emit_exog_decls(modelvar, blk)
    stmts = _block_statements(blk)
    pairs = _with_docs(stmts)
    out = Expr(:block)
    for (doc, stmt) in pairs
        # An inline `@log name` on an exogenous entry is tolerated: peel
        # it off and keep the name. Exogenous vars are pure data inputs.
        inner = stmt
        while inner isa Expr && inner.head === :macrocall
            inner = inner.args[end]
        end
        for name in _split_names(inner)
            push!(out.args, :(IR.add_exog!($(esc(modelvar)),
                IR.ShockDecl($(QuoteNode(name)), $doc, true))))
        end
    end
    return out
end

macro variables(modelvar, blk)
    _emit_var_decls(modelvar, blk, IR.VAR_NORMAL)
end

macro exogenous(modelvar, blk)
    _emit_exog_decls(modelvar, blk)
end

macro logvariables(modelvar, blk)
    _emit_var_decls(modelvar, blk, IR.VAR_LOG)
end

macro shocks(modelvar, blk_or_name)
    _emit_shock_decls(modelvar, blk_or_name)
end

# ----------------------------------------------------------------------
# @parameters - supports scalars, arrays, and `@link expr`
# ----------------------------------------------------------------------

"""
Classify a `@parameters` RHS. Returns `(:scalar, value_expr)`,
`(:array, value_expr)`, or `(:linked, defining_expr)`.
"""
function _classify_param_rhs(rhs)
    if rhs isa Expr && rhs.head === :macrocall &&
       (rhs.args[1] === Symbol("@link") || rhs.args[1] === GlobalRef(@__MODULE__, Symbol("@link")))
        # rhs = :(@link expr)  ->  args = [Symbol("@link"), LineNumberNode, expr]
        defining = rhs.args[end]
        return (:linked, defining)
    elseif rhs isa Expr && rhs.head === :vect
        return (:array, rhs)
    elseif rhs isa Expr && rhs.head === :vcat
        return (:array, rhs)
    else
        return (:scalar, rhs)
    end
end

function _emit_param_decls(modelvar, blk)
    stmts = _block_statements(blk)
    pairs = _with_docs(stmts)
    out = Expr(:block)
    for (doc, stmt) in pairs
        if !(stmt isa Expr && stmt.head === :(=))
            error("expected `name = value` in @parameters, got: $stmt")
        end
        name = stmt.args[1]
        rhs = stmt.args[2]
        name isa Symbol || error("parameter name must be a Symbol, got $name")
        kind, value_expr = _classify_param_rhs(rhs)
        if kind === :scalar
            push!(out.args, :(IR.add_param!($(esc(modelvar)),
                IR.ParamDecl($(QuoteNode(name)), IR.PARAM_SCALAR,
                    Float64($(esc(value_expr))), $doc))))
        elseif kind === :array
            push!(out.args, :(IR.add_param!($(esc(modelvar)),
                IR.ParamDecl($(QuoteNode(name)), IR.PARAM_ARRAY,
                    Vector{Float64}($(esc(value_expr))), $doc))))
        else  # :linked - defer evaluation; store the Expr verbatim
            push!(out.args, :(IR.add_param!($(esc(modelvar)),
                IR.ParamDecl($(QuoteNode(name)), IR.PARAM_LINKED,
                    $(QuoteNode(value_expr)), $doc))))
        end
    end
    return out
end

macro parameters(modelvar, blk)
    _emit_param_decls(modelvar, blk)
end

# ----------------------------------------------------------------------
# @autoexogenize
# ----------------------------------------------------------------------

function _emit_autoexog(modelvar, blk)
    stmts = _block_statements(blk)
    out = Expr(:block)
    for stmt in stmts
        if !(stmt isa Expr && stmt.head === :(=))
            error("expected `var = shock` in @autoexogenize, got: $stmt")
        end
        v = stmt.args[1]
        s = stmt.args[2]
        v isa Symbol && s isa Symbol ||
            error("@autoexogenize entries must be Symbol = Symbol, got: $stmt")
        push!(out.args, :(IR.add_autoexog!($(esc(modelvar)),
            IR.AutoexogPair($(QuoteNode(v)), $(QuoteNode(s))))))
    end
    return out
end

macro autoexogenize(modelvar, blk)
    _emit_autoexog(modelvar, blk)
end

# ----------------------------------------------------------------------
# @autoshocks
# ----------------------------------------------------------------------
# `@autoshocks model [suffix]` - for every endogenous variable `v`
# (declared via @variables, i.e. not exogenous, not a shock) create a
# shock named `Symbol(v, suffix)` and an autoexogenize pair `v => v<suffix>`.
# Default suffix is `_shk`. FRBUS_VAR uses `@autoshocks model _a`.
#
# Must run AFTER @variables / @exogenous so the variable list is complete,
# and before @equations so the generated shocks resolve as tsrefs.
# ----------------------------------------------------------------------

"""
    @autoshocks model [suffix]

Generate one shock per endogenous variable, named by appending `suffix`
to the variable name (default `_shk`), plus the matching autoexogenize
pair. Exogenous variables and pre-existing shocks are skipped.
"""
macro autoshocks(modelvar, suffix=:_shk)
    suf = suffix isa QuoteNode ? suffix.value :
          suffix isa Symbol ? suffix :
          error("@autoshocks suffix must be a symbol, got: $suffix")
    quote
        for _v in $(esc(modelvar)).vars
            _sname = Symbol(_v.name, $(QuoteNode(suf)))
            IR.add_shock!($(esc(modelvar)), IR.ShockDecl(_sname, nothing, false))
            IR.add_autoexog!($(esc(modelvar)),
                IR.AutoexogPair(_v.name, _sname))
        end
        nothing
    end
end

# ----------------------------------------------------------------------
# @equations
# ----------------------------------------------------------------------
# Each statement may be:
#   LHS = RHS                    (a plain equation)
#   @lin LHS = RHS               (annotated)
#   @log LHS = RHS               (annotated)
#   @lin @log LHS = RHS          (multi-annotated; future-proof)
#
# Docstrings preceding an equation are attached.
# ----------------------------------------------------------------------

"""
Strip leading equation-flag macros (`@lin`, `@log`) off `stmt`, returning
`(flags::Set{EquationFlag}, inner_stmt)`. Recurses through nested macrocalls.
"""
function _peel_eq_flags(stmt)
    flags = Set{IR.EquationFlag}()
    cur = stmt
    while cur isa Expr && cur.head === :macrocall
        macname = cur.args[1]
        sym = macname isa GlobalRef ? macname.name : macname
        if sym === Symbol("@lin")
            push!(flags, IR.EQ_LIN)
            cur = cur.args[end]
        elseif sym === Symbol("@log")
            push!(flags, IR.EQ_LOG)
            cur = cur.args[end]
        else
            break
        end
    end
    return flags, cur
end

_is_pair_call(ex) =
    ex isa Expr && ex.head === :call && length(ex.args) == 3 &&
    ex.args[1] === :(=>)

function _take_tag!(tags::Vector{Symbol}, tag_expr)
    if !(tag_expr isa QuoteNode && tag_expr.value isa Symbol)
        error("@equations: equation tag must be a Symbol literal (e.g. `:identity => LHS = RHS`), got: $tag_expr")
    end
    push!(tags, tag_expr.value)
end

"""
Strip leading `:tag => ...` prefix(es) off `stmt`. Returns
`(tags::Vector{Symbol}, inner_stmt)`. Each tag must be a
`QuoteNode(::Symbol)`; anything else raises.

Two parse shapes arise from Julia precedence:

- Plain `:tag => LHS = RHS` parses as `Expr(:(=), Expr(:call, :=>, :tag, LHS), RHS)`
  (because `=` binds looser than `=>`). The tag lives on the LHS of `:(=)`.
- `:tag => @flag LHS = RHS` parses as `Expr(:call, :=>, :tag, macrocall(...))`
  because the macrocall absorbs the trailing assignment. The tag lives at the top.

This function handles both, then handles chained tags
(`:t1 => :t2 => eqn`).
"""
function _peel_eq_tags(stmt)
    tags = Symbol[]
    cur = stmt
    while true
        if _is_pair_call(cur)
            _take_tag!(tags, cur.args[2])
            cur = cur.args[3]
            continue
        elseif cur isa Expr && cur.head === :(=) && _is_pair_call(cur.args[1])
            pair = cur.args[1]
            _take_tag!(tags, pair.args[2])
            # Rebuild the assignment without the consumed tag wrapper.
            cur = Expr(:(=), pair.args[3], cur.args[2])
            continue
        end
        break
    end
    return tags, cur
end

function _emit_equation(modelvar, doc, stmt, src::LineNumberNode)
    tags, after_tags = _peel_eq_tags(stmt)
    flags, inner = _peel_eq_flags(after_tags)
    if !(inner isa Expr && inner.head === :(=))
        error("expected `LHS = RHS` equation at $src, got: $inner")
    end
    # Expand @lag/@lead/@d/@dlog meta-functions on both sides before the
    # residual reaches the symbolic core - they are syntactic operators
    # over the equation Expr, not callable functions.
    lhs = expand_metafuncs(inner.args[1])
    rhs = expand_metafuncs(inner.args[2])
    residual = Expr(:call, :-, lhs, rhs)
    flags_expr = Expr(:call, Set{IR.EquationFlag},
                      Expr(:vect, [QuoteNode(f) for f in flags]...))
    tags_expr = Expr(:call, Vector{Symbol},
                     Expr(:vect, [QuoteNode(t) for t in tags]...))
    return :(IR.add_equation!($(esc(modelvar)),
        IR.EquationAST($(QuoteNode(residual)), $flags_expr, $doc,
                       $tags_expr, $(QuoteNode(src)))))
end

function _emit_equations(modelvar, blk)
    if !(blk isa Expr && blk.head === :block)
        error("@equations requires a `begin ... end` block")
    end
    out = Expr(:block)
    pending_doc = nothing
    last_src = LineNumberNode(0, :unknown)
    # Find the LineNumberNode that immediately precedes each statement so
    # error messages point at the right line. We walk the raw block once
    # to track sources, then dispatch each (non-LineNumber, non-doc)
    # statement through _block_statements' Core.@doc-flattening so the
    # docstring/equation pair surfaces naturally.
    for s in blk.args
        if s isa LineNumberNode
            last_src = s
            continue
        end
        # Flatten any Core.@doc wrapper this statement carries.
        flattened = _is_core_doc(s) ? Any[s.args[3], s.args[4]] : Any[s]
        for fs in flattened
            if fs isa String
                pending_doc = fs
            else
                push!(out.args, _emit_equation(modelvar, pending_doc, fs, last_src))
                pending_doc = nothing
            end
        end
    end
    return out
end

macro equations(modelvar, blk)
    _emit_equations(modelvar, blk)
end

# ----------------------------------------------------------------------
# @steadystate
# ----------------------------------------------------------------------
# Surface:
#
#   @steadystate model lhs = rhs
#   @steadystate model @level lhs = rhs
#   @steadystate model @slope lhs = rhs        # -> error (not yet supported)
#   @steadystate model begin
#       lhs = rhs
#       @level lhs = rhs
#       @delete _SSEQ1 _SSEQ2
#   end
#   @steadystate model @delete _SSEQ1 _SSEQ2
#
# Variable names appear *without* `[t]` (legacy syntax). The residual
# stored on the `SSEquationAST` is `lhs - rhs` verbatim; the kernel
# build rewrites bare-name references to `name[t]` so the existing
# symbolic pipeline can lower them like any other dynamic equation.
# ----------------------------------------------------------------------

"""
True iff `ex` is `@delete sym sym ...` - used at top level and inside
`@steadystate` blocks to drop previously-added constraints.
"""
function _is_ss_delete(ex)
    ex isa Expr && ex.head === :macrocall || return false
    head = ex.args[1]
    sym = head isa GlobalRef ? head.name : head
    return sym === Symbol("@delete")
end

function _ss_delete_names(ex)
    # macrocall args: [@delete, LineNumberNode, name1, name2, ...]
    names = Symbol[]
    for a in ex.args[3:end]
        if a isa Symbol
            push!(names, a)
        elseif a isa QuoteNode && a.value isa Symbol
            push!(names, a.value)
        else
            error("@steadystate @delete: expected bare symbol(s), got $a")
        end
    end
    return names
end

"""
Peel a leading `@level` / `@slope` qualifier off a single `@steadystate`
entry. Returns `(kind::IR.SSEquationKind, inner_stmt)`. Default kind is
`SS_LEVEL`. A `@slope` qualifier raises during macro expansion; it is not
yet supported.
"""
function _peel_ss_qualifier(stmt)
    if stmt isa Expr && stmt.head === :macrocall
        head = stmt.args[1]
        sym = head isa GlobalRef ? head.name : head
        if sym === Symbol("@level")
            return IR.SS_LEVEL, stmt.args[end]
        elseif sym === Symbol("@slope")
            error("@steadystate @slope is not yet supported; @level only")
        end
    end
    return IR.SS_LEVEL, stmt
end

function _emit_ss_equation(modelvar, kind, eqn_stmt, src::LineNumberNode,
                            name_expr)
    if !(eqn_stmt isa Expr && eqn_stmt.head === :(=))
        error("@steadystate at $src: expected `lhs = rhs`, got $eqn_stmt")
    end
    lhs = eqn_stmt.args[1]
    rhs = eqn_stmt.args[2]
    residual = Expr(:call, :-, lhs, rhs)
    # name_expr may be a Symbol literal or `nothing` (then auto-assigned at runtime).
    name_runtime = name_expr === nothing ?
        :(IR.next_ss_eqn_name($(esc(modelvar)))) :
        :($(QuoteNode(name_expr)))
    return :(IR.add_ss_equation!($(esc(modelvar)),
        IR.SSEquationAST($name_runtime, $(QuoteNode(residual)),
                         $kind, nothing, $(QuoteNode(src)))))
end

# Top-level form: `@steadystate model EXPR`
macro steadystate(modelvar, ex)
    src = __source__
    # `@steadystate model @delete a b c`
    if _is_ss_delete(ex)
        names = _ss_delete_names(ex)
        return :(IR.delete_ss_equations!($(esc(modelvar)),
            $(Expr(:vect, [QuoteNode(n) for n in names]...))))
    end
    # Block form: `@steadystate model begin ... end`
    if ex isa Expr && ex.head === :block
        out = Expr(:block)
        last_src = src
        for s in ex.args
            if s isa LineNumberNode
                last_src = s
                continue
            end
            if _is_ss_delete(s)
                names = _ss_delete_names(s)
                push!(out.args,
                    :(IR.delete_ss_equations!($(esc(modelvar)),
                        $(Expr(:vect, [QuoteNode(n) for n in names]...)))))
                continue
            end
            kind, inner = _peel_ss_qualifier(s)
            kind_expr = kind === IR.SS_LEVEL ? :(IR.SS_LEVEL) : :(IR.SS_SLOPE)
            push!(out.args, _emit_ss_equation(modelvar, kind_expr, inner,
                                              last_src, nothing))
        end
        return out
    end
    # Single-equation form (with optional @level/@slope qualifier).
    kind, inner = _peel_ss_qualifier(ex)
    kind_expr = kind === IR.SS_LEVEL ? :(IR.SS_LEVEL) : :(IR.SS_SLOPE)
    return _emit_ss_equation(modelvar, kind_expr, inner, src, nothing)
end

end # module Macros
