module Symbolic

using ..IR
using ..Validate
using Symbolics: Symbolics, Num

export TimeRef, ParamRef
export EquationKernel, build_equation_kernels
export build_ss_equation_kernels
export resolved_link_table, equation_residual_symbolic

# ----------------------------------------------------------------------
# Reference types - describe a slot in the per-equation x or params vector
# ----------------------------------------------------------------------

"""
A `(variable, time-offset)` pair. Time offsets are integers relative to `t`,
so `K[t-1]` is `TimeRef(:K, -1)`, `C[t+1]` is `TimeRef(:C, +1)`. Shocks are
also represented as `TimeRef`s.
"""
struct TimeRef
    name::Symbol
    offset::Int
end

"""
A reference into the parameter vector. For `PARAM_SCALAR` and `PARAM_LINKED`
(after resolution), `index` is `nothing`. For `PARAM_ARRAY`, `index` is the
1-based offset into the array.
"""
struct ParamRef
    name::Symbol
    index::Union{Int, Nothing}
end

# ----------------------------------------------------------------------
# EquationKernel - symbolic output for one equation
# ----------------------------------------------------------------------

"""
Symbolic representation of one equation, ready for codegen.

`tsrefs`  - vector of `(var, offset)` slots, in the canonical order that
            the residual / gradient functions will accept as their `x` input.
`params`  - vector of `ParamRef` slots in the canonical order that the
            evaluator will accept as its `params` input. Includes only
            *root* parameters (linked params have been substituted away).
`residual` - `Symbolics.Num` representing F such that F = 0.
`gradient` - `Vector{Num}` with `length == length(tsrefs)`. ∂F/∂x_i.
`hessian`  - `Matrix{Num}` (size `n×n`) when `max_hod_order ≥ 2`, else `nothing`.
`hod`      - `Vector{Array{Num,N}}` for orders 3..max_hod_order, else `nothing`.
            Order N tensor stored as a full Array{Num,N} for now; sparse
            symmetric storage is a Chunk D refinement (PLAN_v1 §5).
`flags`    - copied from `EquationAST.flags`.
`doc`      - copied from `EquationAST.doc`.
`src`      - copied from `EquationAST.src`.
"""
struct EquationKernel
    tsrefs::Vector{TimeRef}
    params::Vector{ParamRef}
    residual::Num
    gradient::Vector{Num}
    hessian::Union{Matrix{Num}, Nothing}
    hod::Union{Vector{Array{Num}}, Nothing}
    flags::Set{IR.EquationFlag}
    doc::Union{String, Nothing}
    tags::Vector{Symbol}
    src::LineNumberNode
    # The Symbolics scalar Num corresponding to each tsref slot, for
    # downstream substitution / build_function. Same length as tsrefs.
    x_syms::Vector{Num}
    # The Symbolics scalar Num corresponding to each params slot.
    p_syms::Vector{Num}
end

# ----------------------------------------------------------------------
# Param table: scalar Symbolics.Num per *root* parameter (post @link)
# ----------------------------------------------------------------------

"""
For each param in `model.params`, return the Symbolics expression it
ultimately resolves to. Linked params are substituted to their defining
expression in topological order, so a chain `c = @link b*2; b = @link a*2`
gives `c → 4a` (or `4*a` after Symbolics canonicalization).

Returns `(table::Dict{Symbol,Num}, root_layout::Vector{ParamRef})`:

- `table[name]` is the `Num` to substitute for any reference to that name in
  an equation (works for scalar, array-flattened, and linked).
- `root_layout` is the ordered list of slots in the flat `params::Vector{Float64}`
  the evaluator will receive - only includes scalar and array root params.
"""
function resolved_link_table(model::IR.ModelDef)
    link_order, _ = Validate.link_topology(model)
    table = Dict{Symbol, Any}()        # values are Num or Vector{Num}
    root_layout = ParamRef[]

    for idx in link_order
        p = model.params[idx]
        if p.kind === IR.PARAM_SCALAR
            sym = Symbolics.variable(p.name)::Num
            table[p.name] = sym
            push!(root_layout, ParamRef(p.name, nothing))
        elseif p.kind === IR.PARAM_ARRAY
            arr = Vector{Num}(undef, length(p.value))
            for i in 1:length(p.value)
                arr[i] = Symbolics.variable(Symbol(p.name, "_", i))::Num
                push!(root_layout, ParamRef(p.name, i))
            end
            table[p.name] = arr
        elseif p.kind === IR.PARAM_LINKED
            # The defining Expr only references names already resolved
            # (because of topo order). Convert the Expr to a Num using the
            # current table, then store it.
            num = _expr_to_num_with_table(p.value, table; allow_indexed=false,
                                          src=LineNumberNode(0, :link))
            table[p.name] = num
        else
            error("unhandled ParamKind: $(p.kind)")
        end
    end
    return table, root_layout
end

# ----------------------------------------------------------------------
# Per-equation: build x layout, x_syms, then walk the residual Expr
# ----------------------------------------------------------------------

"""
Walk an equation's residual `Expr`, collecting every `var[t±k]` reference.
Returns `Vector{TimeRef}` deduplicated, in (declaration_order, offset_ascending)
order using `model` as the source of declaration order.
"""
function _collect_tsrefs(model::IR.ModelDef, eq::IR.EquationAST)
    seen = Set{TimeRef}()
    var_names  = Set(v.name for v in model.vars)
    shock_names = Set(s.name for s in model.shocks)
    function walk(ex)
        if ex isa Expr && ex.head === :ref
            head = ex.args[1]
            head isa Symbol || error("unexpected indexed expr at $(eq.src): $ex")
            if head in var_names || head in shock_names
                # Time-shifted variable/shock reference.
                offset = _eval_time_offset(ex.args[2], eq.src)
                push!(seen, TimeRef(head, offset))
                # Do not descend; the index is not a variable reference.
            else
                # Indexed expression on something else (e.g. an array
                # parameter `arr[1]`). Fall through to walk children for
                # nested refs, but the index itself is plain data.
                for a in ex.args[2:end]
                    walk(a)
                end
            end
        elseif ex isa Expr
            for a in ex.args
                walk(a)
            end
        end
    end
    walk(eq.residual)
    # Stable order: by declaration index in [vars; shocks], then by offset.
    decl_order = Dict{Symbol, Int}()
    for (i, v) in pairs(model.vars);   decl_order[v.name] = i;            end
    nvars = length(model.vars)
    for (i, s) in pairs(model.shocks); decl_order[s.name] = nvars + i;    end
    refs = collect(seen)
    sort!(refs, by = r -> (get(decl_order, r.name, typemax(Int)), r.offset))
    return refs
end

"""
Evaluate the index expression of `var[idx]` to an integer offset relative to `t`.
Accepts: `t`, `t+k`, `t-k`, `k+t` for integer literal `k`.
"""
function _eval_time_offset(idx, src)
    if idx === :t
        return 0
    elseif idx isa Expr && idx.head === :call && length(idx.args) == 3
        op, a, b = idx.args
        if op === :+
            if a === :t && b isa Integer
                return Int(b)
            elseif b === :t && a isa Integer
                return Int(a)
            end
        elseif op === :-
            if a === :t && b isa Integer
                return -Int(b)
            end
        end
    end
    error("unsupported time index `$idx` at $src - expected t, t+/-k for integer k")
end

# ----------------------------------------------------------------------
# Expr → Num conversion
# ----------------------------------------------------------------------

"""
Convert a Julia `Expr` (already validated by `Validate.validate`) into a
`Symbolics.Num`. The `table` argument maps Symbols to their replacement
values (which may be `Num`, `Vector{Num}`, or numeric).

When `allow_indexed=true`, `name[idx]` is looked up as `var_table[(name, offset)]`
where `var_table` is supplied via `tsref_table`. When `allow_indexed=false`,
encountering an indexed expression is an error (used for @link RHS where
indexing should not appear).
"""
function _expr_to_num_with_table(ex, table::Dict{Symbol,Any};
                                  tsref_table::Union{Dict{TimeRef,Num}, Nothing} = nothing,
                                  allow_indexed::Bool = true,
                                  src::LineNumberNode)
    function go(e)
        if e isa Number
            return Num(e)
        elseif e isa Symbol
            haskey(table, e) || error("unbound symbol `$e` at $src")
            v = table[e]
            v isa Num && return v
            v isa Number && return Num(v)
            error("symbol `$e` at $src has unsupported binding $v")
        elseif e isa Expr && e.head === :ref
            allow_indexed || error("unexpected indexed expression at $src: $e")
            head = e.args[1]
            head isa Symbol || error("unexpected indexed expr at $src: $e")
            # Check if `head` resolves to an array binding (e.g. parameter
            # `arr = [1.0, 2.0, 3.0]`). If so, treat as plain integer
            # indexing into a Vector{Num}; otherwise it's a (var, t±k) ref.
            if haskey(table, head) && table[head] isa Vector
                idx = e.args[2]
                idx isa Integer ||
                    error("array parameter `$head` requires integer index, got $idx at $src")
                vec = table[head]::Vector
                1 <= idx <= length(vec) ||
                    error("array index out of bounds for `$head[$idx]` at $src")
                return vec[idx]::Num
            end
            offset = _eval_time_offset(e.args[2], src)
            tsref_table === nothing && error("indexed expr without tsref_table at $src")
            tref = TimeRef(head, offset)
            haskey(tsref_table, tref) ||
                error("unresolved time-ref $(tref) at $src")
            return tsref_table[tref]
        elseif e isa Expr && e.head === :call
            fn = e.args[1]
            fn isa Symbol ||
                error("unsupported call form at $src: $e")
            fn in Validate.REGISTERED_FUNCTIONS ||
                error("unknown function `$fn` at $src")
            args = [go(a) for a in e.args[2:end]]
            return _apply_registered(fn, args)
        elseif e isa Expr && e.head === :block
            # Single-statement block - usually wraps a parenthesized expr.
            stmts = [a for a in e.args if !(a isa LineNumberNode)]
            length(stmts) == 1 || error("unsupported block at $src: $e")
            return go(stmts[1])
        elseif e isa Expr && e.head === :if
            # Convert `if c; a else b end` to ifelse(c, a, b)
            length(e.args) == 3 ||
                error("if-expression at $src must have an else branch")
            return _apply_registered(:ifelse,
                [go(e.args[1]), go(e.args[2]), go(e.args[3])])
        else
            error("unsupported expression at $src: $e")
        end
    end
    return go(ex)
end

"""
Apply a registered function symbol to `Num` arguments. For most operators we
defer to Julia's normal dispatch on `Num`; for a few (e.g. `heaviside`) we
synthesize as `ifelse`.
"""
function _apply_registered(fn::Symbol, args::Vector)
    # Standard ops dispatch through Julia's resolver.
    if fn === :+
        return Base.:+(args...)
    elseif fn === :-
        return length(args) == 1 ? Base.:-(args[1]) : Base.:-(args...)
    elseif fn === :*
        return Base.:*(args...)
    elseif fn === :/
        return Base.:/(args...)
    elseif fn === :^
        return Base.:^(args[1], args[2])
    elseif fn === :%
        return Base.rem(args[1], args[2])
    elseif fn === :÷
        return Base.div(args[1], args[2])
    elseif fn === :ifelse
        # `Base.ifelse` lifts to a symbolic `ifelse` term on Num args
        # (Symbolics ≥ 7 dropped the old `Symbolics.IfElse` submodule).
        return ifelse(args[1], args[2], args[3])
    elseif fn === :min
        return min(args...)
    elseif fn === :max
        return max(args...)
    elseif fn === :heaviside
        # Step function → CTarget-friendly ifelse. Single-arg only.
        return ifelse(args[1] >= 0, Num(1.0), Num(0.0))
    elseif fn in (:(==), :(!=), :<, :>, :<=, :>=, :&, :|, :!)
        return getfield(Base, fn)(args...)
    else
        # log, exp, sqrt, sin, cos, ... - Symbolics already defines these on Num
        f = getfield(Base, fn)
        return length(args) == 1 ? f(args[1]) : f(args...)
    end
end

# ----------------------------------------------------------------------
# Build EquationKernel
# ----------------------------------------------------------------------

"""
Build all equation kernels for a validated model. Returns
`Vector{EquationKernel}`, one per equation in `model.equations`, in the
same order.

`max_hod_order ≥ 2` enables Hessian; `≥ 3` enables higher-order tensors.
"""
function build_equation_kernels(model::IR.ModelDef; max_hod_order::Int = 1)
    Validate.validate(model)
    param_table, root_layout = resolved_link_table(model)
    p_syms = Num[param_table[ref.name] isa Vector{Num} ?
                 param_table[ref.name][ref.index] :
                 param_table[ref.name]
                 for ref in root_layout]
    kernels = EquationKernel[]
    for eq in model.equations
        kernel = _build_one_kernel(model, eq, param_table, root_layout, p_syms,
                                    max_hod_order)
        push!(kernels, kernel)
    end
    return kernels, root_layout
end

function _build_one_kernel(model, eq, param_table, root_layout, p_syms,
                           max_hod_order)
    tsrefs = _collect_tsrefs(model, eq)
    # Per-equation x symbols, named so derivative output is readable.
    x_syms = Num[Symbolics.variable(_tsref_symname(r)) for r in tsrefs]
    # `@log` variable transform (PLAN_v2 §7 subtask): a VAR_LOG variable's
    # solver unknown `x` *is* the log of the variable. Every appearance of
    # such a variable in an equation therefore sees `exp(x)`. The residual
    # is built against this transformed table; the gradient is still taken
    # w.r.t. the raw `x_syms`, so the Newton step happens in log space -
    # exactly legacy MBE's `need_transform`/`inverse_transformation`.
    log_var_names = Set(v.name for v in model.vars if v.kind === IR.VAR_LOG)
    tsref_table = Dict{TimeRef, Num}()
    for (r, xs) in zip(tsrefs, x_syms)
        tsref_table[r] = r.name in log_var_names ? exp(xs) : xs
    end
    # Build a flat name→Num table that includes both per-(var,offset)
    # entries (handled via tsref_table during walk) and parameter entries.
    name_table = Dict{Symbol, Any}()
    for (name, val) in param_table
        name_table[name] = val
    end
    F = _expr_to_num_with_table(eq.residual, name_table;
                                tsref_table = tsref_table,
                                allow_indexed = true,
                                src = eq.src)
    grad = Symbolics.gradient(F, x_syms)
    hess = max_hod_order >= 2 ?
        Symbolics.jacobian(grad, x_syms) :
        nothing
    hod = nothing
    if max_hod_order >= 3
        hod = Array{Num}[]
        prev = hess  # rank-2 starting point
        for n in 3:max_hod_order
            # Differentiate prev (an Array{Num,n-1}) along x_syms to get rank n.
            new = _next_deriv_tensor(prev, x_syms)
            push!(hod, new)
            prev = new
        end
    end
    return EquationKernel(tsrefs, root_layout, F, grad, hess, hod,
                          eq.flags, eq.doc, eq.tags, eq.src, x_syms, p_syms)
end

function _tsref_symname(r::TimeRef)
    if r.offset == 0
        return Symbol(r.name, "_t")
    elseif r.offset > 0
        return Symbol(r.name, "_tp", r.offset)
    else
        return Symbol(r.name, "_tm", -r.offset)
    end
end

"""
Differentiate a rank-N symbolic tensor along `vars` to produce a rank-(N+1)
tensor. Output has shape `(size(prev)..., length(vars))`.
"""
function _next_deriv_tensor(prev::AbstractArray{Num}, vars::Vector{Num})
    nv = length(vars)
    new = Array{Num}(undef, size(prev)..., nv)
    for I in CartesianIndices(prev)
        for j in 1:nv
            new[I, j] = Symbolics.derivative(prev[I], vars[j])
        end
    end
    return new
end

# ----------------------------------------------------------------------
# Convenience: extract residual as Num for one equation (used by tests)
# ----------------------------------------------------------------------

function equation_residual_symbolic(model::IR.ModelDef, eq_index::Int;
                                    max_hod_order::Int = 1)
    kernels, _ = build_equation_kernels(model; max_hod_order)
    return kernels[eq_index]
end

# ----------------------------------------------------------------------
# Steady-state user equations (v2.1 G4)
#
# An SS equation's residual stores bare variable names (`c`, not `c[t]`).
# Rewrite them to `name[t]` so the existing dynamic-equation kernel
# pipeline accepts them unchanged. Param names and reserved symbols
# (`t`, callables) pass through.
# ----------------------------------------------------------------------

"""
Rewrite an SS residual `Expr` so bare variable / shock references
(`c`, not `c[t]`) become `c[t]`. Param symbols and non-name calls
pass through.
"""
function _rewrite_ss_residual(ex, var_names::Set{Symbol},
                               param_names::Set{Symbol})
    if ex isa Symbol
        if ex in var_names && !(ex in param_names)
            return Expr(:ref, ex, :t)
        end
        return ex
    elseif ex isa Expr
        if ex.head === :ref
            # Already indexed - leave alone (user wrote c[t]).
            return ex
        elseif ex.head === :call
            # Don't rewrite the function name in args[1].
            new_args = Any[ex.args[1]]
            for a in ex.args[2:end]
                push!(new_args, _rewrite_ss_residual(a, var_names, param_names))
            end
            return Expr(:call, new_args...)
        else
            return Expr(ex.head,
                        Any[_rewrite_ss_residual(a, var_names, param_names)
                            for a in ex.args]...)
        end
    else
        return ex
    end
end

"""
Build `EquationKernel`s for every `def.ss_equations` entry. Returns
`(kernels::Vector{EquationKernel}, root_layout)` - the layout matches
the dynamic-equation build so the SS equations can share the same
flat parameter vector at solve time.

The kernel build expects an `EquationAST`-shaped residual that uses
`var[t]` indexing. We rewrite the bare-name SS residual to that form
and synthesize a one-shot `EquationAST` per SS equation so the
existing `_build_one_kernel` path applies unchanged.
"""
function build_ss_equation_kernels(model::IR.ModelDef; max_hod_order::Int = 1)
    Validate.validate(model)
    param_table, root_layout = resolved_link_table(model)
    p_syms = Num[param_table[ref.name] isa Vector{Num} ?
                 param_table[ref.name][ref.index] :
                 param_table[ref.name]
                 for ref in root_layout]
    var_names = Set(v.name for v in model.vars)
    for s in model.shocks
        push!(var_names, s.name)
    end
    param_names = Set(p.name for p in model.params)

    kernels = EquationKernel[]
    for sseq in model.ss_equations
        rewritten = _rewrite_ss_residual(sseq.residual, var_names, param_names)
        rewritten isa Expr || error("SS equation $(sseq.name): residual lowered to non-Expr $rewritten")
        # Synthetic EquationAST so the existing kernel build applies.
        ast = IR.EquationAST(rewritten,
                             Set{IR.EquationFlag}(),
                             sseq.doc,
                             Symbol[Symbol(sseq.kind), :ss_user],
                             sseq.src)
        kernel = _build_one_kernel(model, ast, param_table, root_layout, p_syms,
                                    max_hod_order)
        push!(kernels, kernel)
    end
    return kernels, root_layout
end

end # module Symbolic
