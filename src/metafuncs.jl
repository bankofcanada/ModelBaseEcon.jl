module MetaFuncs

# ----------------------------------------------------------------------
# Equation meta-functions: @lag, @lead, @d, @dlog.
#
# These are *syntactic* operators expanded over the equation `Expr` at
# `@equations` parse time, before the residual is handed to the symbolic
# core. Each rewrites time-indexed sub-expressions into explicit lead/lag
# arithmetic - nothing here survives into the IR as a meta-call.
#
# Semantics are ported verbatim from legacy ModelBaseEcon's
# `src/metafuncs.jl` (`at_lag`, `at_lead`, `at_d`, `at_dlog`). FRBUS_VAR
# uses `@d` and `@dlog` heavily (146 call sites); `@lag`/`@lead` round
# out the set a model author expects alongside them.
#
# The moving-sum/average family (`@movsum`, `@movav`, `@movsumw`, ...) is
# intentionally NOT ported - neither SW07 nor FRBUS_VAR uses it. Add it
# when a model needs it.
# ----------------------------------------------------------------------

using MacroTools: @capture

export expand_metafuncs, METAFUNC_NAMES

# The meta-function names recognised inside `@equations`. A `@<name>`
# macrocall whose name is in this set is expanded; anything else is left
# for the symbolic core to reject as an unknown call.
const METAFUNC_NAMES = (:lag, :lead, :d, :dlog)

# `true` if the expression contains the time symbol `t`.
_has_t(::Any) = false
_has_t(sym::Symbol) = sym === :t
_has_t(expr::Expr) = any(_has_t, expr.args)

# A normalised time index for a given integer lag (negative = lag).
_normal_ref(lag::Int) = lag == 0 ? :t : lag > 0 ? :(t + $lag) : :(t - $(-lag))

"""
    at_lag(expr, n=1)

Apply the lag operator: every `t`-reference inside `expr` is shifted back
`n` periods. Non-`Expr` arguments pass through unchanged.
"""
at_lag(any, ::Any...) = any
function at_lag(expr::Expr, n::Int=1)
    n == 0 && return expr
    if expr.head === :ref
        var, index... = expr.args
        index = collect(index)
        for i in eachindex(index)
            ind = index[i]
            _has_t(ind) || continue
            if @capture(ind, t + lag_)
                index[i] = _normal_ref(lag - n)
            elseif ind === :t
                index[i] = _normal_ref(-n)
            elseif @capture(ind, t - lag_)
                index[i] = _normal_ref(-lag - n)
            else
                error("time index must be `t`, `t+k` or `t-k`, got: $ind")
            end
        end
        return Expr(:ref, var, index...)
    end
    return Expr(expr.head, (at_lag(a, n) for a in expr.args)...)
end

"""
    at_lead(expr, n=1)

Apply the lead operator. Equivalent to `at_lag(expr, -n)`.
"""
at_lead(expr::Expr, n::Int=1) = at_lag(expr, -n)

"""
    at_d(expr, n=1, s=0)

Difference operator. With `L` the lag operator:
`at_d(x) = (1-L)x`, `at_d(x,n) = (1-L)^n x`,
`at_d(x,n,s) = (1-L)^n (1-L^s) x`.
"""
function at_d(expr::Expr, n::Int=1, s::Int=0)
    (n < 0 || s < 0) && error("@d: `n` and `s` must not be negative")
    coefs = zeros(Int, 1 + n + s)
    coefs[1:n+1] .= binomial.(n, 0:n) .* (-1) .^ (0:n)
    if s > 0
        coefs[1+s:end] .-= coefs[1:n+1]
    end
    ret = expr
    for (l, c) in zip(1:n+s, coefs[2:end])
        if c == 0
            continue
        elseif c == 1
            ret = :($ret + $(at_lag(expr, l)))
        elseif c == -1
            ret = :($ret - $(at_lag(expr, l)))
        elseif c > 0
            ret = :($ret + $c * $(at_lag(expr, l)))
        else
            ret = :($ret - $(-c) * $(at_lag(expr, l)))
        end
    end
    return ret
end

"""
    at_dlog(expr, args...)

Difference of `log(expr)`. Equivalent to `at_d(log(expr), args...)`.
"""
at_dlog(expr::Expr, args...) = at_d(:(log($expr)), args...)

const _METAFUNC_IMPL = Dict{Symbol,Function}(
    :lag => at_lag, :lead => at_lead, :d => at_d, :dlog => at_dlog,
)

# Peel a macrocall head down to its bare `Symbol` name (handles GlobalRef
# and `Mod.@name`). Returns `nothing` if not a recognised meta-function.
function _metafunc_name(head)
    sym = head isa GlobalRef ? head.name :
          (head isa Expr && head.head === :(.) && head.args[end] isa QuoteNode) ?
              head.args[end].value : head
    sym isa Symbol || return nothing
    name = Symbol(replace(string(sym), "@" => ""))
    return name in METAFUNC_NAMES ? name : nothing
end

"""
    expand_metafuncs(expr)

Recursively expand every `@lag`/`@lead`/`@d`/`@dlog` macrocall in `expr`.
Inner expressions are expanded first so nesting (e.g. `@d(@dlog(x))`)
resolves bottom-up. Integer literal arguments are required for the
shift/order arguments - they are syntactic, not runtime, quantities.
"""
function expand_metafuncs(expr)
    expr isa Expr || return expr
    # Expand children first.
    args = Any[expand_metafuncs(a) for a in expr.args]
    rebuilt = Expr(expr.head, args...)
    rebuilt.head === :macrocall || return rebuilt
    name = _metafunc_name(rebuilt.args[1])
    name === nothing && return rebuilt
    # macrocall args: [head, LineNumberNode, operand, extra_int_args...]
    call_args = filter(a -> !(a isa LineNumberNode), rebuilt.args[2:end])
    isempty(call_args) && error("@$name requires an expression argument")
    operand = call_args[1]
    operand isa Expr ||
        error("@$name expects an expression operand, got: $operand")
    shifts = Int[]
    for a in call_args[2:end]
        a isa Integer ||
            error("@$name shift/order arguments must be integer literals, got: $a")
        push!(shifts, Int(a))
    end
    return expand_metafuncs(_METAFUNC_IMPL[name](operand, shifts...))
end

end # module MetaFuncs
