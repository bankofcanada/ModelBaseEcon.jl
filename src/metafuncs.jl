
has_t(any) = false
has_t(sym::Symbol) = sym == :t
has_t(expr::Expr) = any(has_t.(expr.args))

normal_ref(var, lag) = Expr(:ref, var, lag == 0 ? :t : lag > 0 ? :(t + $lag) : :(t - $(-lag)))


"""
    @lag(expr[, n=1])

Apply the lag operator to the given expression.
"""
at_lag(any, ::Any...) = any
function at_lag(expr::Expr, n=1)
    if expr.head == :ref 
        var, index = expr.args
        if has_t(index)
            return Expr(:ref, var, :($index - $n))
        end
    end
    return Expr(expr.head, at_lag.(expr.args, n)...)
end

"""
    @lead(expr[, n=1])

Apply the lead operator to the given expression. Equivalent to `@lag(expr, -n)`.

See also [`@lag`](@ref).
"""
at_lead(e::Expr, n::Int=1) = at_lag(e, -n)

"""
    @d(expr)
    @d(expr, n)
    @d(expr, n, s)

Apply the difference operator to the given expression. If `L` represents the lag
operator, then we have the following definitions.
```
@d(x[t]) = (1-L)x = x[t]-x[t-1]
@d(x[t], n) = (1-L)^n x
@d(x[t], n, s) = (1-L)^n (1-L^s) x
```

See also [`@lag`](@ref), [`@dlog`](@ref)
"""
function at_d(expr::Expr, n=1, s=0)
    if s > 0
        expr = :($expr - $(at_lag(expr, s)))
    end
    for i = 1:n
        expr = :($expr - $(at_lag(expr)))
    end
    return expr
end

"""
    @dlog(expr)
    @dlog(expr, n)
    @dlog(expr, n, s)

Apply the difference operator on the log() of the given expression. Equivalent to @d(log(expr), n, s).

See also [`@lag`](@ref), [`@d`](@ref)
"""
at_dlog(expr::Expr, args...) = at_d(:(log($expr)), args...)

"""
    @movsum(expr, n)

Apply moving sum with n periods backwards on the given expression.
For example: `@movsum(x[t], 3) = x[t] + x[t-1] + x[t-2]`.
"""
at_movsum(expr::Expr, n) = Expr(:call, :+, expr, (at_lag(expr, i) for i = 1:n - 1)...)

"""
    @movav(expr, n)

Apply moving average with n periods backwards on the given expression.
For example: `@movav(x[t], 3) = (x[t] + x[t-1] + x[t-2]) / 3`.
"""
at_movav(expr::Expr, n) = :( $(at_movsum(expr, n)) / $n )

for sym in (:lag, :lead, :d, :dlog, :movsum, :movav)
    fsym = Symbol("at_$sym")
    msym = Symbol("@$sym")
    eval(quote
        macro $sym(args...)
            return Meta.quot($fsym(args...))
        end
        export $msym, $fsym
        @doc (@doc $fsym) $msym
    end)
end

