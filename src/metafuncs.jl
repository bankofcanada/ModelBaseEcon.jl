##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

# return `true` if expression contains t
has_t(any) = false
has_t(first, many...) = has_t(first) || has_t(many...)
has_t(sym::Symbol) = sym == :t
has_t(expr::Expr) = has_t(expr.args...)

# normalized :ref expression
normal_ref(var, lag) = Expr(:ref, var, lag == 0 ? :t : lag > 0 ? :(t + $lag) : :(t - $(-lag)))

"""
    at_lag(expr[, n=1])

Apply the lag operator to the given expression.
"""
at_lag(any, ::Any...) = any
function at_lag(expr::Expr, n=1)
    if n == 0
        return expr
    elseif expr.head == :ref
        var, index = expr.args
        if @capture(index, t + lag_) && lag isa Number
            return normal_ref(var, lag - n)
        elseif @capture(index, t - lag_) && lag isa Number
            return normal_ref(var, -lag - n)
        else
            return Expr(:ref, var, :($index - $n))
        end
    else
        return Expr(expr.head, at_lag.(expr.args, n)...)
    end
end

"""
    at_lead(expr[, n=1])

Apply the lead operator to the given expression. Equivalent to
`at_lag(expr, -n)`.
"""
at_lead(e::Expr, n::Int=1) = at_lag(e, -n)

"""
    at_d(expr[, n=1 [, s=0 ]])

Apply the difference operator to the given expression. If `L` represents the lag
operator, then we have the following definitions.
```
at_d(x[t]) = (1-L)x = x[t]-x[t-1]
at_d(x[t], n) = (1-L)^n x
at_d(x[t], n, s) = (1-L)^n (1-L^s) x
```

See also [`at_lag`](@ref), [`at_d`](@ref).
"""
function at_d(expr::Expr, n=1, s=0)
    if n < 0 || s < 0
        error("In @d call `n` and `s` must not be negative.")
    end
    coefs = zeros(Int, 1 + n + s)
    coefs[1:n+1] .= binomial.(n, 0:n) .* (-1).^(0:n)
    if s > 0
        coefs[1+s:end] .-= coefs[1:n+1]
    end
    ret = expr
    for (l, c) in zip(1:n+s, coefs[2:end])
        if abs(c) < 1e-12
            continue
        elseif isapprox(c, 1)
            ret = :($ret + $(at_lag(expr, l)))
        elseif isapprox(c, -1)
            ret = :($ret - $(at_lag(expr, l)))
        elseif c > 0
            ret = :($ret + $c * $(at_lag(expr, l)))
        else
            ret = :($ret - $(-c) * $(at_lag(expr, l)))
        end
    end
    return ret
    #### old implementation
    # if s > 0
    #     expr = :($expr - $(at_lag(expr, s)))
    # end
    # for i = 1:n
    #     expr = :($expr - $(at_lag(expr)))
    # end
    # return expr
end

"""
    at_dlog(expr[, n=1 [, s=0 ]])

Apply the difference operator on the log() of the given expression. Equivalent to at_d(log(expr), n, s).

See also [`at_lag`](@ref), [`at_d`](@ref)
"""
at_dlog(expr::Expr, args...) = at_d(:(log($expr)), args...)

"""
    at_movsum(expr, n)

Apply moving sum with n periods backwards on the given expression.
For example: `at_movsum(x[t], 3) = x[t] + x[t-1] + x[t-2]`.

See also [`at_lag`](@ref).
"""
at_movsum(expr::Expr, n::Integer) = MacroTools.unblock(
    split_nargs(Expr(:call, :+, expr, (at_lag(expr, i) for i = 1:n-1)...))
)

"""
    at_movav(expr, n)

Apply moving average with n periods backwards on the given expression.
For example: `at_movav(x[t], 3) = (x[t] + x[t-1] + x[t-2]) / 3`.

See also [`at_lag`](@ref).
"""
at_movav(expr::Expr, n::Integer) = MacroTools.unblock(:($(at_movsum(expr, n)) / $n))

"""
    at_movsumew(expr, n, r)

Apply moving sum with exponential weights with ratio `r`.
For example: `at_movsumew(x[t], 3, 0.7) = x[t] + 0.7*x[t-1] + 0.7^2x[t-2]`

See also [`at_movavew`](@ref)
"""
at_movsumew(expr::Expr, n::Integer, r) =
    MacroTools.unblock(split_nargs(Expr(:call, :+, expr, (Expr(:call, :*, :(($r)^($i)), at_lag(expr, i)) for i = 1:n-1)...)))
at_movsumew(expr::Expr, n::Integer, r::Real) =
    isapprox(r, 1.0) ? at_movsum(expr, n) :
    MacroTools.unblock(split_nargs(Expr(:call, :+, expr, (Expr(:call, :*, r^i, at_lag(expr, i)) for i = 1:n-1)...)))

"""
    at_movavew(expr, n, r)

Apply moving average with exponential weights with ratio `r`.
For example: `at_moveavew(x[t], 3, 0.7) = (x[t] + 0.7*x[t-1] + 0.7^2x[t-2]) / (1 + 0.7 + 0.7^2)`

See also [`at_movsumew`](@ref)
"""
at_movavew(expr::Expr, n::Integer, r::Real) =
    isapprox(r, 1.0) ? at_movav(expr, n) : begin
        s = (1 - r^n) / (1 - r)
        MacroTools.unblock(:($(at_movsumew(expr, n, r)) / $s))
    end
at_movavew(expr::Expr, n::Integer, r) =
    MacroTools.unblock(:($(at_movsumew(expr, n, r)) * (1 - $r) / (1 - $r^$n)))    #=  isapprox($r, 1.0) ? $(at_movav(expr, n)) :  =#

for sym in (:lag, :lead, :d, :dlog, :movsum, :movav, :movsumew, :movavew)
    fsym = Symbol("at_$sym")
    msym = Symbol("@$sym")
    doc_str = replace(string(eval(:(@doc $fsym))), "at_" => "@")
    qq = quote
        @doc $(doc_str) macro $sym(args...)
            return Meta.quot($fsym(args...))
        end
        export $msym
    end
    eval(qq)
end
