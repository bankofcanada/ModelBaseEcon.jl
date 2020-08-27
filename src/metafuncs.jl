
has_t(any) = false
has_t(sym::Symbol) = sym == :t
has_t(expr::Expr) = any(has_t.(expr.args))

normal_ref(var, lag) = Expr(:ref, var, lag == 0 ? :t : lag > 0 ? :(t + $lag) : :(t - $(-lag)))

meta_lag(any, ::Any...) = any
function meta_lag(expr::Expr, n=1)
    if expr.head == :ref 
        var, index = expr.args
        if has_t(index)
            return normal_ref(var, eval(:(let t = 0; $index - $n end)))
        end
    end
    return Expr(expr.head, meta_lag.(expr.args, n)...)
end

meta_d(any, ::Any...) = any
function meta_d(expr::Expr, n=1, s=0)
    if s > 0
        expr = :($expr - $(meta_lag(expr, s)))
    end
    for i = 1:n
        expr = :($expr - $(meta_lag(expr)))
    end
    return expr
end

meta_movsum(any, ::Any...) = any
meta_movsum(expr::Expr, n) = Expr(:call, :+, expr, (meta_lag(expr, i) for i=1:n-1)...)

meta_movav(any, ::Any...) = any
meta_movav(expr::Expr, n) = :( $(meta_movsum(expr, n)) / $n )

for sym in (:d, :lag, :movsum, :movav)
    fsym = Symbol("meta_$sym")
    msym = Symbol("@$sym")
    eval(quote
        macro $sym(args...)
            return Meta.quot($fsym(args...))
        end
        export @msym
    end)
end

