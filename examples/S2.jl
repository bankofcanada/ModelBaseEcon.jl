"""
Example of @sstate with log variables.
"""
module S2
using ModelBaseEcon

model = Model()
model.flags.linear = false

@parameters model begin
    α = 0.5
    x_ss = 3.1
end

@variables model begin
    y
    @log x
    @shock x_shk
end

@autoexogenize model begin
    x = x_shk
end

@equations model begin
    y[t] = (1 - α) * 2 * @sstate(x) + (α) * @movav(y[t-1], 4)
    log(x[t]) = (1 - α) * log(x_ss) + (α) * @movav(log(x[t-1]), 2) + x_shk[t]
end

@initialize model

end
