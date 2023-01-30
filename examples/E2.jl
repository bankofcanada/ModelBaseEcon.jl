
"""
Simple model with 3 variables, 3 shocks and 3 transition equations
"""
module  E2

using ModelBaseEcon

# start with an empty model
model = Model()
model.flags.linear = true

# add parameters
@parameters model begin
    cp = [0.5, 0.02]
    cr = [0.75, 1.5, 0.5]
    cy = [0.5, -0.02]
end

# add variables: a list of symbols
@variables model begin
    pinf
    rate
    ygap
end

# add shocks: a list of symbols
@shocks model begin
    pinf_shk
    rate_shk
    ygap_shk
end

# autoexogenize: define a mapping of variables to shocks
@autoexogenize model begin
    pinf = pinf_shk
    rate = rate_shk
    ygap = ygap_shk
end

# add equations: a sequence of expressions, such that
# use y[t+1] for expectations/leads
# use y[t] for contemporaneous
# use y[t-1] for lags
# each expression must have exactly one "="
@equations model begin
    pinf[t]=cp[1]*pinf[t-1]+(.98-cp[1])*pinf[t+1]+cp[2]*ygap[t]+pinf_shk[t]
    rate[t]=cr[1]*rate[t-1]+(1-cr[1])*(cr[2]*pinf[t]+cr[3]*ygap[t])+rate_shk[t]
    ygap[t]=cy[1]*ygap[t-1]+(.98-cy[1])*ygap[t+1]+cy[2]*(rate[t]-pinf[t+1])+ygap_shk[t]
end

# call initialize! to build internal structures
@initialize model

end