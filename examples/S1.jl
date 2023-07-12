"""
A simple example of a model with steady state used in the dynamic equations.
"""
module S1

using ModelBaseEcon
model = Model()
model.flags.linear = true

@variables model a b c

@shocks model b_shk c_shk

@parameters model begin
    a_ss = 1.2 
    α = 0.5 
    β = 0.8 
    q = 2
end

@equations model begin
    a[t] = b[t] + c[t]
    b[t] = @sstate(b) * (1 - α) + α * b[t-1] + b_shk[t]
    c[t] = q * @sstate(b) * (1 - β) + β * c[t-1] + c_shk[t]
end

@initialize model

@steadystate model a = a_ss

newmodel() = deepcopy(model)

end
