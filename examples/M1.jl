
"""
Simplest example of model with 1 variable, 1 shock and 1 transition equation.
It shows the boilerplate code for creating models.
"""
module M1

using ModelBaseEcon

model = Model()

@parameters model begin
    α = 0.5
    β = 0.5
end

@variables model y

@shocks model y_shk

@autoexogenize model y = y_shk

@equations model begin
    y[t] = α * y[t - 1] + β * y[t + 1] + y_shk[t]
end

@initialize model

end

