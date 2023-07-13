
"""
Simplest example of model with 1 variable, 1 shock and 1 transition equation.
It shows the boilerplate code for creating models.
"""
module E1_noparams

using ModelBaseEcon

model = Model()
model.flags.linear = true

@variables model y

@shocks model y_shk

@autoexogenize model y = y_shk

@equations model begin
    :maineq => y[t] = 0.5 * y[t - 1] + 0.5 * y[t + 1] + y_shk[t]
end

@initialize model

newmodel() = deepcopy(model)

end

