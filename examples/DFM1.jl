module DFM1

using LinearAlgebra  # diagm()

using ModelBaseEcon
using ModelBaseEcon.DFMModels

const model = DFM(:example_dfm1)
add_observed!(model, :a, :b)
add_components!(model, F=CommonComponents("F"))
map_loadings!(model, (:a, :b) => :F)
add_shocks!(model, :a, :b)
initialize_dfm!(model)

function default_parametrization!(model)
    P = model.params

    P.observed.mean .= [2.3, -1.5]
    P.observed.covar = [0.4, 0.9]
    P.observed.loadings.F = [-0.2, 0.9]

    P.F.coefs = [0.8;;;]
    P.F.covar = [1.0;;]
    model
end

function newmodel()
    global Model
    return default_parametrization!(deepcopy(model))
end

end