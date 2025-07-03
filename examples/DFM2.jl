module DFM2

using LinearAlgebra  # diagm()

using ModelBaseEcon
using ModelBaseEcon.DFMModels

const model = DFM(:example_dfm2)

add_components!(model,
    F=CommonComponents("F", order=2),
    G=CommonComponents("G", order=2),
    ic=IdiosyncraticComponents()
)

map_loadings!(model,
    (:a, :b) => :F,
    (:a, :c, :d) => :G,
    (:c, :d) => :ic
)

add_shocks!(model, :a, :b, :c, :d)

initialize_dfm!(model)

function default_parametrization!(model)
    P = model.params

    fill!(P, NaN)

    P.observed.mean .= [2.3, -1.5, 1.2, 0]
    P.observed.covar = [1, 1, 1, 1] ./ 1e7
    P.observed.loadings.F = [-0.2, 0.9]
    P.observed.loadings.G = [1.1, 0.6, -0.3]

    P.F.coefs = [1.1;;; -0.3]
    P.G.coefs = [0.45;;; 0.1]
    P.ic.coefs = [0.85; 0.3]

    P.F.covar = diagm([1.0])
    P.G.covar = diagm([1.0])
    P.ic.covar = [0.5, 0.1]

    model
end

function newmodel()
    global Model
    return default_parametrization!(deepcopy(model))
end

end