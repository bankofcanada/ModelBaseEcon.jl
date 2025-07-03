module DFM3

"""
Example DFM. The variables and block structure is identical to DFM3MQ, except
that this one is not a mixed-frequency model.
"""
DFM3

using LinearAlgebra  # diagm()

using ModelBaseEcon
using ModelBaseEcon.DFMModels

const model = DFM(:example_dfm3)

"Monthly observable variables"
const M_VARS = (:a, :b, :c)

"Quarterly observable variables"
const Q_VARS = (:y, :z)

"Target (dependent) variable - always quarterly"
const D_VAR = :k

"Factor blocks"
const BLOCKS = (; U=(:a, :b, :y), G=(:c, :z))

add_observed!(model,
    :obsM => ObservedBlock(M_VARS),
    :obsQ => ObservedBlock(Q_VARS),
    :obsQ => :k
)

add_components!(model,
    F=CommonComponents((:U, :G), order=2),
    corM=IdiosyncraticComponents(),
    corQ=IdiosyncraticComponents(),
)

map_loadings!(model,
    (:a, :b, :y, :k) => :U,
    (:c, :z, :k) => :G,
    M_VARS => :corM,
    Q_VARS => :corQ,
    :k => :corQ,
)

add_shocks!(model, M_VARS, Q_VARS, D_VAR)

initialize_dfm!(model)

function default_parametrization!(model)
    P = model.params

    fill!(P, NaN)

    P.obsM.mean .= 0.0
    P.obsM.loadings.F = [0.9, 0.7, 1.1]
    P.obsM.covar .= 0.001

    P.obsQ.mean .= 0.0
    P.obsQ.loadings.F = [-0.8, 0.91, -1.22, -0.55]
    P.obsQ.covar .= 0.001

    NM = length(M_VARS)
    P.corM.coefs .= (0.5 .+ (1:NM) / NM / 5)
    P.corM.covar .= 0.015

    NQ = length(Q_VARS) + 1 # D_VAR is also quarterly
    P.corQ.coefs .= (0.5 .+ (1:NQ) / NQ / 5)
    P.corQ.covar .= 0.015

    P.F.coefs = [1.1 0.01; -0.02 0.9;;; -0.3 0.0; 0.0 0.08]
    P.F.covar = 0.7I(2)

    model
end

function newmodel()
    global Model
    return default_parametrization!(deepcopy(model))
end

end