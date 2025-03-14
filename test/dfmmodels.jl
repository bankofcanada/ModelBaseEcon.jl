##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2025, Bank of Canada
# All rights reserved.
##################################################################################

using Test
using ModelBaseEcon
using ModelBaseEcon.DFMModels
using LinearAlgebra

@testset "dfm.1" begin

    dfm = DFMModel(:test)
    @test dfm isa DFMModel

    @test_throws ErrorException DFMModels.check_dfm(dfm)
    @test_throws ErrorException initialize_dfm!(dfm)

    add_components!(dfm, F=CommonComponents("F"), IC=IdiosyncraticComponents())
    @test length(dfm.components) == 2 && keys(dfm.components) == Set((:F, :IC))

    add_observed!(dfm, (:a, :b, :c))
    @test length(dfm.observed) == 1 && haskey(dfm.observed, :observed)

    add_observed!(dfm, q=(:x, :y, :z))
    @test length(dfm.observed) == 2 && haskey(dfm.observed, :q)

    @test_throws ErrorException add_observed!(dfm, (:w, :h))

    empty!(dfm.observed)
    add_observed!(dfm, (:a, :b, :c))
    obs = first(values(dfm.observed))

    map_loadings!(dfm, (:a, :b) => :F, (:b, :c) => :IC)
    @test length(obs.var2comps) == 3 && keys(obs.var2comps) == Set((:a, :b, :c))
    @test_throws ErrorException initialize_dfm!(dfm)

    add_shocks!(dfm, :b)
    @test length(obs.var2shk) == 1 && haskey(obs.var2shk, :b)
    @test_throws ErrorException initialize_dfm!(dfm)
    @test_throws ErrorException add_shocks!(dfm, :b)

    empty!(obs.var2shk)
    add_shocks!(dfm, :a, :b)
    @test_logs (:warn, r".*more than one.*"i) initialize_dfm!(dfm)

    empty!(obs.var2shk)
    add_shocks!(dfm, :a)
    @test length(obs.var2shk) == 1 && haskey(obs.var2shk, :a)
    @test (initialize_dfm!(dfm); true)

    @test observed(dfm) == [:a, :b, :c]
    @test nobserved(dfm) == 3
    @test states(dfm) == [:F, :b_cor, :c_cor]
    @test nstates(dfm) == 3
    @test shocks(dfm) == [:a_shk, :F_shk, :b_cor_shk, :c_cor_shk]
    @test nshocks(dfm) == 4
    @test varshks(dfm) == [:a, :b, :c, :F, :b_cor, :c_cor, :a_shk, :F_shk, :b_cor_shk, :c_cor_shk]
    @test nvarshks(dfm) == 10
    @test endog(dfm) == [:a, :b, :c, :F, :b_cor, :c_cor]
    @test nendog(dfm) == 6
    @test exog(dfm) == []
    @test nexog(dfm) == 0
    @test leads(dfm) == 0  # always 0 for DFM models
    @test lags(dfm) == 1
    @test all(order.(values(dfm.components)) .== 1)
    @test all(isempty.(observed.(values(dfm.components))))
    @test all(nobserved.(values(dfm.components)) .== 0)
    @test all(isempty.(states.(values(dfm.observed))))
    @test all(nstates.(values(dfm.observed)) .== 0)

    add_observed!(dfm, :z)
    @test_throws ErrorException initialize_dfm!(dfm)

end

## 

@testset "dfm.2" begin
# begin
    dfm = DFM(:test)
    @test dfm isa DFM && dfm.model isa DFMModel && dfm.params isa DFMParams

    add_components!(dfm, C1=CommonComponents("A", 2, order=2))
    add_components!(dfm, C2=CommonComponents(["U", "V"], order=3))
    add_components!(dfm, IC=IdiosyncraticComponents(order=1))
    add_observed!(dfm, O1=(:x, :y), O2=(:z,))

    @test_throws ErrorException initialize_dfm!(dfm)
    @test_throws ErrorException map_loadings!(dfm, :x => :NOSUCHBLOCK)

    map_loadings!(dfm,
        (:x, :z) => (:A¹, :IC),
        :x => (:U,),
        (:z,) => :C2,
        :y => :C1,
    )

    add_shocks!(dfm, :y)
    initialize_dfm!(dfm)

    O1 = dfm.model.observed[:O1]
    O2 = dfm.model.observed[:O2]
    C1 = dfm.model.components[:C1]
    C2 = dfm.model.components[:C2]
    IC = dfm.model.components[:IC]

    @test endog(C1) == [:A¹, :A²]
    @test nendog(C1) == 2
    @test exog(C1) == []
    @test nexog(C1) == 0
    @test shocks(C1) == [:A¹_shk, :A²_shk]
    @test nshocks(C1) == 2
    @test varshks(C1) == [:A¹, :A², :A¹_shk, :A²_shk]
    @test nvarshks(C1) == 4
    @test leads(C1) == 0
    @test lags(C1) == 2

    @test endog(C2) == [:U, :V]
    @test nendog(C2) == 2
    @test exog(C2) == []
    @test nexog(C2) == 0
    @test shocks(C2) == [:U_shk, :V_shk]
    @test nshocks(C2) == 2
    @test varshks(C2) == [:U, :V, :U_shk, :V_shk]
    @test nvarshks(C2) == 4
    @test leads(C2) == 0
    @test lags(C2) == 3

    @test endog(IC) == [:x_cor, :z_cor]
    @test nendog(IC) == 2
    @test exog(IC) == []
    @test nexog(IC) == 0
    @test shocks(IC) == [:x_cor_shk, :z_cor_shk]
    @test nshocks(IC) == 2
    @test varshks(IC) == [:x_cor, :z_cor, :x_cor_shk, :z_cor_shk]
    @test nvarshks(IC) == 4
    @test leads(IC) == 0
    @test lags(IC) == 1

    @test endog(O1) == [:x, :y]
    @test nendog(O1) == 2
    @test exog(O1) == [:A¹, :A², :x_cor, :U]
    @test nexog(O1) == 4
    @test shocks(O1) == [:y_shk]
    @test nshocks(O1) == 1
    @test varshks(O1) == [:x, :y, :A¹, :A², :x_cor, :U, :y_shk]
    @test nvarshks(O1) == 7
    @test lags(O1) == leads(O1) == 0

    @test endog(O2) == [:z]
    @test nendog(O2) == 1
    @test exog(O2) == [:A¹, :z_cor, :U, :V]
    @test nexog(O2) == 4
    @test shocks(O2) == []
    @test nshocks(O2) == 0
    @test varshks(O2) == [:z, :A¹, :z_cor, :U, :V]
    @test nvarshks(O2) == 5
    @test lags(O2) == leads(O2) == 0

    @test states_with_lags(O1) == []
    @test nstates_with_lags(O1) == 0
    @test states_with_lags(O2) == []
    @test nstates_with_lags(O2) == 0

    @test states_with_lags(C1) == [:A¹ₜ₋₁, :A²ₜ₋₁, :A¹, :A²]
    @test nstates_with_lags(C1) == 4
    @test states_with_lags(C2) == [:Uₜ₋₂, :Vₜ₋₂, :Uₜ₋₁, :Vₜ₋₁, :U, :V]
    @test nstates_with_lags(C2) == 6
    @test states_with_lags(IC) == [:x_cor, :z_cor]
    @test nstates_with_lags(IC) == 2

    @test states_with_lags(dfm) == [:A¹ₜ₋₁, :A²ₜ₋₁, :A¹, :A², :Uₜ₋₂, :Vₜ₋₂, :Uₜ₋₁, :Vₜ₋₁, :U, :V, :x_cor, :z_cor]
    @test nstates_with_lags(dfm) == 12

    params = dfm.params
    copyto!(params, 1:length(params))

    @test get_covariance(O1, params.O1) == Float64[7;;]
    @test get_covariance(O2, params.O2) == Float64[;;]
    @test get_covariance(C1, params.C1) == Float64[20 22;22 23;]
    @test get_covariance(C2, params.C2) == Float64[36 38;38 39;]
    @test get_covariance(IC, params.IC) == diagm(Float64[42,43])
    let 
        C = zeros(7,7)
        C[1,1] = 7
        C[2:3,2:3] = [20 22;22 23;]
        C[4:5,4:5] = [36 38;38 39;]
        C[6,6] = 42
        C[7,7] = 43
        @test get_covariance(dfm) == C
    end

    @test DFMModels.get_mean!(zeros(2), O1, params.O1) == Float64[1,2]
    @test DFMModels.get_mean!(zeros(1), O2, params.O2) == Float64[8]
    @test DFMModels.get_mean!(zeros(3), dfm) == Float64[1,2,8]
    @test DFMModels.get_mean(dfm) == Float64[1,2,8]

    
    @test DFMModels.get_loading!(zeros(2,4), O1, params.O1, :C1=>C1) == [zeros(2,2) [3 0; 4 5]]
    @test DFMModels.get_loading!(zeros(2,6), O1, params.O1, :C2=>C2) == [zeros(2,4) [6 0; 0 0]]
    @test DFMModels.get_loading!(zeros(2,2), O1, params.O1, :IC=>IC) == [1 0; 0 0]
    @test DFMModels.get_loading!(zeros(2,12), O1, params.O1) == [zeros(2,2) [3 0; 4 5] [1 0; 0 0] zeros(2,4) [6 0; 0 0]]

    @test DFMModels.get_loading!(zeros(1,4), O2, params.O2, :C1=>C1) == [zeros(1,2) [9 0]]
    @test DFMModels.get_loading!(zeros(1,6), O2, params.O2, :C2=>C2) == [zeros(1,4) [10 11]]
    @test DFMModels.get_loading!(zeros(1,2), O2, params.O2, :IC=>IC) == [0 1]
    @test DFMModels.get_loading!(zeros(1,12), O2, params.O2) == [zeros(1,2) [9 0] [0 1] zeros(1,4) [10 11]]

    @test DFMModels.get_loading!(zeros(3,12), dfm) == [zeros(3,2) [3 0; 4 5; 9 0] zeros(3,4) [6 0; 0 0; 10 11] [1 0; 0 0; 0 1]]
    @test DFMModels.get_loading(dfm) == [zeros(3,2) [3 0; 4 5; 9 0] zeros(3,4) [6 0; 0 0; 10 11] [1 0; 0 0; 0 1]]

end

##

