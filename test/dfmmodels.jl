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
using SparseArrays

##

@testset "dfm.internals" begin
    @test_throws r"Number of names .* must match factor size .*\."i DFMModels.ComponentsBlock{:Dense,DFMModels.MixFreq{:MQ}}(["A", "B"], 3, 1, 1)

    # we can change the default, but `:observed` is already hardcoded in too many places...
    @test DFMModels._default_ObservedBlock_name == :observed

    # empty _BlockRef() refers to all components, but doesn't store the names, so anything goes
    R = DFMModels._BlockRef()
    @test DFMModels.comp_ref(R) == R
    @test DFMModels.comp_ref(R, Val(:A)) == R
    @test_throws "Cannot determine" DFMModels.n_comp_refs(R)
    @test_throws "Cannot determine" DFMModels.inds_comp_refs(R)
    @test_throws "Cannot determine" DFMModels.vars_comp_refs(R)

    # non-empty _BlockRef() must not refer to invalid names
    R1 = DFMModels._BlockRef((:B,))
    @test_throws r".* is not a component." DFMModels.comp_ref(R1, Val(:A))

    # 
    @test DFMModels.mf_ncoefs(ObservedBlock()) == 1
    @test DFMModels.mf_coefs(ObservedBlock()) == (1,)

end

##

@testset "dfm.0" begin



    dfm = DFMModel(:simple)
    add_observed!(dfm, :x)
    add_components!(dfm, :factors => CommonComponents("F"))
    map_loadings!(dfm, :observed => :F)
    add_shocks!(dfm, :observed)

    @test keys(dfm.observed) == Set([:observed])
    @test dfm.observed[:observed].vars == []
    @test dfm.observed[:observed].shks == []
    @test dfm.observed[:observed].size == 0
    @test keys(dfm.observed[:observed].components) == Set([:factors])
    @test keys(dfm.observed[:observed].var2comps) == Set([:x])
    @test keys(dfm.observed[:observed].var2shk) == Set([:x])
    @test keys(dfm.observed[:observed].comp2vars) == Set()

    @test keys(dfm.components) == Set([:factors])
    @test dfm.components[:factors].vars == [:F]
    @test dfm.components[:factors].shks == [:F_shk]
    @test dfm.components[:factors].size == 1
    @test dfm.components[:factors].order == 1
    @test dfm.components[:factors].nlags == 1

    initialize_dfm!(dfm)

    @test keys(dfm.observed) == Set([:observed])
    @test dfm.observed[:observed].vars == [:x]
    @test dfm.observed[:observed].shks == [:x_shk]
    @test dfm.observed[:observed].size == 1
    @test keys(dfm.observed[:observed].components) == Set([:factors])
    @test keys(dfm.observed[:observed].var2comps) == Set([:x])
    @test keys(dfm.observed[:observed].var2shk) == Set([:x])
    @test keys(dfm.observed[:observed].comp2vars) == Set([:factors])

    @test keys(dfm.components) == Set([:factors])
    @test dfm.components[:factors].vars == [:F]
    @test dfm.components[:factors].shks == [:F_shk]
    @test dfm.components[:factors].size == 1
    @test dfm.components[:factors].order == 1
    @test dfm.components[:factors].nlags == 1

    params = init_params(dfm)
    params .= 1:length(params)

    @test DFMModels.get_mean(dfm, params) == [1]
    @test (DFMModels.set_mean!(params, dfm, [200]); params.observed.mean[:] == [200])

end

## 

@testset "dfm.1" begin

    dfm = DFMModel(:test)
    @test dfm isa DFMModel

    @test_throws "Model must be initialized" DFMModels.check_dfm(dfm)
    @test_throws "Model does not have any observed variables" initialize_dfm!(dfm)

    add_components!(dfm, F=CommonComponents("F"), IC=IdiosyncraticComponents())
    @test length(dfm.components) == 2 && keys(dfm.components) == Set((:F, :IC))

    add_observed!(dfm, (:a, :b, :c))
    @test length(dfm.observed) == 1 && haskey(dfm.observed, :observed)

    add_observed!(dfm, q=(:x, :y, :z))
    @test length(dfm.observed) == 2 && haskey(dfm.observed, :q)

    @test_throws "block must be explicitly given" add_observed!(dfm, (:w, :h))
    @test_throws r"Variable .* not found" add_shocks!(dfm, (:w, :h))

    empty!(dfm.observed)
    @test isempty(dfm.observed)
    add_shocks!(dfm, :a, :b, :c)
    @test length(dfm.observed) == 1 && haskey(dfm.observed, :observed)

    empty!(dfm.observed)
    add_observed!(dfm, (:a, :b, :c))
    obs = first(values(dfm.observed))

    map_loadings!(dfm, (:a, :b) => :F, (:b, :c) => :IC)
    @test length(obs.var2comps) == 3 && keys(obs.var2comps) == Set((:a, :b, :c))
    @test_throws "neither a shock nor an idiosyncratic" initialize_dfm!(dfm)

    add_shocks!(dfm, :b)
    @test length(obs.var2shk) == 1 && haskey(obs.var2shk, :b)
    @test_throws ErrorException initialize_dfm!(dfm)
    # @test_throws ErrorException add_shocks!(dfm, :b)
    @test_logs (:warn, r".*`b`.*has a shock") add_shocks!(dfm, :b)

    empty!(obs.var2shk)
    add_shocks!(dfm, :a, :b)
    # @test_logs (:warn, r".*more than one.*"i) initialize_dfm!(dfm)
    @test_nowarn initialize_dfm!(dfm)

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

    @test (map_loadings!(dfm, (:observed) => :F); true)

    add_observed!(dfm, :obs2 => (:a,))
    @test_throws r"duplicate variable"i initialize_dfm!(dfm)

    delete!(dfm.observed, :obs2)
    add_observed!(dfm, :z)
    @test_throws "does not load any component and does not have a shock" initialize_dfm!(dfm)

    dfm = DFMModel(:test)
    add_components!(dfm, IC=IdiosyncraticComponents())
    # test that we can add observed variables without explicitly creating any ObservedBlock
    @test (map_loadings!(dfm, :a => :IC); true)
    # test that we can map the entire default observed block when it is the only one
    @test (map_loadings!(dfm, (:observed,) => :IC); true)
    # test that we can map an entire observed block when there's more than one
    add_observed!(dfm, obs2=(:b))
    @test (map_loadings!(dfm, (:obs2,) => :IC); true)
    @test_throws r"variables not assigned to an observed block"i map_loadings!(dfm, :c => :IC)
    #  initialize_dfm!(dfm)

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

    @test exog(dfm) == []
    @test nexog(dfm) == 0
    @test observed(dfm) == [:x, :y, :z]
    @test nobserved(dfm) == 3
    @test states(dfm) == [:A¹, :A², :U, :V, :x_cor, :z_cor]
    @test nstates(dfm) == 2 + 2 + 2
    @test states_with_lags(dfm) == [:A¹ₜ₋₁, :A²ₜ₋₁, :A¹, :A², :Uₜ₋₂, :Vₜ₋₂, :Uₜ₋₁, :Vₜ₋₁, :U, :V, :x_cor, :z_cor]
    @test nstates_with_lags(dfm) == 2 * 2 + 3 * 2 + 2
    @test leads(dfm) == 0
    @test lags(dfm) == 3

    params = dfm.params
    copyto!(params, 1:length(params))

    @test get_covariance(O1, params.O1) == Float64[7;;]
    @test get_covariance(O2, params.O2) == Float64[;;]
    @test get_covariance(C1, params.C1) == Float64[20 22; 22 23]
    @test get_covariance(C2, params.C2) == Float64[36 38; 38 39]
    @test get_covariance(IC, params.IC) == diagm(Float64[42, 43])
    begin
        C = [7 0 0 0 0 0 0
            0 20 22 0 0 0 0
            0 22 23 0 0 0 0
            0 0 0 36 38 0 0
            0 0 0 38 39 0 0
            0 0 0 0 0 42 0
            0 0 0 0 0 0 43]
        @test get_covariance(dfm) == C
        @test get_covariance(dfm, :O1) == [7;;]
        @test get_covariance(dfm, :O2) == [;;]
        @test get_covariance(dfm, :C1) == [20 22; 22 23]
        @test get_covariance(dfm, :C2) == [36 38; 38 39]
        @test get_covariance(dfm, :IC) == [42 0; 0 43]
        #
        Cobs = zeros(3, 3)
        Cobs[2, 2] = C[1, 1]
        Csts = zeros(12, 12)
        Csts[[3, 4, 9, 10, 11, 12], [3, 4, 9, 10, 11, 12]] = C[2:end, 2:end]
        @test get_covariance(dfm, Val(:Observed)) == Cobs
        @test get_covariance(dfm, Val(:State)) == Csts
    end

    @test DFMModels.get_mean!(zeros(2), O1, params.O1) == Float64[1, 2]
    @test DFMModels.get_mean!(zeros(1), O2, params.O2) == Float64[8]
    @test DFMModels.get_mean!(zeros(3), dfm) == Float64[1, 2, 8]
    @test DFMModels.get_mean(dfm) == Float64[1, 2, 8]


    @test DFMModels.get_loading!(zeros(2, 4), O1, params.O1, :C1 => C1) == [zeros(2, 2) [3 0; 4 5]]
    @test DFMModels.get_loading!(zeros(2, 6), O1, params.O1, :C2 => C2) == [zeros(2, 4) [6 0; 0 0]]
    @test DFMModels.get_loading!(zeros(2, 2), O1, params.O1, :IC => IC) == [1 0; 0 0]
    @test DFMModels.get_loading!(zeros(2, 12), O1, params.O1) == [zeros(2, 2) [3 0; 4 5] [1 0; 0 0] zeros(2, 4) [6 0; 0 0]]

    @test DFMModels.get_loading!(zeros(1, 4), O2, params.O2, :C1 => C1) == [zeros(1, 2) [9 0]]
    @test DFMModels.get_loading!(zeros(1, 6), O2, params.O2, :C2 => C2) == [zeros(1, 4) [10 11]]
    @test DFMModels.get_loading!(zeros(1, 2), O2, params.O2, :IC => IC) == [0 1]
    @test DFMModels.get_loading!(zeros(1, 12), O2, params.O2) == [zeros(1, 2) [9 0] [0 1] zeros(1, 4) [10 11]]

    @test DFMModels.get_loading!(zeros(3, 12), dfm) == [zeros(3, 2) [3 0; 4 5; 9 0] zeros(3, 4) [6 0; 0 0; 10 11] [1 0; 0 0; 0 1]]
    @test DFMModels.get_loading(dfm) == [zeros(3, 2) [3 0; 4 5; 9 0] zeros(3, 4) [6 0; 0 0; 10 11] [1 0; 0 0; 0 1]]

    begin
        T_C1 = [0 0 1 0; 0 0 0 1; 16 18 12 14; 17 19 13 15]
        T_C2 = [0 0 1 0 0 0; 0 0 0 1 0 0; 0 0 0 0 1 0; 0 0 0 0 0 1; 32 34 28 30 24 26; 33 35 29 31 25 27]
        T_IC = [40 0; 0 41]
        T_full = [T_C1 zeros(4, 8); zeros(6, 4) T_C2 zeros(6, 2); zeros(2, 10) T_IC]
        @test DFMModels.get_transition!(zeros(4, 4), C1, params.C1) == T_C1
        @test DFMModels.get_transition!(zeros(6, 6), C2, params.C2) == T_C2
        @test DFMModels.get_transition!(zeros(2, 2), IC, params.IC) == T_IC
        @test DFMModels.get_transition(dfm) == T_full
    end

    @test (1 + lags(C1) == 3) && (nvarshks(C1) == 4)
    @test DFMModels.eval_resid(ones(3, 4), C1, params.C1) == [-60; -64]
    @test begin
        R, J = DFMModels.eval_RJ(ones(3, 4), C1, params.C1)
        (R == [-60; -64]) &&
            (J == [-16 -12 1 -18 -14 0 0 0 -1 0 0 0; -17 -13 0 -19 -15 1 0 0 0 0 0 -1])
    end

    @test (1 + lags(C2) == 4) && (nvarshks(C2) == 4)
    @test DFMModels.eval_resid(ones(4, 4), C2, params.C2) == [-174; -180]
    @test begin
        R, J = DFMModels.eval_RJ(ones(4, 4), C2, params.C2)
        (R == [-174; -180]) &&
            (J == [-32.0 -28.0 -24.0 1.0 -34.0 -30.0 -26.0 0.0 0.0 0.0 0.0 -1.0 0.0 0.0 0.0 0.0;
                -33.0 -29.0 -25.0 0.0 -35.0 -31.0 -27.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -1.0])
    end

    @test (1 + lags(IC) == 2) && (nvarshks(IC) == 4)
    @test DFMModels.eval_resid(ones(2, 4), IC, params.IC) == [-40; -41]
    @test begin
        R, J = DFMModels.eval_RJ(ones(2, 4), IC, params.IC)
        (R == [-40; -41]) && (J == [-40 1 0 0 0 -1 0 0; 0 0 -41 1 0 0 0 -1])
    end

    @test (1 + lags(O1) == 1) && (nvarshks(O1) == 7)
    @test DFMModels.eval_resid(ones(1, 7), O1, params.O1) == [-10; -11]
    @test begin
        R, J = DFMModels.eval_RJ(ones(1, 7), O1, params.O1)
        (R == [-10; -11]) && (J == [1 0 -3 0 -1 -6 0; 0 1 -4 -5 0 0 -1])
    end

    @test (1 + lags(O2) == 1) && (nvarshks(O2) == 5)
    @test DFMModels.eval_resid(ones(1, 5), O2, params.O2) == [-38]
    @test begin
        R, J = DFMModels.eval_RJ(ones(1, 5), O2, params.O2)
        (R == [-38]) && (J == [1 -9 -1 -10 -11])
    end

    let CR = zeros(9), CJ = zeros(9, 64)
        CR[:] = [-10, -11, -38, -60, -64, -174, -180, -40, -41]
        CJ[1, [4, 16, 24, 32]] = [1 -3 -6 -1]
        CJ[2, [8, 16, 20, 40]] = [1 -4 -5 -1]
        CJ[3, [12, 16, 24, 28, 36]] = [1 -9 -10 -11 -1]
        CJ[4, [13:20..., 44]] = [0 -16 -12 1 0 -18 -14 0 -1]
        CJ[5, [13:20..., 48]] = [0 -17 -13 0 0 -19 -15 1 -1]
        CJ[6, [21:28..., 52]] = [-32 -28 -24 1 -34 -30 -26 0 -1]
        CJ[7, [21:28..., 56]] = [-33 -29 -25 0 -35 -31 -27 1 -1]
        CJ[8, [31, 32, 60]] = [-40 1 -1]
        CJ[9, [35, 36, 64]] = [-41 1 -1]
        @test (1 + lags(dfm) == 4) && (nvarshks(dfm) == 16)
        @test DFMModels.eval_resid(ones(4, 16), dfm) == CR
        @test begin
            R, J = DFMModels.eval_RJ(ones(4, 16), dfm)
            (R == CR) && (J == CJ)
        end
        @test DFMModels.eval_R!(similar(CR), ones(4, 16), dfm) == CR
        @test begin
            R, J = DFMModels.eval_RJ!(similar(CR), similar(CJ), ones(4, 16), dfm)
            (R == CR) && (J == CJ)
        end
    end


    pp = deepcopy(dfm.params)
    fill!(dfm.params, NaN)

    DFMModels.set_mean!(dfm, [1, 2, 8])
    @test DFMModels.get_mean(dfm) == Float64[1, 2, 8]

    DFMModels.set_loading!(dfm, [zeros(3, 2) [3 0; 4 5; 9 0] zeros(3, 4) [6 0; 0 0; 10 11] [1 0; 0 0; 0 1]])
    @test DFMModels.get_loading(dfm) == Float64[zeros(3, 2) [3 0; 4 5; 9 0] zeros(3, 4) [6 0; 0 0; 10 11] [1 0; 0 0; 0 1]]

    DFMModels.set_transition!(dfm, T_full)
    @test dfm.params.C1.coefs == reshape(12:19, 2, 2, 2)
    @test dfm.params.C2.coefs == reshape(24:35, 2, 2, 3)
    @test dfm.params.IC.coefs == [40; 41;;]

    @test_throws AssertionError DFMModels.set_covariance!(dfm, Csts, Val(:Observed))
    DFMModels.set_covariance!(dfm, Cobs, Val(:Observed))
    @test dfm.params.O1.covar == [7]
    @test dfm.params.O2.covar == []

    @test_throws AssertionError DFMModels.set_covariance!(dfm, Cobs, Val(:State))
    DFMModels.set_covariance!(dfm, Csts, Val(:State))
    @test dfm.params.C1.covar == [20 22; 22 23]
    @test dfm.params.C2.covar == [36 38; 38 39]
    @test dfm.params.IC.covar == [42, 43]


    params = dfm.params
    params .= NaN
    params.O1.loadings.C1[2] = 1.1
    Λ = DFMModels.get_loading!(zeros(2, 12), O1, params.O1)
    lc = DFMModels.loading_constraint(Λ, O1)
    @test lc.blk === O1
    @test findall(lc.estimcols) == [3, 4, 11, 12]
    @test all(lc.estimcols .!= lc.fixedcols)
    @test length(lc.estimblocks) == 2 && haskey(lc.estimblocks, :C1) && haskey(lc.estimblocks, :C2)
    lc1 = lc.estimblocks[:C1]
    @test size(lc1.W) == (2, 4) && lc1.W == [0 1 0 0; 0 0 1 0]
    @test size(lc1.q) == (2,) && lc1.q == [1.1, 0]
    lc1 = lc.estimblocks[:C2]
    @test size(lc1.W) == (3, 4) && lc1.W == [[0; 0; 0] I]
    @test size(lc1.q) == (3,) && all(iszero, lc1.q)

end

##

@testset "dfm.3.mq" begin

    # mixed frequency example
    dfm = DFM(:test_mq)

    add_observed!(dfm,
        :OM => ObservedBlock(:a, :b, :c),             # monthly observed variables
        :OQ => ObservedBlock(MixFreq{:MQ}, :y, :z),   # quarterly observed variables
        :OQ => :k,       # add another quarterly variable to existing block
    )

    add_components!(dfm,
        :F => CommonComponents(MixFreq{:MQ}, [:U, :G], order=2),    # factor with two blocks - U and G
        :CM => IdiosyncraticComponents(),               # auto-correlated noise in monthly variables
        :CQ => IdiosyncraticComponents(MixFreq{:MQ}),   # auto-correlated noise in quarterly variables
    )

    map_loadings!(dfm,
        (:a, :b, :y) => :U,
        (:c, :z) => :G,
        :k => :F,
        :OM => :CM,
        :OQ => :CQ,
    )

    add_shocks!(dfm)
    initialize_dfm!(dfm)

    @test observed(dfm) == [:a, :b, :c, :y, :z, :k]
    @test nobserved(dfm) == 6
    @test states(dfm) == [:U, :G, :a_cor, :b_cor, :c_cor, :y_cor, :z_cor, :k_cor]
    @test nstates(dfm) == 8  # 2 + 3 + 3
    @test states_with_lags(dfm) == [:Uₜ₋₄, :Gₜ₋₄, :Uₜ₋₃, :Gₜ₋₃, :Uₜ₋₂, :Gₜ₋₂, :Uₜ₋₁, :Gₜ₋₁, :U, :G, :a_cor, :b_cor, :c_cor, :y_corₜ₋₄, :z_corₜ₋₄, :k_corₜ₋₄, :y_corₜ₋₃, :z_corₜ₋₃, :k_corₜ₋₃, :y_corₜ₋₂, :z_corₜ₋₂, :k_corₜ₋₂, :y_corₜ₋₁, :z_corₜ₋₁, :k_corₜ₋₁, :y_cor, :z_cor, :k_cor]
    @test nstates_with_lags(dfm) == 28  # 2*5 + 3 + 3*5
    @test shocks(dfm) == [:a_shk, :b_shk, :c_shk, :y_shk, :z_shk, :k_shk, :U_shk, :G_shk, :a_cor_shk, :b_cor_shk, :c_cor_shk, :y_cor_shk, :z_cor_shk, :k_cor_shk]
    @test nshocks(dfm) == 14
    @test exog(dfm) == []
    @test nexog(dfm) == 0

    OM = dfm.model.observed[:OM]
    OQ = dfm.model.observed[:OQ]
    F = dfm.model.components[:F]
    CM = dfm.model.components[:CM]
    CQ = dfm.model.components[:CQ]

    # make sure the lags are correct according to mixed frequency
    @test lags(OM) == 0
    @test lags(OQ) == 4
    @test lags(F) == 5
    @test lags(CM) == 1
    @test lags(CQ) == 5
    @test leads(OM) == 0
    @test leads(OQ) == 0
    @test leads(F) == 0
    @test leads(CM) == 0
    @test leads(CQ) == 0

    params = dfm.params
    copyto!(params, 1:length(params))

    @test DFMModels.get_mean(dfm) == [1, 2, 3, 10, 11, 12]
    T2 = [4 0; 5 0; 0 6]
    @test DFMModels.get_loading!(zeros(3, 13), OM, params.OM) == [zeros(3, 8) T2 I]
    T1 = [13 0; 0 14; 15 16]
    @test DFMModels.get_loading!(zeros(3, 25), OQ, params.OQ) == [T1 2T1 3T1 2T1 T1 I 2I 3I 2I I]
    @test DFMModels.get_loading(dfm) == [zeros(3, 8) T2 I zeros(3, 15); T1 2T1 3T1 2T1 T1 zeros(3, 3) I 2I 3I 2I I]
    @test DFMModels.get_covariance(dfm) == [Diagonal([7, 8, 9]) zeros(3, 11); 0I Diagonal([17, 18, 19]) zeros(3, 8); zeros(2, 6) [28 30; 30 31] zeros(2, 6); zeros(6, 8) Diagonal([35, 36, 37, 41, 42, 43])]

    T3 = spzeros(28, 28)
    T3[1:8, 3:10] = I(8)
    T3[9:10, 7:10] = [24 26 20 22; 25 27 21 23]
    T3[11:13, 11:13] = Diagonal([32, 33, 34])
    T3[14:25, 17:28] = I(12)
    T3[26:28, 26:28] = Diagonal([38, 39, 40])
    DFMModels.get_transition(dfm)
    @test DFMModels.get_transition(dfm) == T3


    params .= NaN
    Λ = DFMModels.get_loading!(zeros(3, 13), OM, params.OM)
    lc = DFMModels.loading_constraint(Λ, OM)
    @test lc.blk === OM
    @test findall(lc.estimcols) == [9, 10]
    @test all(lc.estimcols .!= lc.fixedcols)
    @test length(lc.estimblocks) == 1 && haskey(lc.estimblocks, :F)
    lc1 = lc.estimblocks[:F]
    @test size(lc1.W) == (3, 6) && lc1.W == [0 0 1 0 0 0; 0 0 0 1 0 0; 0 0 0 0 1 0]
    @test size(lc1.q) == (3,) && all(iszero, lc1.q)

    Λ = DFMModels.get_loading!(zeros(3, 25), OQ, params.OQ)
    lc = DFMModels.loading_constraint(Λ, OQ)
    @test lc.blk === OQ
    @test findall(lc.estimcols) == 1:10
    @test all(lc.estimcols .!= lc.fixedcols)
    @test length(lc.estimblocks) == 1 && haskey(lc.estimblocks, :F)
    lc1 = lc.estimblocks[:F]
    @test size(lc1.W) == (26, 30) && lc1.W == [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 2 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 0 0 0 3 0 0 0 0 0; 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0; -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 2 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 0 0 0 3 0 0 0 0; 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0; 0 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 2 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 0 0 0 3 0 0 0; 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0; 0 0 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 2 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 0 0 0 3 0 0; 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0; 0 0 0 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 2 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 0 0 0 3 0; 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0; 0 0 0 0 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 2; 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 0 0 0 3; 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2; 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
    @test size(lc1.q) == (26,) && all(iszero, lc1.q)

end

