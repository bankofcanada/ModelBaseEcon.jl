using ModelBaseEcon
using SparseArrays
using Test

@testset "Options" begin
    o = Options(tol = 1e-7, maxiter = 25)
    @test getoption(o, tol = 1e7) == 1e-7
    @test getoption(o, abstol = 1e-10) == 1e-10
    @test "abstol" ∉ o
    @test getoption!(o, abstol = 1e-11) == 1e-11
    @test :abstol ∈ o
    @test setoption!(o, reltol = 1e-3, linear = false) isa Options
    @test "reltol" ∈ o 
    @test :linear ∈ o
    @test getoption!(o, tol = nothing, linear = true, name = "Zoro") == (1e-7, false, "Zoro")
    @test "name" ∈ o && o.name == "Zoro"
end

module E
    using ModelBaseEcon
end
@testset "Evaluations" begin
    ModelBaseEcon.initfuncs(E)
    E.eval(ModelBaseEcon.makefuncs(:(x + 3 * y), [:x, :y], mod = E))
    @test :resid_1 ∈ names(E, all = true)
    @test :RJ_1 ∈ names(E, all = true)
    @test E.resid_1([1.1, 2.3]) == 8.0
    @test E.RJ_1([1.1, 2.3]) == (8.0, [1.0, 3.0])
end

############################################################################

function test_eval_RJ(m::Model, known_R, known_J)
    nrows = 1 + m.maxlag + m.maxlead
    ncols = length(m.allvars)
    R, J = eval_RJ(zeros(nrows, ncols), m)
    @test R ≈ known_R
    @test J ≈ known_J
end

function compare_RJ_R!_(m::Model)
    nrows = 1 + m.maxlag + m.maxlead
    ncols = length(m.variables) + length(m.shocks) + length(m.auxvars)
    point = rand(nrows, ncols)
    R, J = eval_RJ(point, m)
    S = similar(R)
    eval_R!(S, point, m)
    @test R ≈ S
end

@using_example M1
@testset "M1" begin
    @test length(M1.model.parameters) == 2
    @test length(M1.model.variables) == 1
    @test length(M1.model.shocks) == 1
    @test length(M1.model.equations) == 1
    @test M1.model.maxlag == 1
    @test M1.model.maxlead == 1
    test_eval_RJ(M1.model, [0.0], [-0.5 1.0 -0.5 0.0 -1.0 0.0])
    compare_RJ_R!_(M1.model)
end

@testset "M1.sstate" begin
    @test issssolved(M1.model) == false
    M1.model.sstate.mask .= true
    @test issssolved(M1.model) == true
end

@testset "M1.lin" begin
    m = deepcopy(M1.model)
    with_linearized(m) do lm
        @test islinearized(lm)
        test_eval_RJ(lm, [0.0], [-0.5 1.0 -0.5 0.0 -1.0 0.0])
        compare_RJ_R!_(lm)
    end
    @test !islinearized(m)
    lm = linearized(m)
    test_eval_RJ(lm, [0.0], [-0.5 1.0 -0.5 0.0 -1.0 0.0])
    compare_RJ_R!_(lm)
    @test islinearized(lm)
    @test !islinearized(m)
    linearize!(m)
    @test islinearized(m)
end

@testset "M1.params" begin
    let m = M1.model
        for α = 0.0:0.1:1.0
            β = 1.0 - α
            m.α = α
            m.β = β
            test_eval_RJ(m, [0.0], [-α 1.0 -β 0.0 -1.0 0.0;])
        end
    end
end

@using_example M2
@testset "M2" begin
    @test length(M2.model.parameters) == 3
    @test length(M2.model.variables) == 3
    @test length(M2.model.shocks) == 3
    @test length(M2.model.equations) == 3
    @test M2.model.maxlag == 1
    @test M2.model.maxlead == 1
    test_eval_RJ(M2.model, [0.0, 0.0, 0.0], 
        [-.5      1  -.48     0    0  0    0   -.02     0  0  -1  0  0  0 0 0  0 0;
           0  -.375     0  -.75    1  0    0  -.125     0  0   0  0  0 -1 0 0  0 0;
           0      0  -.02     0  .02  0  -.5      1  -.48  0   0  0  0  0 0 0 -1 0])
    compare_RJ_R!_(M2.model)
end

@using_example M3
@testset "M3" begin
    @test length(M3.model.parameters) == 3
    @test length(M3.model.variables) == 3
    @test length(M3.model.shocks) == 3
    @test length(M3.model.equations) == 3
    @test M3.model.maxlag == 2
    @test M3.model.maxlead == 3
    compare_RJ_R!_(M3.model)
    test_eval_RJ(M3.model, [0.0, 0.0, 0.0], 
        sparse(
            [1, 1, 2, 1, 3, 1, 1, 2, 2, 3,  3,  3,  1,  2,  3,  3,  1,  2,  3],
            [2, 3, 3, 4, 4, 5, 6, 8, 9, 9, 13, 14, 15, 15, 15, 16, 21, 27, 33],
            [-0.5, 1.0, -0.375, -0.3, -0.02, -0.05, -0.05, -0.75, 1.0, 0.02, -0.25, 
             -0.25, -0.02, -0.125, 1.0, -0.48, -1.0, -1.0, -1.0],
            3, 36,
        )
    )
end

@using_example M6
@testset "M6" begin
    @test length(M6.model.parameters) == 2
    @test length(M6.model.variables) == 6
    @test length(M6.model.shocks) == 2
    @test length(M6.model.equations) == 6
    @test M6.model.maxlag == 2
    @test M6.model.maxlead == 3
    compare_RJ_R!_(M6.model)
    nt = 1 + M6.model.maxlag + M6.model.maxlead
    test_eval_RJ(M6.model, [-0.0027, -0.0025, 0.0, 0.0, 0.0, 0.0], 
        sparse(
            [2, 2, 2, 3, 5, 2, 2, 2, 1, 1, 3, 4, 1, 3, 6, 5, 5, 4, 4, 6, 6, 2, 1],
            [1, 2, 3, 3, 3, 4, 5, 6, 8, 9, 9, 9, 10, 15, 15, 20, 21, 26, 27, 32, 33, 39, 45],
            [-0.1, -0.1, 1.0, -1.0, -1.0, -0.1, -0.1, -0.1, -0.2, 1.0, -1.0, -1.0, -0.2, 1.0, 
             -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0],
            6, 6 * 8,
        ))
end

