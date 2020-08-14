using ModelBaseEcon
using SparseArrays
using Test

@testset "Options" begin
    o = Options(tol=1e-7, maxiter=25)
    @test getoption(o, tol=1e7) == 1e-7
    @test getoption(o, abstol=1e-10) == 1e-10
    @test "abstol" ∉ o
    @test getoption!(o, abstol=1e-11) == 1e-11
    @test :abstol ∈ o
    @test setoption!(o, reltol=1e-3, linear=false) isa Options
    @test "reltol" ∈ o 
    @test :linear ∈ o
    @test getoption!(o, tol=nothing, linear=true, name="Zoro") == (1e-7, false, "Zoro")
    @test "name" ∈ o && o.name == "Zoro"
    z = Options(o)
    z.name = "Oro"
    @test o.name == "Zoro"
end

module E
    using ModelBaseEcon
end
@testset "Evaluations" begin
    ModelBaseEcon.initfuncs(E)
    E.eval(ModelBaseEcon.makefuncs(:(x + 3 * y), [:x, :y], mod=E))
    @test :resid_1 ∈ names(E, all=true)
    @test :RJ_1 ∈ names(E, all=true)
    @test E.resid_1([1.1, 2.3]) == 8.0
    @test E.RJ_1([1.1, 2.3]) == (8.0, [1.0, 3.0])
end

@testset "Misc" begin
    m = Model()
    out = let io = IOBuffer()
        print(io, m.flags)
        readlines(seek(io, 0))
    end
    @test length(out) == 3
    for line in out[2:end]
        sline = strip(line)
        @test isempty(sline) || length(split(sline, "=")) == 2
    end
    @test_throws ModelBaseEcon.ModelErrorBase ModelBaseEcon.modelerror()
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
    let m = M1.model
        @test issssolved(m) == false
        M1.model.sstate.mask .= true
        @test issssolved(m) == true
        @test neqns(m.sstate) == 2
        @steadystate m y = 5
        @test length(m.sstate.constraints) == 1
        @test neqns(m.sstate) == 3
        @test length(alleqns(m.sstate)) == 3
    end
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


module AUX
using ModelBaseEcon
model = Model()
@variables model x y
@equations model begin
    x[t + 1] = log(x[t] - x[t - 1])
    y[t + 1] = y[t] + log(y[t - 1])
end
@initialize model
end
@testset "AUX" begin
    let m = AUX.model
        @test m.nvars == 2
        @test m.nshks == 0
        @test m.nauxs == 2
        @test length(m.auxeqns) == 2
        x = ones(2, 2)
        @test_throws ErrorException ModelBaseEcon.update_auxvars(x, m)
        x = 2 .* ones(4, 2)
        ax = ModelBaseEcon.update_auxvars(x, m; default=0.1)
        @test size(ax) == (4, 4)
        @test x == ax[:, 1:2]
        @test ax[:, 3:4] == [0.0 0.0; 0.1 log(2.0); 0.1 log(2.0); 0.0 0.0]
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


@testset "sstate" begin
    m = M2.model
    ss = m.sstate
    empty!(ss.constraints)
    out = let io = IOBuffer()
        print(io, ss)
        readlines(seek(io, 0))
    end
    @test length(out) == 2
    @steadystate m pinf = rate + 1
    out = let io = IOBuffer()
        print(io, ss)
        readlines(seek(io, 0))
    end
    @test length(out) == 3
    @test length(split(out[end], "=")) == 2
    #
    @test propertynames(ss) == tuple(variables(m)...)
    @test ss.pinf.level == ss.pinf[1]
    @test ss.pinf.slope == ss.pinf[2]
    ss.pinf = (level=2.3, slope=0.7)
    @test ss.values[1:2] == [2.3, 0.7]
    @test ss[:rate] == ss["rate"]
    ss["rate"].level = 21
    ss[:rate].slope = 0.21
    @test ss[:rate].level == 21 && ss["rate"].slope == 0.21
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

