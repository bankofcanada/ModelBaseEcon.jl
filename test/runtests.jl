using ModelBaseEcon
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
end

module E
    using ModelBaseEcon
end
@testset "Evaluations" begin
    ModelBaseEcon.initfuncs(E)
    E.eval(ModelBaseEcon.makefuncs(:(x+3*y), [:x, :y], mod=E))
    @test :resid_1 ∈ names(E, all=true)
    @test :RJ_1 ∈ names(E, all=true)
    @test E.resid_1([1.1, 2.3]) == 8.0
    @test E.RJ_1([1.1, 2.3]) == (8.0, [1.0, 3.0])
end
