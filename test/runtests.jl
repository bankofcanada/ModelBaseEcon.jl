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

