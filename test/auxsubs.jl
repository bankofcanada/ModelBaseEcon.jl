##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020, Bank of Canada
# All rights reserved.
##################################################################################

using ModelBaseEcon
using Test

@testset "AUXSUBS" begin

    ASUBS = @test_logs(
        (:info, "Found log(s), which is a shock or exogenous variable. Make sure s data is positive."),
        (:info, "Found log(s), which is a shock or exogenous variable. Make sure s data is positive."),
        (:info, "Found log(s), which is a shock or exogenous variable. Make sure s data is positive."),
        (:info, "Found log(s), which is a shock or exogenous variable. Make sure s data is positive."),
        (:info, "Found log(lx). Consider making lx a log variable."),
        (:info, "Found log(lx). Consider making lx a log variable."),
        (:warn, "Model contains different numbers of equations (6) and endogenous variables (2)."),
        include_string(@__MODULE__, """module ASUBS

using ModelBaseEcon

model = Model()
model.verbose = true
model.substitutions = true
@variables model begin
    @log x
    lx
    @exog p
    @shock s
end
@equations model begin
    log(x[t]) = lx[t] + log(1.0 * p[t - 1])
    log(x[t] / x[t - 1]) = 1.01 + log(s[t])
    log(x[t] + x[t - 1]) = 1.01 + log(s[t])
    log(x[t] * x[t - 1]) = 1.01 + log(s[t])
    log(x[t] - x[t - 1]) = 1.01 + log(s[t])
    log(lx[t]) - log(lx[t - 1]) = log(0.0 + 1.0)
end

@initialize model 

newmodel() = deepcopy(model)

end"""))
    m = ASUBS.newmodel()
    @test length(m.variables) == 3
    @test length(m.shocks) == 1
    @test length(m.equations) == 6
    @test length(m.auxeqns) == length(m.auxvars) == 4
    text = let io = IOBuffer()
        m.verbose = false
        export_model(m, "ASUBS1", io)
        seekstart(io)
        read(io, String)
    end
    @test occursin("@variables", text)
    @test occursin("@exogenous", text)
    @test !occursin("@exog ", text)
    @test occursin("@shocks", text)
    @test !occursin("@shock ", text)
    @test_warn "Model contains different numbers of equations (10) and endogenous variables (6)." include_string(@__MODULE__, text)
    m1 = ASUBS1.model
    @test Set(m1.variables) == Set(vcat(m.variables, m.auxvars))
    @test m1.shocks == m.shocks
    @test isempty(m1.auxvars)
    m1_set = Set(values(m1.equations))
    @test Set(values(m1.equations)) == Set(values(m.alleqns))
    @test isempty(m1.auxeqns)
end

