##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2025, Bank of Canada
# All rights reserved.
##################################################################################

using ModelBaseEcon
using Test

@testset "AUXSUBS" begin

    ASUBS = Module(:ASUBS)
    @eval ASUBS using ModelBaseEcon

    @test_logs(
        (:info, "Found log(s), which is a shock or exogenous variable. Make sure s data is positive."),
        (:info, "Found log(s), which is a shock or exogenous variable. Make sure s data is positive."),
        (:info, "Found log(s), which is a shock or exogenous variable. Make sure s data is positive."),
        (:info, "Found log(s), which is a shock or exogenous variable. Make sure s data is positive."),
        (:info, "Found log(lx). Consider making lx a log variable."),
        (:info, "Found log(lx). Consider making lx a log variable."),
        (:warn, "Model contains different numbers of equations (6) and endogenous variables (2)."),
        include_string(ASUBS, """

            const model = Model()
            newmodel() = (global model; deepcopy(model))

            model.verbose = true
            model.substitutions = true
            @variables model begin
                @log x
                lx
                @exog p
                @shock s
            end
            @equations model begin
                log(x[t]) = lx[t] + log(1.0 * p[t-1])
                log(x[t] / x[t-1]) = 1.01 + log(s[t])
                log(x[t] + x[t-1]) = 1.01 + log(s[t])
                log(x[t] * x[t-1]) = 1.01 + log(s[t])
                log(x[t] - x[t-1]) = 1.01 + log(s[t])
                log(lx[t]) - log(lx[t-1]) = log(0.0 + 1.0)
            end

            @initialize model
        """)
    )

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
    m1 = ASUBS1.newmodel()
    @test Set(m1.variables) == Set(vcat(m.variables, m.auxvars))
    @test m1.shocks == m.shocks
    @test isempty(m1.auxvars)
    m1_set = Set(values(m1.equations))
    @test Set(values(m1.equations)) == Set(values(m.alleqns))
    @test isempty(m1.auxeqns)
end

@testset "update_auxvars world age" begin
    # This test guards against a world age regression in update_auxvars.
    #
    # The bug: update_auxvars calls eqn.eval_resid(...) directly. After the
    # codegen refactor, eval_resid is an EquationEvaluatorFD instance whose
    # callable method is created via Core.eval at model-initialization time.
    # If update_auxvars was already compiled (first call), it runs under the
    # old world age and cannot see the eval_resid method of a model that was
    # initialized later — causing a silent MethodError.
    # The fix is invokelatest(eqn.eval_resid, ...) in update_auxvars.
    #
    # To reproduce: compile update_auxvars by calling it on a first model,
    # then initialize a second model (advances the world age via Core.eval),
    # then call update_auxvars on the second model's data.

    make_log_model(modname::Symbol) = let M = Module(modname)
        @eval M using ModelBaseEcon
        include_string(M, """
            const model = Model()
            model.substitutions = true
            @variables model begin
                @log x
            end
            @shocks model x_shk
            @equations model begin
                log(x[t] / x[t-1]) = 0.0 + x_shk[t]
            end
            @initialize model
            newmodel() = deepcopy(model)
        """)
        M
    end

    check_update_auxvars(m) = let
        nvarshk = length(m.variables) + length(m.shocks)
        nt = 1 + m.maxlag + m.maxlead + 3
        data = fill(1.0, nt, nvarshk)
        result = ModelBaseEcon.update_auxvars(data, m)
        @test size(result, 2) == nvarshk + length(m.auxvars)
    end

    # First model: compiles update_auxvars at the current world age
    M1 = make_log_model(:AuxWorldAge1)
    check_update_auxvars(M1.newmodel())

    # Second model: Core.eval in @initialize advances the world age.
    # Without invokelatest, calling update_auxvars here would throw:
    #   MethodError: applicable method may be too new
    M2 = make_log_model(:AuxWorldAge2)
    check_update_auxvars(M2.newmodel())
end

