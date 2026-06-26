##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

@testset "export_model round-trip" begin

    # Build a simple_RBC-shaped model end-to-end.
    function _build_rbc()
        m = ModelDef(:simple_RBC)
        @parameters m begin
            α = 0.33
            δ = 0.1
            ρ = 0.03
            λ = 0.97
            γ = 0.5
            g = 0.015
            β = @link 1 / (1 + ρ)
        end
        @logvariables m begin
            "Consumption"
            C
            "Capital Stock"
            K
            "Labour"
            L
            "Real Wage"
            w
            "Real Rental Rate"
            r
            "Technological shock"
            A
        end
        @shocks m ea
        @autoexogenize m begin
            A = ea
        end
        @equations m begin
            @log C[t+1] * (1 + g) = β * C[t] * (r[t+1] + 1 - δ)
            @log (L[t])^γ * C[t] = w[t]
            @log r[t] * (K[t-1] / (1 + g))^(1 - α) = α * A[t] * L[t]^(1 - α)
            @log w[t] * L[t]^(α) = (1 - α) * A[t] * (K[t-1] / (1 + g))^α
            @lin K[t] + C[t] = A[t] * (K[t-1] / (1 + g))^α * (L[t])^(1 - α) + (1 - δ) * (K[t-1] / (1 + g))
            log(A[t]) = λ * log(A[t-1]) + ea[t]
        end
        return initialize_model(m)
    end

    @testset "simple_RBC: expr_hash equality and param byte-equality" begin
        mdl = _build_rbc()
        dir = mktempdir()
        path = export_model(mdl, :simple_RBC, dir)
        @test isfile(path)

        # Re-include the emitted file inside a fresh anonymous module so
        # `m`, `build_simple_RBC`, and the DSL bindings don't clobber
        # this test scope.
        sandbox = Module(:rbc_sandbox)
        Base.eval(sandbox, :(using ModelBaseEcon))
        Base.include(sandbox, path)
        rebuilt = Base.eval(sandbox, :(build_simple_RBC()))

        @test length(rebuilt.eqns) == length(mdl.eqns)
        for (a, b) in zip(mdl.eqns, rebuilt.eqns)
            @test a.expr_hash == b.expr_hash
        end

        # Flat parameter vector: identical layout, byte-identical values.
        @test length(rebuilt.param_layout) == length(mdl.param_layout)
        for (a, b) in zip(mdl.param_layout, rebuilt.param_layout)
            @test a == b
        end
    end

    @testset "@link parameter survives as @link syntax" begin
        mdl = _build_rbc()
        dir = mktempdir()
        path = export_model(mdl, :simple_RBC, dir)
        src = read(path, String)
        # The β = @link 1 / (1 + ρ) line must round-trip with the @link
        # macrocall syntax - i.e. not be pre-resolved to a numeric.
        @test occursin("@link", src)
        @test occursin("β", src)
    end

    @testset "doc and tag metadata emitted" begin
        m = ModelDef(:eqmeta)
        @variables m begin; y; i; g; c; end
        @equations m begin
            "GDP identity"
            :identity => c[t] = y[t] - i[t] - g[t]
        end
        mdl = initialize_model(m)
        dir = mktempdir()
        path = export_model(mdl, :eqmeta, dir)
        src = read(path, String)
        @test occursin("GDP identity", src)
        @test occursin(":identity", src)

        sandbox = Module(:meta_sandbox)
        Base.eval(sandbox, :(using ModelBaseEcon))
        Base.include(sandbox, path)
        rebuilt = Base.eval(sandbox, :(build_eqmeta()))
        @test doc(rebuilt.eqns[1]) == "GDP identity"
        @test :identity in tags(rebuilt.eqns[1])
        @test mdl.eqns[1].expr_hash == rebuilt.eqns[1].expr_hash
    end

    @testset "invalid model name rejected" begin
        m = ModelDef(:foo)
        @variables m begin; y; end
        @equations m begin; y[t] = 0; end
        mdl = initialize_model(m)
        dir = mktempdir()
        @test_throws ArgumentError export_model(mdl, Symbol("not-an-identifier"), dir)
    end

    @testset "float parameters round-trip at full precision" begin
        m = ModelDef(:precision)
        @parameters m begin
            # An awkward fraction that would lose bits at %.15g.
            θ = 1.0 / 3.0
        end
        @variables m begin; y; end
        @equations m begin
            y[t] = θ * y[t-1]
        end
        mdl = initialize_model(m)
        dir = mktempdir()
        path = export_model(mdl, :precision, dir)
        sandbox = Module(:prec_sandbox)
        Base.eval(sandbox, :(using ModelBaseEcon))
        Base.include(sandbox, path)
        rebuilt = Base.eval(sandbox, :(build_precision()))
        # Locate θ in both param layouts.
        for (a, b) in zip(mdl.param_layout, rebuilt.param_layout)
            @test a == b
        end
        # Find θ's flat slot and confirm it's bit-identical.
        θ_orig = mdl.defs.params[findfirst(p -> p.name === :θ, mdl.defs.params)].value
        θ_back = rebuilt.defs.params[findfirst(p -> p.name === :θ, rebuilt.defs.params)].value
        @test reinterpret(UInt64, θ_orig) == reinterpret(UInt64, θ_back)
    end

end
