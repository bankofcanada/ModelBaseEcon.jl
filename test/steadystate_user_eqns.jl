# v2.1 Chunk G4 - @steadystate user-supplied SS constraints (MBE side).

@testset "G4: @steadystate macro (MBE)" begin

    @testset "single-equation @level (implicit default)" begin
        m = ModelDef(:g4_single)
        @parameters m begin; a = 0.5; end
        @variables m begin; y; z; end
        @equations m begin
            y[t] = a * y[t-1] + z[t]
            z[t] = a * z[t-1]
        end
        @steadystate m y = 1.0
        @test length(m.ss_equations) == 1
        sseq = m.ss_equations[1]
        @test sseq.name === :_SSEQ1
        @test sseq.kind === SS_LEVEL
        # residual is `y - 1.0` as an Expr
        @test sseq.residual.head === :call
        @test sseq.residual.args[1] === :-
    end

    @testset "explicit @level qualifier" begin
        m = ModelDef(:g4_qual)
        @parameters m begin; a = 0.5; end
        @variables m begin; y; z; end
        @equations m begin
            y[t] = a * y[t-1] + z[t]
            z[t] = a * z[t-1]
        end
        @steadystate m @level y = 2.0
        @test length(m.ss_equations) == 1
        @test m.ss_equations[1].kind === SS_LEVEL
    end

    @testset "@slope not yet supported (raises)" begin
        m = ModelDef(:g4_slope)
        @variables m begin; y; end
        # Macro expansion itself raises.
        @test_throws Exception @eval (@steadystate $m @slope y = 1.0)
    end

    @testset "block form, multiple constraints" begin
        m = ModelDef(:g4_blk)
        @parameters m begin; a = 0.4; end
        @variables m begin; y; z; w; end
        @equations m begin
            y[t] = a * y[t-1]
            z[t] = a * z[t-1]
            w[t] = a * w[t-1]
        end
        @steadystate m begin
            y = 1.0
            @level z = 2.0
            w = y + z
        end
        @test length(m.ss_equations) == 3
        @test [e.name for e in m.ss_equations] == [:_SSEQ1, :_SSEQ2, :_SSEQ3]
        @test all(e.kind === SS_LEVEL for e in m.ss_equations)
    end

    @testset "@delete removes by name" begin
        m = ModelDef(:g4_del)
        @variables m begin; y; z; end
        @equations m begin
            y[t] = 0.5 * y[t-1]
            z[t] = 0.5 * z[t-1]
        end
        @steadystate m y = 1.0
        @steadystate m z = 2.0
        @test length(m.ss_equations) == 2

        @steadystate m @delete _SSEQ1
        @test length(m.ss_equations) == 1
        @test m.ss_equations[1].name === :_SSEQ2

        # Block form @delete works too.
        @steadystate m begin
            @delete _SSEQ2
        end
        @test isempty(m.ss_equations)
    end

    @testset "augmented model builds: residual + RJ functions land" begin
        m = ModelDef(:g4_build)
        @parameters m begin; a = 0.5; c0 = 1.0; end
        @variables m begin; y; z; end
        @equations m begin
            y[t] = a * y[t-1] + c0
            z[t] = a * z[t-1] + y[t]
        end
        @steadystate m y = 3.0
        compiled = @initialize m
        @test length(compiled.ss_eqns) == 1
        sseq = compiled.ss_eqns[1]
        @test sseq.name === :_SSEQ1
        @test sseq.kind === SS_LEVEL
        # The Equation's tsrefs are the bare variables, all offset 0
        # (set by the rewrite `y` -> `y[t]`).
        refs = sseq.eqn.tsrefs
        @test all(r -> r.offset == 0, refs)
        @test :y in [r.name for r in refs]
        # Residual evaluates to `y - 3.0` at any p, for x = [y_value].
        # Slot map: tsrefs has one entry (y), so x = [y].
        @test isapprox(sseq.eqn.eval_resid([5.0], Float64[0.5, 1.0]), 2.0; atol=1e-12)
    end

    @testset "export_model round-trip with @steadystate" begin
        m = ModelDef(:g4_export)
        @parameters m begin; a = 0.5; c0 = 1.0; end
        @variables m begin; y; z; end
        @equations m begin
            y[t] = a * y[t-1] + c0
            z[t] = a * z[t-1] + y[t]
        end
        @steadystate m y = 3.0
        compiled = @initialize m

        dir = mktempdir()
        path = export_model(compiled, :g4_roundtrip, dir)
        @test isfile(path)
        text = read(path, String)
        @test occursin("@steadystate m begin", text)
        @test occursin("@level y = 3.0", text)

        # Reload: include the emitted file and call build_g4_roundtrip()
        # in a fresh module so we don't leak names. `include` isn't
        # auto-imported into a bare `Module()`, so use `Base.include`.
        mod = Module()
        Core.eval(mod, :(using ModelBaseEcon))
        Base.include(mod, path)
        rebuilt = Core.eval(mod, :(build_g4_roundtrip()))
        @test length(rebuilt.ss_eqns) == 1
        @test rebuilt.ss_eqns[1].name === :_SSEQ1
        @test rebuilt.ss_eqns[1].kind === SS_LEVEL
    end
end
