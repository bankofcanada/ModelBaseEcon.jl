# Legacy (against RW internals) — the BoC-terms half of the dual-green gate.
#
# The Symbolics rewrite removed the legacy model-level evaluator
# (`eval_RJ`/`eval_R!`), the in-MBE steady-state solver (`m.sstate`, now in
# StateSpaceEcon), the `Options`/`evaldata`/`ModelSymbol` machinery, and the
# ForwardDiff backend. The legacy `test/codegen.jl` probed those internals
# directly, so it cannot run verbatim (M-i decision #1: don't re-shim dead
# internals). Per the curated-rewrite decision, this file re-asserts the
# BEHAVIORAL legacy claims on the PUBLIC alias surface (compat.jl): the
# legacy `Model()` builder, `@initialize`, property access, accessors (A2),
# predicates (A3), `selectively_linearize` (A6), model edits + `@reinitialize`,
# the example models, and the permanent `@auxvar` drop.
#
# This is parity *in BoC's own terms*: the public API a legacy caller uses
# still builds the same models and answers the same questions.

using ModelBaseEcon
using Test

@testset "legacy (against RW internals)" begin

    # ------------------------------------------------------------------
    # Build with the legacy `Model()` idiom + accessor/predicate surface.
    # (legacy "Misc"/"Vars"/"VarTypes" behavioral core, on the alias layer)
    # ------------------------------------------------------------------
    @testset "Model() builder + accessors + predicates" begin
        m = Model(:legacy_build)
        @test m isa ModelDef
        @parameters m begin
            α = 0.5
            β = 0.5
        end
        @variables m begin
            x
            @log k
        end
        @shocks m begin
            e
        end
        # accessors (A2)
        @test nvariables(m) == 2
        @test nshocks(m) == 1
        @test nparameters(m) == 2
        @test [v.name for v in variables(m)] == [:x, :k]
        @test [s.name for s in shocks(m)] == [:e]
        @test length(allvars(m)) == 3
        @test nallvars(m) == 3
        # predicates (A3)
        @test islin(m, :x)
        @test islog(m, :k)
        @test !islog(m, :x)
        @test isshock(m, :e)
        @test !isshock(m, :x)
        # legacy property surface on the def
        @test [v.name for v in m.variables] == [:x, :k]
        @test [p.name for p in m.parameters] == [:α, :β]
    end

    # ------------------------------------------------------------------
    # @initialize lifecycle: in-place idiom + cached compiled model.
    # (legacy "E1"/"Model edits" lifecycle core)
    # ------------------------------------------------------------------
    @testset "@initialize lifecycle + property forwarding" begin
        m = Model(:lc)
        @parameters m begin; α = 0.5; β = 0.5; end
        @variables m begin; y; end
        @shocks m begin; e; end
        @equations m begin
            y[t] = α * y[t-1] + β * y[t+1] + e[t]
        end
        # legacy in-place statement form
        @initialize m
        @test m.initialized
        @test m.compiled isa CompiledModel
        # property access forwards through the def to the cached model
        @test m.name === :lc
        @test length(m) == 1
        @test m[1] isa Equation
        @test m.maxlag == 1
        @test m.maxlead == 1
        @test nequations(m) == 1
        # `cm = @initialize m` binding form returns the compiled model
        m2 = Model(:lc2)
        @variables m2 begin; z; end
        @equations m2 begin; z[t] = z[t-1]; end
        cm = @initialize m2
        @test cm isa CompiledModel
        @test cm.name === :lc2
        @test length(cm) == 1
    end

    # ------------------------------------------------------------------
    # model-level flags (legacy `m.flags.linear` / `m.linear`).
    # ------------------------------------------------------------------
    @testset "model flags" begin
        m = Model()
        @test m.linear == m.flags.linear == false
        m.linear = true
        @test m.linear
        @test m.flags.linear
        m.flags.linear = false
        @test !m.linear
    end

    # ------------------------------------------------------------------
    # @log equation builds and is flagged (legacy "@log eqn").
    # ------------------------------------------------------------------
    @testset "@log equation" begin
        m = Model()
        @parameters m rho = 0.1
        @variables m X
        @shocks m EX
        @equations m begin
            @log X[t] = rho * X[t-1] + EX[t]
        end
        cm = @initialize m
        @test length(cm) == 1
        @test ModelBaseEcon.EQ_LOG in cm[1].flags
    end

    # ------------------------------------------------------------------
    # selectively_linearize (legacy "lin"/"sel_lin", via A6).
    # ------------------------------------------------------------------
    @testset "selectively_linearize" begin
        m = Model(:lin)
        @parameters m begin; a = 0.5; end
        @variables m begin; y; end
        @shocks m begin; e; end
        @equations m begin
            @lin y[t] = a * y[t-1] + e[t]
        end
        cm = @initialize m
        x_ss = zeros(nvariables(m))
        lm = selectively_linearize(cm, x_ss)
        @test lm isa CompiledModel
        # linear equation: residual matches the original at a probe point.
        x = Float64[0.3, 0.7, 0.1]      # y[t-1], y[t], e[t] slots
        p = Float64[0.5]
        @test cm[1].eval_resid(x, p) ≈ lm[1].eval_resid(x, p) atol = 1e-12
    end

    # ------------------------------------------------------------------
    # Model edits + @reinitialize (legacy "E1.equation change"/"Model edits").
    # ------------------------------------------------------------------
    @testset "@reinitialize after edit" begin
        m = Model(:edit)
        @parameters m begin; α = 0.5; β = 0.5; end
        @variables m begin; y; z; end
        @shocks m begin; e; end
        @equations m begin
            y[t] = α * y[t-1] + e[t]
            z[t] = β * z[t-1]
        end
        @initialize m
        prev1 = m.compiled[1]
        # one-arg legacy form rebuilds in place; unchanged eqns are reused.
        @reinitialize m
        @test m.compiled[1] === prev1
    end

    # ------------------------------------------------------------------
    # Example models load and build (legacy "E1"/"E2"/"E3"/"E6"/"E7").
    # ------------------------------------------------------------------
    @testset "example models build" begin
        @include_example E1 force
        @include_example E2 force
        @include_example E3 force
        @include_example E6 force
        @include_example E7 force

        mE1 = E1.newmodel()
        @test nparameters(mE1) == 2
        @test nvariables(mE1) == 1
        @test nshocks(mE1) == 1
        @test nequations(mE1) == 1
        @test mE1.maxlag == 1
        @test mE1.maxlead == 1
        @test mE1.linear            # E1 sets flags.linear = true

        @test nvariables(E2.newmodel()) == 3
        @test nvariables(E3.newmodel()) == 3
        @test nvariables(E6.newmodel()) == 6
        @test nequations(E7.newmodel()) == 7
    end

    # ------------------------------------------------------------------
    # @include_example error handling (legacy "include_example").
    # ------------------------------------------------------------------
    @testset "include_example errors" begin
        @test_throws Exception (@include_example NOSUCHEXAMPLE)
    end

    # ------------------------------------------------------------------
    # @auxvar drop is permanent (legacy "AUX"/auxsubs → drop-enforcement).
    # See decision_auxvar_never_existed: the auto aux-substitution engine is
    # removed; the Symbolics core differentiates subexpressions directly.
    # ------------------------------------------------------------------
    @testset "@auxvar / aux-substitution dropped" begin
        @test !isdefined(ModelBaseEcon, Symbol("@auxvar"))
        @test !isdefined(ModelBaseEcon, :update_auxvars)
        @test !isdefined(ModelBaseEcon, :auxvars)
        @test !isdefined(ModelBaseEcon, :auxeqns)
        # The subexpression that used to require aux vars builds directly.
        m = Model()
        @variables m begin; x; end
        @shocks m begin; e; end
        @equations m begin
            x[t] = log(x[t-1] + x[t-2]) + e[t]
        end
        cm = @initialize m
        @test length(cm) == 1
    end

end
