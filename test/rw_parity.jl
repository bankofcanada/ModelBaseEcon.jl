# RW parity suite — the Symbolics-rewrite guard tests ported in as-is from
# RW-ModelBaseEcon (the ≤1e-8 / bit-identical-vs-legacy-reference evidence,
# the C-ledger guards). Ported mechanically: module RWModelBaseEcon ->
# ModelBaseEcon, compiled type Model -> CompiledModel. This is the "RW
# parity" half of the dual-green gate (api_and_tests §5).
using ModelBaseEcon
using ModelBaseEcon: IR, Validate, Symbolic
using Symbolics: Symbolics, Num, substitute
using Test

# Minimum allocation over a few warm runs, using only Base.@allocated.
# Replaces BenchmarkTools.@ballocated for the hot-loop guard so the test
# target carries no extra dependency. The min over repeats discards
# one-time GC bookkeeping noise the same way @ballocated's sampling does.
function _min_alloc(f, args...; repeats = 20)
    f(args...)  # warm up
    best = typemax(Int)
    for _ in 1:repeats
        a = @allocated f(args...)
        a < best && (best = a)
    end
    best
end

@testset "parity" begin

    @testset "module loads" begin
        @test isdefined(ModelBaseEcon, :ModelBaseEcon)
    end

    @testset "@variables, @logvariables, @shocks" begin
        m = ModelDef(:test1)
        @variables m begin
            "Output"
            y
            x
        end
        @logvariables m begin
            "Consumption"
            c
        end
        @shocks m begin
            ea; eb
        end

        @test length(m.vars) == 3
        @test m.vars[1].name === :y
        @test m.vars[1].kind === IR.VAR_NORMAL
        @test m.vars[1].doc == "Output"
        @test m.vars[2].name === :x
        @test m.vars[2].doc === nothing
        @test m.vars[3].name === :c
        @test m.vars[3].kind === IR.VAR_LOG
        @test m.vars[3].doc == "Consumption"
        @test [s.name for s in m.shocks] == [:ea, :eb]
    end

    @testset "@shocks single-name form" begin
        m = ModelDef()
        @shocks m ea
        @test length(m.shocks) == 1
        @test m.shocks[1].name === :ea
    end

    @testset "duplicate name rejected" begin
        m = ModelDef()
        @variables m begin; y; end
        @test_throws ErrorException @eval @variables $m begin; y; end
    end

    @testset "@parameters scalar/array/linked" begin
        m = ModelDef()
        @parameters m begin
            α = 0.33
            δ = 0.1
            β = @link 1 / (1 + ρ)
            ρ = 0.03
            arr = [1.0, 2.0, 3.0]
        end
        names = [p.name for p in m.params]
        @test names == [:α, :δ, :β, :ρ, :arr]
        @test m.params[1].kind === IR.PARAM_SCALAR
        @test m.params[1].value === 0.33
        @test m.params[3].kind === IR.PARAM_LINKED
        # Linked value is the defining expression, preserved verbatim.
        @test m.params[3].value isa Expr
        @test m.params[5].kind === IR.PARAM_ARRAY
        @test m.params[5].value == [1.0, 2.0, 3.0]
    end

    @testset "@equations, residual rewrite, flag peeling" begin
        m = ModelDef()
        @parameters m begin
            α = 0.5
            β = 0.5
        end
        @variables m begin; y; end
        @shocks m begin; e; end
        @equations m begin
            y[t] = α * y[t-1] + β * y[t+1] + e[t]
            "doc"
            @lin y[t] = α * y[t-1]
            @log y[t] = exp(y[t-1])
        end
        @test length(m.equations) == 3
        # First eq: residual is `LHS - RHS` form, not LHS = RHS
        @test m.equations[1].residual.head === :call
        @test m.equations[1].residual.args[1] === :-
        @test isempty(m.equations[1].flags)
        @test m.equations[2].doc == "doc"
        @test IR.EQ_LIN in m.equations[2].flags
        @test IR.EQ_LOG in m.equations[3].flags
    end

    @testset "@autoexogenize" begin
        m = ModelDef()
        @variables m begin; A; B; end
        @shocks m begin; ea; eb; end
        @autoexogenize m begin
            A = ea
            B = eb
        end
        @test length(m.autoexog) == 2
        @test m.autoexog[1].var === :A
        @test m.autoexog[1].shock === :ea
    end

    @testset "validate: full simple_RBC-style model" begin
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

        # Hits all macros end-to-end and validates cleanly.
        link_order, link_deps = validate(m)
        @test length(m.equations) == 6
        @test :β in keys(link_deps)
        @test link_deps[:β] == [:ρ]
        # ρ must come before β in topo order
        β_idx = findfirst(==(:β), [m.params[i].name for i in link_order])
        ρ_idx = findfirst(==(:ρ), [m.params[i].name for i in link_order])
        @test ρ_idx < β_idx
    end

    @testset "validate: unknown symbol error" begin
        m = ModelDef()
        @variables m begin; y; end
        @equations m begin
            y[t] = z[t-1]   # z is undeclared
        end
        @test_throws Validate.ValidationError validate(m)
    end

    @testset "validate: unknown function error" begin
        m = ModelDef()
        @variables m begin; y; end
        @equations m begin
            y[t] = mystery_fn(y[t-1])
        end
        @test_throws Validate.ValidationError validate(m)
    end

    @testset "validate: registered functions accepted" begin
        m = ModelDef()
        @variables m begin; y; end
        @shocks m begin; e; end
        @equations m begin
            y[t] = log(exp(y[t-1])) + ifelse(y[t-1] > 0, 1, 0) + max(e[t], 0)
        end
        @test_nowarn validate(m)
    end

    @testset "validate: @link cycle detected" begin
        m = ModelDef()
        @parameters m begin
            a = @link b + 1
            b = @link a + 1
        end
        # Cycles now raise the typed LinkCycleError.
        @test_throws Validate.LinkCycleError validate(m)
        err = try
            validate(m)
            nothing
        catch e
            e
        end
        @test err isa Validate.LinkCycleError
        # Cycle path: a -> b -> a (entry repeated to close the loop).
        @test err.cycle == [:a, :b, :a]
        # showerror produces a readable message.
        buf = IOBuffer()
        showerror(buf, err)
        @test occursin("circular @link chain", String(take!(buf)))
    end

    @testset "validate: deep @link chain resolves" begin
        m = ModelDef()
        @parameters m begin
            a = 1.0
            b = @link a * 2
            c = @link b * 2
            d = @link c * 2
        end
        order, deps = Validate.link_topology(m)
        names_in_order = [m.params[i].name for i in order]
        # a must come first, d must come last
        @test names_in_order[1] === :a
        @test names_in_order[end] === :d
        @test deps[:c] == [:b]
        @test deps[:d] == [:c]
    end

    # ------------------------------------------------------------------
    # @link deep-chain resolution + typed cycle error
    # ------------------------------------------------------------------

    @testset "ChunkV: 3-level @link chain resolves to a number" begin
        # a = b + 1, b = c * 2, c = 3.0  ->  b = 6, a = 7.
        m = ModelDef()
        @parameters m begin
            a = @link b + 1
            b = @link c * 2
            c = 3.0
        end
        table, root_layout = Symbolic.resolved_link_table(m)
        # Only c is a root param.
        @test [pr.name for pr in root_layout] == [:c]
        # Resolved entries: substitute the single root (c) and evaluate.
        sub = Dict(table[:c] => 3.0)
        @test Symbolics.value(substitute(table[:b], sub)) ≈ 6.0
        @test Symbolics.value(substitute(table[:a], sub)) ≈ 7.0
    end

    @testset "ChunkV: deep chain resolves inside an equation" begin
        # The residual must end up expressed in the root param only.
        m = ModelDef()
        @parameters m begin
            a = @link b + 1
            b = @link c * 2
            c = 3.0
        end
        @variables m begin; y; end
        @equations m begin
            y[t] = a * y[t-1]
        end
        kernels, params = build_equation_kernels(m)
        @test [p.name for p in params] == [:c]
        k = kernels[1]
        free = Set(string(Symbolics.tosymbol(v))
                   for v in Symbolics.get_variables(k.residual))
        @test "c" in free
        @test !("a" in free) && !("b" in free)
        funcs = build_equation_functions(k)
        # F = y[t] - a*y[t-1], a = (c*2)+1 = 7 at c=3.
        # tsrefs: y[t-1], y[t]. x = [2.0, 5.0] -> F = 5 - 7*2 = -9.
        @test funcs.eval_resid([2.0, 5.0], [3.0]) ≈ -9.0 atol = 1e-12
    end

    @testset "ChunkV: cycle raises LinkCycleError with the path" begin
        m = ModelDef()
        @parameters m begin
            a = @link b + 1
            b = @link a - 1
        end
        err = try
            Validate.link_topology(m); nothing
        catch e; e; end
        @test err isa Validate.LinkCycleError
        @test err.cycle == [:a, :b, :a]
    end

    @testset "ChunkV: 3-node cycle reports the full loop" begin
        m = ModelDef()
        @parameters m begin
            a = @link b + 1
            b = @link c + 1
            c = @link a + 1
        end
        err = try
            Validate.link_topology(m); nothing
        catch e; e; end
        @test err isa Validate.LinkCycleError
        # First node visited is `a`; loop closes back on `a`.
        @test err.cycle[1] === :a && err.cycle[end] === :a
        @test Set(err.cycle) == Set([:a, :b, :c])
    end

    @testset "ChunkV: SW07-style nested chain round-trips" begin
        # SW07's link chain is several levels deep with shared sub-terms.
        # Model a representative shape and check the resolved values.
        m = ModelDef()
        @parameters m begin
            ρ      = 0.03
            cgamma = 1.004
            cbeta  = @link 1 / (1 + ρ / 100)
            cpie   = 1.005
            cr     = @link cpie / (cbeta * cgamma^(-1.0))
            crk    = @link cr - 1 + 0.025
        end
        table, root_layout = Symbolic.resolved_link_table(m)
        roots = Set(pr.name for pr in root_layout)
        @test roots == Set([:ρ, :cgamma, :cpie])
        sub = Dict(table[:ρ] => 0.03, table[:cgamma] => 1.004,
                   table[:cpie] => 1.005)
        cbeta_v = 1 / (1 + 0.03 / 100)
        cr_v    = 1.005 / (cbeta_v * 1.004^(-1.0))
        crk_v   = cr_v - 1 + 0.025
        @test Symbolics.value(substitute(table[:cbeta], sub)) ≈ cbeta_v
        @test Symbolics.value(substitute(table[:cr],    sub)) ≈ cr_v
        @test Symbolics.value(substitute(table[:crk],   sub)) ≈ crk_v
    end

    # ------------------------------------------------------------------
    # Symbolic kernels
    # ------------------------------------------------------------------

    @testset "symbolic: linear equation gradient" begin
        m = ModelDef()
        @parameters m begin
            α = 0.5
            β = 0.5
        end
        @variables m begin; y; end
        @shocks m begin; e; end
        @equations m begin
            y[t] = α * y[t-1] + β * y[t+1] + e[t]
        end
        kernels, params = build_equation_kernels(m)
        @test length(kernels) == 1
        k = kernels[1]
        # tsrefs: y declared first, then shock e. Within name, sort by offset.
        @test k.tsrefs == [TimeRef(:y, -1), TimeRef(:y, 0), TimeRef(:y, +1),
                            TimeRef(:e, 0)]
        # params: α, β in declaration order.
        @test [p.name for p in params] == [:α, :β]
        @test all(p.index === nothing for p in params)
        # F = y[t] - α*y[t-1] - β*y[t+1] - e[t]
        # ∂F/∂y[t-1] = -α, ∂F/∂y[t] = 1, ∂F/∂y[t+1] = -β, ∂F/∂e[t] = -1
        # Substitute α=0.5, β=0.5 to get numeric values.
        sub = Dict(k.p_syms[1] => 0.5, k.p_syms[2] => 0.5)
        grad_vals = [Symbolics.value(substitute(g, sub)) for g in k.gradient]
        @test grad_vals ≈ [-0.5, 1.0, -0.5, -1.0]
        @test k.hessian === nothing
    end

    @testset "symbolic: nonlinear equation gradient" begin
        m = ModelDef()
        @parameters m begin; α = 0.33; end
        @variables m begin; A; K; L; end
        @equations m begin
            r[t] = α * A[t] * (L[t] / K[t-1])^(1 - α)
        end
        # r is undeclared above on purpose - it should be caught.
        # Instead, build a valid one:
        m2 = ModelDef()
        @parameters m2 begin; α = 0.33; end
        @variables m2 begin; A; K; L; r; end
        @equations m2 begin
            r[t] = α * A[t] * (L[t] / K[t-1])^(1 - α)
        end
        kernels, _ = build_equation_kernels(m2)
        k = kernels[1]
        # Evaluate gradient at A=1, K=1, L=1, α=0.33; F = r - α*A*(L/K)^(1-α)
        # At those values F = r - 0.33, ∂F/∂r = 1, ∂F/∂A = -0.33, etc.
        sub = Dict{Num,Float64}()
        for (i, ref) in enumerate(k.tsrefs)
            sub[k.x_syms[i]] = 1.0
        end
        sub[k.p_syms[1]] = 0.33
        # Find ∂F/∂r by index of TimeRef(:r, 0)
        r_idx = findfirst(==(TimeRef(:r, 0)), k.tsrefs)
        @test r_idx !== nothing
        @test Symbolics.value(substitute(k.gradient[r_idx], sub)) ≈ 1.0
        # ∂F/∂A at A=K=L=1, α=0.33  =  -α * (L/K)^(1-α)  =  -0.33
        A_idx = findfirst(==(TimeRef(:A, 0)), k.tsrefs)
        @test Symbolics.value(substitute(k.gradient[A_idx], sub)) ≈ -0.33
    end

    @testset "symbolic: @link substitution into equation" begin
        m = ModelDef()
        @parameters m begin
            ρ = 0.03
            β = @link 1 / (1 + ρ)
        end
        @variables m begin; C; end
        @shocks m begin; e; end
        @equations m begin
            C[t+1] = β * C[t] + e[t]
        end
        kernels, params = build_equation_kernels(m)
        # Param layout should contain root params only (ρ), not β.
        @test [p.name for p in params] == [:ρ]
        k = kernels[1]
        # F should depend on ρ symbolically (since β was substituted away).
        # Check by collecting variables in F's expression.
        free = Symbolics.get_variables(k.residual)
        free_names = Set(string(Symbolics.tosymbol(v)) for v in free)
        @test "ρ" in free_names
        @test !("β" in free_names)
    end

    @testset "symbolic: hessian for nonlinear equation" begin
        m = ModelDef()
        @variables m begin; x; y; end
        @equations m begin
            x[t] * y[t] = 1.0
        end
        # Disable validation circular checking by validating manually first
        # the equation is fine, no @link, no shocks.
        kernels, _ = build_equation_kernels(m; max_hod_order = 2)
        k = kernels[1]
        @test k.hessian !== nothing
        # F = x*y - 1. ∂²F/∂x∂y = 1, ∂²F/∂x² = 0.
        @test size(k.hessian) == (2, 2)
        # Find indices of x[t] and y[t] in tsrefs.
        ix = findfirst(==(TimeRef(:x, 0)), k.tsrefs)
        iy = findfirst(==(TimeRef(:y, 0)), k.tsrefs)
        @test Symbolics.value(k.hessian[ix, iy]) == 1
        @test Symbolics.value(k.hessian[ix, ix]) == 0
    end

    @testset "symbolic: array parameter expands to flat slots" begin
        m = ModelDef()
        @parameters m begin
            arr = [1.0, 2.0, 3.0]
            s = 0.5
        end
        @variables m begin; y; end
        @equations m begin
            y[t] = arr[1] * y[t-1] + s
        end
        kernels, params = build_equation_kernels(m)
        # Layout: arr_1, arr_2, arr_3, s
        @test [(p.name, p.index) for p in params] == [
            (:arr, 1), (:arr, 2), (:arr, 3), (:s, nothing)]
    end

    @testset "symbolic: simple_RBC end-to-end builds without error" begin
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
            C; K; L; w; r; A
        end
        @shocks m ea
        @equations m begin
            @log C[t+1] * (1 + g) = β * C[t] * (r[t+1] + 1 - δ)
            @log (L[t])^γ * C[t] = w[t]
            @log r[t] * (K[t-1] / (1 + g))^(1 - α) = α * A[t] * L[t]^(1 - α)
            @log w[t] * L[t]^(α) = (1 - α) * A[t] * (K[t-1] / (1 + g))^α
            @lin K[t] + C[t] = A[t] * (K[t-1] / (1 + g))^α * (L[t])^(1 - α) + (1 - δ) * (K[t-1] / (1 + g))
            log(A[t]) = λ * log(A[t-1]) + ea[t]
        end
        kernels, params = build_equation_kernels(m)
        @test length(kernels) == 6
        # Gradient lengths match per-equation tsref count.
        for k in kernels
            @test length(k.gradient) == length(k.tsrefs)
        end
        # Param layout has all root scalars (β is linked, excluded).
        @test Set(p.name for p in params) == Set([:α, :δ, :ρ, :λ, :γ, :g])
        # The Euler equation uses β symbolically - its residual must contain ρ.
        free_names_eq1 = Set(string(Symbolics.tosymbol(v))
                             for v in Symbolics.get_variables(kernels[1].residual))
        @test "ρ" in free_names_eq1
        @test !("β" in free_names_eq1)
    end

    @testset "symbolic: lead/lag offset parsing" begin
        m = ModelDef()
        @variables m begin; y; end
        @equations m begin
            y[t] = y[t-1] + y[t+2] + y[t-3]
        end
        kernels, _ = build_equation_kernels(m)
        offsets = sort([r.offset for r in kernels[1].tsrefs])
        @test offsets == [-3, -1, 0, 2]
    end

    # ------------------------------------------------------------------
    # Codegen (RGF emission)
    # ------------------------------------------------------------------

    @testset "codegen: residual RGF matches manual evaluation" begin
        m = ModelDef()
        @parameters m begin
            α = 0.5
            β = 0.5
        end
        @variables m begin; y; end
        @shocks m begin; e; end
        @equations m begin
            y[t] = α * y[t-1] + β * y[t+1] + e[t]
        end
        kernels, params = build_equation_kernels(m)
        funcs = build_equation_functions(kernels[1])
        # tsrefs order: y[t-1], y[t], y[t+1], e[t]; param order: α, β.
        x = [0.7, 1.0, 1.2, 0.05]
        p = [0.5, 0.5]
        # F = y[t] - α*y[t-1] - β*y[t+1] - e[t]
        # = 1.0 - 0.5*0.7 - 0.5*1.2 - 0.05 = 1.0 - 0.35 - 0.6 - 0.05 = 0.0
        @test funcs.eval_resid(x, p) ≈ 0.0 atol=1e-12
        # Perturb y[t] to confirm sensitivity.
        x2 = copy(x); x2[2] = 1.5
        @test funcs.eval_resid(x2, p) ≈ 0.5 atol=1e-12
    end

    @testset "codegen: eval_RJ! returns residual and writes gradient" begin
        m = ModelDef()
        @parameters m begin
            α = 0.5
            β = 0.5
        end
        @variables m begin; y; end
        @shocks m begin; e; end
        @equations m begin
            y[t] = α * y[t-1] + β * y[t+1] + e[t]
        end
        kernels, _ = build_equation_kernels(m)
        funcs = build_equation_functions(kernels[1])
        x = [0.7, 1.0, 1.2, 0.05]
        p = [0.5, 0.5]
        J = zeros(funcs.n_x)
        F = funcs.eval_RJ!(J, x, p)
        @test F ≈ 0.0 atol=1e-12
        @test J ≈ [-0.5, 1.0, -0.5, -1.0] atol=1e-12
    end

    @testset "codegen: hot-loop allocation guard" begin
        # The generated RGFs should not allocate when called with
        # preallocated buffers in a hot loop. Verifies the in-place
        # gradient form from Symbolics is wired correctly.
        m = ModelDef()
        @parameters m begin; α = 0.33; end
        @variables m begin; A; K; L; r; end
        @equations m begin
            r[t] = α * A[t] * (L[t] / K[t-1])^(1 - α)
        end
        kernels, _ = build_equation_kernels(m)
        funcs = build_equation_functions(kernels[1])
        x = ones(funcs.n_x)
        p = [0.33]
        J = zeros(funcs.n_x)
        # Warm up so any first-call compilation is excluded.
        funcs.eval_resid(x, p)
        funcs.eval_RJ!(J, x, p)
        # Wrap calls in a function so closure capture is concrete.
        # Symbolics' generated callable plus our small wrapper closures
        # may allocate a few words (e.g. Float64 boxing in older Symbolics);
        # require <= 32 bytes to keep the bound tight while not failing on
        # one-time GC bookkeeping noise. Tighten when feasible.
        resid_alloc = _min_alloc(funcs.eval_resid, x, p)
        rj_alloc = _min_alloc(funcs.eval_RJ!, J, x, p)
        @test resid_alloc <= 32
        @test rj_alloc <= 32
    end

    @testset "codegen: nonlinear residual matches symbolic value" begin
        m = ModelDef()
        @parameters m begin; α = 0.33; end
        @variables m begin; A; K; L; r; end
        @equations m begin
            r[t] = α * A[t] * (L[t] / K[t-1])^(1 - α)
        end
        kernels, _ = build_equation_kernels(m)
        funcs = build_equation_functions(kernels[1])
        # Build x with A=K=L=r=1.0.
        x = ones(funcs.n_x)
        p = [0.33]
        @test funcs.eval_resid(x, p) ≈ 1.0 - 0.33 atol=1e-12
    end

    @testset "codegen: hessian RGF" begin
        m = ModelDef()
        @variables m begin; x; y; end
        @equations m begin
            x[t] * y[t] = 1.0
        end
        kernels, _ = build_equation_kernels(m; max_hod_order = 2)
        funcs = build_equation_functions(kernels[1])
        @test funcs.eval_hess! !== nothing
        H = zeros(funcs.n_x, funcs.n_x)
        funcs.eval_hess!(H, ones(funcs.n_x), Float64[])
        ix = findfirst(==(TimeRef(:x, 0)), kernels[1].tsrefs)
        iy = findfirst(==(TimeRef(:y, 0)), kernels[1].tsrefs)
        @test H[ix, iy] ≈ 1.0
        @test H[iy, ix] ≈ 1.0
        @test H[ix, ix] ≈ 0.0
        @test H[iy, iy] ≈ 0.0
    end

    @testset "codegen: build_model_functions for full simple_RBC" begin
        m = ModelDef(:simple_RBC)
        @parameters m begin
            α = 0.33; δ = 0.1; ρ = 0.03; λ = 0.97; γ = 0.5; g = 0.015
            β = @link 1 / (1 + ρ)
        end
        @logvariables m begin; C; K; L; w; r; A; end
        @shocks m ea
        @equations m begin
            @log C[t+1] * (1 + g) = β * C[t] * (r[t+1] + 1 - δ)
            @log (L[t])^γ * C[t] = w[t]
            @log r[t] * (K[t-1] / (1 + g))^(1 - α) = α * A[t] * L[t]^(1 - α)
            @log w[t] * L[t]^(α) = (1 - α) * A[t] * (K[t-1] / (1 + g))^α
            @lin K[t] + C[t] = A[t] * (K[t-1] / (1 + g))^α * (L[t])^(1 - α) + (1 - δ) * (K[t-1] / (1 + g))
            log(A[t]) = λ * log(A[t-1]) + ea[t]
        end
        kernels, _ = build_equation_kernels(m)
        funcs = build_model_functions(kernels)
        @test length(funcs) == 6
        for f in funcs
            @test f.eval_resid isa Function
            @test f.eval_RJ! isa Function
        end
    end

    @testset "codegen: world-age safe - RGF callable from freshly compiled function" begin
        # Construct an RGF inside this testset, then call it from a function
        # defined here, after construction. If world-age were broken, this
        # would throw MethodError or require invokelatest.
        m = ModelDef()
        @parameters m begin; a = 1.0; end
        @variables m begin; y; end
        @equations m begin; y[t] = a * y[t-1]; end
        kernels, _ = build_equation_kernels(m)
        funcs = build_equation_functions(kernels[1])
        # Define a fresh function that closes over the RGF and call it.
        caller = (xv, pv) -> funcs.eval_resid(xv, pv)
        x = [0.5, 0.7]  # tsrefs ordering: y[t-1], y[t]
        p = [1.0]
        # F = y[t] - a*y[t-1] = 0.7 - 1.0*0.5 = 0.2
        @test caller(x, p) ≈ 0.2 atol=1e-12
    end

    # ------------------------------------------------------------------
    # Equation API + @initialize / @reinitialize
    # ------------------------------------------------------------------

    @testset "compile: @initialize freezes a Model" begin
        m = ModelDef(:rbc_mini)
        @parameters m begin
            α = 0.5
            β = 0.5
        end
        @variables m begin; y; end
        @shocks m begin; e; end
        @equations m begin
            y[t] = α * y[t-1] + β * y[t+1] + e[t]
        end
        compiled = @initialize m
        @test compiled isa CompiledModel
        @test compiled.name === :rbc_mini
        @test length(compiled) == 1
        @test compiled[1] isa Equation
        # eqns storage is Tuple (small model) or Vector{Equation} (large /
        # forced via RW_MBE_TUPLE_THRESHOLD); here we only require it
        # iterates as expected.
        @test compiled.eqns isa Union{Tuple, Vector{Equation}}
        @test length(compiled.eqns) == 1
        # `initialized` flag set on the source ModelDef.
        @test m.initialized
        # Sanity: the eval functions produced match Codegen path.
        x = [0.7, 1.0, 1.2, 0.05]
        p = [0.5, 0.5]
        @test compiled[1].eval_resid(x, p) ≈ 0.0 atol=1e-12
    end

    @testset "compile: expr_hash stable, ignores LineNumberNodes" begin
        # Two equations with identical AST except for src lines should hash equal.
        e1 = IR.EquationAST(
            :(y[t] - α*y[t-1]),
            Set{IR.EquationFlag}(),
            nothing,
            LineNumberNode(1, :a),
        )
        e2 = IR.EquationAST(
            :(y[t] - α*y[t-1]),
            Set{IR.EquationFlag}(),
            nothing,
            LineNumberNode(99, :b),
        )
        @test expr_hash(e1) == expr_hash(e2)
        # Different residual -> different hash.
        e3 = IR.EquationAST(
            :(y[t] - α*y[t-2]),
            Set{IR.EquationFlag}(),
            nothing,
            LineNumberNode(1, :a),
        )
        @test expr_hash(e1) != expr_hash(e3)
        # Different flags -> different hash.
        e4 = IR.EquationAST(
            :(y[t] - α*y[t-1]),
            Set{IR.EquationFlag}([IR.EQ_LIN]),
            nothing,
            LineNumberNode(1, :a),
        )
        @test expr_hash(e1) != expr_hash(e4)
    end

    @testset "compile: @reinitialize reuses unchanged equations" begin
        m = ModelDef(:reuse)
        @parameters m begin; α = 0.5; β = 0.5; end
        @variables m begin; y; z; end
        @shocks m begin; e; end
        @equations m begin
            y[t] = α * y[t-1] + e[t]
            z[t] = β * z[t-1]
        end
        prev = @initialize m

        # Build a fresh ModelDef with the SAME equations and env.
        m2 = ModelDef(:reuse)
        @parameters m2 begin; α = 0.5; β = 0.5; end
        @variables m2 begin; y; z; end
        @shocks m2 begin; e; end
        @equations m2 begin
            y[t] = α * y[t-1] + e[t]
            z[t] = β * z[t-1]
        end
        new_model, n_rebuilt = @reinitialize prev m2
        @test n_rebuilt == 0
        # Object identity: reused equations should be the same Equation
        # instance from `prev`, not a fresh copy.
        @test new_model[1] === prev[1]
        @test new_model[2] === prev[2]
    end

    @testset "compile: @reinitialize rebuilds only changed equations" begin
        m = ModelDef(:partial)
        @parameters m begin; α = 0.5; β = 0.5; end
        @variables m begin; y; z; end
        @shocks m begin; e; end
        @equations m begin
            y[t] = α * y[t-1] + e[t]
            z[t] = β * z[t-1]
        end
        prev = @initialize m

        # Edit equation 2 only.
        m2 = ModelDef(:partial)
        @parameters m2 begin; α = 0.5; β = 0.5; end
        @variables m2 begin; y; z; end
        @shocks m2 begin; e; end
        @equations m2 begin
            y[t] = α * y[t-1] + e[t]
            z[t] = β * z[t-1] + α * z[t-2]    # changed
        end
        new_model, n_rebuilt = @reinitialize prev m2
        @test n_rebuilt == 1
        @test new_model[1] === prev[1]            # eq 1 reused
        @test new_model[2] !== prev[2]            # eq 2 rebuilt
        # Sanity: rebuilt eq has correct n_x (now includes z[t-2]).
        @test new_model[2].n_x == 3
    end

    @testset "compile: @reinitialize invalidates all on env change" begin
        m = ModelDef(:envchg)
        @parameters m begin; α = 0.5; end
        @variables m begin; y; end
        @equations m begin
            y[t] = α * y[t-1]
        end
        prev = @initialize m

        # Same equation, but param value changed -> env signature differs.
        m2 = ModelDef(:envchg)
        @parameters m2 begin; α = 0.7; end       # value changed
        @variables m2 begin; y; end
        @equations m2 begin
            y[t] = α * y[t-1]
        end
        new_model, n_rebuilt = @reinitialize prev m2
        @test n_rebuilt == 1
        @test new_model[1] !== prev[1]
    end

    @testset "compile: full simple_RBC initializes" begin
        m = ModelDef(:simple_RBC)
        @parameters m begin
            α = 0.33; δ = 0.1; ρ = 0.03; λ = 0.97; γ = 0.5; g = 0.015
            β = @link 1 / (1 + ρ)
        end
        @logvariables m begin; C; K; L; w; r; A; end
        @shocks m ea
        @equations m begin
            @log C[t+1] * (1 + g) = β * C[t] * (r[t+1] + 1 - δ)
            @log (L[t])^γ * C[t] = w[t]
            @log r[t] * (K[t-1] / (1 + g))^(1 - α) = α * A[t] * L[t]^(1 - α)
            @log w[t] * L[t]^(α) = (1 - α) * A[t] * (K[t-1] / (1 + g))^α
            @lin K[t] + C[t] = A[t] * (K[t-1] / (1 + g))^α * (L[t])^(1 - α) + (1 - δ) * (K[t-1] / (1 + g))
            log(A[t]) = λ * log(A[t-1]) + ea[t]
        end
        compiled = @initialize m
        @test length(compiled) == 6
        @test all(e -> e.eval_resid isa Function, compiled.eqns)
        # @lin flag preserved through compile.
        @test IR.EQ_LIN in compiled[5].flags
    end

    # ------------------------------------------------------------------
    # Selective linearization
    # ------------------------------------------------------------------

    @testset "linearize: identical results for an exactly linear equation" begin
        # F = y[t] - α*y[t-1] - c is already linear; linearizing it should
        # produce numerically identical evaluations everywhere.
        α_val = 0.4
        c_val = 1.5
        y_ss = c_val / (1 - α_val)
        m = ModelDef()
        @parameters m begin; α = 0.4; c = 1.5; end
        @variables m begin; y; end
        @equations m begin
            @lin y[t] = α * y[t-1] + c
        end
        compiled = @initialize m
        lin_model = selectively_linearize(compiled, [y_ss])
        @test IR.EQ_LIN in lin_model[1].flags

        # Evaluate at the same off-SS point through both eq's eval_RJ!.
        x_off = [3.0, 1.0]    # y[t-1]=3, y[t]=1
        p = [α_val, c_val]
        J_orig = zeros(2); J_lin = zeros(2)
        R_orig = compiled[1].eval_RJ!(J_orig, x_off, p)
        R_lin  = lin_model[1].eval_RJ!(J_lin,  x_off, p)
        @test R_orig ≈ R_lin atol=1e-12
        @test J_orig ≈ J_lin atol=1e-12
    end

    @testset "linearize: nonlinear equation matches at SS, deviates far from SS" begin
        # F = y[t]^2 - α*y[t-1] - c. SS: y_ss^2 = α*y_ss + c.
        # Picking α=1, c=2: y_ss = 2 (since 4 = 2 + 2). F'(y[t]) = 2*y[t] = 4 at SS.
        m = ModelDef()
        @parameters m begin; α = 1.0; c = 2.0; end
        @variables m begin; y; end
        @equations m begin
            @lin y[t]^2 = α * y[t-1] + c
        end
        compiled = @initialize m
        y_ss = 2.0
        lin_model = selectively_linearize(compiled, [y_ss])
        # At SS: residual should match (both ≈ 0), gradients should match exactly.
        x_ss = [y_ss, y_ss]
        p = [1.0, 2.0]
        J_orig = zeros(2); J_lin = zeros(2)
        R_orig = compiled[1].eval_RJ!(J_orig, x_ss, p)
        R_lin  = lin_model[1].eval_RJ!(J_lin,  x_ss, p)
        @test R_orig ≈ 0.0  atol=1e-10
        @test R_lin  ≈ 0.0  atol=1e-10
        @test J_orig ≈ J_lin atol=1e-12
        # Far from SS: linearization differs from the truth.
        x_far = [3.0, 5.0]
        R_orig_far = compiled[1].eval_resid(x_far, p)
        R_lin_far  = lin_model[1].eval_resid(x_far, p)
        @test !(R_orig_far ≈ R_lin_far)
        # Linearized form: R_ss + grad_ss · (x - x_ss) = 0 + [-1, 4] · ([3,5]-[2,2])
        #                = -1*1 + 4*3 = 11
        @test R_lin_far ≈ 11.0  atol=1e-10
    end

    @testset "linearize: hard-error on nonzero SS residual" begin
        m = ModelDef()
        @parameters m begin; α = 0.5; c = 1.0; end
        @variables m begin; y; end
        @equations m begin
            @lin y[t] = α * y[t-1] + c
        end
        compiled = @initialize m
        # Wrong SS: real y_ss = 2.0, supply 5.0 -> residual = 5 - 0.5*5 - 1 = 1.5
        @test_throws LinearizationError selectively_linearize(compiled, [5.0])
    end

    @testset "linearize: leaves non-@lin equations untouched (object identity)" begin
        m = ModelDef()
        @parameters m begin; α = 0.5; β = 0.6; cc = 1.0; end
        @variables m begin; y; z; end
        @equations m begin
            @lin y[t] = α * y[t-1] + cc
            z[t] = β * z[t-1] + cc      # not @lin
        end
        compiled = @initialize m
        y_ss = 1.0 / (1 - 0.5)
        z_ss = 1.0 / (1 - 0.6)
        lin_model = selectively_linearize(compiled, [y_ss, z_ss])
        # Eq 2 (non-@lin) should be the same Equation instance.
        @test lin_model[2] === compiled[2]
        # Eq 1 (@lin) must be a fresh Equation.
        @test lin_model[1] !== compiled[1]
        @test IR.EQ_LIN in lin_model[1].flags
    end

    @testset "linearize: shocks treated as zero in x_ss_per_slot" begin
        α_val = 0.5
        c_val = 1.0
        y_ss = c_val / (1 - α_val)
        m = ModelDef()
        @parameters m begin; α = 0.5; c = 1.0; end
        @variables m begin; y; end
        @shocks m begin; e; end
        @equations m begin
            @lin y[t] = α * y[t-1] + c + e[t]
        end
        compiled = @initialize m
        # SS residual: y_ss - α*y_ss - c - 0 = 0. Should not throw.
        lin_model = selectively_linearize(compiled, [y_ss])
        # Off-SS check: F = y - α*y_lag - c - e, linear -> exact.
        # tsrefs: y[t-1], y[t], e[t]; param: α, c.
        x = [3.0, 5.0, 0.7]
        p = [0.5, 1.0]
        R_orig = compiled[1].eval_resid(x, p)
        R_lin  = lin_model[1].eval_resid(x, p)
        @test R_orig ≈ R_lin  atol=1e-12
    end

    # ------------------------------------------------------------------
    # Model{Vector{Equation}} for large models
    # ------------------------------------------------------------------

    # Build a synthetic model with `n` equations of the form
    #   x_i[t] = α * x_i[t-1] + e[t]
    # independent recursions sharing one shock. Square (n_eq == n_var)
    # so the downstream solvers accept it.
    function _synthetic_model(n::Int; name::Symbol = :synthetic)
        def = ModelDef(name)
        IR.add_param!(def, IR.ParamDecl(:α, 0.5; kind = IR.PARAM_SCALAR))
        for i in 1:n
            IR.add_var!(def, IR.VarDecl(Symbol(:x, i)))
        end
        IR.add_shock!(def, IR.ShockDecl(:e))
        for i in 1:n
            xi = Symbol(:x, i)
            res = Expr(:call, :-,
                       Expr(:ref, xi, :t),
                       Expr(:call, :+,
                            Expr(:call, :*, :α, Expr(:ref, xi, :(t - 1))),
                            Expr(:ref, :e, :t)))
            IR.add_equation!(def,
                IR.EquationAST(res, Set{IR.EquationFlag}(), nothing,
                               LineNumberNode(i, name)))
        end
        return def
    end

    @testset "ChunkT: model_tuple_threshold reads ENV" begin
        old = get(ENV, "RW_MBE_TUPLE_THRESHOLD", nothing)
        try
            delete!(ENV, "RW_MBE_TUPLE_THRESHOLD")
            @test model_tuple_threshold() == DEFAULT_MODEL_TUPLE_THRESHOLD
            ENV["RW_MBE_TUPLE_THRESHOLD"] = "7"
            @test model_tuple_threshold() == 7
            ENV["RW_MBE_TUPLE_THRESHOLD"] = "not-an-int"
            @test_throws ErrorException model_tuple_threshold()
        finally
            if old === nothing
                delete!(ENV, "RW_MBE_TUPLE_THRESHOLD")
            else
                ENV["RW_MBE_TUPLE_THRESHOLD"] = old
            end
        end
    end

    @testset "ChunkT: small model below threshold builds as Tuple" begin
        old = get(ENV, "RW_MBE_TUPLE_THRESHOLD", nothing)
        try
            ENV["RW_MBE_TUPLE_THRESHOLD"] = "50"
            compiled = initialize_model(_synthetic_model(5))
            @test compiled.eqns isa Tuple
            @test length(compiled) == 5
            @test compiled isa CompiledModel
        finally
            old === nothing ? delete!(ENV, "RW_MBE_TUPLE_THRESHOLD") :
                              (ENV["RW_MBE_TUPLE_THRESHOLD"] = old)
        end
    end

    @testset "ChunkT: model above threshold builds as Vector{Equation}" begin
        old = get(ENV, "RW_MBE_TUPLE_THRESHOLD", nothing)
        try
            ENV["RW_MBE_TUPLE_THRESHOLD"] = "10"
            compiled = initialize_model(_synthetic_model(20))
            @test compiled.eqns isa Vector{Equation}
            @test length(compiled) == 20
            # getindex / iteration still work and evaluators are callable.
            @test compiled[1] isa Equation
            x = [0.7, 1.0, 0.05]   # xi[t-1], xi[t], e[t]
            p = [0.5]
            # F = xi[t] - α*xi[t-1] - e[t] = 1.0 - 0.35 - 0.05 = 0.6
            @test compiled[1].eval_resid(x, p) ≈ 0.6 atol = 1e-12
            @test compiled[20].eval_resid(x, p) ≈ 0.6 atol = 1e-12
        finally
            old === nothing ? delete!(ENV, "RW_MBE_TUPLE_THRESHOLD") :
                              (ENV["RW_MBE_TUPLE_THRESHOLD"] = old)
        end
    end

    @testset "ChunkT: threshold=0 forces vector even for a 1-eq model" begin
        old = get(ENV, "RW_MBE_TUPLE_THRESHOLD", nothing)
        try
            ENV["RW_MBE_TUPLE_THRESHOLD"] = "0"
            compiled = initialize_model(_synthetic_model(1))
            @test compiled.eqns isa Vector{Equation}
        finally
            old === nothing ? delete!(ENV, "RW_MBE_TUPLE_THRESHOLD") :
                              (ENV["RW_MBE_TUPLE_THRESHOLD"] = old)
        end
    end

    @testset "ChunkT: synthetic 200-equation model builds" begin
        old = get(ENV, "RW_MBE_TUPLE_THRESHOLD", nothing)
        try
            ENV["RW_MBE_TUPLE_THRESHOLD"] = "50"
            t0 = time()
            compiled = initialize_model(_synthetic_model(200))
            elapsed = time() - t0
            @test compiled.eqns isa Vector{Equation}
            @test length(compiled) == 200
            @test all(e -> e.eval_resid isa Function, compiled.eqns)
            @info "ChunkT: 200-equation build time" elapsed
        finally
            old === nothing ? delete!(ENV, "RW_MBE_TUPLE_THRESHOLD") :
                              (ENV["RW_MBE_TUPLE_THRESHOLD"] = old)
        end
    end

    @testset "ChunkT: @reinitialize preserves vector shape" begin
        old = get(ENV, "RW_MBE_TUPLE_THRESHOLD", nothing)
        try
            ENV["RW_MBE_TUPLE_THRESHOLD"] = "10"
            prev = initialize_model(_synthetic_model(20))
            @test prev.eqns isa Vector{Equation}
            new_model, n_rebuilt = reinitialize_model(prev, _synthetic_model(20))
            @test new_model.eqns isa Vector{Equation}
            @test n_rebuilt == 0
            @test new_model[1] === prev[1]
        finally
            old === nothing ? delete!(ENV, "RW_MBE_TUPLE_THRESHOLD") :
                              (ENV["RW_MBE_TUPLE_THRESHOLD"] = old)
        end
    end

    # ------------------------------------------------------------------
    # @lag / @lead / @d / @dlog meta-function expansion
    # ------------------------------------------------------------------

    @testset "metafunc expansion - syntactic forms" begin
        using ModelBaseEcon: MetaFuncs
        ex = MetaFuncs.expand_metafuncs

        # @lag / @lead shift every t-reference.
        @test ex(:(@lag(x[t]))) == :(x[t - 1])
        @test ex(:(@lag(x[t], 2))) == :(x[t - 2])
        @test ex(:(@lead(x[t]))) == :(x[t + 1])
        @test ex(:(@lag(x[t - 1]))) == :(x[t - 2])
        @test ex(:(@lead(x[t + 1], 2))) == :(x[t + 3])

        # @d first difference: @d(x) = x[t] - x[t-1].
        @test ex(:(@d(x[t]))) == :(x[t] - x[t - 1])
        # @d(x, 0, 1) = (1-L)^0 (1-L^1) x = x[t] - x[t-1].
        @test ex(:(@d(x[t], 0, 1))) == :(x[t] - x[t - 1])

        # @dlog = @d ∘ log.
        @test ex(:(@dlog(x[t]))) == :(log(x[t]) - log(x[t - 1]))
        @test ex(:(@d(log(x[t]), 0, 1))) == :(log(x[t]) - log(x[t - 1]))

        # Operates on whole sub-expressions, not just bare refs.
        @test ex(:(@d(x[t] / y[t - 1]))) ==
              :(x[t] / y[t - 1] - x[t - 1] / y[t - 2])

        # Nested meta-functions resolve bottom-up.
        @test ex(:(@d(@dlog(q[t - 1])))) ==
              :((log(q[t - 1]) - log(q[t - 2])) -
                (log(q[t - 2]) - log(q[t - 3])))

        # Non-meta macrocalls and plain expressions pass through untouched.
        @test ex(:(a[t] + b[t - 1])) == :(a[t] + b[t - 1])

        # Integer-literal guard on shift/order arguments.
        @test_throws ErrorException ex(:(@d(x[t], n)))
    end

    @testset "ChunkY1: @d expands inside @equations residual" begin
        # @d(x[t]) - x_a[t] = c  ⇒  residual carries the expanded lag.
        m = ModelDef(:dtest)
        @parameters m begin; c = 0.5; end
        @variables m begin; x; end
        @shocks m begin; x_a; end
        @equations m begin
            @d(x[t]) - x_a[t] = c
        end
        res = m.equations[1].residual
        @test !occursin("@d", string(res))
        @test occursin("t - 1", string(res))
        compiled = @initialize m
        @test length(compiled) == 1
    end

    @testset "ChunkY1: @d numerics match hand-expanded equation" begin
        # Two models, identical dynamics - one via @dlog, one hand-written.
        # The compiled residual + gradient, evaluated at a probe point,
        # must agree to machine precision.
        function probe(use_meta::Bool)
            m = ModelDef(:dnum)
            @variables m begin; y; end
            if use_meta
                @equations m begin
                    @dlog(y[t]) = 0.02
                end
            else
                @equations m begin
                    log(y[t]) - log(y[t - 1]) = 0.02
                end
            end
            kernels, _ = build_equation_kernels(m)
            k = kernels[1]
            funcs = build_equation_functions(k)
            # Build x in tsref order: y[t-1]=1.4, y[t]=1.65.
            x = [tr.offset == 0 ? 1.65 : 1.4 for tr in k.tsrefs]
            p = Float64[]
            J = zeros(funcs.n_x)
            F = funcs.eval_RJ!(J, x, p)
            # Sort gradient by offset so both models compare slot-aligned.
            order = sortperm([tr.offset for tr in k.tsrefs])
            return F, J[order]
        end
        Fm, Jm = probe(true)
        Fh, Jh = probe(false)
        @test Fm ≈ Fh  atol=1e-14
        @test Jm ≈ Jh  atol=1e-14
    end

    # ------------------------------------------------------------------
    # @exogenous block + inline @log in @variables
    # ------------------------------------------------------------------

    @testset "ChunkY2: @exogenous registers exogenous shock-list entries" begin
        m = ModelDef(:exotest)
        @variables m begin; y; z; end
        @exogenous m begin
            "Policy switch" dswitch
            ramp
        end
        @shocks m begin; e_y; end
        # Exogenous vars land in the shock list, tagged exogenous=true.
        @test length(m.shocks) == 3
        @test [s.name for s in m.shocks] == [:dswitch, :ramp, :e_y]
        @test is_exogenous(m, :dswitch)
        @test is_exogenous(m, :ramp)
        @test !is_exogenous(m, :e_y)
        @test exogenous_names(m) == [:dswitch, :ramp]
        @test m.shocks[1].doc == "Policy switch"
    end

    @testset "ChunkY2: @exogenous names usable in equations as data" begin
        # An exogenous var referenced in an equation resolves like a shock
        # (a data column), so the per-period system stays square.
        m = ModelDef(:exoeq)
        @parameters m begin; α = 0.5; end
        @variables m begin; y; end
        @exogenous m begin; g; end
        @equations m begin
            y[t] = α * y[t-1] + g[t]
        end
        compiled = @initialize m
        @test length(compiled) == 1
        # The equation's tsrefs include the exogenous `g`.
        names = Set(tr.name for tr in compiled[1].tsrefs)
        @test :g in names
        @test :y in names
    end

    @testset "ChunkY2: duplicate name across @exogenous rejected" begin
        m = ModelDef()
        @variables m begin; y; end
        @test_throws ErrorException @eval @exogenous $m begin; y; end
    end

    @testset "ChunkY2: inline @log in @variables sets VAR_LOG per entry" begin
        # FRBUS-style mixed block: some entries @log, some plain.
        m = ModelDef(:mixedlog)
        @variables m begin
            "plain output" y
            "log consumption" @log c
            k
            @log inv
        end
        @test [v.name for v in m.vars] == [:y, :c, :k, :inv]
        @test m.vars[1].kind === IR.VAR_NORMAL
        @test m.vars[2].kind === IR.VAR_LOG
        @test m.vars[2].doc == "log consumption"
        @test m.vars[3].kind === IR.VAR_NORMAL
        @test m.vars[4].kind === IR.VAR_LOG
    end

    # ------------------------------------------------------------------
    # @autoshocks
    # ------------------------------------------------------------------

    @testset "ChunkY3: @autoshocks generates one shock per variable" begin
        m = ModelDef(:astest)
        @variables m begin; y; z; end
        @autoshocks m _a
        @test [s.name for s in m.shocks] == [:y_a, :z_a]
        @test all(!s.exogenous for s in m.shocks)
        # Autoexogenize pairs created alongside.
        @test length(m.autoexog) == 2
        @test m.autoexog[1].var === :y && m.autoexog[1].shock === :y_a
        @test m.autoexog[2].var === :z && m.autoexog[2].shock === :z_a
    end

    @testset "ChunkY3: @autoshocks default suffix is _shk" begin
        m = ModelDef()
        @variables m begin; gdp; end
        @autoshocks m
        @test m.shocks[1].name === :gdp_shk
    end

    @testset "ChunkY3: @autoshocks skips exogenous variables" begin
        # Exogenous vars live in the shock list; @autoshocks only iterates
        # `m.vars` (endogenous), so no shock is made for them.
        m = ModelDef(:asx)
        @variables m begin; y; end
        @exogenous m begin; g; end
        @autoshocks m _a
        # shocks: exogenous g (pre-existing) + generated y_a.
        @test [s.name for s in m.shocks] == [:g, :y_a]
        @test is_exogenous(m, :g)
        @test !is_exogenous(m, :y_a)
        # autoexog pairs only for the endogenous variable.
        @test [p.var for p in m.autoexog] == [:y]
    end

    @testset "ChunkY3: @autoshocks shocks resolve in equations" begin
        # The generated `_a` shock is referenceable as a tsref.
        m = ModelDef(:aseq)
        @parameters m begin; ρ = 0.8; end
        @variables m begin; x; end
        @autoshocks m _a
        @equations m begin
            x[t] = ρ * x[t-1] + x_a[t]
        end
        compiled = @initialize m
        @test length(compiled) == 1
        @test :x_a in Set(tr.name for tr in compiled[1].tsrefs)
    end

    @testset "ChunkY2: @log variable applies exp() transform in codegen" begin
        # `@log y`: the solver unknown is log(y); every equation reference
        # to `y` becomes exp(unknown). A plain model that writes exp(Y)
        # explicitly must produce a bit-identical residual + gradient.
        function probe_log()
            m = ModelDef(:logv)
            @parameters m begin; c = 2.0; end
            @logvariables m begin; y; end
            @equations m begin
                y[t] = c * y[t-1]
            end
            kernels, _ = build_equation_kernels(m)
            return build_equation_functions(kernels[1]), kernels[1].tsrefs
        end
        function probe_plain()
            m = ModelDef(:plainv)
            @parameters m begin; c = 2.0; end
            @variables m begin; Y; end
            @equations m begin
                exp(Y[t]) = c * exp(Y[t-1])
            end
            kernels, _ = build_equation_kernels(m)
            return build_equation_functions(kernels[1]), kernels[1].tsrefs
        end
        flog, tlog = probe_log()
        fpln, tpln = probe_plain()
        # x in tsref order; pick the same log-space point for both.
        xlog = [tr.offset == 0 ? 0.3 : 0.1 for tr in tlog]
        xpln = [tr.offset == 0 ? 0.3 : 0.1 for tr in tpln]
        p = [2.0]
        Jl = zeros(flog.n_x); Jp = zeros(fpln.n_x)
        Fl = flog.eval_RJ!(Jl, xlog, p)
        Fp = fpln.eval_RJ!(Jp, xpln, p)
        @test Fl ≈ Fp  atol=1e-13
        ol = sortperm([tr.offset for tr in tlog])
        op = sortperm([tr.offset for tr in tpln])
        @test Jl[ol] ≈ Jp[op]  atol=1e-13
        # Residual sign check: at y_ss the recursion log-residual is
        # exp(0.3) - 2*exp(0.1) = 1.34986 - 2.21034 = -0.86048.
        @test Fl ≈ exp(0.3) - 2.0 * exp(0.1)  atol=1e-12
    end

    @testset "symbolic: build does not hang on a moderate equation (perf guard)" begin
        m = ModelDef()
        @variables m begin; a; b; c; d; e; end
        @equations m begin
            a[t] = (b[t-1] + c[t]) * (d[t+1] - e[t]) / (b[t] + c[t-1] + 1.0)
        end
        # Sanity timeout: just call it, assert it returns within reason.
        # If Symbolics' simplify hangs we want a hard failure here, not in CI.
        t0 = time()
        kernels, _ = build_equation_kernels(m; max_hod_order = 2)
        elapsed = time() - t0
        @test elapsed < 30.0
        @test length(kernels) == 1
    end

    include("equation_metadata.jl")
    include("export_model.jl")
    include("steadystate_user_eqns.jl")  # @steadystate user constraints
    include("dfm_models.jl")             # DFM DSL / IR / params

    @testset "@auxvar is dropped" begin
        # The legacy @auxvar / update_auxvars surface is dropped permanently.
        # The package must not export it, and using the unqualified macro
        # must fail loudly.
        @test !isdefined(ModelBaseEcon, Symbol("@auxvar"))
        @test !isdefined(ModelBaseEcon, :update_auxvars)
        @test !isdefined(ModelBaseEcon, :auxvars)
        @test !isdefined(ModelBaseEcon, :auxeqns)

        m = ModelDef(:auxvar_drop)
        @test_throws Exception @eval (@auxvar $m c_log = log(c))
    end
end
