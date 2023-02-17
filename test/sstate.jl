
@using_example S1
@testset "dynss" begin
    m = deepcopy(S1.model)
    @test m.dynss
    @test isempty(m.equations[:_EQ1].ssrefs)
    @test !isempty(m.equations[:_EQ2].ssrefs)
    @test !isempty(m.equations[:_EQ3].ssrefs)
    @test_logs (:warn, r".*steady\s+state.*"i) refresh_med!(m)
    ss = m.sstate
    fill!(ss.values, 0)
    fill!(ss.mask, true)
    let
        nrows = 1 + m.maxlag + m.maxlead
        ncols = length(m.allvars)
        ss.a.level = 3
        ss.b.level = 1
        ss.c.level = 2
        @test_logs refresh_med!(m)
        R, J = eval_RJ(zeros(nrows, ncols), m)
        @test R ≈ [0, -0.5, -0.4] atol = 1e-12
        @test J ≈ [0 1 0 -1 0 -1 0 0 0 0
            0 0 -0.5 1.0 0 0 0 -1 0 0
            0 0 0 0 -0.8 1 0 0 0 -1]
        # we can pick up changes in parameters ...
        m.α = 0.6
        m.β = 0.4
        R, J = eval_RJ(zeros(nrows, ncols), m)
        @test R ≈ [0, -0.4, -1.2] atol = 1e-12
        @test J ≈ [0 1 0 -1 0 -1 0 0 0 0
            0 0 -0.6 1.0 0 0 0 -1 0 0
            0 0 0 0 -0.4 1 0 0 0 -1]
        # ... but not changes in steady state ...
        ss.a.level = 6
        ss.b.level = 2
        ss.c.level = 4
        R, J = eval_RJ(zeros(nrows, ncols), m)
        @test R ≈ [0, -0.4, -1.2] atol = 1e-12
        @test J ≈ [0 1 0 -1 0 -1 0 0 0 0
            0 0 -0.6 1.0 0 0 0 -1 0 0
            0 0 0 0 -0.4 1 0 0 0 -1]
        # that requires refresh
        refresh_med!(m)
        R, J = eval_RJ(zeros(nrows, ncols), m)
        @test R ≈ [0, -0.8, -2.4] atol = 1e-12
        @test J ≈ [0 1 0 -1 0 -1 0 0 0 0
            0 0 -0.6 1.0 0 0 0 -1 0 0
            0 0 0 0 -0.4 1 0 0 0 -1]
    end
    let
        seq = ss.equations[3]
        inds = indexin([Symbol("#c#lvl#"), Symbol("#c#slp#"), Symbol("#b#lvl#")], seq.vsyms)
        for i = 1:50
            m.β = β = rand()
            m.α = α = rand()
            m.q = q = 2 + (8 - 2) * rand()
            _, J = seq.eval_RJ(ss.values[seq.vinds])
            @test J[inds] ≈ [1 - β, β, -q * (1 - β)]
        end
    end
    m.sstate.b.slope = 0.1
    @test_logs (:warn, r".*non-zero slope.*"i) (:warn, r".*non-zero slope.*"i) refresh_med!(m)
end

@using_example S2
@testset "dynss2" begin
    m = deepcopy(S2.model)
    # make sure @sstate(x) was transformed
    @test m.equations[:_EQ1].ssrefs[:x] === Symbol("#log#x#ss#")

    xi = ModelBaseEcon._index_of_var(:x, m.variables)
    for i = 1:10
        x = 0.1 + 6*rand()
        m.sstate.x.level = x
        @test m.sstate.values[2xi-1] ≈ log(x)
    end
end