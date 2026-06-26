@testset "G9: equation doc and tag metadata" begin

    @testset "docstring attached to equation" begin
        m = ModelDef()
        @variables m begin; y; i; g; c; end
        @equations m begin
            "GDP identity"
            c[t] = y[t] - i[t] - g[t]
        end
        mdl = initialize_model(m)
        @test length(mdl.eqns) == 1
        @test doc(mdl.eqns[1]) == "GDP identity"
        @test isempty(tags(mdl.eqns[1]))
    end

    @testset "single tag attached to equation" begin
        m = ModelDef()
        @variables m begin; y; i; g; c; end
        @equations m begin
            :identity => c[t] = y[t] - i[t] - g[t]
        end
        mdl = initialize_model(m)
        @test :identity in tags(mdl.eqns[1])
        @test length(tags(mdl.eqns[1])) == 1
        @test doc(mdl.eqns[1]) == ""
    end

    @testset "docstring and tag together" begin
        m = ModelDef()
        @variables m begin; y; i; g; c; end
        @equations m begin
            "GDP identity"
            :identity => c[t] = y[t] - i[t] - g[t]
        end
        mdl = initialize_model(m)
        @test doc(mdl.eqns[1]) == "GDP identity"
        @test :identity in tags(mdl.eqns[1])
    end

    @testset "tag survives with @lin flag" begin
        m = ModelDef()
        @variables m begin; y; i; g; c; end
        @equations m begin
            :identity => @lin c[t] = y[t] - i[t] - g[t]
        end
        mdl = initialize_model(m)
        @test :identity in tags(mdl.eqns[1])
        @test IR.EQ_LIN in mdl.eqns[1].flags
    end

    @testset "no tag is empty Vector{Symbol}" begin
        m = ModelDef()
        @variables m begin; y; i; g; c; end
        @equations m begin
            c[t] = y[t] - i[t] - g[t]
        end
        mdl = initialize_model(m)
        @test tags(mdl.eqns[1]) isa Vector{Symbol}
        @test isempty(tags(mdl.eqns[1]))
    end

    @testset "non-Symbol tag raises at macro expansion" begin
        m = ModelDef()
        @variables m begin; y; i; g; c; end
        @test_throws Exception @eval @equations $m begin
            "not_a_symbol" => c[t] = y[t] - i[t] - g[t]
        end
    end

    @testset "Base.show prints doc and tag prefix" begin
        m = ModelDef()
        @variables m begin; y; i; g; c; end
        @equations m begin
            "GDP identity"
            :identity => c[t] = y[t] - i[t] - g[t]
        end
        mdl = initialize_model(m)
        s = sprint(show, mdl.eqns[1])
        @test occursin("GDP identity", s)
        @test occursin("identity", s)
    end

    @testset "EquationAST 4-arg backward-compat constructor (no tags)" begin
        # Existing call sites that construct EquationAST without tags
        # must keep working - IR exposes a 4-arg fallback.
        ast = IR.EquationAST(:(a - b), Set{IR.EquationFlag}(), nothing, LineNumberNode(0))
        @test ast.tags isa Vector{Symbol}
        @test isempty(ast.tags)
    end

    @testset "tags survive @reinitialize for unchanged equation" begin
        m = ModelDef()
        @variables m begin; y; i; g; c; end
        @equations m begin
            :identity => c[t] = y[t] - i[t] - g[t]
        end
        mdl = initialize_model(m)
        new_mdl, n_rebuilt = reinitialize_model(mdl, m)
        @test :identity in tags(new_mdl.eqns[1])
        @test n_rebuilt == 0
    end

end
