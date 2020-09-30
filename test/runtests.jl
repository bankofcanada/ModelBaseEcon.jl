using ModelBaseEcon
using SparseArrays
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
    z = Options(o)
    z.name = "Oro"
    @test o.name == "Zoro"
end

@testset "Vars" begin
    y1 = :y
    y2 = ModelSymbol(:y)
    y3 = ModelSymbol("y3", :y)
    y4 = ModelSymbol(quote "y4" y end)
    @test hash(y1) == hash(:y)
    @test hash(y2) == hash(:y)
    @test hash(y3) == hash(:y)
    @test hash(y4) == hash(:y)
    @test_throws ArgumentError ModelSymbol(:(x + 5))
    @test y1 == y2
    @test y3 == y1
    @test y1 == y4
    @test y2 == y3
    @test y2 == y4
    @test y3 == y4
    ally = Symbol[y1,y2,y3,y4]
    @test y1 in ally
    @test y2 in ally
    @test y3 in ally
    @test y4 in ally
    @test indexin([y1,y2,y3,y4], ally) == [1,1,1,1]
    ally = ModelSymbol[y1,y2,y3,y4,:y,quote "y5" y end]
    @test indexin([y1,y2,y3,y4], ally) == [1,1,1,1]
    @test length(unique(hash.(ally))) == 1
    ally = Dict{Symbol,Any}()
    get!(ally, y1, "y1")
    get!(ally, y2, "y2")
    @test length(ally) == 1
    @test ally[y3] == "y1"
    ally = Dict{ModelSymbol,Any}()
    get!(ally, y1, "y1")
    get!(ally, y2, "y2")
    @test length(ally) == 1
    @test ally[y3] == "y1"
    @test sprint(print, y2, context=IOContext(stdout, :compact => true)) == "y"
    @test sprint(print, y2, context=IOContext(stdout, :compact => false)) == "y"
    @test sprint(print, y3, context=IOContext(stdout, :compact => true)) == "y"
    @test sprint(print, y3, context=IOContext(stdout, :compact => false)) == "\"y3\" y"

    @test ModelBaseEcon.makesym(Val(:level), y1) == ModelBaseEcon.makesym(Val(:level), y2)
    @test ModelBaseEcon.makesym(Val(:level), y1) == ModelBaseEcon.makesym(Val(:level), y3)
    @test ModelBaseEcon.makesym(Val(:level), y1) == ModelBaseEcon.makesym(Val(:level), "y")
    @test ModelBaseEcon.makesym(Val(:slope), y1) == ModelBaseEcon.makesym(Val(:slope), y2)
    @test ModelBaseEcon.makesym(Val(:slope), y1) == ModelBaseEcon.makesym(Val(:slope), y3)
    @test ModelBaseEcon.makesym(Val(:slope), y1) == ModelBaseEcon.makesym(Val(:slope), "y")
end

@testset "VarTypes" begin
    lvars = ModelSymbol[]
    push!(lvars, :ly)
    push!(lvars, quote "ly" ly end)
    push!(lvars, quote @log ly end)
    push!(lvars, quote "ly" @log ly end)
    push!(lvars, quote @lin ly end)
    push!(lvars, quote "ly" @lin ly end)
    push!(lvars, quote @steady ly end)
    push!(lvars, quote "ly" @steady ly end)
    push!(lvars, ModelSymbol(:ly, :lin))
    for i = 1:length(lvars)
        for j = i + 1:length(lvars)
            @test lvars[i] == lvars[j]
        end
        @test lvars[i] == :ly
    end
    @test lvars[1].type == :lin
    @test lvars[2].type == :lin
    @test lvars[3].type == :log
    @test lvars[4].type == :log
    @test lvars[5].type == :lin
    @test lvars[6].type == :lin
    @test lvars[7].type == :steady
    @test lvars[8].type == :steady
    @test lvars[9].type == :lin
    for i = 1:length(lvars)
        @test sprint(print, lvars[i], context=IOContext(stdout, :compact => true)) == "ly"
    end
    @test sprint(print, lvars[1], context=IOContext(stdout, :compact => false)) == "ly"
    @test sprint(print, lvars[2], context=IOContext(stdout, :compact => false)) == "\"ly\" ly"
    @test sprint(print, lvars[3], context=IOContext(stdout, :compact => false)) == "@log ly"
    @test sprint(print, lvars[4], context=IOContext(stdout, :compact => false)) == "\"ly\" @log ly"
    @test sprint(print, lvars[5], context=IOContext(stdout, :compact => false)) == "ly"
    @test sprint(print, lvars[6], context=IOContext(stdout, :compact => false)) == "\"ly\" ly"
    @test sprint(print, lvars[7], context=IOContext(stdout, :compact => false)) == "@steady ly"
    @test sprint(print, lvars[8], context=IOContext(stdout, :compact => false)) == "\"ly\" @steady ly"

    let m = Model()
        @variables m p q r
        @variables m begin
            x; @log y; @steady z;
        end
        @test [v.type for v in m.allvars] == [:lin, :lin, :lin, :lin, :log, :steady]
    end
    let m = Model()
        @shocks m p q r
        @shocks m begin
            x; @log y; @steady z;
        end
        @test [v.type for v in m.allvars] == [:shock, :shock, :shock, :shock, :shock, :shock]
    end
    let m = Model()
        @logvariables m p q r
        @logvariables m begin
            x; @log y; @steady z;
        end
        @test [v.type for v in m.allvars] == [:log, :log, :log, :log, :log, :log]
    end
    let m = Model()
        @steadyvariables m p q r
        @steadyvariables m begin
            x; @log y; @steady z;
        end
        @test [v.type for v in m.allvars] == [:steady, :steady, :steady, :steady, :steady, :steady]
    end
end

module E
    using ModelBaseEcon
end
@testset "Evaluations" begin
    ModelBaseEcon.initfuncs(E)
    E.eval(ModelBaseEcon.makefuncs(:(x + 3 * y), [:x, :y], mod=E))
    @test :resid_1 ∈ names(E, all=true)
    @test :RJ_1 ∈ names(E, all=true)
    @test E.resid_1([1.1, 2.3]) == 8.0
    @test E.RJ_1([1.1, 2.3]) == (8.0, [1.0, 3.0])
end

@testset "Misc" begin
    m = Model()
    out = let io = IOBuffer()
        print(io, m.flags)
        readlines(seek(io, 0))
    end
    @test length(out) == 3
    for line in out[2:end]
        sline = strip(line)
        @test isempty(sline) || length(split(sline, "=")) == 2
    end
    @test_throws ModelBaseEcon.ModelErrorBase ModelBaseEcon.modelerror()
    @variables m x y z
    @logvariables m k l m
    @steadyvariables m p q r
    @shocks m a b c
    for s in (:a, :b, :c)
        @test m.:($s) isa ModelSymbol && isshock(m.:($s))
    end
    for s in (:x, :y, :z)
        @test m.:($s) isa ModelSymbol && islin(m.:($s))
    end
    for s in (:k, :l, :m)
        @test m.:($s) isa ModelSymbol && islog(m.:($s))
    end
    for s in (:p, :q, :r)
        @test m.:($s) isa ModelSymbol && issteady(m.:($s))
    end
end


module MetaTest
using ModelBaseEcon
    params = @parameters
    custom(x) = x + one(x)
    const val = 12.0
    params.b = custom(val)
    params.a = @link custom(val)
end

@testset "Parameters" begin
    params = Parameters()
    push!(params, :a => 1.0)
    push!(params, :b => @link 1.0 - a)
    push!(params, :c => @alias b)
    push!(params, :e => [1,2,3])
    push!(params, :d => @link (sin(2π / e[3])))
    @test length(params) == 5
    # dot notation evaluates
    @test params.a isa Number
    @test params.b isa Number
    @test params.c isa Number
    @test params.d isa Number
    @test params.e isa Vector{<:Number}
    # [] notation returns the holding structure
    a = params[:a]
    b = params[:b]
    c = params[:c]
    d = params[:d]
    e = params[:e]
    @test a isa ModelParam
    @test b isa ModelParam
    @test c isa ModelParam
    @test d isa ModelParam
    @test e isa ModelParam
    @test a.depends == Set([:b])
    @test b.depends == Set([:c])
    @test c.depends == Set([])
    @test d.depends == Set([])
    @test e.depends == Set([:d])
    # circular dependencies not allowed
    @test_throws ArgumentError push!(params, :a => @alias b)
    # even deep ones
    @test_throws ArgumentError push!(params, :a => @alias c)
    # even when it is in an expr
    @test_throws ArgumentError push!(params, :a => @link 5 + b^2)
    @test_throws ArgumentError push!(params, :a => @link 3 - c)

    @test params.d ≈ √3 / 2.0
    params.e[3] = 2
    update_links!(params)
    @test 1.0 + params.d ≈ 1.0
    
    params.d = @link cos(2π / e[2])
    @test params.d ≈ -1.0

    @test_throws ArgumentError @alias a + 5
    @test_throws ArgumentError @link 28

    @test MetaTest.params.a ≈ 13.0
    @test MetaTest.params.b ≈ 13.0
    MetaTest.eval(quote custom(x) = 2x + one(x) end)
    update_links!(MetaTest.params)
    @test MetaTest.params.a ≈ 25.0
    @test MetaTest.params.b ≈ 13.0

    @test @alias(c) == ModelParam(Set(), :c, nothing)
    @test @link(c) == ModelParam(Set(), :c, nothing)
    @test @link(c + 1) == ModelParam(Set(), :(c + 1), nothing)

    @test_throws ArgumentError params[:contents] = 5
    @test_throws ArgumentError params.abc
end

@testset "ifelse" begin
    m = Model()
    @variables m x
    @equations m begin
        x[t] = 0
    end
    @initialize m
    @test_throws ArgumentError ModelBaseEcon.process_equation(m, :(y[t] = 0))
    @test_throws ArgumentError ModelBaseEcon.process_equation(m, :(x[t] = p))
    @test_throws ArgumentError ModelBaseEcon.process_equation(m, :(x[t] = if false 2 end))
    @test ModelBaseEcon.process_equation(m, :(x[t] = if false 2 else 0 end)) isa Equation
    @test ModelBaseEcon.process_equation(m, :(x[t] = ifelse(false, 2, 0))) isa Equation
end

@testset "Meta" begin
    mod = Model()
    @parameters mod a = 0.1 b = @link(1.0 - a)
    @variables mod x
    @shocks mod sx
    @equations mod begin
    x[t - 1] = sx[t + 1]
        @lag(x[t]) = @lag(sx[t + 2])
        # 
        x[t - 1] + a = sx[t + 1] + 3
        @lag(x[t] + a) = @lag(sx[t + 2] + 3)
        # 
        x[t - 2] = sx[t]
        @lag(x[t], 2) = @lead(sx[t - 2], 2)
        # 
        x[t] - x[t - 1] = x[t + 1] - x[t] + sx[t]
        @d(x[t]) = @d(x[t + 1]) + sx[t]
        # 
        (x[t] - x[t + 1]) - (x[t - 1] - x[t]) = sx[t]
        @d(x[t] - x[t + 1]) = sx[t]
        # 
        x[t] - x[t - 2] = sx[t]
        @d(x[t],0,2) = sx[t]
        # 
        (x[t] - x[t - 1]) - (x[t - 1] - x[t - 2]) = sx[t]
        @d(x[t],2) = sx[t]
        # 
        (x[t] - x[t - 2]) - (x[t - 1] - x[t - 3]) = sx[t]
        @d(x[t],1,2) = sx[t]
        # 
        log(x[t] - x[t - 2]) - log(x[t - 1] - x[t - 3]) = sx[t]
        @dlog(@d(x[t],0,2)) = sx[t]
        # 
        (x[t] + 0.3x[t + 2]) + (x[t - 1] + 0.3x[t + 1]) + (x[t - 2] + 0.3x[t]) = 0
        @movsum(x[t] + 0.3x[t + 2],3) = 0
        # 
        ((x[t] + 0.3x[t + 2]) + (x[t - 1] + 0.3x[t + 1]) + (x[t - 2] + 0.3x[t])) / 3 = 0
        @movav(x[t] + 0.3x[t + 2],3) = 0
    end
    @initialize mod

    compare_resids(e1, e2) = (
        e1.resid.head == e2.resid.head  && (
            (length(e1.resid.args) == length(e2.resid.args) == 2 && e1.resid.args[2] == e2.resid.args[2]) ||
            (length(e1.resid.args) == length(e2.resid.args) == 1 && e1.resid.args[1] == e2.resid.args[1])
        )
    )

    for i = 2:2:length(mod.equations)
        @test compare_resids(mod.equations[i - 1], mod.equations[i])
    end
    # test errors and warnings
    mod.warn.no_t = false
    @test  add_equation!(mod, :(x = sx[t])) isa Model
    @test  add_equation!(mod, :(x[t] = sx)) isa Model
    @test  add_equation!(mod, :(x[t] = sx[t])) isa Model
    @test compare_resids(mod.equations[end], mod.equations[end - 1])
    @test compare_resids(mod.equations[end], mod.equations[end - 2])
    @test_throws ArgumentError add_equation!(mod, :(@notametafunction(x[t]) = 7))
    @test_throws ArgumentError add_equation!(mod, :(x[t] = unknownsymbol))
    @test_throws ArgumentError add_equation!(mod, :(x[t] = unknownseries[t]))
    @test_throws ArgumentError add_equation!(mod, :(x[t] = let c = 5; sx[t + c]; end))

end

############################################################################

@testset "export" begin
    let m = Model()
        m.warn.no_t = false
        @parameters m begin
            a = 0.3
            b = @link 1 - a
            d = [1,2,3]
            c = @link sin(2π / d[3])
        end
        @variables m begin
            "variable x" x
        end
        @shocks m sx
        @equations m begin
            "This equation is super cool"
            a * @d(x) = b * @d(x[t + 1]) + sx
        end
        @initialize m
        @steadystate m x = a + 1

        export_model(m, "TestModel", "../examples/")

        @test isfile("../examples/TestModel.jl")
        @using_example TestModel

        @test parameters(TestModel.model) == parameters(m)
        @test variables(TestModel.model) == variables(m)
        @test shocks(TestModel.model) == shocks(m)
        @test equations(TestModel.model) == equations(m)
        @test sstate(TestModel.model).constraints == sstate(m).constraints

        @test_throws ArgumentError TestModel.model.parameters.d = @alias c

        rm("../examples/TestModel.jl")
    end
end

@testset "@log eqn" begin
    let m = Model()
        @parameters m rho = 0.1
        @variables m X
        @shocks m EX
        @equations m begin
            @log X[t] = rho * X[t-1] + EX[t]
        end
        @initialize m
        @test length(m.equations) == 1 && islog(m.equations[1])
    end
end

############################################################################

function test_eval_RJ(m::Model, known_R, known_J)
    nrows = 1 + m.maxlag + m.maxlead
    ncols = length(m.allvars)
    R, J = eval_RJ(zeros(nrows, ncols), m)
    @test R ≈ known_R
    @test J ≈ known_J
end

function compare_RJ_R!_(m::Model)
    nrows = 1 + m.maxlag + m.maxlead
    ncols = length(m.variables) + length(m.shocks) + length(m.auxvars)
    point = rand(nrows, ncols)
    R, J = eval_RJ(point, m)
    S = similar(R)
    eval_R!(S, point, m)
    @test R ≈ S
end

@using_example E1
@testset "E1" begin
    @test length(E1.model.parameters) == 2
    @test length(E1.model.variables) == 1
    @test length(E1.model.shocks) == 1
    @test length(E1.model.equations) == 1
    @test E1.model.maxlag == 1
    @test E1.model.maxlead == 1
    test_eval_RJ(E1.model, [0.0], [-0.5 1.0 -0.5 0.0 -1.0 0.0])
    compare_RJ_R!_(E1.model)
    @test E1.model.tol == E1.model.options.tol
    tol = E1.model.tol
    E1.model.tol = tol * 10
    @test E1.model.options.tol == E1.model.tol
    E1.model.tol = tol
    @test E1.model.linear == E1.model.flags.linear
    E1.model.linear = true
    @test E1.model.linear
end

@testset "E1.sstate" begin
    let io = IOBuffer(), m = E1.model
        @test issssolved(m) == false
        E1.model.sstate.mask .= true
        @test issssolved(m) == true
        @test neqns(m.sstate) == 2
        @steadystate m y = 5
        @test_throws ErrorException @steadystate m sin(y + 7)
        @test length(m.sstate.constraints) == 1
        @test neqns(m.sstate) == 3
        @test length(alleqns(m.sstate)) == 3
        printsstate(io, m)
        lines = split(String(take!(io)), '\n')
        @test length(lines) == 2 + length(m.allvars)
    end
end

@testset "E1.lin" begin
    m = deepcopy(E1.model)
    with_linearized(m) do lm
        @test islinearized(lm)
        test_eval_RJ(lm, [0.0], [-0.5 1.0 -0.5 0.0 -1.0 0.0])
        compare_RJ_R!_(lm)
    end
    @test !islinearized(m)
    lm = linearized(m)
    test_eval_RJ(lm, [0.0], [-0.5 1.0 -0.5 0.0 -1.0 0.0])
    compare_RJ_R!_(lm)
    @test islinearized(lm)
    @test !islinearized(m)
    linearize!(m)
    @test islinearized(m)
end

@testset "E1.params" begin
    let m = E1.model
        @test propertynames(m.parameters) == (:α, :β)
        m.β = @link 1.0 - α
        m.parameters.beta = @alias β
        for α = 0.0:0.1:1.0
            m.α = α
            test_eval_RJ(m, [0.0], [-α 1.0 -m.beta 0.0 -1.0 0.0;])
        end
    end
    let io = IOBuffer(), m = E1.model
        show(io, m.parameters)
        @test length(split(String(take!(io)), '\n')) == 1
        show(io, MIME"text/plain"(), m.parameters)
        @test length(split(String(take!(io)), '\n')) == 4
    end
    end

module AUX
using ModelBaseEcon
model = Model()
model.substitutions = true
@variables model x y
@equations model begin
    x[t + 1] = log(x[t] - x[t - 1])
    y[t + 1] = y[t] + log(y[t - 1])
end
@initialize model
end
@testset "AUX" begin
    let m = AUX.model
    @test m.nvars == 2
        @test m.nshks == 0
        @test m.nauxs == 2
        @test length(m.auxeqns) == 2
        x = ones(2, 2)
        @test_throws ErrorException ModelBaseEcon.update_auxvars(x, m)
        x = 2 .* ones(4, 2)
        ax = ModelBaseEcon.update_auxvars(x, m; default=0.1)
        @test size(ax) == (4, 4)
        @test x == ax[:, 1:2]
        @test ax[:, 3:4] == [0.0 0.0; 0.1 log(2.0); 0.1 log(2.0); 0.0 0.0]
    end
    end


@using_example E2
@testset "E2" begin
    @test length(E2.model.parameters) == 3
    @test length(E2.model.variables) == 3
    @test length(E2.model.shocks) == 3
    @test length(E2.model.equations) == 3
    @test E2.model.maxlag == 1
    @test E2.model.maxlead == 1
    test_eval_RJ(E2.model, [0.0, 0.0, 0.0],
        [-.5      1  -.48     0    0  0    0   -.02     0  0  -1  0  0  0 0 0  0 0;
           0  -.375     0  -.75    1  0    0  -.125     0  0   0  0  0 -1 0 0  0 0;
           0      0  -.02     0  .02  0  -.5      1  -.48  0   0  0  0  0 0 0 -1 0])
    compare_RJ_R!_(E2.model)
end

@testset "E2.sstate" begin
    m = E2.model
    ss = m.sstate
    empty!(ss.constraints)
    out = let io = IOBuffer()
        print(io, ss)
        readlines(seek(io, 0))
    end
    @test length(out) == 2
    @steadystate m pinf = rate + 1
    out = let io = IOBuffer()
        print(io, ss)
        readlines(seek(io, 0))
    end
    @test length(out) == 3
    @test length(split(out[end], "=")) == 2
    # 
    @test propertynames(ss) == tuple(m.allvars...)
    @test ss.pinf.level == ss.pinf[1]
    @test ss.pinf.slope == ss.pinf[2]
    ss.pinf = (level = 2.3, slope = 0.7)
    @test ss.values[1:2] == [2.3, 0.7]
    @test ss[:rate] == ss["rate"]
    ss["rate"].level = 21
    ss[:rate].slope = 0.21
    @test ss[:rate].level == 21 && ss["rate"].slope == 0.21
end

@using_example E3
@testset "E3" begin
    @test length(E3.model.parameters) == 3
    @test length(E3.model.variables) == 3
    @test length(E3.model.shocks) == 3
    @test length(E3.model.equations) == 3
    @test E3.model.maxlag == 2
    @test E3.model.maxlead == 3
    compare_RJ_R!_(E3.model)
    test_eval_RJ(E3.model, [0.0, 0.0, 0.0],
        sparse(
            [1, 1, 2, 1, 3, 1, 1, 2, 2, 3,  3,  3,  1,  2,  3,  3,  1,  2,  3],
            [2, 3, 3, 4, 4, 5, 6, 8, 9, 9, 13, 14, 15, 15, 15, 16, 21, 27, 33],
            [-0.5, 1.0, -0.375, -0.3, -0.02, -0.05, -0.05, -0.75, 1.0, 0.02, -0.25,
             -0.25, -0.02, -0.125, 1.0, -0.48, -1.0, -1.0, -1.0],
            3, 36,
        )
    )
end

@using_example E6
@testset "E6" begin
    @test length(E6.model.parameters) == 2
    @test length(E6.model.variables) == 6
    @test length(E6.model.shocks) == 2
    @test length(E6.model.equations) == 6
    @test E6.model.maxlag == 2
    @test E6.model.maxlead == 3
    compare_RJ_R!_(E6.model)
    nt = 1 + E6.model.maxlag + E6.model.maxlead
    test_eval_RJ(E6.model, [-0.0027, -0.0025, 0.0, 0.0, 0.0, 0.0],
        sparse(
            [2, 2, 2, 3, 5, 2, 2, 2, 1, 1, 3, 4, 1, 3, 6, 5, 5, 4, 4, 6, 6, 2, 1],
            [1, 2, 3, 3, 3, 4, 5, 6, 8, 9, 9, 9, 10, 15, 15, 20, 21, 26, 27, 32, 33, 39, 45],
            [-0.1, -0.1, 1.0, -1.0, -1.0, -0.1, -0.1, -0.1, -0.2, 1.0, -1.0, -1.0, -0.2, 1.0,
             -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0],
            6, 6 * 8,
        ))
end

@testset "VarTypesSS" begin
    let m = Model()

    m.verbose = !true

        @variables m begin
            p
            @log q
        end
        @equations m begin
            2p[t] = p[t + 1] + 0.1
            q[t] = p[t] + 1
        end
        @initialize m

        # clear_sstate!(m)
        # ret = sssolve!(m)
        # @test ret ≈ [0.1, 0.0, log(1.1), 0.0]

        eq1, eq2, eq3, eq4 = m.sstate.equations
        x = rand(Float64, (4,))
        R, J = eq1.eval_RJ(x[eq1.vinds])
        @test R ≈ x[1] - x[2] - 0.1
        @test J ≈ [1.0, -1.0, 0, 0][eq1.vinds]

        for sh = 1:5
            m.shift = sh
            R, J = eq3.eval_RJ(x[eq3.vinds])
            @test R ≈ x[1] + (sh - 1) * x[2] - 0.1
            @test J ≈ [1.0, sh - 1.0, 0, 0][eq3.vinds]
        end

        R, J = eq2.eval_RJ(x[eq2.vinds])
        @test R ≈ exp(x[3]) - x[1] - 1
        @test J ≈ [-1, 0.0, exp(x[3]), 0.0][eq2.vinds]

        for sh = 1:5
            m.shift = sh
            R, J = eq4.eval_RJ(x[eq4.vinds])
            @test R ≈ exp(x[3] + sh * x[4]) - x[1] - sh * x[2] - 1
            @test J ≈ [-1.0, -sh, exp(x[3] + sh * x[4]), exp(x[3] + sh * x[4]) * sh][eq4.vinds]
        end

    end

    let m = Model()
        @variables m begin
            lx
            @log x
        end
        @shocks m s1 s2
        @equations m begin
            "linear growth with slope 0.2"
            lx[t] = lx[t - 1] + 0.2 + s1[t]
            "exponential with the same rate as the slope of lx"
            log(x[t]) = lx[t] + s2[t + 1]
        end
        @initialize m
        # 
        @test nvariables(m) == 2
        @test nshocks(m) == 2
        @test nequations(m) == 2
        ss = sstate(m)
        @test neqns(ss) == 4
        eq1, eq2, eq3, eq4 = ss.equations
        @test length(ss.values) == 2 * length(m.allvars)
        # 
        # test with eq1
        ss.lx = [1.5, 0.2]
        ss.x = [0.0, 0.2]
        ss.s1 = [0.0, 0.0]
        ss.s2 = [0.0, 0.0]
        for s1 = -2:0.1:2
            ss.s1.level = s1
            @test eq1.eval_resid(ss.values[eq1.vinds]) ≈ -s1
        end
        ss.s1.level = 0.0
        for lxslp = -2:0.1:2
            ss.lx.slope = lxslp
            @test eq1.eval_resid(ss.values[eq1.vinds]) ≈ lxslp - 0.2
        end
        ss.lx.slope = 0.2
        R, J = eq1.eval_RJ(ss.values[eq1.vinds])
        TMP = fill!(similar(ss.values), 0.0)
        TMP[eq1.vinds] .= J
        @test R == 0
        @test TMP[[1,2,5]] ≈ [0.0, 1.0, -1.0]
        # test with eq4
        ss.lx = [1.5, 0.2]
        ss.x = [1.5, 0.2]
        ss.s1 = [0.0, 0.0]
        ss.s2 = [0.0, 0.0]
        for s2 = -2:0.1:2
            ss.s2.level = s2
            @test eq4.eval_resid(ss.values[eq4.vinds]) ≈ -s2
        end
        ss.s2.level = 0.0
        for lxslp = -2:0.1:2
            ss.lx.slope = lxslp
            @test eq4.eval_resid(ss.values[eq4.vinds]) ≈ m.shift * (0.2 - lxslp)
        end
        ss.lx.slope = 0.2
        for xslp = -2:0.1:2
            ss.x.slope = xslp
            @test eq4.eval_resid(ss.values[eq4.vinds]) ≈ m.shift * (xslp - 0.2)
        end
        ss.x.slope = 0.2
        R, J = eq4.eval_RJ(ss.values[eq4.vinds])
        TMP = fill!(similar(ss.values), 0.0)
        TMP[eq4.vinds] .= J
        @test R ≈ 0.0
        @test TMP[[1,2,3,4,7]] ≈ [-1.0, -m.shift, 1.0, m.shift, -1.0]
        for xlvl = 0.1:0.1:2
            ss.x.level = xlvl
            R, J = eq4.eval_RJ(ss.values[eq4.vinds])
            @test R ≈ xlvl - 1.5
            TMP[eq4.vinds] .= J
            @test TMP[[1,2,3,4,7]] ≈ [-1.0, -m.shift, 1.0, m.shift, -1.0]
        end
    end
end
