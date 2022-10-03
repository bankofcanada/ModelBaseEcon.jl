##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

using ModelBaseEcon
using SparseArrays
using Test

@testset "Tranformations" begin
    @test_throws ErrorException transformation(Transformation)
    @test_throws ErrorException inverse_transformation(Transformation)
    let m = Model()
        @variables m begin
            x
            @log lx
            @neglog lmx
        end
        @test length(m.variables) == 3
        @test m.x.tr_type === :none
        @test m.lx.tr_type === :log
        @test m.lmx.tr_type === :neglog
        data = rand(20)
        @test transform(data, m.x) ≈ data
        @test inverse_transform(data, m.x) ≈ data
        @test transform(data, m.lx) ≈ log.(data)
        @test inverse_transform(log.(data), m.lx) ≈ data
        mdata = -data
        @test transform(mdata, m.lmx) ≈ log.(data)
        @test inverse_transform(log.(data), m.lmx) ≈ mdata
        @test !need_transform(:y)
        y = to_lin(:y)
        @test y.tr_type === :none
        @logvariables m lmy
        @neglogvariables m ly
        @test_throws ArgumentError m.ly = 25
        @test_throws ArgumentError m.lmy = -25
        @test_throws ArgumentError m.ly = ModelVariable(:lmy)
        @test_throws ErrorException update(m.ly, tr_type=:log, transformation=NoTransform)
        m.ly = update(m.ly, transformation=LogTransform)
        m.lmy = update(m.lmy, tr_type=:neglog, transformation=NegLogTransform)
        @test m.ly.tr_type === :log
        @test m.lmy.tr_type === :neglog
        
        @test_throws ErrorException m.dummy = nothing
        
    end
end

@testset "Options" begin
    o = Options(tol=1e-7, maxiter=25)
    @test propertynames(o) == (:maxiter, :tol)
    @test getoption(o, tol=1e7) == 1e-7
    @test getoption(o, "name", "") == ""
    @test getoption(o, abstol=1e-10, name="") == (1e-10, "")
    @test all(["abstol","name"] .∉ Ref(o))
    @test getoption!(o, abstol=1e-11) == 1e-11
    @test :abstol ∈ o
    @test setoption!(o, reltol=1e-3, linear=false) isa Options
    @test all(["reltol", :linear] .∈ Ref(o))
    @test getoption!(o, tol=nothing, linear=true, name="Zoro") == (1e-7, false, "Zoro")
    @test "name" ∈ o && o.name == "Zoro"
    z = Options()
    @test merge(z, o) == Options(o...) == Options(o)
    @test merge!(z, o) == Options(Dict(string(k) => v for (k,v) in pairs(o))...)
    @test o == z
    @test Dict(o...) == z
    @test o == Dict(z...)
    z.name = "Oro"
    @test o.name == "Zoro"
    @test setoption!(z, "linear", true) isa Options
    @test getoption!(z, "linear", false) == true
    @test getoption!(z, :name, "") == "Oro"
    @test show(IOBuffer(), o) === nothing
    @test show(IOBuffer(), MIME"text/plain"(), o) === nothing

    @using_example S1
    m = S1.model
    @test getoption(m, "shift", 1) == getoption(m, shift=1) == 10
    @test getoption!(m, "substitutions", true) == getoption!(m, :substitutions, true) == false
    @test getoption(setoption!(m, "maxiter", 25), maxiter = 0) == 25
    @test getoption(setoption!(m, verbose = true), "verbose", false) == true
    @test typeof(setoption!(println, m)) == Options
end

@testset "Vars" begin
    y1 = :y
    y2 = ModelSymbol(:y)
    y3 = ModelSymbol("y3", :y)
    y4 = ModelSymbol(quote
        "y4"
        y
    end)
    @test hash(y1) == hash(:y)
    @test hash(y2) == hash(:y)
    @test hash(y3) == hash(:y)
    @test hash(y4) == hash(:y)
    @test hash(y4, UInt(0)) == hash(:y, UInt(0))
    @test_throws ArgumentError ModelSymbol(:(x + 5))
    @test y1 == y2
    @test y3 == y1
    @test y1 == y4
    @test y2 == y3
    @test y2 == y4
    @test y3 == y4
    ally = Symbol[y1, y2, y3, y4]
    @test y1 in ally
    @test y2 in ally
    @test y3 in ally
    @test y4 in ally
    @test indexin([y1, y2, y3, y4], ally) == [1, 1, 1, 1]
    ally = ModelSymbol[y1, y2, y3, y4, :y, quote
        "y5"
        y
    end]
    @test indexin([y1, y2, y3, y4], ally) == [1, 1, 1, 1]
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

end

@testset "VarTypes" begin
    lvars = ModelSymbol[]
    push!(lvars, :ly)
    push!(lvars, quote
        "ly"
        ly
    end)
    push!(lvars, quote
        @log ly
    end)
    push!(lvars, quote
        "ly"
        @log ly
    end)
    push!(lvars, quote
        @lin ly
    end)
    push!(lvars, quote
        "ly"
        @lin ly
    end)
    push!(lvars, quote
        @steady ly
    end)
    push!(lvars, quote
        "ly"
        @steady ly
    end)
    push!(lvars, ModelSymbol(:ly, :lin))
    for i = 1:length(lvars)
        for j = i+1:length(lvars)
            @test lvars[i] == lvars[j]
        end
        @test lvars[i] == :ly
    end
    @test lvars[1].var_type == :lin
    @test lvars[2].var_type == :lin
    @test lvars[3].var_type == :log
    @test lvars[4].var_type == :log
    @test lvars[5].var_type == :lin
    @test lvars[6].var_type == :lin
    @test lvars[7].var_type == :steady
    @test lvars[8].var_type == :steady
    @test lvars[9].var_type == :lin
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
            x
            @log y
            @steady z
        end
        @test [v.var_type for v in m.allvars] == [:lin, :lin, :lin, :lin, :log, :steady]
    end
    let m = Model()
        @shocks m p q r        
        @shocks m begin
            x
            @log y
            @steady z
        end
        @test [v.var_type for v in m.allvars] == [:shock, :shock, :shock, :shock, :shock, :shock]
        @test (m.r = to_shock(m.r)) == :r
    end
    let m = Model()
        @logvariables m p q r
        @logvariables m begin
            x
            @log y
            @steady z
        end
        @test [v.var_type for v in m.allvars] == [:log, :log, :log, :log, :log, :log]
    end
    let m = Model()
        @neglogvariables m p q r
        @neglogvariables m begin
            x
            @log y
            @steady z
        end
        @test [v.var_type for v in m.allvars] == [:neglog, :neglog, :neglog, :neglog, :neglog, :neglog]
    end
    let m = Model()
        @steadyvariables m p q r
        @steadyvariables m begin
            x
            @log y
            @steady z
        end
        @warn "Test disabled"
        # @test [v.var_type for v in m.allvars] == [:steady, :steady, :steady, :steady, :steady, :steady]

    end
end

module E
using ModelBaseEcon
end
@testset "Evaluations" begin
    ModelBaseEcon.initfuncs(E)
    E.eval(ModelBaseEcon.makefuncs(:(x + 3 * y), [:x, :y], [], [], E))
    @test :resid_1 ∈ names(E, all=true)
    @test :RJ_1 ∈ names(E, all=true)
    @test E.resid_1([1.1, 2.3]) == 8.0
    @test E.RJ_1([1.1, 2.3]) == (8.0, [1.0, 3.0])
end

@testset "Misc" begin
    m = Model(Options(verbose=true))
    out = let io = IOBuffer()
        print(io, m.flags)
        readlines(seek(io, 0))
    end
    @test length(out) == 3
    for line in out[2:end]
        sline = strip(line)
        @test isempty(sline) || length(split(sline, "=")) == 2
    end
    @test fullprint(IOBuffer(), m) === nothing
    @test_throws ModelBaseEcon.ModelError ModelBaseEcon.modelerror()
    sprint(showerror, ModelBaseEcon.ModelError())
    sprint(showerror, ModelBaseEcon.ModelNotInitError()) 
    sprint(showerror, ModelBaseEcon.NotImplementedError(""))
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
    @test_throws ArgumentError m.a = 1
    @test_throws ModelBaseEcon.EqnNotReadyError ModelBaseEcon.eqnnotready()
    sprint(showerror, ModelBaseEcon.EqnNotReadyError())

    @test_throws ErrorException try @eval @equations m :(p[t] = 0) catch err; throw(ErrorException(err.msg)); end

    @equations m begin
        p[t] = 0
    end
    @initialize m
    @test_throws ErrorException @initialize m

    @test Symbol(m.variables[1]) == m.variables[1]

    for (i, v) = enumerate(m.varshks)
        s = convert(Symbol, v)
        @test m.sstate[i] == m.sstate[v] == m.sstate[s] == m.sstate["$s"]
    end

    m.sstate.values .= rand(length(m.sstate.values))
    @test begin
        (l, s) = m.sstate.x.data
        l == m.sstate.x.level && s == m.sstate.x.slope
    end
    @test begin
        (l, s) = m.sstate.k.data
        exp(l) == m.sstate.k.level && exp(s) == m.sstate.k.slope
    end

    xdata = m.sstate.x[1:8, ref=3]
    @test xdata[3] ≈ m.sstate.x.level
    @test xdata ≈ m.sstate.x.level .+ ((1:8) .- 3) .* m.sstate.x.slope
    kdata = m.sstate.k[1:8, ref=3]
    @test kdata[3] ≈ m.sstate.k.level
    @test kdata ≈ m.sstate.k.level .* m.sstate.k.slope .^ ((1:8) .- 3)

    @test_throws Exception m.sstate.x.data = [1, 2]
    @test_throws ArgumentError m.sstate.nosuchvariable

    @steadystate m m = l
    @steadystate m slope m = l
    @test length(m.sstate.constraints) == 2

    let io = IOBuffer()
        show(io, m.sstate.x)
        lines = split(String(take!(io)), '\n')
        @test length(lines) == 1 && occursin('+', lines[1])

        show(io, m.sstate.k)
        lines = split(String(take!(io)), '\n')
        @test length(lines) == 1 && !occursin('+', lines[1]) && occursin('*', lines[1])

        m.sstate.y.slope = 0
        show(io, m.sstate.y)
        lines = split(String(take!(io)), '\n')
        @test length(lines) == 1 && !occursin('+', lines[1]) && !occursin('*', lines[1])

        m.sstate.l.slope = 1
        show(io, m.sstate.l)
        lines = split(String(take!(io)), '\n')
        @test length(lines) == 1 && !occursin('+', lines[1]) && !occursin('*', lines[1])

        show(io, m.sstate.p)
        lines = split(String(take!(io)), '\n')
        @test length(lines) == 1 && !occursin('+', lines[1]) && !occursin('*', lines[1])

        ModelBaseEcon.show_aligned5(io, m.sstate.x, mask=[true, false])
        lines = split(String(take!(io)), '\n')
        @test length(lines) == 1 && length(split(lines[1], '?')) == 2

        ModelBaseEcon.show_aligned5(io, m.sstate.k, mask=[false, false])
        lines = split(String(take!(io)), '\n')
        @test length(lines) == 1 && length(split(lines[1], '?')) == 3

        ModelBaseEcon.show_aligned5(io, m.sstate.l, mask=[false, false])
        println(io)
        ModelBaseEcon.show_aligned5(io, m.sstate.y, mask=[false, false])
        println(io)
        ModelBaseEcon.show_aligned5(io, m.sstate.p, mask=[false, true])
        lines = split(String(take!(io)), '\n')
        @test length(lines) == 3
        for line in lines
            @test length(split(line, '?')) == 2
        end
        @test fullprint(io, m) === nothing
        @test show(io, m) === nothing
        @test show(IOBuffer(), MIME"text/plain"(), m) === nothing
        @test show(io, Model()) === nothing

        @test m.exogenous == ModelVariable[]
        @test m.nexog == 0
        @test_throws ErrorException m.dummy 

        @test show(IOBuffer(), MIME"text/plain"(), m.flags) === nothing
    end
end

@testset "Timer" begin
    @test inittimer() === nothing
    @timer "model" m = Model()
    @timer "model" @variables m x
    @timer "model" @shocks m sx
    @timer "model" @equations m begin
            x[t-1] = sx[t+1]
            @lag(x[t]) = @lag(sx[t+2])
        end
    @timer params = @parameters
    @test printtimer(IOBuffer()) === nothing
    @test_throws ErrorException try @eval @timer catch err; throw(ErrorException(err.msg)); end
    @test stoptimer() === nothing
end

@testset "Abstract" begin
    struct AM <: ModelBaseEcon.AbstractModel end
    m = AM()
    @test_throws ErrorException ModelBaseEcon.alleqns(m)
    @test_throws ErrorException ModelBaseEcon.allvars(m)
    @test_throws ErrorException ModelBaseEcon.nalleqns(m) == 0
    @test_throws ErrorException ModelBaseEcon.nallvars(m) == 0
    # @test_throws ErrorException ModelBaseEcon.modelof(m)
end

@testset "metafuncts" begin
    @test ModelBaseEcon.has_t(1) == false
    @test ModelBaseEcon.has_t(:(x[t]-x[t-1])) == true
    @test ModelBaseEcon.at_lag(:(x[t]),0) == :(x[t])
    @test_throws ErrorException ModelBaseEcon.at_d(:(x[t]),0,-1)
    @test ModelBaseEcon.at_d(:(x[t]),3,0) == :(((x[t] - 3 * x[t - 1]) + 3 * x[t - 2]) - x[t - 3])
    @test ModelBaseEcon.at_movsumew(:(x[t]), 3, 2.0) == :(x[t] + 2.0 * x[t - 1] + 4.0 * x[t - 2])
    @test ModelBaseEcon.at_movsumew(:(x[t]), 3, :y) == :(x[t] + y ^ 1 * x[t - 1] + y ^ 2 * x[t - 2])
    @test ModelBaseEcon.at_movavew(:(x[t]), 3, 2.0) == :((x[t] + 2.0 * x[t - 1] + 4.0 * x[t - 2]) / 7.0)
    @test ModelBaseEcon.at_movavew(:(x[t]), 3, :y) == :(((x[t] + y ^ 1 * x[t - 1] + y ^ 2 * x[t - 2]) * (1 - y)) / (1 - y ^ 3))
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
    push!(params, :e => [1, 2, 3])
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
    MetaTest.eval(quote
        custom(x) = 2x + one(x)
    end)
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
    @test_throws ArgumentError ModelBaseEcon.process_equation(m, :(x[t] = if false
        2
    end))
    @test ModelBaseEcon.process_equation(m, :(x[t] = if false
        2
    else
        0
    end)) isa Equation
    @test ModelBaseEcon.process_equation(m, :(x[t] = ifelse(false, 2, 0))) isa Equation
    p = 0
    @test ModelBaseEcon.process_equation(m, "x=$p") isa Equation
end

@testset "Meta" begin
    mod = Model()
    @parameters mod a = 0.1 b = @link(1.0 - a)
    @variables mod x
    @shocks mod sx
    @equations mod begin
        x[t-1] = sx[t+1]
        @lag(x[t]) = @lag(sx[t+2])
        # 
        x[t-1] + a = sx[t+1] + 3
        @lag(x[t] + a) = @lag(sx[t+2] + 3)
        # 
        x[t-2] = sx[t]
        @lag(x[t], 2) = @lead(sx[t-2], 2)
        # 
        x[t] - x[t-1] = x[t+1] - x[t] + sx[t]
        @d(x[t]) = @d(x[t+1]) + sx[t]
        # 
        (x[t] - x[t+1]) - (x[t-1] - x[t]) = sx[t]
        @d(x[t] - x[t+1]) = sx[t]
        # 
        x[t] - x[t-2] = sx[t]
        @d(x[t], 0, 2) = sx[t]
        # 
        x[t] - 2x[t-1] + x[t-2] = sx[t]
        @d(x[t], 2) = sx[t]
        # 
        x[t] - x[t-1] - x[t-2] + x[t-3] = sx[t]
        @d(x[t], 1, 2) = sx[t]
        # 
        log(x[t] - x[t-2]) - log(x[t-1] - x[t-3]) = sx[t]
        @dlog(@d(x[t], 0, 2)) = sx[t]
        # 
        (x[t] + 0.3x[t+2]) + (x[t-1] + 0.3x[t+1]) + (x[t-2] + 0.3x[t]) = 0
        @movsum(x[t] + 0.3x[t+2], 3) = 0
        # 
        ((x[t] + 0.3x[t+2]) + (x[t-1] + 0.3x[t+1]) + (x[t-2] + 0.3x[t])) / 3 = 0
        @movav(x[t] + 0.3x[t+2], 3) = 0
    end
    @initialize mod

    compare_resids(e1, e2) = (
        e1.resid.head == e2.resid.head && (
            (length(e1.resid.args) == length(e2.resid.args) == 2 && e1.resid.args[2] == e2.resid.args[2]) ||
            (length(e1.resid.args) == length(e2.resid.args) == 1 && e1.resid.args[1] == e2.resid.args[1])
        )
    )

    for i = 2:2:length(mod.equations)
        @test compare_resids(mod.equations[i-1], mod.equations[i])
    end
    # test errors and warnings
    mod.warn.no_t = false
    @test add_equation!(mod, :(x = sx[t])) isa Model
    @test add_equation!(mod, :(x[t] = sx)) isa Model
    @test add_equation!(mod, :(x[t] = sx[t])) isa Model
    @test compare_resids(mod.equations[end], mod.equations[end-1])
    @test compare_resids(mod.equations[end], mod.equations[end-2])
    @test_throws ArgumentError add_equation!(mod, :(@notametafunction(x[t]) = 7))
    @test_throws ArgumentError add_equation!(mod, :(x[t] = unknownsymbol))
    @test_throws ArgumentError add_equation!(mod, :(x[t] = unknownseries[t]))
    @test_throws ArgumentError add_equation!(mod, :(x[t] = let c = 5
        sx[t+c]
    end))
    @test ModelBaseEcon.update_auxvars(ones(2,2), mod) == ones(2,2)
end

############################################################################

@testset "export" begin
    let m = Model()
        m.warn.no_t = false
        @parameters m begin
            a = 0.3
            b = @link 1 - a
            d = [1, 2, 3]
            c = @link sin(2π / d[3])
        end
        @variables m begin
            "variable x"
            x
        end
        @shocks m sx
        @equations m begin
            "This equation is super cool"
            a * @d(x) = b * @d(x[t+1]) + sx
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

        @test export_parameters(TestModel.model) == Dict(:a => 0.3, :b => 0.7, :d => [1, 2, 3], :c => sin(2π/3))
        @test export_parameters!(Dict{Symbol,Any}(), TestModel.model) == export_parameters(TestModel.model.parameters)

        p = deepcopy(parameters(m))
        @test_throws BoundsError assign_parameters!(TestModel.model, d=2.0)
        map!(x->ModelParam(), values(TestModel.model.parameters.contents))
        @test parameters(assign_parameters!(TestModel.model, p)) == p

        ss = Dict(:x => 0.0, :sx => 0.0)
        assign_sstate!(TestModel.model, y = 0.0)
        @test export_sstate(assign_sstate!(TestModel.model,ss)) == ss
        @test export_sstate!(Dict(),TestModel.model.sstate, ssZeroSlope=true) == ss

        ss = sstate(m)
        @test show(IOBuffer(), MIME"text/plain"(), ss) === nothing
        @test geteqn(1,m) == first(m.sstate.constraints)
        @test geteqn(neqns(ss), m) == last(m.sstate.equations)
        @test propertynames(ss, true) == (:x, :sx, :vars, :values, :mask, :equations, :constraints)
        @test fullprint(IOBuffer(), m) === nothing

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
    @test R ≈ known_R atol = 1e-12
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
        @steadystate m y = 3
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
    @test_throws ArgumentError refresh_med!(Model())
end

@testset "E1.params" begin
    let m = E1.model
        @test propertynames(m.parameters) == (:α, :β)
        @test peval(m, :α) == 0.5
        m.β = @link 1.0 - α
        m.parameters.beta = @alias β
        for α = 0.0:0.1:1.0
            m.α = α
            test_eval_RJ(m, [0.0], [-α 1.0 -m.beta 0.0 -1.0 0.0;])
        end
        @test_logs (:warn, r"Model does not have parameters*"i) assign_parameters!(m, γ=0)
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
    x[t+1] = log(x[t] - x[t-1])
    y[t+1] = y[t] + log(y[t-1])
end
@initialize model
end
@testset "AUX" begin
    let m = AUX.model
        @test m.nvars == 2
        @test m.nshks == 0
        @test m.nauxs == 2
        @test_throws ArgumentError m.aux1 = 1
        @test (m.aux1 = update(m.aux1; doc = "aux1")) == :aux1
        @test length(m.auxeqns) == ModelBaseEcon.nauxvars(m) == 2
        x = ones(2, 2)
        @test_throws ErrorException ModelBaseEcon.update_auxvars(x, m)
        x = ones(4, 3)
        @test_throws ErrorException ModelBaseEcon.update_auxvars(x, m)
        x = 2 .* ones(4, 2)
        ax = ModelBaseEcon.update_auxvars(x, m; default=0.1)
        @test size(ax) == (4, 4)
        @test x == ax[:, 1:2]  # exactly equal
        @test ax[:, 3:4] ≈ [0.0 0.0; 0.1 log(2.0); 0.1 log(2.0); 0.1 log(2.0)] # computed values, so ≈ equal
        @test propertynames(AUX.model) == (fieldnames(Model)..., :exogenous, :nvars, :nshks, :nauxs, :nexog, :allvars, :varshks, :alleqns,
            keys(AUX.model.options)..., fieldnames(ModelBaseEcon.ModelFlags)..., Symbol[AUX.model.variables...]..., 
            Symbol[AUX.model.shocks...]..., keys(AUX.model.parameters)...,)
        @test show(IOBuffer(), m) === nothing
        @test show(IOContext(IOBuffer(), :compact => true), m) === nothing
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
        [-0.5 1 -0.48 0 0 0 0 -0.02 0 0 -1 0 0 0 0 0 0 0
            0 -0.375 0 -0.75 1 0 0 -0.125 0 0 0 0 0 -1 0 0 0 0
            0 0 -0.02 0 0.02 0 -0.5 1 -0.48 0 0 0 0 0 0 0 -1 0])
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
    @test ss.pinf.level == ss.pinf.data[1]
    @test ss.pinf.slope == ss.pinf.data[2]
    ss.pinf.data .= [2.3, 0.7]
    @test ss.values[1:2] == [2.3, 0.7]
    ss.rate.level = 21
    ss.rate.slope = 0.21
    @test ss.rate.level == 21 && ss.rate.slope == 0.21
    @test ss.rate.data == [21, 0.21]
end

@using_example E3
@testset "E3" begin
    @test length(E3.model.parameters) == 3
    @test length(E3.model.variables) == 3
    @test length(E3.model.shocks) == 3
    @test length(E3.model.equations) == 3
    @test ModelBaseEcon.nallvars(E3.model) == 6
    @test ModelBaseEcon.allvars(E3.model) == ModelVariable.([:pinf,:rate,:ygap,:pinf_shk,:rate_shk,:ygap_shk])
    @test ModelBaseEcon.nalleqns(E3.model) == 3
    @test E3.model.maxlag == 2
    @test E3.model.maxlead == 3
    compare_RJ_R!_(E3.model)
    test_eval_RJ(E3.model, [0.0, 0.0, 0.0],
        sparse(
            [1, 1, 2, 1, 3, 1, 1, 2, 2, 3, 3, 3, 1, 2, 3, 3, 1, 2, 3],
            [2, 3, 3, 4, 4, 5, 6, 8, 9, 9, 13, 14, 15, 15, 15, 16, 21, 27, 33],
            [-0.5, 1.0, -0.375, -0.3, -0.02, -0.05, -0.05, -0.75, 1.0, 0.02, -0.25,
                -0.25, -0.02, -0.125, 1.0, -0.48, -1.0, -1.0, -1.0],
            3, 36,
        )
    )
    @test_throws ModelBaseEcon.ModelNotInitError eval_RJ(zeros(2,2), ModelBaseEcon.NoModelEvaluationData())
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
            2p[t] = p[t+1] + 0.1
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
            lx[t] = lx[t-1] + 0.2 + s1[t]
            "exponential with the same rate as the slope of lx"
            log(x[t]) = lx[t] + s2[t+1]
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
        ss.lx.data .= [1.5, 0.2]
        ss.x.data .= [0.0, 0.2]
        ss.s1.data .= [0.0, 0.0]
        ss.s2.data .= [0.0, 0.0]
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
        @test TMP[[1, 2, 5]] ≈ [0.0, 1.0, -1.0]
        # test with eq4
        ss.lx.data .= [1.5, 0.2]
        ss.x.data .= [1.5, 0.2]
        ss.s1.data .= [0.0, 0.0]
        ss.s2.data .= [0.0, 0.0]
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
            ss.x.data[2] = xslp
            @test eq4.eval_resid(ss.values[eq4.vinds]) ≈ m.shift * (xslp - 0.2)
        end
        ss.x.slope = exp(0.2)
        R, J = eq4.eval_RJ(ss.values[eq4.vinds])
        TMP = fill!(similar(ss.values), 0.0)
        TMP[eq4.vinds] .= J
        @test R + 1.0 ≈ 0.0 + 1.0
        @test TMP[[1, 2, 3, 4, 7]] ≈ [-1.0, -m.shift, 1.0, m.shift, -1.0]
        for xlvl = 0.1:0.1:2
            ss.x.level = exp(xlvl)
            R, J = eq4.eval_RJ(ss.values[eq4.vinds])
            @test R ≈ xlvl - 1.5
            TMP[eq4.vinds] .= J
            @test TMP[[1, 2, 3, 4, 7]] ≈ [-1.0, -m.shift, 1.0, m.shift, -1.0]
        end
    end
end

@testset "bug #28" begin
    let 
        m = Model()
        @variables m (@log(a); la)
        @equations m begin
            a[t] = exp(la[t])
            la[t] = 20
        end
        @initialize m
        assign_sstate!(m, a = 20, la = log(20))
        @test m.sstate.a.level ≈ 20 atol=1e-14
        @test m.sstate.a.slope == 1.0
        @test m.sstate.la.level ≈ log(20) atol=1e-14
        @test m.sstate.la.slope == 0.0
        assign_sstate!(m, a = (level=20,), la = [log(20), 0])
        @test m.sstate.a.level ≈ 20 atol=1e-14
        @test m.sstate.a.slope == 1.0
        @test m.sstate.la.level ≈ log(20) atol=1e-14
        @test m.sstate.la.slope == 0.0
    end
end

@testset "sel_lin" begin
    let
        m = Model()
        @variables m (la; @log a)
        @equations m begin
            @lin a[t] = exp(la[t])
            @lin la[t] = 2
        end
        @initialize m
        assign_sstate!(m; a = exp(2), la = 2)
        @test_nowarn (selective_linearize!(m); true)
    end
end

include("auxsubs.jl")
include("sstate.jl")
