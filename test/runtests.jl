##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

using ModelBaseEcon
using SparseArrays
using Test

import ModelBaseEcon.update


@testset "Transformations" begin
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
        @test_throws ErrorException m.ly = 25
        @test_throws ErrorException m.lmy = -25
        @test_throws ErrorException m.ly = ModelVariable(:lmy)
        @test_logs (:warn, r".*do not specify transformation directly.*"i) @test_throws ArgumentError update(m.ly, tr_type=:log, transformation=NoTransform)
        @test_logs (:warn, r".*do not specify transformation directly.*"i) update(m.ly, tr_type=:log, transformation=LogTransform)
        @test_logs (:warn, r".*do not specify transformation directly.*"i) @test update(m.ly, transformation=LogTransform).tr_type == :log
        @test_logs (:warn, r".*do not specify transformation directly.*"i) @test update(m.lmy, tr_type=:neglog, transformation=NegLogTransform).tr_type == :neglog

        @test_throws ErrorException m.dummy = nothing

    end
end

@testset "Options" begin
    o = Options(tol=1e-7, maxiter=25)
    @test propertynames(o) == (:maxiter, :tol)
    @test getoption(o, tol=1e7) == 1e-7
    @test getoption(o, "name", "") == ""
    @test getoption(o, abstol=1e-10, name="") == (1e-10, "")
    @test all(["abstol", "name"] .∉ Ref(o))
    @test getoption!(o, abstol=1e-11) == 1e-11
    @test :abstol ∈ o
    @test setoption!(o, reltol=1e-3, linear=false) isa Options
    @test all(["reltol", :linear] .∈ Ref(o))
    @test getoption!(o, tol=nothing, linear=true, name="Zorro") == (1e-7, false, "Zorro")
    @test "name" ∈ o && o.name == "Zorro"
    z = Options()
    @test merge(z, o) == Options(o...) == Options(o)
    @test merge!(z, o) == Options(Dict(string(k) => v for (k, v) in pairs(o))...)
    @test o == z
    @test Dict(o...) == z
    @test o == Dict(z...)
    z.name = "Oro"
    @test o.name == "Zorro"
    @test setoption!(z, "linear", true) isa Options
    @test getoption!(z, "linear", false) == true
    @test getoption!(z, :name, "") == "Oro"
    @test show(IOBuffer(), o) === nothing
    @test show(IOBuffer(), MIME"text/plain"(), o) === nothing

    @using_example S1
    m = S1.newmodel()
    @test getoption(m, "shift", 1) == getoption(m, shift=1) == 10
    @test getoption!(m, "substitutions", true) == getoption!(m, :substitutions, true) == false
    @test getoption(setoption!(m, "maxiter", 25), maxiter=0) == 25
    @test getoption(setoption!(m, verbose=true), "verbose", false) == true
    @test typeof(setoption!(identity, m)) == Options
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
    for i in eachindex(lvars)
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
    for i in eachindex(lvars)
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

@testset "Abstract" begin
    struct AM <: ModelBaseEcon.AbstractModel end
    m = AM()
    @test_throws ErrorException ModelBaseEcon.alleqns(m)
    @test_throws ErrorException ModelBaseEcon.allvars(m)
    @test_throws ErrorException ModelBaseEcon.nalleqns(m) == 0
    @test_throws ErrorException ModelBaseEcon.nallvars(m) == 0
    @test_throws ErrorException ModelBaseEcon.moduleof(m) == @__MODULE__
end

@testset "metafuncts" begin
    @test ModelBaseEcon.has_t(1) == false
    @test ModelBaseEcon.has_t(:(x[t] - x[t-1])) == true
    @test @lag(x[t], 0) == :(x[t])
    @test_throws ErrorException @macroexpand @d(x[t], 0, -1)
    @test @d(x[t], 3, 0) == :(((x[t] - 3 * x[t-1]) + 3 * x[t-2]) - x[t-3])
    @test @movsumew(x[t], 3, 2.0) == :(x[t] + (2.0 * x[t-1] + 4.0 * x[t-2]))
    @test @movsumew(x[t], 3, y) == :(x[t] + (y^1 * x[t-1] + y^2 * x[t-2]))
    @test @movavew(x[t], 3, 2.0) == :((x[t] + (2.0 * x[t-1] + 4.0 * x[t-2])) / 7.0)
    @test @movavew(x[t], 3, y) == :(((x[t] + (y^1 * x[t-1] + y^2 * x[t-2])) * (1 - y)) / (1 - y^3))
    @test @lag(x[t+4]) == :(x[t+3])
    @test @lag(x[t-1]) == :(x[t-2])
    @test @lag(x[3]) == :(x[3])
    @test_throws ErrorException @macroexpand @lag(x[3+t])
    @test @movsumw(a[t] + b[t+1], 2, p) == :(p[1] * (a[t] + b[t+1]) + p[2] * (a[t-1] + b[t]))
    @test @movavw(a[t] + b[t+1], 2, p) == :((p[1] * (a[t] + b[t+1]) + p[2] * (a[t-1] + b[t])) / (p[1] + p[2]))
    @test @movsumw(a[t] + b[t+1], 2, q, p) == :(q * (a[t] + b[t+1]) + p * (a[t-1] + b[t]))
    @test @movavw(a[t] + b[t+1], 2, q, p) == :((q * (a[t] + b[t+1]) + p * (a[t-1] + b[t])) / (q + p))
    @test @lead(v[t, 2]) == :(v[t+1, 2])
    @test @dlog(v[t-1, z, t+2], 1) == :(log(v[t-1, z, t+2]) - log(v[t-2, z, t+1]))
end

module ParamsTests
using ModelBaseEcon
params = @parameters
custom = (x) -> x + one(x)
val = 12.0
pair = :hello => "world"
params.b = custom(val)
params.a = @link custom(val)
params.c = val
params.d = @link val
params.e = @link pair.first
params.f = @link pair[2]
end

@testset "Parameters" begin
    m = Model()
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
    m.parameters = params
    # update_links!(params)
    update_links!(params)
    @test 1.0 + params.d ≈ 1.0

    params.d = @link cos(2π / e[2])
    @test params.d ≈ -1.0

    @test_throws ArgumentError @alias a + 5
    @test_throws ArgumentError @link 28

    @test ParamsTests.params.a ≈ 13.0
    @test ParamsTests.params.b ≈ 13.0
    @test ParamsTests.params.c ≈ 12.0
    @test ParamsTests.params.d ≈ 12.0
    # Core.eval(ParamsTests, :(custom(x) = 2x + one(x)))
    ParamsTests.custom = (x) -> 2x + one(x)
    update_links!(ParamsTests.params)
    @test ParamsTests.params.a ≈ 25.0
    @test ParamsTests.params.b ≈ 13.0
    @test ParamsTests.params.c ≈ 12.0
    @test ParamsTests.params.d ≈ 12.0
    Core.eval(ParamsTests, :(val = 22))
    update_links!(ParamsTests.params)
    @test ParamsTests.params.a == 45
    @test ParamsTests.params.b ≈ 13.0
    @test ParamsTests.params.c ≈ 12.0
    @test ParamsTests.params.d == 22

    @test ParamsTests.params.e == :hello
    @test ParamsTests.params.f == "world"
    Core.eval(ParamsTests, :(pair = 27 => π))
    update_links!(ParamsTests.params)
    @test ParamsTests.params.e == 27
    @test ParamsTests.params.f == π

    @test @alias(c) == ModelParam(Set(), :c, nothing)
    @test @link(c) == ModelParam(Set(), :c, nothing)
    @test @link(c + 1) == ModelParam(Set(), :(c + 1), nothing)

    @test_throws ArgumentError params[:contents] = 5
    @test_throws ArgumentError params.abc

    @test_logs (:error, r"While updating value for parameter b:*"i) begin
        try
            params.a = [1, 2, 3]
        catch E
            if E isa ModelBaseEcon.ParamUpdateError
                io = IOBuffer()
                showerror(io, E)
                seekstart(io)
                @error read(io, String)
            else
                rethrow(E)
            end
        end
    end
end

##==============================================================================

module E
using ModelBaseEcon
end
@testset "DerivsFD" begin
    ModelBaseEcon.initfuncs(E, :forwarddiff)
    @test isdefined(E, :_forwarddiff)
    @test isdefined(E._forwarddiff, :EquationEvaluatorFD)
    @test isdefined(E._forwarddiff, :EquationGradientFD)
    resid, RJ = ModelBaseEcon.DerivsFD.makefuncs(:fd, :(x + 3 * y), [:x, :y], [], [], E)
    @test resid isa E._forwarddiff.EquationEvaluatorFD
    @test RJ isa E._forwarddiff.EquationGradientFD
    @test RJ.fn1 isa ModelBaseEcon.DerivsFD.FunctionWrapper
    @test RJ.fn1.f == resid
    @test ModelBaseEcon.moduleof(resid) === E._forwarddiff
    # @test ModelBaseEcon.moduleof(RJ) === E
    @test resid([1.1, 2.3]) == 8.0
    @test RJ([1.1, 2.3]) == (8.0, [1.0, 3.0])
    # make sure the EquationEvaluator and EquationGradient are reused for identical expressions and arguments
    nnames = length(names(E, all=true))
    resid1, RJ1 = ModelBaseEcon.DerivsFD.makefuncs(:fd, :(x + 3 * y), [:x, :y], [], [], E)
    @test nnames == length(names(E, all=true))
    @test resid === resid1
    @test RJ === RJ1
end

@static if VERSION >= v"1.10"
    @testset "DerivsSym" begin
        ModelBaseEcon.initfuncs(E, :symbolics)
        @test isdefined(E, :_symbolics)
        @test isdefined(E._symbolics, :EquationEvaluatorSym)
        @test isdefined(E._symbolics, :GradientEvaluatorSym)
        resid, RJ = ModelBaseEcon.DerivsSym.makefuncs(:sym, :(x + 3 * y), [:x, :y], [], [], E)
        @test resid isa E._symbolics.EquationEvaluatorSym
        @test RJ isa E._symbolics.GradientEvaluatorSym
        @test ModelBaseEcon.moduleof(resid) === E._symbolics
        @test ModelBaseEcon.moduleof(RJ) === E._symbolics
        @test resid([1.1, 2.3]) == 8.0
        @test RJ([1.1, 2.3]) == (8.0, [1.0, 3.0])
        # make sure the EquationEvaluator and EquationGradient are reused for identical expressions and arguments
        nnames = length(names(E, all=true))
        resid1, RJ1 = ModelBaseEcon.DerivsSym.makefuncs(:sym, :(x + 3 * y), [:x, :y], [], [], E)
        @test nnames == length(names(E, all=true))
        @test resid === resid1
        @test RJ === RJ1
    end
else
    @warn "Skip DerivSym on Julia 1.9"
end

##==============================================================================
# tests that use the codegen infrastructure

@info "Testing `codegen = :forwarddiff`"
module TestForwardDiff
using ModelBaseEcon
ModelBaseEcon.defaultoptions.codegen = :forwarddiff
include("codegen.jl")
end


@info "Testing `codegen = :symbolics`"
module TestSymbolics
using ModelBaseEcon
@static if VERSION >= v"1.10"
    ModelBaseEcon.defaultoptions.codegen = :symbolics
    include("codegen.jl")
else
    @warn "Skip DerivSym on Julia 1.9"
end
end

##==============================================================================
include("dfmmodels.jl")

nothing