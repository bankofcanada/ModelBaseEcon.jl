##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

"""
    ModelBaseEcon

This package is part of the StateSpaceEcon ecosystem.
It provides the basic elements needed for model definition.
StateSpaceEcon works with model objects defined with ModelBaseEcon.
"""
module ModelBaseEcon

using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

using PrecompileTools: @setup_workload, @compile_workload

include("ir.jl")
include("validate.jl")
include("metafuncs.jl")
include("macros.jl")
include("symbolic.jl")
include("codegen.jl")
include("compile.jl")
include("linearize.jl")
include("export_model.jl")
include("dfm/dfm.jl")
include("compat.jl")

using .IR
using .Validate
using .MetaFuncs
using .Macros
using .Symbolic
using .Codegen
# Compile exports `@initialize`/`@reinitialize`; the compat layer shadows
# them with the caching variants, so import Compile's surface WITHOUT those
# two macros, then bring in the whole Compat module below.
using .Compile: Equation, CompiledModel, SteadyStateUserEquation,
                expr_hash, initialize_model, reinitialize_model,
                model_tuple_threshold, DEFAULT_MODEL_TUPLE_THRESHOLD,
                doc, tags
using .Linearize
using .Export
using .Compat

# Re-export the public surface
export ModelDef
export var"@variables", var"@logvariables", var"@shocks", var"@parameters",
       var"@equations", var"@autoexogenize", var"@exogenous", var"@autoshocks",
       var"@steadystate"
export SSEquationAST, SS_LEVEL, SS_SLOPE, SteadyStateUserEquation
export add_ss_equation!, delete_ss_equations!, next_ss_eqn_name
export is_exogenous, exogenous_names
export validate, ValidationError, LinkCycleError
export TimeRef, ParamRef, EquationKernel, build_equation_kernels
export EquationFunctions, build_equation_functions, build_model_functions
export Equation, CompiledModel, expr_hash, initialize_model, reinitialize_model
export doc, tags
export export_model
export model_tuple_threshold, DEFAULT_MODEL_TUPLE_THRESHOLD
export var"@initialize", var"@reinitialize"
export LinEqnEvalData, LinearizationError, linearize_equation, selectively_linearize

# Backward-compatibility layer (v0.8.0 alias surface, compat.jl).
# `shocks`/`nshocks`/`allvars`/`nallvars`/`isshock` are exported below via
# the DFM subsystem - the compat layer extends those same generics.
export Model
export parameters, variables, equations
export nvariables, nparameters, nequations, alleqns
export islog, islin
export update_links!, moduleof
export var"@using_example", var"@include_example"

# DFM subsystem. Self-contained above the equation-kernel layer; accessed as
# `ModelBaseEcon.DFMModels` (matches the legacy access path). We export the
# submodule plus the user-facing DSL surface; the model-internal state-space
# accessors stay qualified (DFMModels.get_loading etc.) to avoid clashing with
# the equation-model surface.
using .DFMModels
export DFMModels
export DFM, DFMModel, DFMParams
export DFMBlock, ComponentsBlock, ObservedBlock, CommonComponents, IdiosyncraticComponents
export MixFreq, NoMixFreq, ismixfreq
export add_observed!, add_components!, map_loadings!, add_shocks!, initialize_dfm!
export init_params, init_params!
export observed, nobserved, states, nstates, varshks, nvarshks
export shocks, nshocks, allvars, nallvars, isshock
export endog, nendog, exog, nexog, lags, leads, order
export states_with_lags, nstates_with_lags

# ---------------------------------------------------------------------------
# Precompile workload
#
# The dominant cost of using this package is one-time Symbolics JIT, split
# across two non-overlapping subsystems: (1) @link resolution via
# Symbolics.substitute and (2) symbolic differentiation + build_function /
# RuntimeGeneratedFunctions codegen. Both fire the first time a model is
# built, regardless of model size - a 1-equation model pays the same JIT as
# SW07. Exercising them here moves that compilation into precompilation
# (cached on disk, paid once per install / dep change) so downstream use and
# the test suite start warm. This mirrors the @compile_workload blocks in our
# upstream deps Symbolics.jl and ModelingToolkit.jl.
@setup_workload begin
    @compile_workload begin
        # Subsystem 1: deep @link chain -> resolved_link_table -> substitute.
        m = ModelDef()
        @parameters m begin
            a = @link b + 1
            b = @link c * 2
            c = 3.0
        end
        @variables m begin
            y
        end
        @equations m begin
            y[t] = a * y[t-1]
        end
        validate(m)

        # Subsystem 2: build kernels (Symbolics.jacobian) + RGF codegen, then
        # evaluate so the generated residual/gradient functions are compiled.
        # A nonlinear term (^) and a @log variable cover the transform paths.
        m2 = ModelDef()
        @parameters m2 begin
            α = 0.33
        end
        @logvariables m2 begin
            A
        end
        @variables m2 begin
            K
            L
            r
        end
        @equations m2 begin
            r[t] = α * A[t] * (L[t] / K[t-1])^(1 - α)
        end
        kernels, params = build_equation_kernels(m2)
        funcs = build_equation_functions(kernels[1])
        x = ones(funcs.n_x)
        p = [0.33]
        J = zeros(funcs.n_x)
        funcs.eval_resid(x, p)
        funcs.eval_RJ!(J, x, p)

        # Subsystem 3: the DFM DSL -> build -> params -> state-space-accessor
        # paths. These are a wholly separate compile graph from the equation-model
        # kernels above (ComponentArrays-typed params, NamedList block plumbing,
        # get_loading/get_transition/get_covariance). The first DFM the test suite
        # builds otherwise pays ~3s of this JIT (dfm_models.jl dfm.2 ~15s cold);
        # building two small models here - one with a CommonComponents+observed
        # block, one mixing IdiosyncraticComponents + MixFreq - covers the block
        # kinds the tests exercise and moves that compile into the cached image.
        d1 = DFM(:precompile_dfm1)
        add_components!(d1, C1=CommonComponents("A", 2, order=2))
        add_components!(d1, C2=CommonComponents(["U", "V"], order=3))
        add_components!(d1, IC=IdiosyncraticComponents(order=1))
        add_observed!(d1, O1=(:x, :y), O2=(:z,))
        map_loadings!(d1, (:x, :z) => (:A¹, :IC), :x => (:U,), (:z,) => :C2, :y => :C1)
        add_shocks!(d1, :y)
        initialize_dfm!(d1)
        init_params(d1.model)
        m1, p1 = d1.model, d1.params
        copyto!(p1, 1:length(p1))
        DFMModels.get_mean(m1, p1)
        Λ = DFMModels.get_loading(m1, p1)
        DFMModels.get_transition(m1, p1)
        get_covariance(m1, p1)
        DFMModels.get_covariance(m1, p1, Val(:Observed))
        DFMModels.get_covariance(m1, p1, Val(:State))
        nvs = nvarshks(m1)
        pt = ones(1 + lags(m1), nvs)
        R, J = DFMModels.eval_RJ(pt, m1, p1)
        DFMModels.eval_resid(pt, m1, p1)
        DFMModels.eval_R!(similar(R), pt, m1, p1)
        DFMModels.eval_RJ!(similar(R), similar(J), pt, m1, p1)
        DFMModels.set_mean!(d1, DFMModels.get_mean(m1, p1))
        DFMModels.set_loading!(d1, Λ)
        DFMModels.set_transition!(d1, DFMModels.get_transition(m1, p1))
        O1 = m1.observed[:O1]
        fill!(p1, NaN)
        Λ1 = DFMModels.get_loading!(zeros(2, size(Λ, 2)), O1, p1.O1)
        DFMModels.loading_constraint(Λ1, O1)

        # d2 covers the MixFreq{:MQ} block kinds the DFM3MQ models use.
        d2 = DFM(:precompile_dfm2)
        add_observed!(d2, :obsQ => ObservedBlock(MixFreq{:MQ}, (:y, :z)))
        add_components!(d2,
            G=CommonComponents(MixFreq{:MQ}, "G", order=2),
            ic=IdiosyncraticComponents(MixFreq{:MQ}),
        )
        map_loadings!(d2, (:y, :z) => :G, (:y, :z) => :ic)
        add_shocks!(d2, :y, :z)
        initialize_dfm!(d2)
        init_params(d2.model)
        DFMModels.get_loading(d2.model, d2.params)
        DFMModels.get_transition(d2.model, d2.params)
    end
end

end # module
