##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

###########################################################
# Part 1: Helper functions


"""
    precompilefuncs(resid, RJ, ::Val{N}, tag) where N

Pre-compiles the given `resid` and `RJ` functions together
with the dual-number arithmetic required by ForwardDiff.

!!! warning
    Internal function. Do not call directly

"""
function precompilefuncs(resid, RJ, ::Val{N}, tag) where {N}
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing

    # tag = MyTag # ForwardDiff.Tag{resid,Float64}
    dual = ForwardDiff.Dual{tag,Float64,N}
    duals = Array{dual,1}
    cfg = ForwardDiff.GradientConfig{tag,Float64,N,duals}
    mdr = DiffResults.MutableDiffResult{1,Float64,Tuple{Array{Float64,1}}}

    precompile(resid, (Array{Float64,1},)) || error("precompile")
    precompile(resid, (duals,)) || error("precompile")
    precompile(RJ, (Array{Float64,1},)) || error("precompile")

    for pred in Symbol[:isinf, :isnan, :isfinite, :iseven, :isodd, :isreal, :isinteger, :-, :+, :log, :exp]
        pred ∈ (:iseven, :isodd) || precompile(getfield(Base, pred), (Float64,)) || error("precompile")
        precompile(getfield(Base, pred), (dual,)) || error("precompile")
    end

    for pred in Symbol[:isequal, :isless, :<, :>, :(==), :(!=), :(<=), :(>=), :+, :-, :*, :/, :^]
        precompile(getfield(Base, pred), (Float64, Float64)) || error("precompile")
        precompile(getfield(Base, pred), (dual, Float64)) || error("precompile")
        precompile(getfield(Base, pred), (Float64, dual)) || error("precompile")
        precompile(getfield(Base, pred), (dual, dual)) || error("precompile")
    end

    # precompile(ForwardDiff.extract_gradient!, (Type{tag}, mdr, dual)) || error("precompile")
    # precompile(ForwardDiff.vector_mode_gradient!, (mdr, typeof(resid), Array{Float64,1}, cfg)) || error("precompile")

    # precompile(Tuple{typeof(ForwardDiff.extract_gradient!), Type{tag}, mdr, dual}) || error("precompile")
    # precompile(Tuple{typeof(ForwardDiff.vector_mode_gradient!), mdr, resid, Array{Float64, 1}, cfg}) || error("precompile")

    return nothing
end

"""
    funcsyms(mod::Module)

Create a pair of identifiers that does not conflict with existing identifiers in
the given module.

!!! warning
    Internal function. Do not call directly.

### Implementation (for developers)
We need two identifiers `resid_N` and `RJ_N` where "N" is some integer number.
The first is going to be the name of the function that evaluates the equation
and the second is going to be the name of the function that evaluates both the
equation and its gradient.
"""
function funcsyms end

function funcsyms(mod::Module)
    if !isdefined(mod, :__counter)
        Base.eval(mod, :(__counter = Ref{Int}(1)))
    end
    num = mod.__counter[]
    fn1 = Symbol("resid_", num)
    fn2 = Symbol("RJ_", num)
    mod.__counter[] += 1
    return fn1, fn2
end

# Can be changed to MAX_CHUNK_SIZE::Bool = 4 when support for Julia 1.7
# is dropped.
const MAX_CHUNK_SIZE = Ref(4)

# Used to avoid specialzing the ForwardDiff functions on
# every equation.
struct FunctionWrapper <: Function
    f::Function
end
(f::FunctionWrapper)(x) = f.f(x)

"""
    makefuncs(expr, tssyms, sssyms, psyms, mod)

Create two functions that evaluate the residual and its gradient for the given
expression.

!!! warning
    Internal function. Do not call directly.

### Arguments
- `expr`: the expression
- `tssyms`: list of time series variable symbols
- `sssyms`: list of steady state symbols
- `psyms`: list of parameter symbols

### Return value
Return a quote block to be evaluated in the module where the model is being
defined. The quote block contains definitions of the residual function (as a
callable `EquationEvaluator` instance) and a second function that evaluates both
the residual and its gradient (as a callable `EquationGradient` instance).
"""
function makefuncs(expr, tssyms, sssyms, psyms, mod)
    fn1, fn2 = funcsyms(mod)
    x = gensym("x")
    nargs = length(tssyms) + length(sssyms)
    chunk = min(nargs, MAX_CHUNK_SIZE[])
    return quote
        function (ee::EquationEvaluator{$(QuoteNode(fn1))})($x::Vector{<:Real})
            ($(tssyms...), $(sssyms...),) = $x
            ($(psyms...),) = values(ee.params)
            $expr
        end
        const $fn1 = EquationEvaluator{$(QuoteNode(fn1))}(UInt(0),
            $(@__MODULE__).LittleDict(Symbol[$(QuoteNode.(psyms)...)], fill(nothing, $(length(psyms)))))
        const $fn2 = EquationGradient($FunctionWrapper($fn1), $nargs, Val($chunk))
        $(@__MODULE__).precompilefuncs($fn1, $fn2, Val($chunk), MyTag)
        ($fn1, $fn2)
    end
end

"""
    initfuncs(mod::Module)

Initialize the given module before creating functions that evaluate residuals
and thier gradients.

!!! warning
    Internal function. Do not call directly.

### Implementation (for developers)
Declare the necessary types in the module where the model is being defined.
There are two such types. First is `EquationEvaluator`, which is callable and
stores a collection of parameters. The call will be defined in
[`makefuncs`](@ref) and will evaluate the residual. The other type is
`EquationGradient`, which is also callable and stores the `EquationEvaluator`
together with a `DiffResult` and a `GradientConfig` used by `ForwardDiff`. Its
call is defined here and computes the residual and the gradient.
"""
function initfuncs(mod::Module)
    if :MyTag ∉ names(mod; all=true)
        mod.eval(quote
            struct MyTag end
            struct EquationEvaluator{FN} <: Function
                rev::Ref{UInt}
                params::$(@__MODULE__).LittleDict{Symbol,Any}
            end
            struct EquationGradient{DR,CFG} <: Function
                fn1::Function
                dr::DR
                cfg::CFG
            end
            EquationGradient(fn1::Function, nargs::Int, ::Val{N}) where {N} = EquationGradient(fn1,
                $(@__MODULE__).DiffResults.DiffResult(zero(Float64), zeros(Float64, nargs)),
                $(@__MODULE__).ForwardDiff.GradientConfig(fn1, zeros(Float64, nargs), $(@__MODULE__).ForwardDiff.Chunk{N}(), MyTag))
            function (s::EquationGradient)(x::Vector{Float64})
                $(@__MODULE__).ForwardDiff.gradient!(s.dr, s.fn1, x, s.cfg)
                return s.dr.value, s.dr.derivs[1]
            end
        end)
    end
    return nothing
end

_index_of_var(var, allvars) = indexin([var], allvars)[1]

###########################################################
# Part 2: Evaluation data for models and equations

#### Equation evaluation data

# It's not needed for the normal case. It'll be specialized later for
# selectively linearized equations.

abstract type AbstractEqnEvalData end
eval_RJ(eqn::AbstractEquation, x) = eqn.eval_RJ(x)
eval_resid(eqn::AbstractEquation, x) = eqn.eval_resid(x)

abstract type DynEqnEvalData <: AbstractEqnEvalData end
struct DynEqnEvalData0 <: DynEqnEvalData end
struct DynEqnEvalDataN <: DynEqnEvalData
    ss::Vector{Float64}
end

function _fill_ss_values(eqn, ssvals, allvars)
    ret = fill(0.0, length(eqn.ssrefs))
    bad = ModelSymbol[]
    for (i, v) in enumerate(keys(eqn.ssrefs))
        vi = _index_of_var(v, allvars)
        ret[i] = ssvals[2vi-1]
        if !isapprox(ssvals[2vi], 0, atol=1e-12)
            push!(bad, v)
        end
    end
    if !isempty(bad)
        nzslope = tuple(unique(bad)...)
        @warn "@sstate used with non-zero slope" eqn nzslope
    end
    return ret
end
function DynEqnEvalData(eqn, model)
    return length(eqn.ssrefs) == 0 ? DynEqnEvalData0() : DynEqnEvalDataN(
        _fill_ss_values(eqn, model.sstate.values, model.allvars)
    )
end

eval_resid(eqn::AbstractEquation, x, ed::DynEqnEvalDataN) = eqn.eval_resid(vcat(x, ed.ss))
@inline function eval_RJ(eqn::AbstractEquation, x, ed::DynEqnEvalDataN)
    R, J = eqn.eval_RJ(vcat(x, ed.ss))
    return (R, J[1:length(x)])
end
eval_resid(eqn::AbstractEquation, x, ::DynEqnEvalData0) = eqn.eval_resid(x)
eval_RJ(eqn::AbstractEquation, x, ::DynEqnEvalData0) = eqn.eval_RJ(x)


"""
    AbstractModelEvaluationData

Base type for all model evaluation structures.
Specific derived types would specialize in different types of models.

### Implementaion (for developers)
Derived types must specialize two functions
  * [`eval_R!`](@ref) - evaluate the residual
  * [`eval_RJ`](@ref) - evaluate the residual and its Jacobian
"""
abstract type AbstractModelEvaluationData end

"""
    eval_R!(res::AbstractArray{Float64,1}, point::AbstractArray{Float64, 2}, ::MED) where MED <: AbstractModelEvaluationData

Evaluate the model residual at the given point using the given model evaluation
structure. The residual is stored in the provided vector.

### Implementation details (for developers)
When creating a new type of model evaluation data, you must define a method of
this function specialized to it.

The `point` argument will be a 2d array, with the number of rows equal to
`maxlag+maxlead+1` and the number of columns equal to the number of
`variables+shocks+auxvars` of the model. The `res` vector will have the same
length as the number of equations + auxiliary equations. Your implementation
must not modify `point` and must update `res`.

See also: [`eval_RJ`](@ref)
"""
function eval_R! end
export eval_R!
eval_R!(res::AbstractVector{Float64}, point::AbstractMatrix{Float64}, ::AMED) where {AMED<:AbstractModelEvaluationData} = modelerror(NotImplementedError, AMED)

"""
    eval_RJ(point::AbstractArray{Float64, 2}, ::MED) where MED <: AbstractModelEvaluationData

Evaluate the model residual and its Jacobian at the given point using the given
model evaluation structure. Return a tuple, with the first element being the
residual and the second element being the Jacobian.

### Implementation details (for developers)
When creating a new type of model evaluation data, you must define a method of
this function specialized to it.

The `point` argument will be a 2d array, with the number of rows equal to
`maxlag+maxlead+1` and the number of columns equal to the number of
`variables+shocks+auxvars` of the model. Your implementation must not modify
`point` and must return the tuple of (residual, Jacobian) evaluated at the given
`point`. The Jacobian is expected to be `SparseMatrixCSC` (*this might change in
the future*).

See also: [`eval_R!`](@ref)
"""
function eval_RJ end
export eval_RJ
eval_RJ(point::AbstractMatrix{Float64}, ::AMED) where {AMED<:AbstractModelEvaluationData} = modelerror(NotImplementedError, AMED)

##### The standard Model Evaluation Data used in the general case.
"""
    ModelEvaluationData <: AbstractModelEvaluationData

The standard model evaluation data used in the general case and by default.
"""
struct ModelEvaluationData{E<:AbstractEquation,I,D<:DynEqnEvalData} <: AbstractModelEvaluationData
    params::Ref{Parameters{ModelParam}}
    eedata::Vector{D}
    alleqns::Vector{E}
    allinds::Vector{I}
    "Placeholder for the Jacobian matrix"
    J::SparseMatrixCSC{Float64,Int64}
    "Placeholder for the residual vector"
    R::Vector{Float64}
    rowinds::Vector{Vector{Int64}}
end

@inline function _update_eqn_params!(ee, params)
    if ee.rev[] !== params.rev[]
        for k in keys(ee.params)
            ee.params[k] = getproperty(params, k)
        end
        ee.rev[] = params.rev[]
    end
end

"""
    ModelEvaluationData(model::AbstractModel)

Create the standard evaluation data structure for the given model.
"""
function ModelEvaluationData(model::AbstractModel)
    time0 = 1 + model.maxlag
    alleqns = model.alleqns
    neqns = length(alleqns)
    allvars = model.allvars
    nvars = length(allvars)
    allinds = [[CartesianIndex((time0 + ti, _index_of_var(var, allvars))) for (var, ti) in keys(eqn.tsrefs)] for eqn in alleqns]
    ntimes = 1 + model.maxlag + model.maxlead
    LI = LinearIndices((ntimes, nvars))
    II = reduce(vcat, (fill(i, length(eqn.tsrefs)) for (i, eqn) in enumerate(alleqns)))
    JJ = [LI[inds] for inds in allinds]
    M = SparseArrays.sparse(II, reduce(vcat, JJ), similar(II), neqns, ntimes * nvars)
    M.nzval .= 1:length(II)
    rowinds = [copy(M[i, LI[inds]].nzval) for (i, inds) in enumerate(JJ)]
    eedata = [DynEqnEvalData(eqn, model) for eqn in alleqns]
    if model.dynss && !issssolved(model)
        @warn "Steady state not solved."
    end
    ModelEvaluationData(Ref(model.parameters), eedata,
        alleqns, allinds, similar(M, Float64), Vector{Float64}(undef, neqns), rowinds)
end

function eval_R!(res::AbstractVector{Float64}, point::AbstractMatrix{Float64}, med::ModelEvaluationData)
    for (i, eqn, inds, ed) in zip(1:length(med.alleqns), med.alleqns, med.allinds, med.eedata)
        _update_eqn_params!(eqn.eval_resid, med.params[])
        res[i] = eval_resid(eqn, point[inds], ed)
    end
    return nothing
end

function eval_RJ(point::Matrix{Float64}, med::ModelEvaluationData)
    neqns = length(med.alleqns)
    res = similar(med.R)
    jac = med.J
    for (i, eqn, inds, ri, ed) in zip(1:neqns, med.alleqns, med.allinds, med.rowinds, med.eedata)
        _update_eqn_params!(eqn.eval_resid, med.params[])
        res[i], jac.nzval[ri] = eval_RJ(eqn, point[inds], ed)
    end
    return res, jac
end

##################################################################################
# PART 3: Selective linearization

##### Linearized equation

# specialize equation evaluation data for linearized equation
mutable struct LinEqnEvalData <: AbstractEqnEvalData
    # Taylor series expansion:
    #    f(x) = f(s) + ∇f(s) ⋅ (x-s) + O(|x-s|^2)
    # we store s in sspt, f(s) in resid and ∇f(s) in grad
    # we expect that f(s) should be 0 (because steady state is a solution) and
    #    warn it in isn't
    # we store it and use it because even with ≠0 it's still a valid Taylor
    #    expansion.
    resid::Float64
    grad::Vector{Float64}
    sspt::Vector{Float64}   # point about which we linearize
    LinEqnEvalData(r, g, s) = new(Float64(r), Float64[g...], Float64[s...])
end

eval_resid(eqn::AbstractEquation, x, led::LinEqnEvalData) = led.resid + sum(led.grad .* (x - led.sspt))
eval_RJ(eqn::AbstractEquation, x, led::LinEqnEvalData) = (eval_resid(eqn, x, led), led.grad)

function LinEqnEvalData(eqn, sspt, ed::DynEqnEvalData)
    return LinEqnEvalData(eval_RJ(eqn, sspt, ed)..., sspt)
end

mutable struct SelectiveLinearizationMED <: AbstractModelEvaluationData
    sspt::Matrix{Float64}
    eedata::Vector{AbstractEqnEvalData}
    med::ModelEvaluationData
end

function SelectiveLinearizationMED(model::AbstractModel)

    sstate = model.sstate
    if !issssolved(sstate)
        linearizationerror("Steady state solution is not available.")
    end
    if maximum(abs, sstate.values[2:2:end]) > getoption(model, :tol, 1e-12)
        linearizationerror("Steady state solution has non-zero slope. Not yet implemented.")
    end

    med = ModelEvaluationData(model)

    sspt = Matrix{Float64}(undef, 1 + model.maxlag + model.maxlead, length(model.varshks))
    for (i, v) in enumerate(model.varshks)
        sspt[:, i] = transform(sstate[v][-model.maxlag:model.maxlead, ref=0], v)
    end
    eedata = Vector{AbstractEqnEvalData}(undef, length(med.alleqns))
    num_lin = 0
    for (i, (eqn, inds)) in enumerate(zip(med.alleqns, med.allinds))
        _update_eqn_params!(eqn.eval_resid, model.parameters)
        ed = DynEqnEvalData(eqn, model)
        if islin(eqn)
            num_lin += 1
            eedata[i] = LinEqnEvalData(eqn, sspt[inds], ed)
            resid = eedata[i].resid
            if abs(resid) > getoption(model, :tol, 1e-12)
                @warn "Non-zero steady state residual in equation E$i" eqn resid
            end
        else
            eedata[i] = ed
        end
    end
    if num_lin == 0
        @warn "\nNo equations were linearized.\nAnnotate equations for selective linearization with `@lin`."
    end
    return SelectiveLinearizationMED(sspt, eedata, med)
end


function eval_R!(res::AbstractVector{Float64}, point::AbstractMatrix{Float64}, slmed::SelectiveLinearizationMED)
    med = slmed.med
    for (i, eqn, inds, eed) in zip(1:length(med.alleqns), med.alleqns, med.allinds, slmed.eedata)
        islin(eqn) || _update_eqn_params!(eqn.eval_resid, med.params[])
        res[i] = eval_resid(eqn, point[inds], eed)
    end
    return nothing
end

function eval_RJ(point::Matrix{Float64}, slmed::SelectiveLinearizationMED)
    med = slmed.med
    neqns = length(med.alleqns)
    res = similar(med.R)
    jac = med.J
    for (i, eqn, inds, ri, eed) in zip(1:neqns, med.alleqns, med.allinds, med.rowinds, slmed.eedata)
        islin(eqn) || _update_eqn_params!(eqn.eval_resid, med.params[])
        res[i], jac.nzval[ri] = eval_RJ(eqn, point[inds], eed)
    end
    return res, jac
end

"""
    selective_linearize!(model)

Instruct the model instance to use selective linearization. Only equations
annotated with `@lin` in the model definition will be linearized about the
current steady state solution while the rest of the eq

"""
function selective_linearize!(model::AbstractModel)
    setevaldata!(model, selective_linearize=SelectiveLinearizationMED(model))
    return model
end
export selective_linearize!


"""
    refresh_med!(model)

Refresh the model evaluation data stored within the given model instance. Most
notably, this is necessary when the steady state is used in the dynamic
equations.

Normally there's no need for the end-used to call this function. It should be
called when necessay by the solver.
"""
function refresh_med! end
export refresh_med!

# dispatcher
refresh_med!(model::AbstractModel, variant::Symbol=model.options.variant) = model.dynss ? refresh_med!(model, Val(variant)) : model
# catch all and issue a meaningful error message
refresh_med!(::AbstractModel, V::Val{VARIANT}) where VARIANT = modelerror("Missing method to update model variant: $VARIANT")
# specific cases
# refresh_med!(m::AbstractModel, ::Type{NoModelEvaluationData}) = (m.evaldata = ModelEvaluationData(m); m)
refresh_med!(model::AbstractModel, ::Val{:default}) = (setevaldata!(model, default = ModelEvaluationData(model)); model)
refresh_med!(model::AbstractModel, ::Val{:selective_linearize}) = selective_linearize!(model)
