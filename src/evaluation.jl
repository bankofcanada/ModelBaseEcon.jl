##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

###########################################################
# Part 1: Helper functions


"""
    precompilefuncs(resid, RJ, ::Val{N}) where N

Add code that precompiles the given `resid` and `RJ` functions together
with the dual-number arythmetic required by ForwardDiff.

!!! warning
    Internal function. Do not call directly

# Implementation (for developers)
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

    for pred in (ForwardDiff.UNARY_PREDICATES ∪ Symbol[:-, :+, :log, :exp])
        pred ∈ (:iseven, :isodd) || precompile(getfield(Base, pred), (Float64,)) || error("precompile")
        precompile(getfield(Base, pred), (dual,)) || error("precompile")
    end

    for pred in ForwardDiff.BINARY_PREDICATES ∪ Symbol[:+, :-, :*, :/, :^]
        precompile(getfield(Base, pred), (Float64, Float64)) || error("precompile")
        precompile(getfield(Base, pred), (dual, Float64)) || error("precompile")
        precompile(getfield(Base, pred), (Float64, dual)) || error("precompile")
        precompile(getfield(Base, pred), (dual, dual)) || error("precompile")
    end

    precompile(ForwardDiff.extract_gradient!, (Type{tag}, mdr, dual)) || error("precompile")
    precompile(ForwardDiff.vector_mode_gradient!, (mdr, typeof(resid), Array{Float64,1}, cfg)) || error("precompile")

    # precompile(Tuple{typeof(ForwardDiff.extract_gradient!), Type{tag}, mdr, dual}) || error("precompile")
    # precompile(Tuple{typeof(ForwardDiff.vector_mode_gradient!), mdr, resid, Array{Float64, 1}, cfg}) || error("precompile")

    return nothing
end

"""
    funcsyms(mod::Module)

Create a pair of identifiers that does not conflict with existing identifiers
in the given module.

!!! warning
    Internal function. Do not call directly.

### Implementation (for developers)
We need two identifiers `resid_N` and `RJ_N` where "N" is some integer number.
The first is going to be the name of the function that evaluates the equation and
the second is going to be the name of the function that evaluates both the equation
and its gradient.
"""
function funcsyms end

let funcsyms_state = 0
    global funcsyms_counter() = (funcsyms_state += 1)
end
function funcsyms(mod::Module)
    fn1, fn2 = mod.eval(quote
        let nms = names(@__MODULE__; all = true)
            num = $(@__MODULE__).funcsyms_counter()
            local fn1 = Symbol("resid_$num")
            local fn2 = Symbol("RJ_$num")
            while fn1 ∈ nms || fn2 ∈ nms
                num = $(@__MODULE__).funcsyms_counter()
                fn1 = Symbol("resid_$num")
                fn2 = Symbol("RJ_$num")
            end
            fn1, fn2
        end
    end)
end

"""
    makefuncs(expr, vsyms [, params_expr]; mod::Module)

Create two functions that evaluate the residual and its gradient for the given
expression.

!!! warning
    Internal function. Do not call directly.

### Arguments
- `expr`: the expression
- `vsyms`: a list of variables in the expression.
- `params_expr`: an expression to be included in the function code before evaluating 
    the expression. This allows for assigning values of parameters in the expression,
    i.e. variables that are not arguments to the funcion.

### Return value
Return a quote block to be evaluated in the module where the model is being defined. 
The quote block contains definitions of the residual function and a second function
that evaluates both the residual and its gradient.
"""
function makefuncs(expr, vsyms, params_expr = nothing; mod::Module)
    fn1, fn2 = funcsyms(mod)
    x = gensym("x")
    nargs = length(vsyms)
    return quote
        function $fn1($x::AbstractVector{T}) where {T<:Real}
            ($(vsyms...),) = $x
            $(params_expr)
            $expr
        end
        const $fn2 = EquationGradient($fn1, Val($nargs))
        $(@__MODULE__).precompilefuncs($fn1, $fn2, Val($nargs), MyTag)
        ($fn1, $fn2)
    end
end

"""
    initfuncs(mod::Module)

Initialize the given module before creating functions that evaluate residuals
and thier gradients.


"""
function initfuncs(mod::Module)
    if :MyTag ∉ names(mod; all = true)
        mod.eval(quote
            struct MyTag end
            struct EquationGradient{DR,CFG} <: Function
                fn1::Function
                dr::DR
                cfg::CFG
            end
            EquationGradient(fn1::Function, ::Val{N}) where {N} = EquationGradient(fn1,
                $(@__MODULE__).DiffResults.DiffResult(zero(Float64), zeros(Float64, N)),
                $(@__MODULE__).ForwardDiff.GradientConfig(fn1, zeros(Float64, N), $(@__MODULE__).ForwardDiff.Chunk{N}(), MyTag))
            function (s::EquationGradient)(x::AbstractVector{Float64})
                $(@__MODULE__).ForwardDiff.gradient!(s.dr, s.fn1, x, s.cfg)
                return s.dr.value, s.dr.derivs[1]
            end
        end)
    end
    return nothing
end

###########################################################
# Part 2: Evaluation data for models and equations

#### Equation evaluation data 

# It's not needed for the normal case. It'll be specialized later for
# selectively linearized equations.

abstract type AbstractEqnEvalData end
@inline eval_RJ(eqn, x) = eqn.eval_RJ(x)
@inline eval_resid(eqn, x) = eqn.eval_resid(x)

struct EqnEvalData <: AbstractEqnEvalData end
@inline eval_RJ(eqn, x, ::EqnEvalData) = eqn.eval_RJ(x)
@inline eval_resid(eqn, x, ::EqnEvalData) = eqn.eval_resid(x)


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

Evaluate the model residual at the given point using the given model evaluation structure.
The residual is stored in the provided vector.

### Implementation details (for developers)
When creating a new type of model evaluation data, you must define a
method of this function specialized to it.

The `point` argument will be a 2d array, with the number of rows equal to
`maxlag+maxlead+1` and the number of columns equal to the number of `variables+shocks+auxvars` of the model.
The `res` vector will have the same length as the number of equations + auxiliary equations. 
Your implementation must not modify `point` and must update `res`.

See also: [`eval_RJ`](@ref)
"""
function eval_R! end
export eval_R!
eval_R!(res::AbstractVector{Float64}, point::AbstractMatrix{Float64}, ::AMED) where {AMED<:AbstractModelEvaluationData} = throw(NotImplementedError(AMED))

"""
    eval_RJ(point::AbstractArray{Float64, 2}, ::MED) where MED <: AbstractModelEvaluationData

Evaluate the model residual and its Jacobian at the given point using the given model evaluation structure.
Return a tuple, with the first element being the residual and the second element being the Jacobian.

### Implementation details (for developers)
When creating a new type of model evaluation data, you must define a
method of this function specialized to it.

The `point` argument will be a 2d array, with the number of rows equal to
`maxlag+maxlead+1` and the number of columns equal to the number of `variables+shocks+auxvars` of the model.
Your implementation must not modify `point` and must return the tuple of (residual, Jacobian) evaluated
at the given `point`. The Jacobian is expected to be `SparseMatrixCSC` (*this might change in the future*).

See also: [`eval_R!`](@ref)
"""
function eval_RJ end
export eval_RJ
eval_RJ(point::AbstractMatrix{Float64}, ::AMED) where {AMED<:AbstractModelEvaluationData} = throw(NotImplementedError(AMED))


##### Model Evaluation Data that doesn't exist.
"""
    struct NoModelEvaluationData <: AbstractModelEvaluationData

Specific type that indicates that the model cannot be evaluated.
This is used as a placeholder while the model is being defined.
During initialization, the actual model evaluation data is created.
"""
struct NoModelEvaluationData <: AbstractModelEvaluationData end
const NoMED = NoModelEvaluationData()
eval_R!(res::AbstractVector{Float64}, point::AbstractMatrix{Float64}, ::NoModelEvaluationData) = throw(ModelNotInitError())
eval_RJ(point::AbstractMatrix{Float64}, ::NoModelEvaluationData) = throw(ModelNotInitError())

##### The standard Model Evaluation Data used in the general case.
"""
    ModelEvaluationData <: AbstractModelEvaluationData

The standard model evaluation data used in the general case and by default.
"""
struct ModelEvaluationData{E<:AbstractEquation,I} <: AbstractModelEvaluationData
    alleqns::Vector{E}
    allinds::Vector{I}
    "Placeholder for the Jacobian matrix"
    J::SparseMatrixCSC{Float64,Int64}
    "Placeholder for the residual vector"
    R::Vector{Float64}
    rowinds::Vector{Vector{Int64}}
end

"""
    ModelEvaluationData(model::AbstractModel)

Create the standard evaluation data structure for the given model.
"""
function ModelEvaluationData(model::AbstractModel)
    time0 = 1 + model.maxlag
    alleqns = model.alleqns
    neqns = length(alleqns)
    allinds = @timer [[CartesianIndex((time0 + ti, vi)) for (ti, vi) in eqn.vinds] for eqn in alleqns]
    ntimes = 1 + model.maxlag + model.maxlead
    nvars = length(model.allvars)
    LI = LinearIndices((ntimes, nvars))
    II = @timer reduce(vcat, (fill(Int64(i), length(eqn.vinds)) for (i, eqn) in enumerate(alleqns)))
    JJ = @timer [LI[inds] for inds in allinds]
    M = @timer SparseArrays.sparse(II, reduce(vcat, JJ), similar(II), neqns, ntimes * nvars)
    M.nzval .= @timer 1:length(II)
    rowinds = @timer [copy(M[i, LI[inds]].nzval) for (i, inds) in enumerate(JJ)]
    ModelEvaluationData(alleqns, allinds, similar(M, Float64), Vector{Float64}(undef, neqns), rowinds)
end

function eval_R!(res::AbstractVector{Float64}, point::AbstractMatrix{Float64}, med::ModelEvaluationData)
    # med === NoMED && throw(ModelNotInitError())
    for (i, eqn, inds) in zip(1:length(med.alleqns), med.alleqns, med.allinds)
        res[i] = eval_resid(eqn, point[inds])
    end
    return nothing
end

function eval_RJ(point::AbstractMatrix{Float64}, med::ModelEvaluationData)
    # med === NoMED && throw(ModelNotInitError())
    neqns = length(med.alleqns)
    res = similar(med.R)
    jac = med.J
    for (i, eqn, inds, ri) in zip(1:neqns, med.alleqns, med.allinds, med.rowinds)
        res[i], jac.nzval[ri] = eval_RJ(eqn, point[inds])
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

@inline eval_resid(eqn, x, eed::LinEqnEvalData) = eed.resid + sum(eed.grad .* (x - eed.sspt))
@inline eval_RJ(eqn, x, eed::LinEqnEvalData) = (eval_resid(eqn, x, eed), eed.grad)

function LinEqnEvalData(eqn, sspt)
    return LinEqnEvalData(eqn.eval_RJ(sspt)..., sspt)
end

mutable struct SelectiveLinearizationMED <: AbstractModelEvaluationData
    sspt::Matrix{Float64}
    eedata::Vector{AbstractEqnEvalData}
    med::ModelEvaluationData
end

function SelectiveLinearizationMED(model::AbstractModel)

    sstate = model.sstate
    if !issssolved(sstate)
        error("Steady state solution is not available.")
    end
    if maximum(abs, sstate.values[2:2:end]) > getoption(model, :tol, 1e-12)
        error("Steady state solution has non-zero slope. Not yet implemented.")
    end

    med = model.evaldata
    if !isa(med, ModelEvaluationData)
        if hasproperty(med, :med)
            med = med.med
        else
            med = ModelEvaluationData(model)
        end
    end

    sspt = Matrix{Float64}(undef, 1 + model.maxlag + model.maxlead, length(model.varshks))
    for (i, v) in enumerate(model.varshks)
        sspt[:, i] = sstate[v][-model.maxlag:model.maxlead, ref = 0]
    end
    eedata = Vector{AbstractEqnEvalData}(undef, length(med.alleqns))
    num_lin = 0
    for (i, (eqn, inds)) in enumerate(zip(med.alleqns, med.allinds))
        if islin(eqn)
            num_lin += 1
            eedata[i] = LinEqnEvalData(eqn, sspt[inds])
            resid = eedata[i].resid
            if abs(resid) > getoption(model, :tol, 1e-12)
                @warn "Non-zero steady state residual in equation E$i" eqn resid
            end
        else
            eedata[i] = EqnEvalData()
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
        res[i] = eval_resid(eqn, point[inds], eed)
    end
    return nothing
end

function eval_RJ(point::AbstractMatrix{Float64}, slmed::SelectiveLinearizationMED)
    med = slmed.med
    neqns = length(med.alleqns)
    res = similar(med.R)
    jac = med.J
    for (i, eqn, inds, ri, eed) in zip(1:neqns, med.alleqns, med.allinds, med.rowinds, slmed.eedata)
        res[i], jac.nzval[ri] = eval_RJ(eqn, point[inds], eed)
    end
    return res, jac
end

"""
    selective_linearize!(model)

Instruct the model instance to use selective linearization. 
Only equations annotated with `@lin` in the model definition will be
linearized about the current steady state solution while the rest of the eq
    
"""
function selective_linearize!(model::AbstractModel)
    model.evaldata = SelectiveLinearizationMED(model)
    return model
end
export selective_linearize!
