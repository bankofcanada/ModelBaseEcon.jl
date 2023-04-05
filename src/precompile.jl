
"""
    precompilefuncs(N::Int)

Pre-compiles functions used by models for a `ForwardDiff.Dual` numbers
with chunk size `N`.

!!! warning
    Internal function. Do not call directly

"""
function precompile_dual_funcs(N::Int)
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing

    tag = ModelBaseEconTag
    dual = ForwardDiff.Dual{tag,Float64,N}
    duals = Vector{dual}
    cfg = ForwardDiff.GradientConfig{tag,Float64,N,duals}
    mdr = DiffResults.MutableDiffResult{1,Float64,Tuple{Vector{Float64}}}

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

    precompile(ForwardDiff.extract_gradient!, (Type{tag}, mdr, dual)) || error("precompile")
    precompile(ForwardDiff.vector_mode_gradient!, (mdr, FunctionWrapper, Vector{Float64}, cfg)) || error("precompile")
    precompile(ForwardDiff.gradient!, (DiffResults.MutableDiffResult{1, Float64, Tuple{Vector{Float64}}},
                                       FunctionWrapper, Vector{Float64}, ForwardDiff.GradientConfig{ModelBaseEconTag, Float64, N, Vector{ForwardDiff.Dual{ModelBaseEconTag, Float64, N}}})) || error("precompile")

    return nothing
end

for i in 1:MAX_CHUNK_SIZE
    precompile_dual_funcs(i)
end

function precompile_other()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{typeof(eval_RJ), Matrix{Float64}, Model}) || error("precompile")
    precompile(Tuple{typeof(_update_eqn_params!), Function, Parameters{ModelParam}}) || error("precompile")
    precompile(Tuple{typeof(eval_RJ), Matrix{Float64}, ModelEvaluationData{Equation, Vector{CartesianIndex{2}}, DynEqnEvalData0}}) || error("precompile")
end

precompile_other()
