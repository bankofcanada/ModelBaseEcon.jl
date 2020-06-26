
"""
    precompilefuncs(resid, RJ, ::Val{N}) where N

Add code that precompiles the given `resid` and `RJ` functions together
with the dual-number arythmetic required by ForwardDiff.

!!! warning
    Internal function. Do not call directly

# Implementation (for developers)
"""
function precompilefuncs(resid, RJ, ::Val{N}) where N
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing

    tag = ForwardDiff.Tag{resid, Float64}
    dual = ForwardDiff.Dual{tag, Float64, N}
    duals = Array{dual, 1}
    cfg = ForwardDiff.GradientConfig{tag, Float64, N, duals}
    mdr = DiffResults.MutableDiffResult{1, Float64, Tuple{Array{Float64, 1}}}

    precompile(resid, (Array{Float64,1},)) || error("precompile")
    precompile(resid, (duals,)) || error("precompile")
    precompile(RJ, (Array{Float64,1},)) || error("precompile")

    for pred in (ForwardDiff.UNARY_PREDICATES ∪ Symbol[:-, :+, :log, :exp]) 
        pred ∈ (:iseven, :isodd) || precompile(getfield(Base,pred), (Float64,)) || error("precompile")
        precompile(getfield(Base,pred), (dual,)) || error("precompile")
    end

    for pred in ForwardDiff.BINARY_PREDICATES ∪ Symbol[:+, :-, :*, :/, :^]
        precompile(getfield(Base,pred), (Float64, Float64)) || error("precompile")
        precompile(getfield(Base,pred), (dual, Float64)) || error("precompile")
        precompile(getfield(Base,pred), (Float64, dual)) || error("precompile")
        precompile(getfield(Base,pred), (dual, dual)) || error("precompile")
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
        let nms = names(@__MODULE__; all=true)
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
function makefuncs(expr, vsyms, params_expr=nothing; mod::Module)
    fn1, fn2 = funcsyms(mod)
    x = gensym("x")
    nargs = length(vsyms)
    return quote
        function $fn1($x::AbstractVector{T}) where T<:Real
            ($(vsyms...),) = $x
            $(params_expr)
            $expr
        end
        const $fn2 = EquationGradient($fn1, Val{$nargs}())
        $(@__MODULE__).precompilefuncs($fn1, $fn2, Val{$nargs}())
        ($fn1, $fn2)
    end
end

"""
    initfuncs(mod::Module)

Initialize the given module before creating functions that evaluate residuals
and thier gradients.


"""
function initfuncs(mod::Module)
    mod.eval(quote
        struct MyTag end
        struct EquationGradient{DR,CFG} <: Function
            fn1::Function
            dr::DR
            cfg::CFG
        end
        EquationGradient(fn1::Function, ::Val{N}) where N = EquationGradient(fn1, 
                $(@__MODULE__).DiffResults.DiffResult(zero(Float64), zeros(Float64, N)),
                $(@__MODULE__).ForwardDiff.GradientConfig(fn1, zeros(Float64, N), $(@__MODULE__).ForwardDiff.Chunk{N}(), MyTag))
        function (s::EquationGradient)(x::AbstractVector{Float64})
            $(@__MODULE__).ForwardDiff.gradient!(s.dr, s.fn1, x, s.cfg)
            return s.dr.value, s.dr.derivs[1]
        end
    end)
    return nothing
end

