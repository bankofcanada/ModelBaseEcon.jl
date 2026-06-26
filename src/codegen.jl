##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

module Codegen

using ..IR
using ..Symbolic
using Symbolics: Symbolics, Num
using RuntimeGeneratedFunctions: RuntimeGeneratedFunctions, RuntimeGeneratedFunction,
                                  drop_expr

# RGF needs init in every module that creates RuntimeGeneratedFunctions.
RuntimeGeneratedFunctions.init(@__MODULE__)

export EquationFunctions, build_equation_functions, build_model_functions

"""
Callable evaluators for one equation, all `RuntimeGeneratedFunction`s
wrapped behind `Function` to keep the field type uniform across equations.

- `eval_resid(x, p)` returns the scalar residual `F(x, p)`.
- `eval_RJ!(J, x, p)` writes the gradient `∇F(x, p)` into the preallocated
  vector `J` (length `n_x`) and returns the residual `F(x, p)`.
- `eval_hess!(H, x, p)` writes the Hessian into `H` (size `n_x x n_x`) when
  `kernel.hessian !== nothing`; `nothing` otherwise.
- `eval_hod` is a vector of Hessian-and-up RGFs for orders 3..max_hod_order,
  or `nothing`.
"""
struct EquationFunctions
    eval_resid::Function
    eval_RJ!::Function
    eval_hess!::Union{Function, Nothing}
    eval_hod::Union{Vector{Function}, Nothing}
    n_x::Int
    n_p::Int
end

# ----------------------------------------------------------------------
# RGF construction. We use this Codegen module as both the cache and
# context module - both of which need RuntimeGeneratedFunctions.init,
# done above.
# ----------------------------------------------------------------------

@inline function _rgf(expr::Expr)
    f = RuntimeGeneratedFunction(@__MODULE__, @__MODULE__, expr)
    return drop_expr(f)
end

# ----------------------------------------------------------------------
# Build per-equation evaluators from an EquationKernel
# ----------------------------------------------------------------------

"""
Build RGF evaluators for one `EquationKernel`. Returns `EquationFunctions`.

The argument layout for every generated function is `(x, p)`:
    `x::Vector{Float64}` length `length(kernel.tsrefs)`
    `p::Vector{Float64}` length `length(kernel.params)` (root layout)
"""
function build_equation_functions(kernel::Symbolic.EquationKernel)
    n_x = length(kernel.tsrefs)
    n_p = length(kernel.params)
    x_syms = kernel.x_syms
    p_syms = kernel.p_syms

    # Residual: scalar Num -> single Expr.
    resid_expr = Symbolics.build_function(kernel.residual, x_syms, p_syms;
                                          expression = Val{true})
    eval_resid_rgf = _rgf(resid_expr)
    eval_resid = (x, p) -> eval_resid_rgf(x, p)::Float64

    # Gradient: Vector{Num} -> (oop_expr, ip_expr). Use in-place form.
    _grad_oop, grad_ip_expr = Symbolics.build_function(
        kernel.gradient, x_syms, p_syms; expression = Val{true})
    eval_grad_rgf = _rgf(grad_ip_expr)

    eval_RJ! = let resid_rgf = eval_resid_rgf, grad_rgf = eval_grad_rgf
        (J, x, p) -> begin
            grad_rgf(J, x, p)
            return resid_rgf(x, p)::Float64
        end
    end

    eval_hess! = nothing
    if kernel.hessian !== nothing
        _h_oop, h_ip = Symbolics.build_function(
            kernel.hessian, x_syms, p_syms; expression = Val{true})
        h_rgf = _rgf(h_ip)
        eval_hess! = (H, x, p) -> (h_rgf(H, x, p); nothing)
    end

    eval_hod = nothing
    if kernel.hod !== nothing
        eval_hod = Function[]
        for tensor in kernel.hod
            _t_oop, t_ip = Symbolics.build_function(
                tensor, x_syms, p_syms; expression = Val{true})
            t_rgf = _rgf(t_ip)
            push!(eval_hod, (T, x, p) -> (t_rgf(T, x, p); nothing))
        end
    end

    return EquationFunctions(eval_resid, eval_RJ!, eval_hess!, eval_hod, n_x, n_p)
end

"""
Build evaluators for every equation in a model. Returns
`Vector{EquationFunctions}` aligned with the kernel ordering.
"""
function build_model_functions(kernels::Vector{Symbolic.EquationKernel})
    return [build_equation_functions(k) for k in kernels]
end

end # module Codegen
