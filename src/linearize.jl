module Linearize

using ..IR
using ..Symbolic
using ..Compile

export LinEqnEvalData, linearize_equation, selectively_linearize
export LinearizationError

"""
Hard error raised when a `@lin`-flagged equation has a nonzero residual
at the supplied steady-state point. Per REQUIREMENTS.md §11 #3 this
replaces the warn-and-continue behavior of the legacy MBE.
"""
struct LinearizationError <: Exception
    eq_index::Int
    residual::Float64
    tol::Float64
    msg::String
end
Base.showerror(io::IO, e::LinearizationError) =
    print(io, "LinearizationError(eq=$(e.eq_index), R_ss=$(e.residual), tol=$(e.tol)): ",
              e.msg)

"""
Cached data for one linearized equation. The linearization is

    F_lin(x_eqn, p) = R_ss + grad_ss · (x_eqn - x_ss_per_slot)

where `R_ss` and `grad_ss` were evaluated at the steady-state point
once, and held fixed for all subsequent evaluations.
"""
struct LinEqnEvalData
    R_ss::Float64
    grad_ss::Vector{Float64}
    x_ss_per_slot::Vector{Float64}
end

# ----------------------------------------------------------------------
# Build per-slot SS values from the global x_ss + the equation's tsrefs
# ----------------------------------------------------------------------

function _x_ss_per_slot(eqn::Compile.Equation,
                        var_index::Dict{Symbol, Int},
                        shock_names::Set{Symbol},
                        x_ss_global::AbstractVector{Float64})
    x = Vector{Float64}(undef, eqn.n_x)
    for (k, ref) in pairs(eqn.tsrefs)
        if haskey(var_index, ref.name)
            x[k] = x_ss_global[var_index[ref.name]]
        elseif ref.name in shock_names
            x[k] = 0.0
        else
            error("linearize: equation references unknown name `$(ref.name)`")
        end
    end
    return x
end

# ----------------------------------------------------------------------
# Linearize one equation
# ----------------------------------------------------------------------

"""
    linearize_equation(eq, x_ss_per_slot, p; tol=1e-8) -> Equation

Return a new `Equation` whose `eval_resid` / `eval_RJ!` are replaced
with the linearized form around `x_ss_per_slot`. Throws
`LinearizationError` if `|R_ss| > tol`.

The returned equation keeps the same `residual::Num`, `tsrefs`, etc.,
so all downstream wiring (slot maps, sparsity patterns) is unchanged -
only the evaluators differ.
"""
function linearize_equation(eq::Compile.Equation,
                            x_ss_per_slot::Vector{Float64},
                            p::Vector{Float64};
                            eq_index::Int = 0,
                            tol::Float64 = 1e-8)
    grad_ss = Vector{Float64}(undef, eq.n_x)
    R_ss = eq.eval_RJ!(grad_ss, x_ss_per_slot, p)
    if !isfinite(R_ss) || abs(R_ss) > tol
        throw(LinearizationError(eq_index, R_ss, tol,
            "@lin equation has nonzero residual at the supplied steady-state point"))
    end

    data = LinEqnEvalData(R_ss, copy(grad_ss), copy(x_ss_per_slot))

    # Build linearized closures.
    eval_resid_lin = let d = data
        (x, _p) -> begin
            r = d.R_ss
            @inbounds for k in 1:length(x)
                r += d.grad_ss[k] * (x[k] - d.x_ss_per_slot[k])
            end
            return r::Float64
        end
    end
    eval_RJ_lin! = let d = data
        (J, x, _p) -> begin
            r = d.R_ss
            @inbounds for k in 1:length(x)
                J[k] = d.grad_ss[k]
                r += d.grad_ss[k] * (x[k] - d.x_ss_per_slot[k])
            end
            return r::Float64
        end
    end

    return Compile.Equation(
        eq.residual, eq.tsrefs, eq.params,
        eq.flags, eq.doc, eq.tags, eq.src,
        eval_resid_lin, eval_RJ_lin!,
        eq.eval_hess!, eq.eval_hod,
        eq.n_x, eq.n_p, eq.expr_hash,
        eq.x_syms, eq.p_syms,
    )
end

# ----------------------------------------------------------------------
# Linearize a whole model selectively
# ----------------------------------------------------------------------

"""
    selectively_linearize(model::Model, x_ss::Vector{Float64}; tol=1e-8) -> Model

Return a new `Model` in which every equation flagged `EQ_LIN` has been
replaced with its linearization around `x_ss`. Non-linearized equations
are kept by reference (object identity preserved). `x_ss` must have
length `length(model.defs.vars)`.

Throws `LinearizationError` on the first `@lin` equation whose residual
at `x_ss` exceeds `tol` - per REQUIREMENTS.md §11 #3, this is a hard
failure, not a warning.
"""
function selectively_linearize(model::Compile.Model,
                                x_ss::AbstractVector{Float64};
                                tol::Float64 = 1e-8)
    def = model.defs
    length(x_ss) == length(def.vars) ||
        error("selectively_linearize: x_ss has length $(length(x_ss)), expected $(length(def.vars))")

    var_index = Dict{Symbol, Int}()
    for (i, v) in pairs(def.vars); var_index[v.name] = i; end
    shock_names = Set(s.name for s in def.shocks)

    p = _resolve_param_values(def, model.param_layout)

    new_eqs = Compile.Equation[]
    for (i, eq) in pairs(model.eqns)
        if IR.EQ_LIN in eq.flags
            x_local = _x_ss_per_slot(eq, var_index, shock_names, x_ss)
            push!(new_eqs, linearize_equation(eq, x_local, p; eq_index = i, tol = tol))
        else
            push!(new_eqs, eq)
        end
    end
    return Compile.Model(def.name, Compile._pack_eqns(new_eqs),
                         model.param_layout, def, model.ss_eqns)
end

# Local copy of the resolver - we'd ideally share with SteadyState but that
# lives in the downstream RWStateSpaceEcon package.
function _resolve_param_values(def::IR.ModelDef,
                                layout::Vector{Symbolic.ParamRef})
    by_name = Dict{Symbol, IR.ParamDecl}()
    for p in def.params; by_name[p.name] = p; end
    vals = Vector{Float64}(undef, length(layout))
    for (i, ref) in pairs(layout)
        p = by_name[ref.name]
        if p.kind === IR.PARAM_SCALAR
            vals[i] = Float64(p.value)
        elseif p.kind === IR.PARAM_ARRAY
            vals[i] = Float64(p.value[ref.index])
        else
            error("root layout should only hold scalar/array params, got $(p.kind)")
        end
    end
    return vals
end

end # module Linearize
