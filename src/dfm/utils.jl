##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################


"""spell a number using subscript digits e.g., `num2sub(5)` returns "₅"`"""
_num2sub(n::Integer) = n < 0 ? '₋' * _num2sub(-n) :
                       n < 10 ? string('₀' + n) :
                       _num2sub(n ÷ 10) * _num2sub(n % 10)

const sup_nums = ('⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹')
"""spell a number using superscript digits e.g., `num2sup(5)` returns "⁵"`"""
_num2sup(n::Integer) = n < 0 ? '⁻' * _num2sup(-n) :
                       n < 10 ? sup_nums[n+1] :
                       _num2sup(n ÷ 10) * _num2sup(n % 10)

@inline _make_factor_names(name::Sym, N::Integer)::Vector{Symbol} = (N == 1) ? Symbol[Symbol(name)] : Symbol[Symbol(name, _num2sup(i)) for i = 1:N]

@inline _tosymvec(::A) where {A} = error("No method available to convert from $A to Vector{Symbol}.")
@inline _tosymvec(s::Symbol)::Vector{Symbol} = Symbol[s]
@inline _tosymvec(s::Sym)::Vector{Symbol} = Symbol[Symbol(s)]
@inline _tosymvec(v::Vector{Symbol})::Vector{Symbol} = v
@inline _tosymvec(v::LikeVec{Symbol})::Vector{Symbol} = collect(v)
@inline _tosymvec(v::Vector{<:Sym})::Vector{Symbol} = Symbol[Symbol(s) for s in v]
@inline _tosymvec(v::SymVec)::Vector{Symbol} = Symbol[Symbol(s) for s in v]

"default name of shock associated with a variable"
@inline _make_shock(name::Sym) = to_shock(Symbol(name, "_shk"))

"default names of shocks associated with a list of variables"
@inline _make_shocks(names) = ModelVariable[_make_shock(name) for name in names]

"""default name for an idiosyncratic component associated with a variable"""
@inline _make_ic_name(name::Sym) = Symbol(name, "_cor")

"name given to the lag of a variable"
@inline _make_lag_name(name::Sym, lag::Int) =
    lag < 0 ? error("Negative lag") :
    lag == 0 ? Symbol(name) : Symbol(name, "ₜ₋", _num2sub(lag))

@inline _enumerate_vars(vars) = (; (Symbol(v) => n for (n, v) = enumerate(vars))...)

# _do_wrap(::Nothing, AXES...) = nothing
_do_wrap(X::AbstractArray, AXES...) = ComponentArray(X, AXES...)
# _do_wrap(X, AXES...) = error("Unable to wrap $(nameof(typeof(X))) in a ComponentArray")

function _wrap_arrays(bm::DFMBlockOrModel, R, J, point)
    # number of equations (same as number of endogenous variables)
    ne = nendog(bm)
    # total number of variables
    nv = nvarshks(bm)
    # number of time periods
    nt = lags(bm) + 1 + leads(bm)

    a1vars = _enumerate_vars(endog(bm))
    a3vars = _enumerate_vars(varshks(bm))

    # rows - equations correspond to endogenous variables
    A1 = Axis{a1vars}
    # columns - time periods - by - variables
    A2 = FlatAxis
    A3 = Axis{a3vars}

    if !isnothing(R) && size(R) !== (ne,)
        throw(DimensionMismatch("Wrong size of R. Expected ($ne,), got $(size(R))"))
    end
    if !isnothing(J) && size(J) !== (ne, nt * nv)
        throw(DimensionMismatch("Wrong size of J. Expected ($ne, $(nt*nv)), got $(size(J))"))
    end
    if !isnothing(point) && size(point) !== (nt, nv)
        throw(DimensionMismatch("Wrong size of data point. Expected ($nt,$nv), got $(size(point))"))
    end

    CR = _do_wrap(R, A1())
    CJ = isnothing(J) ? nothing : _do_wrap(reshape(J, ne, nt, nv), A1(), A2(), A3())
    Cpoint = _do_wrap(point, A2(), A3())

    return CR, CJ, Cpoint
end

@inline ComponentArrays.toval(v::ModelVariable) = ComponentArrays.toval(Symbol(v))
@inline ComponentArrays.toval(tv::NTuple{N,ModelVariable}) where {N} = ComponentArrays.toval(Symbol[t fot t in tv])
@inline ComponentArrays.toval(av::AbstractArray{<:ModelVariable}) = ComponentArrays.toval(Symbol[a for a in av])

"Check if the reference includes the entire block (`true`) or only some components in it (`false`)"
isa_BlockRef(x) = x isa _BlockRef
all_BlockRef(x::NamedList) = all(isa_BlockRef, x.vals)

