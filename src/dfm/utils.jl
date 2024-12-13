##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2024, Bank of Canada
# All rights reserved.
##################################################################################



_num2sub(n::Integer) = n < 0 ? '₋' * _num2sub(-n) :
                       n < 10 ? string('₀' + n) :
                       _num2sub(n ÷ 10) * _num2sub(n % 10)

@inline _make_factor_names(name::Sym, size::Integer)::Vector{Symbol} = _make_factor_names(name, Val(Int(size)))
@inline _make_factor_names(name::Sym, ::Val{1})::Vector{Symbol} = Symbol[Symbol(name)]
@inline _make_factor_names(name::Sym, ::Val{N}) where {N} = Symbol[Symbol(name, _num2sub(i)) for i = 1:N]

@inline _tosymvec(s::Symbol)::Vector{Symbol} = Symbol[s]
@inline _tosymvec(s::Sym)::Vector{Symbol} = Symbol[Symbol(s)]
@inline _tosymvec(v::Vector{Symbol})::Vector{Symbol} = v
@inline _tosymvec(v::LikeVec{Symbol})::Vector{Symbol} = collect(v)
@inline _tosymvec(v::Vector{<:Sym})::Vector{Symbol} = Symbol[Symbol(s) for s in v]
@inline _tosymvec(v::SymVec)::Vector{Symbol} = Symbol[Symbol(s) for s in v]

@inline _make_shock(name::Sym) = to_shock(Symbol(name, "_shk"))
@inline _make_shocks(names) = ModelVariable[_make_shock(name) for name in names]

@inline _enumerate_vars(vars) = (; (Symbol(v) => n for (n, v) = enumerate(vars))...)

_do_wrap(::Nothing, AXES...) = nothing
_do_wrap(X::AbstractArray, AXES...) = ComponentArray(X, AXES...)
_do_wrap(X, AXES...) = error("Unable to wrap $(nameof(typeof(X))) in a ComponentArray")

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
    CJ = _do_wrap(reshape(J, ne, nt, nv), A1(), A2(), A3())
    Cpoint = _do_wrap(point, A2(), A3())

    return CR, CJ, Cpoint
end

@inline ComponentArrays.toval(v::ModelVariable) = ComponentArrays.toval(Symbol(v))
@inline ComponentArrays.toval(tv::NTuple{N,ModelVariable}) where {N} = ComponentArrays.toval(Symbol.(tv))
@inline ComponentArrays.toval(av::AbstractArray{<:ModelVariable}) = ComponentArrays.toval(Symbol.(av))

