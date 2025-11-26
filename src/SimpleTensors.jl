module SimpleTensors

"""
Module `SimpleTensors` provides a simple implementation of **square** tensors,
that is, tensors where the sizes in all dimensions are equal.

It provides dense and sparse storage.

It also handles efficiently the case of full symmetry, meaning tensors that are
invariant to permutations of dimensions.

N.B. not all tensor functionality is available, the main focus is storage.
"""
SimpleTensors


export Tensor, DenseTensor, SparseTensor
export SymmetricTensor, DenseSymmetricTensor, SparseSymmetricTensor
export SymmetricIndices
export TaylorPolyFunc, DerivsContainer, derivs_container, degreeof, nvars
export gettheta, settheta!, numtheta, d_dtheta, d_dtheta!

"""
There is no fundamental limit on N in the code. However, the code in this
module has been written with the assumption that N is small.
"""
const MAX_N::Int = 5

using SparseArrays
using LinearAlgebra
using OrderedCollections

import ..LittleDictVec

abstract type AbstractTensor{T,N} <: AbstractArray{T,N} end

Base.size(A::AbstractTensor) = ((A.dim for _ in 1:ndims(A))...,)

Base.@propagate_inbounds function Base.getindex(A::AbstractTensor, I::Integer...)
    @boundscheck all(isone, I[ndims(A)+1:end]) || throw(BoundsError(A, I))
    getindex(A.data, idx_N2O(A, I[1:ndims(A)]))
end

Base.@propagate_inbounds function Base.setindex!(A::AbstractTensor, val, I::Integer...)
    @boundscheck all(isone, I[ndims(A)+1:end]) || throw(BoundsError(A, I))
    setindex!(A.data, val, idx_N2O(A, I[1:ndims(A)]))
end

"convert a tuple of indexes to a linear index; NB: our tensors are alwauys \"square\"."
idx_tup2lin(::Val{0}, dim::Integer, I::NTuple{0}) = 1
idx_tup2lin(::Val{1}, dim::Integer, I::NTuple{1}) = I[1]
idx_tup2lin(::Val{N}, dim::Integer, I::NTuple{N}) where N = I[1] + dim * (idx_tup2lin(Val(N - 1), dim, Base.tail(I)) - 1)
idx_tup2lin(v::Val{N}, dim::Integer, idx::CartesianIndex) where N = idx_tup2lin(v, dim, Tuple(idx))

"""
    n_store(::Type{<:AbstractTensor}, dim::Integer)

Return the number of elements of the tensor that need to be stored. Default
implementation returns `dim^N`. Can be overloaded for tensor types with
specialized storage scheme.
"""
@inline n_store(x::AbstractTensor, dim::Integer=_dim(x)) = dim^ndims(x)
@inline n_store(X::Type{<:AbstractTensor}, dim::Integer) = dim^ndims(X)

@generated function _dim(A::AbstractArray)
    if A <: AbstractTensor && hasfield(A, :dim)
        return :(getfield(A, :dim))
    elseif A <: AbstractTensor
        return :(size(A, 1))
    else # for general array we check that all sizes are equal
        return quote
            all(size(A) .== size(A, 1)) || error("Array must be square.")
            size(A, 1)
        end
    end
end

"""
    idx_N2O(x::AbstractTensor, I::NTuple)

A generic function that takes a tuple of indexes and returns the index where
that element is stored in the data vector. Default implementation simply
converts Cartesian to Linear index. Can be overloaded for tensor types with
specialized storage scheme.
"""
Base.@propagate_inbounds idx_N2O(A::AbstractTensor, I::NTuple) = idx_tup2lin(Val(ndims(A)), _dim(A), I)

################################

struct Tensor{T,N,D<:AbstractVector{T}} <: AbstractTensor{T,N}
    dim::Int
    data::D
    Tensor{T,N}(dim::Int, data::D) where {T,N,D} = begin
        @assert (N isa Integer) && (0 <= N <= MAX_N)
        @assert length(data) == n_store(Tensor{T,N}, dim)
        return new{T,N,D}(dim, data)
    end
end
Tensor(A::AbstractArray{T,N}, ::Val{:sparse}=Val(:sparse)) where {T,N} = Tensor{T,N}(_dim(A), sparsevec(A))
Tensor(A::AbstractArray{T,N}, ::Val{:dense}) where {T,N} = Tensor{T,N}(_dim(A), vec(A))
Tensor(N::Int, dim::Int, kind::Val=Val(:sparse)) = Tensor{Float64,N}(dim, kind)
Tensor(T::Type, N::Int, dim::Int, kind::Val=Val(:sparse)) = Tensor{T,N}(dim, kind)
Tensor{T}(N::Int, dim::Int, kind::Val=Val(:sparse)) where T = Tensor{T,N}(dim, kind)
Tensor{T,N}(dim::Int, ::Val{:sparse}=Val(:sparse)) where {T,N} = Tensor{T,N}(dim, SparseVector{T,Int}(n_store(Tensor{T,N}, dim), Int[], T[]))
Tensor{T,N}(dim::Int, ::Val{:dense}) where {T,N} = Tensor{T,N}(dim, zeros(T, n_store(Tensor{T,N}, dim)))

const DenseTensor{T,N} = Tensor{T,N,Vector{T}}
DenseTensor(args...) = Tensor(args..., Val(:dense))

const SparseTensor{T,N} = Tensor{T,N,SparseVector{T,Int}}
SparseTensor(args...) = Tensor(args..., Val(:sparse))

################################

abstract type AbstractSymmetricTensor{T,N} <: AbstractTensor{T,N} end

# N.B. disabling broadcasting for symmetric tensors because things go wrong with
# the default Julia machinery due to symmetric storage. TODO: fix it, rather than disable

# struct SymmetricTensorStyle <: Broadcast.BroadcastStyle end
# Base.BroadcastStyle(::Type{<:AbstractSymmetricTensor}) = SymmetricTensorStyle()
Base.BroadcastStyle(::Type{<:AbstractSymmetricTensor}) = error("Broadcasting not allowed")


"Method for symmetric tensors, where only the unique elements are stored."
@inline n_store(T::Type{<:AbstractSymmetricTensor}, dim::Int) = _n_stored_sym(Val(ndims(T)), Val(dim))
@inline n_store(x::AbstractSymmetricTensor, dim::Integer=_dim(x)) = _n_stored_sym(Val(ndims(x)), Val(dim))
# Let the compiler build a look-up table for these values and hardcode them
@generated _n_stored_sym(::Val{N}, ::Val{dim}) where {N,dim} = binomial(dim + N - 1, N)

sort_ntuple(::Tuple{}) = ()
sort_ntuple(x::Tuple{Int}) = x
sort_ntuple(x::Tuple{Int,Int}) = x[1] > x[2] ? x : (x[2], x[1])
function sort_ntuple(x::NTuple{N,Int}) where N
    K = div(N, 2)
    a = sort_ntuple(x[begin:K])
    b = sort_ntuple(x[K+1:end])
    return _merge_sorted(a, b)
end
_merge_sorted(::Tuple{}, ::Tuple{}) = ()
_merge_sorted(::Tuple{}, x::NTuple) = x
_merge_sorted(x::NTuple, ::NTuple{0}) = x
_merge_sorted(a::NTuple, b::NTuple) = a[1] > b[1] ? (a[1], _merge_sorted(Base.tail(a), b)...) : (b[1], _merge_sorted(a, Base.tail(b))...)

_is_sorted_idx(::Tuple{}) = true
_is_sorted_idx(::Tuple{Int}) = true
_is_sorted_idx(idx::Tuple{Int,Int}) = idx[1] >= idx[2]
_is_sorted_idx(idx::NTuple{N,Int}) where N = idx[1] >= idx[2] && _is_sorted_idx(Base.tail(idx))

"Method for symmetric tensors where only the unique elements are stored."
Base.Base.@propagate_inbounds function idx_N2O(A::AbstractSymmetricTensor, I::NTuple)
    sortedI = sort_ntuple(I)
    return idx_t2l_sym(Val(ndims(A)), _dim(A), sortedI)
end

# Idea.
# We store only the elements on or below the main diagonal (N=2 case) or
# where the N-tuple of indexes is sorted in descending order.
# The algorithm below is recursive. The recursion goes along N based on
# the last index.
# If the last index (call it k) is 1, then the indexing is the same as in the
# case N-1 with the first N-1 indices.
# If the last index is k=2, then we have to add the number of elements we had
# from the k=1 hyper-plane (which is n_store(N-1,dim)) plus the indexing in the
# current (k=2) hyper-plane, which is the case N-1 with dim-1 and the first
# N-1 indices reduced by 1. This is because all of the first N-1 indexes of
# stored elements start from 2, rather than 1, so if we subtract 1 we have normal
# indexing.
# And so on.
# In the general case of k>2 we have to add up all elements stored in the first
# k-1 planes, plus the indexing in the current plane.
# So we have the elements from the k=1 plane (which are n_store(N-1, dim))
# plus all elements for k=2 plane, which are n_store(N-1,dim-1),
# plus, and so on, all the way up to all elements for k-1-plane, which are
# n_store(N-1,dim-(k-2)), plus the indexing in the k-plane, which is the same
# as for the case N-1 with dim-(k-1) and using the first N-1 indices reduced by
# k-1 (because stored elements in the k-plane have their first N-1 indexes
# running from k to dim, so we translate it to 1 to dim-(k-1)).
# Voilà

idx_t2l_sym(v::Val, dim::Int, I::CartesianIndex) = idx_t2l_sym(v, dim, Tuple(I))
idx_t2l_sym(::Val{0}, dim::Int, ::Tuple{}) = 1
idx_t2l_sym(::Val{1}, dim::Int, idx::Tuple{Int}) = idx[1]
function idx_t2l_sym(::Val{N}, dim::Int, idx::NTuple{N,Int}) where N
    if _is_alldim(dim, idx)
        # this case is called repeatedly.
        # luckily we have a direct formula for it, no need for recursion
        return _n_stored_sym(Val(N), Val(dim))
    else
        # split off the last index and call _impl
        return _idx_t2l_sym_impl(dim, Val(last(idx)), Base.front(idx))
    end
end
_is_alldim(dim::Int, ind::Tuple{Int}) = (ind[1] == dim)
_is_alldim(dim::Int, ind::NTuple) = ((ind[1] == dim) && _is_alldim(dim, Base.tail(ind)))
@generated function _idx_t2l_sym_impl(dim::Int, ::Val{k}, idx1::NTuple{N,Int}) where {N,k}
    # N.B. The N here is actually N-1, since the last index, k, has been separated
    #      so using Val($N) is actually a recursive call to the N-1 case
    if k == 1
        return :(idx_t2l_sym(Val($N), dim, idx1))
    end
    if k == 2
        return :(idx_t2l_sym(Val($N), dim - 1, idx1 .- 1) + _n_stored_sym(Val($N), Val(dim)))
    end
    @assert k > 2
    ret = :(idx_t2l_sym(Val($N), dim - $(k - 1), idx1 .- $(k - 1)))
    for s = 0:k-2
        ret = :($ret + _n_stored_sym(Val($N), Val(dim - $s)))
    end
    return ret
end

next_sym_idx(dim::Int) = error()
next_sym_idx(dim::Int, i::Int) = (i + 1,)
function next_sym_idx(dim::Int, i::Int, J::Int...)
    i < dim && return (i + 1, J...)
    # Logic.
    # i == dim means that i+1 "overflows".
    # So, we move up I without i and set i to I[1] (to maintain it being sorted)
    J = next_sym_idx(dim, J...)
    return (J[1], J...)
end

struct SymmetricIndices{N,dim}
    transform
end
SymmetricIndices{N,dim}() where {N,dim} = SymmetricIndices{N,dim}(identity)
SymmetricIndices(A::AbstractArray, transform=identity) = SymmetricIndices{ndims(A),_dim(A)}(transform)
function Base.iterate(x::SymmetricIndices{N,dim}, state::NTuple{N,Int}=ntuple(one, N)) where {N,dim}
    first(state) > dim && return nothing
    return (x.transform(state), next_sym_idx(dim, state...))
end
Base.iterate(::SymmetricIndices{0}, ::Tuple{}=()) = ((), nothing)
Base.iterate(::SymmetricIndices{0}, ::Nothing) = nothing
Base.length(x::SymmetricIndices{N,dim}) where {N,dim} = _n_stored_sym(Val(N), Val(dim))

################################

struct SymmetricTensor{T,N,D<:AbstractVector{T}} <: AbstractSymmetricTensor{T,N}
    dim::Int
    data::D
    SymmetricTensor{T,N}(dim::Int, data::D) where {T,N,D} = begin
        @assert (N isa Integer) && (0 <= N <= MAX_N)
        @assert length(data) == n_store(SymmetricTensor{T,N}, dim)
        return new{T,N,D}(dim, data)
    end
end


# This function helps instantiate symmetric tensors (invariant to permutations
# of its indices). It takes a full Array that is symmetric and extracts the
# unique elements from it into the given data storage vector. As it goes, it
# also checks to make sure it is indeed symmetric.
function _take_sym!(::Val{0.0}, data::AbstractVector, A::Array, dim::Int=_dim(A))
    # the case of no check
    @assert length(data) == n_store(AbstractSymmetricTensor{eltype(A),ndims(A)}, dim)
    @inbounds for (i, I) in enumerate(SymmetricIndices(A, CartesianIndex))
        data[i] = A[I]
    end
    return data
end
function _take_sym!(v::Val{tol}, data::AbstractVector, A::Array, dim::Int=_dim(A)) where tol
    @nospecialize(v)
    @assert length(data) == n_store(AbstractSymmetricTensor{eltype(A),ndims(A)}, dim)
    @inbounds for I in CartesianIndices(A)
        if _is_sorted_idx(Tuple(I))
            idx = idx_t2l_sym(Val(ndims(A)), dim, I)
            data[idx] = A[I]
        else
            idx = idx_t2l_sym(Val(ndims(A)), dim, sort_ntuple(Tuple(I)))
            if abs(data[idx] - A[I]) > tol
                error("Not symmetric: data[$idx] = $(data[idx]), A[$(Tuple(I)...)] = $(A[I])")
            end
        end
    end
    return data
end

function SymmetricTensor(A::AbstractArray, kind::Val=Val(:sparse);
    check::Bool=false, tol::AbstractFloat=eps(Float64) * 1e3)
    dim = _dim(A)
    x = SymmetricTensor{eltype(A),ndims(A)}(dim, kind)
    _take_sym!(check ? Val(tol) : Val(0.0), x.data, A, dim)
    return x
end

SymmetricTensor(N::Int, dim::Int, kind::Val=Val(:sparse)) = SymmetricTensor{Float64,N}(dim, kind)
SymmetricTensor(T::Type, N::Int, dim::Int, kind::Val=Val(:sparse)) = SymmetricTensor{T,N}(dim, kind)
SymmetricTensor{T}(N::Int, dim::Int, kind::Val=Val(:sparse)) where T = SymmetricTensor{T,N}(dim, kind)
SymmetricTensor{T,N}(dim::Int, ::Val{:sparse}=Val(:sparse)) where {T,N} = SymmetricTensor{T,N}(dim, SparseVector{T,Int}(n_store(SymmetricTensor{T,N}, dim), Int[], T[]))
SymmetricTensor{T,N}(dim::Int, ::Val{:dense}) where {T,N} = SymmetricTensor{T,N}(dim, zeros(T, n_store(SymmetricTensor{T,N}, dim)))

const DenseSymmetricTensor{T,N} = SymmetricTensor{T,N,Vector{T}}
DenseSymmetricTensor(args...) = SymmetricTensor(args..., Val(:dense))

const SparseSymmetricTensor{T,N} = SymmetricTensor{T,N,SparseVector{T,Int}}
SparseSymmetricTensor(args...) = SymmetricTensor(args..., Val(:sparse))


############################################

const DerivsContainer{T} = LittleDict{Int,SymmetricTensor{T},UnitRange{Int},Vector{SymmetricTensor{T}}}
derivs_container(T::Type, D::Integer, nvars::Integer, kind::Symbol) = LittleDict{Int,SymmetricTensor{T}}(0:D, SymmetricTensor{T}[SymmetricTensor(T, N, nvars, Val(kind)) for N = 0:D])

# Function defined as a Taylor polynomial of degree `D` about a point (`x̄`)
struct TaylorPolyFunc{D,T} <: Function
    x̄::Vector{T}
    derivs::DerivsContainer{T}
    function TaylorPolyFunc{D,F}(nvars::Int) where {D,F}
        @assert 0 <= D <= MAX_N "Maximum degree supported is $MAX_N."
        new{D,F}(zeros(F, nvars), derivs_container(F, D, nvars, :dense))
    end
end
TaylorPolyFunc(D::Integer, dim::Integer) = TaylorPolyFunc{D,Float64}(Int(dim))
TaylorPolyFunc{D}(dim::Integer) where D = TaylorPolyFunc{D,Float64}(Int(dim))
degreeof(f::TaylorPolyFunc{D}) where D = D
degreeof(::Type{<:TaylorPolyFunc{D}}) where D = D
nvars(f::TaylorPolyFunc) = length(f.x̄)

numtheta(x::TaylorPolyFunc) = sum(n_store, values(x.derivs))
gettheta(x::TaylorPolyFunc{D,T}) where {D,T} = gettheta!(Vector{T}(undef, numtheta(x)), x, 1)
function gettheta!(θ::AbstractVector, x::TaylorPolyFunc{D,T}, offset::Int=1) where {D,T}
    for der in values(x.derivs)
        n = length(der.data)
        copyto!(θ, offset, der.data, 1, n)
        offset = offset + n
    end
    return θ
end
function settheta!(x::TaylorPolyFunc, θ::AbstractVector, offset::Int=1)
    for der in values(x.derivs)
        n = length(der.data)
        copyto!(der.data, 1, θ, offset, n)
        offset = offset + n
    end
    return θ
end


function Base.show(io::IO, ::MIME"text/plain", f::TaylorPolyFunc{D}) where {D}
    println(io, nameof(typeof(f)), " of degree ", D)
    print(io, "x̄ = ", f.x̄)
    print(io, "\nD0 = ", f.derivs[0][])
    D > 0 && print(io, "\nD1 = ", f.derivs[1][:])
    for (d, dd) in f.derivs
        d < 2 && continue
        print(io, "\nD$d = ", iszero(dd.data) ? "zero" : dd)
        D == d && return
    end
    # print(io, "\nD1 = ", f.derivs[1][:])
    # D == 1 && return
    # print(io, "\nD2 = ", iszero(f.derivs[2]) ? "zero" : f.derivs[2][:,:])
    # D == 2 && return
    # print(io, "\n...")
end

(f::TaylorPolyFunc)(x::Number...) = f([x...,])
@generated function (f::TaylorPolyFunc{D,T})(x::AbstractVector{S}, ::Val{deriv}=Val(0)) where {S,T,D,deriv}
    ST = promote_type(S, T)
    if deriv > D
        return :(SymmetricTensor{$ST,$deriv}(length(x), Val(:sparse)))
    end
    ret = quote
        pt = iszero(f.x̄) ? x : x - f.x̄
        der = f.derivs
        result = SymmetricTensor{$ST,$deriv}(length(x), Val(:sparse))
    end
    for N = deriv:D
        # push!(ret.args, :(add_degree!(result, Val($(N-deriv)), der[$N], pt)))
        push!(ret.args, :(add_degree!(result, der[$N], pt)))
    end
    push!(ret.args, :(return result))
    return ret
end

function eval_hod(f::TaylorPolyFunc{D,T}, x::AbstractVector{S}) where {D,T,S}
    TS = promote_type(T, S)
    result = derivs_container(TS, D, nvars(f), :dense)
    eval_hod!(result, f, x)
end

function eval_hod!(result::DerivsContainer, f::TaylorPolyFunc, x::AbstractVector)
    pt = iszero(f.x̄) ? x : x - f.x̄
    for i in keys(result)
        res = result[i]
        fill!(res.data, zero(eltype(res)))
        for j = i:degreeof(f)
            add_degree!(res, f.derivs[j], pt)
        end
    end
    return result
end


# multinomial coefficients formula using formula based on binomial coefficients
# cf. https://en.wikipedia.org/wiki/Multinomial_theorem#Multinomial_coefficients
function _multinom_coeff(deg::AbstractVector{T}, der::AbstractVector{S}=T[]) where {T,S}
    # deg is a vector of integer powers of the mulinomial term we're constructing
    # der is a vector of integer powers of the derivative we're taking of this term
    TS = promote_type(T, S)
    result = one(TS)
    s = zero(Int)
    if iszero(der)
        for ind = eachindex(deg)
            @inbounds k = deg[ind]
            k < 0 && return zero(TS)
            s += k
            result *= binomial(s, k)
        end
    else
        @assert axes(deg) == axes(der)
        for ind = eachindex(deg)
            @inbounds k = deg[ind] - der[ind]
            k < 0 && return zero(TS)
            s += k
            result *= binomial(s, k)
        end
    end
    return result
end


function _multinom_pow(pt::AbstractVector{R}, deg::AbstractVector{T}, der::AbstractVector{S}=T[]) where {R,T,S}
    # deg is a vector of integer powers of the multinomial term we're constructing
    # der is a vector of integer powers of the derivative we're taking of this term
    result = one(R)
    @assert axes(pt) == axes(deg)
    if iszero(der)
        for ind in eachindex(pt)
            @inbounds k = deg[ind]
            k < 0 && return zero(R)
            k > 0 && (result *= @inbounds pt[ind]^k)
        end
    else
        @assert axes(pt) == axes(der)
        for ind in eachindex(pt)
            @inbounds k = deg[ind] - der[ind]
            k < 0 && return zero(R)
            k > 0 && (result *= @inbounds pt[ind]^k)
        end
    end
    return result
end

count_degrees(::Val{dim}, I::NTuple{N,Int}) where {dim,N} = count_degrees!(zeros(Int, dim), I)
count_degrees!(x::Vector{Int}, ::Tuple{}) = x
function count_degrees!(x::Vector{Int}, I::NTuple{N,Int}) where N
    i, rest... = I
    x[i] += 1
    count_degrees!(x, rest)
end

# return the coefficient count times the power for the given monomial
function _coeff_pow(pt::AbstractVector, deg_idx::NTuple{N,Int}, der_idx::NTuple{d,Int}=()) where {N,d}
    # deg_idx -- index of the monomial, i.e. (1,1,2) means x*x*y
    # der_idx -- index of derivative we are taking, e.g., (1,2) means second mixed derivative d^2/dxdy
    dim = length(pt)
    x = count_degrees(Val(dim), deg_idx)
    d == 0 && return _multinom_coeff(x) * _multinom_pow(pt, x)
    y = count_degrees(Val(dim), der_idx)
    return _multinom_coeff(x, y) * _multinom_pow(pt, x, y)
end


"""
    result = add_degree!(result, deriv, pt)

Accumulate into result the contribution to a derivative of
a poly-function from its derivative.

    result::AbstractSymmetricTensor{T,d}
    deriv::AbstractSymmetricTensor{T,N}
    pt::Vector

`result` is a symmetric tensor that accumulates the d-th derivative
of a poly-function at point `x`, such that `pt = x-x̄`. `deriv` contains
the `N`-th derivative of the poly-function at `x̄`.

Note that we assert `N >= d`.

"""
function add_degree! end

function add_degree!(result::AbstractSymmetricTensor{T1,d},
    deriv::AbstractSymmetricTensor{T2,d}, ::Vector) where {T1,T2,d}
    result.data .+= deriv.data
    return result
end

function add_degree!(result::AbstractSymmetricTensor{T1,d},
    deriv::AbstractSymmetricTensor{T2,N}, pt::Vector) where {d,T1,T2,N}
    @assert d <= N
    dim = length(pt)
    @assert dim == deriv.dim == result.dim
    coeff1 = (N - d) < 2 ? one(T1) : one(T1) / prod(2:(N-d))
    for (i, idx) in enumerate(SymmetricIndices{N,dim}())
        dval = deriv.data[i]
        iszero(dval) && continue
        for (j, jdx) in enumerate(SymmetricIndices{d,dim}())
            coeff2 = _coeff_pow(pt, idx, jdx)
            iszero(coeff2) && continue
            result.data[j] += dval * coeff1 * coeff2
        end
    end
    return result
end

"""
    d_dtheta(f, x, Val(d))

This function computes the derivatives of a PolyFunc 'f' with respect to its
parameters (called θ here) at a given point `x`

The value of `d` determines which x-derivative of `f` is considered here. That
is, for `d=0` we compute the θ-gradient of `f` itself, if `d=1` we compute the
θ-Jacobian of the x-gradient of `f`, if `d=2` we compute the θ-Jacobian of the
unique elements of the x-Hessian of `f` and so on.

Return a M-by-N matrix where `M = n_store(f.derivs[d])` is the number of unique
mixed derivatives of f of order `d` and `N = numtheta(f)` is the number of
parameters in `f`.

Note that `f` and all of its x-derivatives depend linearly on θ, so higher
derivatives w.r.t. θ are zero.

"""
function d_dtheta(f::TaylorPolyFunc{D,T}, x::AbstractVector{S}, ::Val{DX}=Val(0)) where {D,DX,T,S}
    TS = promote_type(T, S)
    # result: axis 1 is the derivatives wrt x, axis 2 is the derivative wrt θ
    result = spzeros(TS, n_store(AbstractSymmetricTensor{TS,DX}, nvars(f)), numtheta(f))
    return d_dtheta!(result, f, x, Val(DX), 0, 0)
end

function d_dtheta!(result::AbstractMatrix{TS}, f::TaylorPolyFunc, x::AbstractVector, ::Val{DX}=Val(0), x_offset::Int=0, θ_offset::Int=0) where {TS,DX}
    dim = nvars(f)
    pt = x ≈ f.x̄ ? spzeros(TS, sizeof(x)) : iszero(f.x̄) ? x : x - f.x̄
    ind = 1
    fill!(result, zero(TS))
    for (d, deriv) in pairs(f.derivs)
        # N.B. this is the derivative of add_degree! w.r.t. dval
        coeff1 = (d - DX) < 2 ? one(TS) : one(TS) / prod(2:(d-DX))
        for idx in SymmetricIndices(deriv)
            # loop over all unique elements of deriv
            if d < DX
                # derivatives with respect to x have eliminated these
                # coefficients, which are already zero as per fill!() above
                # for j in axes(result, 1)
                #     result[x_offset+j, θ_offset+ind] = zero(TS)
                # end
            else
                for (j, jdx) in enumerate(SymmetricIndices{DX,dim}())
                    # loop over all derivative w.r.t x that we're taking
                    coeff2 = _coeff_pow(pt, idx, jdx)
                    iszero(coeff2) && continue
                    result[x_offset+j, θ_offset+ind] += coeff1 * coeff2
                end
            end
            ind = ind + 1
        end
    end
    return result
end


###################################################
end
