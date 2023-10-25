

const LittleDictVec{K,V} = LittleDict{K,V,Vector{K},Vector{V}}

const Sym = Union{AbstractString,Symbol}
const LikeVec{T} = Union{Vector{T},NTuple{N,T} where {N},NamedTuple{NT,NTuple{N,T} where {N}} where {NT}}
const SymVec = LikeVec{<:Sym}


_num2sub(n::Integer) = n < 0 ? '₋' * _num2sub(-n) :
                       n < 10 ? string('₀' + n) :
                       _num2sub(n ÷ 10) * _num2sub(n % 10)

_make_factor_names(name::Sym, size::Integer)::Vector{Symbol} = _make_factor_names(name, Val(Int(size)))
_make_factor_names(name::Sym, ::Val{1})::Vector{Symbol} = Symbol[Symbol(name)]
_make_factor_names(name::Sym, ::Val{N}) where {N} = Symbol[Symbol(name, _num2sub(i)) for i = 1:N]

@inline _tosymvec(s::Symbol)::Vector{Symbol} = [s]
@inline _tosymvec(s::Sym)::Vector{Symbol} = [Symbol(s)]
@inline _tosymvec(v::Vector{Symbol})::Vector{Symbol} = v
@inline _tosymvec(v::LikeVec{Symbol})::Vector{Symbol} = collect(v)
@inline _tosymvec(v::Vector{<:Sym})::Vector{Symbol} = Symbol.(v)
@inline _tosymvec(v::SymVec)::Vector{Symbol} = collect(Symbol.(v))

