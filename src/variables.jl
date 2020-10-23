##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020, Bank of Canada
# All rights reserved.
##################################################################################

export ModelVariable, ModelSymbol
export update

const doc_macro = MacroTools.unblock(quote
    "hello"
    world
end).args[1]


struct ModelVariable{T <: Transformation}
    doc::String
    name::Symbol
    var_type::Symbol
    index::Int
end

function ModelVariable(d, s, t)
    T = ifelse(t == :log, LogTransform, ifelse(t == :neglog, NegLogTransform, NoTransform))
    return ModelVariable{T}(d, s, t, -1)
end

# for compatibility with old code. will be removed soon.
const ModelSymbol = ModelVariable

# !!! must not update v.name.
@inline update(v::ModelVariable{T};
    doc=v.doc,
    var_type=v.var_type,
    index=v.index,
    transformation::Type{<:Transformation}=T,
) where {T <: Transformation} = ModelVariable{transformation}(string(doc), v.name, Symbol(var_type), Int(index))

@inline ModelVariable(s::Symbol) = ModelVariable("", s, :lin)
@inline ModelVariable(d::String, s::Symbol) = ModelVariable(d, s, :lin)
@inline ModelVariable(s::Symbol, t::Symbol) = ModelVariable("", s, t)

function ModelVariable(s::Expr)
    s = MacroTools.unblock(s)
    if MacroTools.isexpr(s, :macrocall) && s.args[1] == doc_macro
        return ModelVariable(s.args[3], s.args[4])
    else
        return ModelVariable("", s)
    end
end

function ModelVariable(doc::String, s::Expr)
    s = MacroTools.unblock(s)
    if MacroTools.isexpr(s, :macrocall)
        t = Symbol(String(s.args[1])[2:end])
        return ModelVariable(doc, s.args[3], t)
    else
        throw(ArgumentError("Invalid variable or shock expression $s."))
    end
end

for sym ∈ (:shock, :log, :neglog, :lin, :steady, :exog)
    to_sym = Symbol("to_", sym)
    issym = Symbol("is", sym)
    eval(quote
        @inline $(to_sym)(s::ModelVariable) = ModelVariable(s.doc, s.name, $(QuoteNode(sym)))
        @inline $(to_sym)(any) = $(to_sym)(convert(ModelVariable, any))
        export $(to_sym)
        @inline $(issym)(any) = false
        @inline $(issym)(s::ModelVariable) = s.var_type == $(QuoteNode(sym))
        export $(issym)
    end)
end


Symbol(v::ModelVariable) = v.name
Base.convert(::Type{Symbol}, v::ModelVariable) = v.name
Base.convert(::Type{ModelVariable}, v::Symbol) = ModelVariable(v)
Base.convert(::Type{ModelVariable}, v::Expr) = ModelVariable(v)
Base.:(==)(a::ModelVariable, b::ModelVariable) = a.name == b.name
Base.:(==)(a::ModelVariable, b::Symbol) = a.name == b
Base.:(==)(a::Symbol, b::ModelVariable) = a == b.name

# The hash must be the same as the hash of the symbol, so that we can use
# ModelVariable as index in a Dict with Symbol keys
Base.hash(v::ModelVariable, h::UInt) = hash(v.name, h)

function Base.show(io::IO, v::ModelVariable)
    if get(io, :compact, false)
        print(io, v.name)
    else
        doc = isempty(v.doc) ? "" : "\"$(v.doc)\" "
        type = v.var_type ∈ (:lin, :shock) ? "" : "@$(v.var_type) "
        print(io, doc, type, v.name)
    end
end

#############################################################################
# Transformations stuff

export transform, inverse_transform

@inline transformation(::ModelVariable{T}) where {T <: Transformation} = transformation(T)
@inline inverse_transformation(::ModelVariable{T}) where {T <: Transformation} = inverse_transformation(T)

# redirect to the stored transform
transform(x, m::ModelVariable) = broadcast(transformation(m), x)
inverse_transform(x, m::ModelVariable) = broadcast(inverse_transformation(m), x)

need_transform(a) = need_transform(convert(ModelVariable, a))
need_transform(::ModelVariable{T}) where {T <: Transformation} = T != NoTransform
