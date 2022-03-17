##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################

export ModelVariable, ModelSymbol
export update

const doc_macro = MacroTools.unblock(quote
    "hello"
    world
end).args[1]

const variable_types = (:var, :shock, :exog)
const transformation_types = (:none, :log, :neglog)
const steadystate_types = (:const, :growth)

struct ModelVariable
    doc::String
    name::Symbol
    vr_type::Symbol   # one of :var, :shock, :exog
    tr_type::Symbol    # transformation, one of :none, :log, :neglog
    ss_type::Symbol    # behaviour as t → ∞, one of :const, :growth
    # index::Int
    ModelVariable(d, n, vt, tt, st) = begin
        vt ∈ variable_types || error("Unknown variable type $vt. Expected one of $variable_types")
        tt ∈ transformation_types || error("Unknown transformation type $tt. Expected one of $transformation_types")
        st ∈ steadystate_types || error("Unknown steady state type $st. Expected one of $steadystate_types")
        new(d, n, vt, tt, st)
    end
end

function Base.getproperty(v::ModelVariable, s::Symbol)
    if s === :var_type
        vt = getfield(v, :vr_type)
        if vt === :shock || vt === :exog
            return vt
        end
        tt = getfield(v, :tr_type)
        if tt !== :none 
            return tt
        end
        if getfield(v, :ss_type) === :const
            return :steady
        end
        return :lin
    end
    return getfield(v, s)
end

function ModelVariable(d, s, t)
    if t ∈ (:log, :neglog)
        return ModelVariable(d, s, :var, t, :growth, )
    elseif t === :steady
        return ModelVariable(d, s, :var, :none, :const, )
    elseif t == :lin
        return ModelVariable(d, s, :var, :none, :growth, )
    elseif t ∈ (:shock, :exog)
        return ModelVariable(d, s, t, :none, :growth, )
    end
    # T = ifelse(t == :log, LogTransform, ifelse(t == :neglog, NegLogTransform, NoTransform))
end

_sym2trans(s::Symbol) = _sym2trans(Val(s))
_sym2trans(::Val) = NoTransform
_sym2trans(::Val{:log}) = LogTransform
_sym2trans(::Val{:neglog}) = NegLogTransform

_trans2sym(::Type{NoTransform}) = :none
_trans2sym(::Type{LogTransform}) = :log
_trans2sym(::Type{NegLogTransform}) = :neglog

# for compatibility with old code. will be removed soon.
const ModelSymbol = ModelVariable

# !!! must not update v.name.
function update(v::ModelVariable; doc = v.doc,
    vr_type::Symbol = v.vr_type, tr_type::Symbol = v.tr_type, ss_type::Symbol = v.ss_type,
    transformation = nothing)
    if transformation !== nothing
        trsym = _trans2sym(transformation)
        if (tr_type == v.tr_type)
            # only transformation is explicitly given
            tr_type = trsym
        elseif (tr_type == trsym)
            # both given and they match
            tr_type = trsym
        else
            # both given and don't match
            error("Given `transformation` is incompatible with the given `ss_type`.")
        end
    end
    ModelVariable(string(doc), v.name, vr_type, tr_type, ss_type, )
end

ModelVariable(s::Symbol) = ModelVariable("", s, :var, :none, :growth,)
ModelVariable(d::String, s::Symbol) = ModelVariable(d, s, :var, :none, :growth,)
ModelVariable(s::Symbol, t::Symbol) = ModelVariable("", s, t)

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

to_shock(v) = update(convert(ModelVariable, v); vr_type = :shock)
to_exog(v) = update(convert(ModelVariable, v); vr_type = :exog)
to_steady(v) = update(convert(ModelVariable, v); ss_type = :const)
to_lin(v) = update(convert(ModelVariable, v); tr_type = :none)
to_log(v) = update(convert(ModelVariable, v); tr_type = :log)
to_neglog(v) = update(convert(ModelVariable, v); tr_type = :neglog)
isshock(v) = v.vr_type == :shock
isexog(v) = v.vr_type == :exog
issteady(v) = v.ss_type == :const
islin(v) = v.tr_type == :none
islog(v) = v.tr_type == :log
isneglog(v) = v.tr_type == :neglog
export to_shock, to_exog, to_steady, to_lin, to_log, to_neglog
export isshock, isexog, issteady, islin, islog, isneglog

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
Base.hash(v::ModelVariable) = hash(v.name)

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

transformation(v::ModelVariable) = transformation(_sym2trans(v.tr_type))
inverse_transformation(v::ModelVariable) = inverse_transformation(_sym2trans(v.tr_type))

# redirect to the stored transform
transform(x, m::ModelVariable) = broadcast(transformation(m), x)
inverse_transform(x, m::ModelVariable) = broadcast(inverse_transformation(m), x)

need_transform(a) = need_transform(convert(ModelVariable, a))
need_transform(v::ModelVariable) = _sym2trans(v.tr_type) != NoTransform
