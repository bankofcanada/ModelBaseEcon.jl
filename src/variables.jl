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

"""
    struct ModelVariable ⋯ end

Data type for model variables. `ModelVariable` functions like a `Symbol` in many
respects, but also holds meta-information about the variable, such as doc
string, the variable type, transformation, steady state behaviour.

Variable types include
 * `:var` - a regular variable is endogenous by default, but can be exogenized. 
 * `:shock` - a shock variable is exogenous by default, but can be endogenized.
   Steady state is 0.
 * `:exog` - an exogenous variable is always exogenous. 

 These can be declared with [`@variables`](@ref), [`@shocks`](@ref), and
 [`@exogenous`](@ref) blocks. You can also use `@exog` within an
 `@variables` block to declare an exogenous variable.

 Transformations include
 * `:none` - no transformation. This is the default. In steady state these
   variables exhibit linear growth.
 * `:log` - logarithm. This is useful for variables that must be always strictly
   positive. Internally the solver work with the logarithm of the variable. in
   steady state these variables exhibit exponential growth (the log variable
   grows linearly).
* `:neglog` - same as `:log` but for variables that are strictly negative. 

These can be declared with [`@logvariables`](@ref), [`@neglogvariables`](@ref),
`@log`, `@neglog`.

Steady state behaviours include
* `:const` - these variables have zero slope in steady state and final
  conditions.
* `:growth` - these variables have constant slope in steady state and final
  conditions. The meaning of "slope" changes depending on the transformation.
  For `:log` and `:neglog` variables this is the growth rate, while for `:none`
  variables it is the usual slope of linear growth. 

Shock variables are always `:const` while regular variables are assumed
`:growth`. They can be declared `:const` using `@steady`.
 
"""
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
            error("Given `transformation` is incompatible with the given `tr_type`.")
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

"""
    to_shock(v)

Make a shock `ModelVariable` from `v`.
"""
to_shock(v) = update(convert(ModelVariable, v); vr_type = :shock)
"""
    to_exog(v)

Make an exogenous `ModelVariable` from `v`.
"""
to_exog(v) = update(convert(ModelVariable, v); vr_type = :exog)
"""
    to_steady(v)

Make a zero-slope `ModelVariable` from `v`.
"""
to_steady(v) = update(convert(ModelVariable, v); ss_type = :const)
"""
    to_lin(v)

Make a no-transformation `ModelVariable` from `v`.
"""
to_lin(v) = update(convert(ModelVariable, v); tr_type = :none)
"""
    to_log(v)

Make a log-transformation `ModelVariable` from `v`.
"""
to_log(v) = update(convert(ModelVariable, v); tr_type = :log)
"""
    to_neglog(v)

Make a negative-log-transformation `ModelVariable` from `v`.
"""
to_neglog(v) = update(convert(ModelVariable, v); tr_type = :neglog)
"""
    isshock(v)

Return `true` if the given `ModelVariable` is a shock, otherwise return `false`.
"""
isshock(v::ModelVariable) = v.vr_type == :shock
"""
    isexog(v)

Return `true` if the given `ModelVariable` is exogenous, otherwise return
`false`.
"""
isexog(v::ModelVariable) = v.vr_type == :exog
"""
    issteady(v)

Return `true` if the given `ModelVariable` is zero-slope, otherwise return
`false`.
"""
issteady(v::ModelVariable) = v.ss_type == :const
"""
    islin(v)

Return `true` if the given `ModelVariable` is a no-transformation variable,
otherwise return `false`.
"""
islin(v::ModelVariable) = v.tr_type == :none
"""
    islog(v)

Return `true` if the given `ModelVariable` is a log-transformation variable,
otherwise return `false`.
"""
islog(v::ModelVariable) = v.tr_type == :log
"""
    isneglog(v)

Return `true` if the given `ModelVariable` is a negative-log-transformation
variable, otherwise return `false`.
"""
isneglog(v::ModelVariable) = v.tr_type == :neglog
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

"""
    transform(x, m::ModelVariable)

Apply the transformation associated with model variable `m` to data `x`.

See also [`transformation`](@ref).
"""
function transform end
export transform

"""
    inverse_transform(x, m::ModelVariable)

Apply the inverse transformation associated with model variable `m` to data `x`.

See also [`inverse_transformation`](@ref)
"""
function inverse_transform end
export inverse_transform

transformation(v::ModelVariable) = transformation(_sym2trans(v.tr_type))
inverse_transformation(v::ModelVariable) = inverse_transformation(_sym2trans(v.tr_type))

# redirect to the stored transform
transform(x, m::ModelVariable) = broadcast(transformation(m), x)
inverse_transform(x, m::ModelVariable) = broadcast(inverse_transformation(m), x)

"""
    need_transform(v)

Return `true` if there is a transformation associated with model variable `v`,
otherwise return `false`.
"""
function need_transform end
export need_transform

need_transform(a) = need_transform(convert(ModelVariable, a))
need_transform(v::ModelVariable) = _sym2trans(v.tr_type) != NoTransform
