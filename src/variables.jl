
export ModelSymbol

const doc_macro = MacroTools.unblock(quote
    "hello"
    world
end).args[1]

struct ModelSymbol
    doc::String
    name::Symbol
    type::Symbol
    ModelSymbol(d::String, n::Symbol, t::Symbol) =
                t ∉ (:steady, :lin, :log, :shock) ?
                    throw(ArgumentError("Invalid symbol type $t.")) :
                    new(d, n, t)
end

ModelSymbol(s::Symbol) = ModelSymbol("", s, :lin)
ModelSymbol(d::String, s::Symbol) = ModelSymbol(d, s, :lin)
ModelSymbol(s::Symbol, t::Symbol) = ModelSymbol("", s, t)

function ModelSymbol(s::Expr)
    s = MacroTools.unblock(s)
    if MacroTools.isexpr(s, :macrocall) && s.args[1] == doc_macro
        return ModelSymbol(s.args[3], s.args[4])
    else
        return ModelSymbol("", s)
    end
end

function ModelSymbol(doc::String, s::Expr)
    s = MacroTools.unblock(s)
    if MacroTools.isexpr(s, :macrocall)
        t = Symbol(String(s.args[1])[2:end])
        return ModelSymbol(doc, s.args[3], t)
    else
        throw(ArgumentError("Invalid variable or shock expression $s."))
    end
end

for sym ∈ (:shock, :log, :lin, :steady)
    to_sym = Symbol("to_$sym")
    eval(quote
        $(to_sym)(s::ModelSymbol) = ModelSymbol(s.doc, s.name, $(QuoteNode(sym)))
        $(to_sym)(any) = $(to_sym)(convert(ModelSymbol, any))
        export $(to_sym)
    end)
end

Base.convert(::Type{Symbol}, v::ModelSymbol) = v.name
Base.convert(::Type{ModelSymbol}, v::Symbol) = ModelSymbol(v)
Base.convert(::Type{ModelSymbol}, v::Expr) = ModelSymbol(v)
Base.:(==)(a::ModelSymbol, b::ModelSymbol) = a.name == b.name
Base.:(==)(a::ModelSymbol, b::Symbol) = a.name == b
Base.:(==)(a::Symbol, b::ModelSymbol) = a == b.name

# The hash must be the same as the hash of the symbol, so that we can use
# ModelSymbol as index in a Dict with Symbol keys
Base.hash(v::ModelSymbol, h::UInt) = hash(v.name, h)

function Base.show(io::IO, v::ModelSymbol)
    if get(io, :compact, false)
        print(io, v.name)
    else
        doc = isempty(v.doc) ? "" : "\"$(v.doc)\" "
        type = v.type ∈ (:lin, :shock) ? "" : "@$(v.type) "
        print(io, doc, type, v.name)
    end
end


