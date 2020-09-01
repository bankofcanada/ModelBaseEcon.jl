
export ModelSymbol

struct ModelSymbol
    doc::String
    name::Symbol
end

# A symbol is a ModelSymbol with no description
ModelSymbol(sym::Symbol) = ModelSymbol("", sym)

# An attempt to make the parsing of docstring expression future proof.
find_macrocalls(any) = []
find_macrocalls(expr::Expr) = expr.head == :macrocall ? [expr] : vcat([], [find_macrocalls(a) for a in expr.args]...)
const docvar = find_macrocalls(quote
    "docstring"
    varname
end)[1]

# An expression is a ModelSymbol only if it is a docstring for a symbol.
function ModelSymbol(expr::Expr)
    if expr.head == :macrocall && expr.args[1] == docvar.args[1]
        return ModelSymbol(expr.args[3], expr.args[4])
    end
    if expr.head == :block
        args = filter(a -> !isa(a, LineNumberNode), expr.args)
        if length(args) == 1
            return ModelSymbol(args[1])
        end
    end
    throw(ArgumentError("Invalid variable declaration $(expr)."))
end

Base.convert(::Type{Symbol}, v::ModelSymbol) = v.name
Base.convert(::Type{ModelSymbol}, v::Symbol) = ModelSymbol(v)
Base.convert(::Type{ModelSymbol}, v::Expr) = ModelSymbol(v)
Base.:(==)(a::ModelSymbol, b::ModelSymbol) = a.name == b.name
Base.:(==)(a::ModelSymbol, b::Symbol) = a.name == b
Base.:(==)(a::Symbol, b::ModelSymbol) = a == b.name
Base.hash(v::ModelSymbol, h::UInt) = hash(v.name, h)


function Base.show(io::IO, v::ModelSymbol) 
    if isempty(v.doc) || get(io, :compact, false)
        print(io, v.name)
    else
        print(io, "\"", v.doc, "\" ", v.name)
    end
end


