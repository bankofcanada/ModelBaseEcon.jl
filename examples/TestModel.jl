module TestModel

using ModelBaseEcon

const model = Model()

# options
model.warn.no_t = false

# flags

@parameters model begin
    a = 0.3
    b = 1 - a
    c = [1, 2, 3]
    d = sin((2π) / 3)
end # parameters

@variables model begin
    "variable x" x
end # variables

@shocks model begin
    sx
end # shocks

@equations model begin
    "This equation is super cool" 
    a * @d(x) = b * @d(x[t + 1]) + sx
end # equations

@initialize model

@steadystate model level x = a + 1

end # module TestModel

