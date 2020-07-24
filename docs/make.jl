# https://juliadocs.github.io/Documenter.jl/stable/man/guide/#Package-Guide
# push!(LOAD_PATH,"../src/") 

# Run these locally to build docs/build folder:
# PS C:\Users\akua\repos\github\ModelBaseEcon.jl> julia --color=yes --project=docs/ -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'       
# PS C:\Users\akua\repos\github\ModelBaseEcon.jl> julia --project=docs/ docs/make.jl

using Documenter, ModelBaseEcon

# Workaround for JuliaLang/julia/pull/28625
if Base.HOME_PROJECT[] !== nothing
    Base.HOME_PROJECT[] = abspath(Base.HOME_PROJECT[])
end

makedocs(sitename = "ModelBaseEcon.jl",
         format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
         modules = [ModelBaseEcon],
         doctest = false,
         pages = [
        "Home" => "index.md",
        "Examples" => "examples.md"
    ]
)

# deploydocs(
#     repo = "github.com/bankofcanada/ModelBaseEcon.jl.git",
# )