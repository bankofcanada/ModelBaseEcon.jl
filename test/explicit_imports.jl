##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

# Import hygiene checks, https://github.com/JuliaTesting/ExplicitImports.jl
using ExplicitImports

@testset "ExplicitImports" begin
    test_explicit_imports(ModelBaseEcon;
        # crayon"..." appears only in code that prettyprint_equation builds as a string and evaluates
        no_stale_explicit_imports=(ignore=(Symbol("@crayon_str"),),),
        # use of non-public names is checked later
        all_explicit_imports_are_public=false,
        all_qualified_accesses_are_public=false,
    )
end
