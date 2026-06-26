##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

# Test entry. Three groups:
#  1. "parity"  - the Symbolics-rewrite guard suites (the <=1e-8 /
#     bit-identical-vs-reference numerical evidence).
#  2. "legacy compatibility" - the behavioral legacy claims
#     re-asserted on the public compatibility layer (compat.jl), proving
#     parity in the legacy package's own terms. The internals-coupled
#     legacy testsets (model-level eval_RJ/eval_R!, the in-package SS
#     solver, Options/evaldata/ModelSymbol, the ForwardDiff backend) were
#     removed along with the machinery they exercised.
#  3. "@slope (red)" - @steadyvariables and dynamic-steady-state references,
#     not yet implemented, carried as @test_broken so they are visibly
#     pending rather than silently skipped.

using ModelBaseEcon
using Test

include("rw_parity.jl")
include("legacy_compat.jl")
include("transform_subexpr.jl")

@testset "@slope (red)" begin
    # @steadyvariables and dynamic-steady-state references (dynss) are not
    # yet implemented. Marked broken so a future implementation flips them
    # green; NOT deleted (they track a real, planned capability).
    @test_broken isdefined(ModelBaseEcon, Symbol("@steadyvariables"))
    @test_broken isdefined(ModelBaseEcon, :dynss)
end

nothing
