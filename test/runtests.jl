##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

# Dual-green test entry (PLAN_v3 Chunk M, api_and_tests §5).
#
# Three groups:
#  1. "RW parity"  — the Symbolics-rewrite guard suites ported in as-is
#     (the ≤1e-8 / bit-identical-vs-legacy-reference evidence, ledger C).
#  2. "legacy (against RW internals)" — the behavioral legacy claims
#     re-asserted on the public alias layer (compat.jl), proving parity in
#     BoC's own terms. The internals-coupled legacy testsets (model-level
#     eval_RJ/eval_R!, the in-MBE SS solver, Options/evaldata/ModelSymbol,
#     the ForwardDiff backend) were removed with their machinery (ledger D);
#     they are not re-shimmed (M-i decision #1).
#  3. "v2.3 @slope (red)" — features deferred to v2.3 (@steadyvariables /
#     dynamic-steady-state references), carried as @test_broken so they are
#     visibly pending, not silently skipped (D4).
#
# Dual-green = groups 1 and 2 both green, modulo @test_broken in group 3.

using ModelBaseEcon
using Test

include("rw_parity.jl")
include("legacy_compat.jl")
include("transform_subexpr.jl")

@testset "v2.3 @slope (red)" begin
    # @steadyvariables and dynamic-steady-state references (dynss) are
    # deferred to v2.3 (REQUIREMENTS §6 / D4). Marked broken so v2.3 flips
    # them green; NOT deleted (they track a real, planned capability).
    @test_broken isdefined(ModelBaseEcon, Symbol("@steadyvariables"))
    @test_broken isdefined(ModelBaseEcon, :dynss)
end

nothing
