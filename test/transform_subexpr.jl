##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2025, Bank of Canada
# All rights reserved.
##################################################################################

# Guard for the removal of the automatic auxiliary-variable substitution engine
# (v0.8.0; replaces the legacy test/auxsubs.jl).
#
# The legacy engine (model.substitutions = true / update_auxvars / auxvars /
# auxeqns) existed only to make equations containing a transform of a non-trivial
# subexpression - e.g. log(x[t] + x[t-1]) - tractable for the ForwardDiff backend,
# by rewriting them into extra auxiliary variables and identity equations. The
# Symbolics core differentiates through such expressions directly, so the engine
# is gone. This test pins that behavior: the same equations the legacy auxsubs
# test used must build and evaluate correctly with NO auxiliary variables.

@testset "transform-of-subexpression equations (no aux substitution)" begin
    m = ModelDef(:auxfree)
    @variables m begin
        @log x
        lx
    end
    @shocks m s
    @equations m begin
        log(x[t]) = lx[t] + log(1.0 * s[t-1])
        log(x[t] / x[t-1]) = 1.01 + log(s[t])
        log(x[t] + x[t-1]) = 1.01 + log(s[t])
        log(x[t] * x[t-1]) = 1.01 + log(s[t])
        log(x[t] - x[t-1]) = 1.01 + log(s[t])
        log(lx[t]) - log(lx[t-1]) = log(0.0 + 1.0)
    end

    kernels, params = build_equation_kernels(m)
    # All six equations build directly - none is split into auxiliary equations.
    @test length(kernels) == 6

    # The compiled model carries no auxiliary variables/equations: the variable
    # count is exactly what was declared (no aux machinery exists anymore).
    @test length(m.vars) == 2

    # The equation log(x[t] + x[t-1]) = 1.01 + log(s[t]) is differentiated through
    # the (x[t] + x[t-1]) subexpression directly - under the legacy ForwardDiff
    # backend this required an auxiliary variable. It references x[t-1], x[t] and
    # s[t] (three slots), and the residual/gradient evaluate to finite numbers with
    # a nonzero sensitivity to both x slots (proving the chain rule ran through the
    # inner sum rather than failing or zeroing out).
    keq = kernels[3]
    @test sort([r.offset for r in keq.tsrefs]) == [-1, 0, 0]
    f = build_equation_functions(keq)
    xv = fill(2.0, f.n_x)
    J = zeros(f.n_x)
    r, = f.eval_RJ!(J, xv, Float64[])
    @test isfinite(r)
    @test all(isfinite, J)
    @test count(!iszero, J) == 3
end
