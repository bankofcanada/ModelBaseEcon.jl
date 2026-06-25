# ModelBaseEcon.jl release notes

## v0.8.0 — Symbolics-based core

This release replaces the internal equation-evaluation machinery with a
Symbolics.jl-based core. Model definitions, steady-state handling, and the
high-level API are preserved (legacy names are kept as thin compatibility
aliases), but the way residuals, gradients, and higher-order derivatives are
generated changed substantially. The change unlocks C-code export of equation
kernels (see the new `ModelBaseEconC.jl` sibling package) and gives faster model
builds and simulations.

### Breaking change: satellite models / `@replaceparameterlinks` removed

This is the one behavior change that is not source-compatible.

Parameter `@link`s are now **resolved eagerly** at `@initialize` — each link is
substituted down to the underlying root parameters when the model is compiled,
rather than being re-evaluated lazily at runtime. Eager resolution is what makes
the equation kernels self-contained and exportable to C; it is the load-bearing
improvement of this release.

The cost is that **runtime-resolved cross-model links are no longer possible**.
Specifically:

- **Satellite models** — a model whose parameters `@link` into a *separate*
  parent model and track that parent's parameters as they change — are no longer
  supported. The links are resolved against the parent's values at build time and
  do not follow subsequent edits to the parent.
- **`@replaceparameterlinks`** is removed. It existed to repoint a satellite's
  links at a different parent model after construction; with eager resolution
  there is nothing to repoint.

**Migration.** If you relied on a satellite tracking a parent at runtime, fold
the shared parameters into a single model, or rebuild (re-`@initialize`) the
dependent model whenever the upstream parameters change. Within a single model,
`@link` continues to work exactly as before — only *cross-model* runtime links
are affected.

### Removed internals (no migration needed for normal use)

These were implementation details, not part of the supported surface; their
behavior is preserved by the new core or guarded by the existing tests:

- The ForwardDiff derivative backend (`codegen = :forwarddiff`) and the on-disk
  code cache. Derivatives now come from a single Symbolics path; precompilation
  and `RuntimeGeneratedFunctions` replace the cache.
- The `SimpleTensors` higher-order-derivative storage, replaced by iterated
  symbolic differentiation.
- The `Transformation` type hierarchy, replaced by a variable-kind enum; `@log`
  / `@neglog` variable behavior is unchanged.
- The runtime `Parameters` container and the `@peval` / `@alias` macros. `@alias`
  is subsumed by `@link`. Parameter accessors keep their names.

### Automatic auxiliary-variable substitution removed

The automatic substitution engine (`model.substitutions = true`, `update_auxvars`,
and the internal `auxvars` / `auxeqns`) is removed. It existed only to make
equations containing a transform of a non-trivial subexpression — e.g.
`log(x[t] + x[t-1])` — tractable for the ForwardDiff backend, by rewriting them
into extra auxiliary variables and identity equations behind the scenes.

The Symbolics-based core differentiates through such expressions directly, so no
auxiliary substitution is needed: write the equation as-is and it builds and
evaluates correctly. There is no migration — model definitions are unchanged; the
extra auxiliary variables/equations simply no longer appear in the built model.
