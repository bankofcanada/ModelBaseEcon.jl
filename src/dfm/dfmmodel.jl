##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2022, Bank of Canada
# All rights reserved.
##################################################################################


export DFMModel, AbstractFactorBlock,
    ARFactorBlock, FactorBlock, IdiosyncraticComponents,
    factors, factorshocks, observed, observedshocks,
    add_icblock!, add_factorblock!,
    varshks



# """
# DFM Model
# 
#     # observed data follows a factor model
#     x_t = μ + Λ f_t + ε_t + η_t
#     # observation shock η_{i,t} ~ N(0, σ_i^2) i.i.d.
#     # idiosyncratic components follow AR(s) processes
#     ε_{i,t} = α_i ε_{i,t-1} + ⋯ + α_s ε_{i,t-s} + e_{i,t}
#     e_{i,t} ~ N(0, σ_i^2), i.i.d.
#     # factors follow VAR(p) process
#     f_t = A_1 f_{t-1} + ⋯ + A_p f_{t-p} + u_t
#     u_t ~ N(0, Ω)
# 
#   Each observed variable x_i must have either a shock (η_i) or an 
#   idiosyncratic component (ε_i, with shock e_i) associated with it.
# """

# The factors are organized into blocks. 
# Each observed variable should depend on at least one factor block.
# Each factor belongs to exactly one block and is uncorrelated with
# any factors in other blocks. 
# Within the same block, the factors may be correlated - this 
# will depend on the type of block.
# The transition of each factors block is decoupled from 
# the transitions of other blocks.
# We implement two types of blocks - an AR block (transition is VAR(p) process) and 
# a block containing idiosyncratic components (transition and covariance matrices are diagonal)
#

"""
    abstract type AbstractFactorBlock end

Factor blocks can be used when the transition system is in fact several
decoupled systems.

Each factor block gets its own set of AR coefficient matrices and a covariance
matrix.
"""
abstract type AbstractFactorBlock end

########################################
#  API common to all factor blocks

# The default implementations return a field by the appropriate name.
# If the concrete type doesn't have such field, we need a specific method.
"Return the number of factors in this block"
_nfactors(fb::AbstractFactorBlock) = fb.nfactors
"Return the number of state variables in this block"
_nstates(fb::AbstractFactorBlock) = error("Not implemented for $(typeof(fb))")
# Note on the difference between factors and states:
#   The state space transition equation is always S_t = F(S_{t-1}, U_t)
#   If the factor block dynamic depends on higher order lags of the factor, then
#   some of the lags will be states.
"Return the number of state shocks in this block"
_nstateshocks(fb::AbstractFactorBlock) = error("Not implemented for $(typeof(fb))")
#   The number of shocks U_t does not have to match neither the number of 
#   factors nor the number of states.  We leave tha to the model developer, just tell us what it is.
"Return the names of the observed variables affected by this block"
_observed(fb::AbstractFactorBlock) = fb.variables
"Return the number of observed variables that this block affects"
_nobserved(fb::AbstractFactorBlock) = length(_observed(fb))
# Note: Factor blocks do not assign shocks to the observed variables. 
# The observation equation is controlled in the DFMModel instance and
# observation shocks are assigned there.

"""
    block_transition(fb::AbstractFactorBlock)

Return the transition dynamic in the given block. The
specifics of what is returned will depend on the concrete type of factor block.
"""
block_transition(fb::AbstractFactorBlock) = error("Not implemented for $(typeof(fb))")

"""
    block_covariance(fb::AbstractFactorBlock)

Return the covariance matrix of the shocks in this factor block.
"""
block_covariance(fb::AbstractFactorBlock) = error("Not implemented for $(typeof(fb))")


"""
    block_observation(fb::AbstractFactorBlock)

Return the contribution of the given block to the observation equation. 
The specifics of what is returned will depend on the concrete type of factor block.
"""
block_observation(fb::AbstractFactorBlock) = error("Not implemented for $(typeof(fb))")

##############################################################################

"""
    abstract type ARFactorBlock <: AbstractFactorBlock end

A factor block in which the internal dynamic of the factors is a VAR process.
"""
abstract type ARFactorBlock <: AbstractFactorBlock end

# Factor dynamic is given by VAR(p)
#   F_t = A_1*F_{t-1} + A_2*F_{t-2} + . . . + A_p*F_{t-p} + U_t
#
# We assign state S1_t = F_t, S2_t = F_{t-1}, . . . , Sp_t = F{t-p+1}
# Then the state-space transition for this block is 
#   S1_t = A1*S1_{t-1} + A2*S2_{t-1} + . . . + A_p*Sp_{t-1} + U_t
#   S2_t = S1_{t-1}
#     . . . 
#   Sp_t = Sp-1_{t-1}
#
# or
# 
#   S_t = F*S_{t-1} + G*U_t
# where S_t collects all states, state shocks U_t are the same as factor shocks above and 
# matrices F and G are constructed by assigning relevant
# blocks in the right places according to the above formulas.


########################################
#  API common to all AR factor blocks

"Return `p` if the factor block is a VAR(p) process."
_order(fb::ARFactorBlock) = length(fb.ARcoefs)
# we have a state variable for each relevant lag of each factor
_nstates(fb::ARFactorBlock) = _nfactors(fb) * _order(fb)
# we have a shock for each facto, but no shocks for their lags
_nstateshocks(fb::ARFactorBlock) = _nfactors(fb)
"""Return a matrix containing the AR coefficient for the given `lag ∈ 1:_order(fb)`."""
_arcoefs(fb::ARFactorBlock) = fb.ARcoefs
_arcoefs(fb::ARFactorBlock, lag::Integer) = fb.ARcoefs[lag]
"Return the noise covariance matrix"
_covariance(fb::ARFactorBlock) = fb.covariance
"Return the loadings matrix for the observed variables and the factors in this block"
_loadings(fb::ARFactorBlock) = fb.loadings


"""
    block_transition(b::ARFactorBlock)

Return the matrices associated with the transition equation of the 
given block.  
``
    S_t = F * S_{t-1} + G * U_t
``
Since [`ARFactorBlock`](@ref) is linear, it is 
sufficient to return two matrices: F and G.
"""
function block_transition(b::ARFactorBlock)
    # return:
    # F = [ AR1 AR2 ... ARp-1 ARp
    #        I   0  ...   0    0
    #        0   I  ...   0    0
    #       ... ... ...  ...  ...
    #        0   0  ...   I    0]
    # G = [ I; 0; ...; 0]
    #
    #   S = [f[t], f[t-1], ..., f[t-p+1]]
    #   v = [u_t, 0,  ..., 0] 
    #   S[t] = F * S[t-1] + G * u[t]
    #
    ns = _nstates(b)
    nss = _nstateshocks(b)
    nf = _nfactors(b)
    @assert nss == nf "Inconsistent dimensions for number of factors and number of state shocks."
    F = zeros(ns, ns)
    F[1:nf, 1:nf] = _arcoefs(b, 1)
    for i = 2:_order(b)
        F[1:nf, (i-1)*nf.+(1:nf)] = _arcoefs(b, i)
        F[(i-1)*nf.+(1:nf), (i-2)*nf.+(1:nf)] .= I(nf)
    end
    G = zeros(ns, nss)
    G[1:nss, :] = I(nss)
    return F, G
end

block_covariance(b::ARFactorBlock) = _covariance(b)

""" Return the factor loadings for the given factor block """
@inline function block_observation(b::ARFactorBlock)
    H = zeros(_nobserved(b), _nstates(b))
    H[:, 1:_nfactors(b)] = _loadings(b)
    return H
end

#########################################################################################
#  Two concrete factor blocks: FactorBlock and IdiosyncraticComponents

"""
    struct FactorBlock <: ARFactorBlock ... end

A concrete type representing a typical factor block in a factor model. 
"""
struct FactorBlock <: ARFactorBlock
    name::Symbol
    # known_params::Vector{Symbol}
    nfactors::Int
    loadings::Matrix{Float64}           # Λ
    ARcoefs::Vector{Matrix{Float64}}    # A_1 ... A_p
    covariance::Matrix{Float64}         # Ω
    variables::Vector{ModelSymbol}      # names of variables participating in this factor block
    # FactorBlock(n, nf, l, a, s, v) = new(n, Symbol[], nf, l, a, s, v)
end

# all default implementations of the API should work for FactorBlock

"""
    struct IdiosyncraticComponents <: ARFactorBlock ... end

A concrete type representing a block of idiosyncratic components in a factor
model.

The block contains one component for each variable. All components in the same
block are independent AR(p) processes of the same order p. It is allowed to have
multiple IdiosyncraticComponents blocks in the same factor model, possibly with
different AR orders. Each observed variable must belong to at most one
IdiosyncraticComponents block. Variables that do not belong to any
IdiosyncraticComponents block will have an observation shock instead.
"""
struct IdiosyncraticComponents <: ARFactorBlock
    name::Symbol
    # known_params::Vector{Symbol}
    # all matrices are diagonal
    ARcoefs::Vector{Diagonal{Float64,Vector{Float64}}}    # α_1, ..., α_s
    covariance::Diagonal{Float64,Vector{Float64}}  # diag = [σ_i^2 for i = 1:length(variables)]
    variables::Vector{ModelSymbol}
    # IdiosyncraticComponents(n, a, s, v) = new(n, Symbol[], a, s, v)
end
# Constructor in case matrix-diagonals are given in Vectors
IdiosyncraticComponents(n::Symbol, arc::AbstractVector{<:AbstractVector{<:Real}},
    sig2::AbstractVector{<:Real}, vars) =
    IdiosyncraticComponents(n, [Diagonal(convert(Vector{Float64}, ar)) for ar in arc],
        Diagonal(convert(Vector{Float64}, sig2)), vars)

# one factor for each variable
_nfactors(b::IdiosyncraticComponents) = length(b.variables)
# loadings are always 1
_loadings(b::IdiosyncraticComponents) = I(length(b.variables))

#######################################################################################
# Finally, we can define our DFMModel 

struct DFMModel <: AbstractModel
    factorblocks::LittleDict{Symbol,AbstractFactorBlock}
    # known_params::Vector{Symbol}
    mean::Vector{Float64}
    covariance::Ref{Matrix{Float64}}
    variables::Vector{ModelSymbol}
    # always create an empty model and fill with stuff later
    DFMModel() = new(LittleDict{Symbol,AbstractFactorBlock}(), Float64[],
        Ref(zeros(0, 0)), ModelSymbol[])
end

_blocks(m) = values(m.factorblocks)
_get_block(m, name::Symbol) = m.factorblocks[name]

# define exported UI for DFMModel
nobserved(m) = length(observed(m))
nobservedshocks(m) = length(observedshocks(m))
nfactors(m::DFMModel) = sum(_nfactors, _blocks(m.factorblocks); init=0)
nstates(m::DFMModel) = sum(_nstates, _blocks(m.factorblocks); init=0)
nstateshocks(m::DFMModel) = sum(_nstateshocks, _blocks(m.factorblocks); init=0)


########################
#  Functions that create factor blocks and add them to a DFMModel instance

"Throw an error if block by this given name already exists."
_check_fbname(model, name) = begin
    if haskey(model.factorblocks, name)
        error("Factor block $name already exists")
    end
end
"Throw an error if the given integer is not a valid AR order value."
_check_fborder(order) = order > 0 || error("Factor block order must be positive")
"Throw an error if given variables are not variables in the given model."
@inline _check_fbvars(model, vars) = begin
    mis = setdiff(vars, model.variables)
    isempty(mis) || error("""These variables are not in the model: $(join(mis, ", "))""")
end
"Throw an error if given variables already have idiosyncratic components."
@inline _check_icvars(model, vars) = begin
    _check_fbvars(model, vars)
    dup = mapreduce(fb -> intersect(vars, _observed(fb)), union, _blocks(model); init=())
    isempty(dup) || error("""These variables already have idiosyncratic components: $(join(dup, ", "))""")
end

_make_name(model, blocktype, prefix) = begin
    num = sum(fb -> fb isa blocktype, _blocks(model); init=0)
    return Symbol(prefix, num + 1)
end


"""
    add_icblock(model::DFMModel; <options>...)

Create a new [`IdiosyncraticComponents`](@ref) instance and add it to the list
of factor blocks in the given [`DFMModel`](@ref).

Options:
 * `order::Integer` - the AR order, must be positive, default is 1.
 * `vars::Vector{Symbol}` - default is `m.variables`
 * `name::Symbol` - If not given, name is assigned automatically
"""
function add_icblock!(model::DFMModel;
    order::Integer=1,
    vars=:all,
    name=nothing
)
    if name === nothing
        name = _make_name(model, IdiosyncraticComponents, "ic")
    end
    _check_fbname(model, name)
    _check_fborder(order)
    if vars === :all
        vars = model.variables
    elseif vars isa Symbol
        vars = Symbol[vars,]
    end
    _check_icvars(model, vars)

    nvars = length(vars)
    push!(model.factorblocks, name => IdiosyncraticComponents(
        name, [zeros(nvars) for _ = 1:order],
        ones(nvars), collect(vars)
    ))
    return _get_block(model, name)
end

"""
    add_factorblock(model::DFMModel; <options>...)

Create a new [`IdiosyncraticComponents`](@ref) instance and add it to the list
of factor blocks in the given [`DFMModel`](@ref).

Options:
 * `nfactors::Integer` - number of factors. No default value, so it must be given
 * `order::Integer` - the AR order, must be positive, default is 1.
 * `vars::Vector{Symbol}` - default is `m.variables`
 * `name::Symbol` - If not given, name is assigned automatically
"""
function add_factorblock!(model;
    name::Union{Symbol,Nothing}=nothing,
    vars=:all,
    order::Integer=1,
    nfactors # no default value, mandatory
)

    if name === nothing
        name = _make_name(model, FactorBlock, "fb")
    end
    _check_fbname(model, name)
    _check_fborder(order)
    if vars === :all
        vars = model.variables
    elseif vars isa Symbol
        vars = Symbol[vars,]
    end
    _check_fbvars(model, vars)
    nvars = length(vars)
    push!(model.factorblocks, name => FactorBlock(
        Symbol(name), convert(Int, nfactors), zeros(nvars, nfactors),
        Matrix{Float64}[zeros(nfactors, nfactors) for _ = 1:order],
        diagm(ones(nfactors)), collect(vars)
    ))
    return _get_block(model, name)
end

############################################################################
# Functions that enumerate DFMModel variables of every kind

nfactors(fb::ARFactorBlock) = _nfactors(fb)
factors(fb::FactorBlock) = ModelVariable[
    Symbol(fb.name, "_", i) for i = 1:_nfactors(fb)
]

factors(ic::IdiosyncraticComponents) = ModelVariable[
    Symbol(v, "_ic") for v in _observed(ic)
]

function factors(model::DFMModel)
    vcat(map(factors, _blocks(model))...,)
end

shocks(fb::ARFactorBlock) = factorshocks(fb)
factorshocks(fb::ARFactorBlock) = ModelVariable[
    to_shock(Symbol(f, "_shk")) for f in factors(fb)
]

function factorshocks(model::DFMModel) 
    return vcat(map(factorshocks, _blocks(model))...)
end

observed(fb::ARFactorBlock) = _observed(fb)
observed(model::DFMModel) = model.variables
observedshocks(model::DFMModel) = begin
    # take all variables
    vars = copy(model.variables)
    # remove those that have idiosyncratic components
    for blk in _blocks(model)
        if blk isa IdiosyncraticComponents
            setdiff!(vars, blk.variables)
        end
    end
    # make shocks for the remaining ones
    return [to_shock(Symbol(v, "_shk")) for v in vars]
end

shocks(model::DFMModel) = vcat(observedshocks(model),
    map(factorshocks, _blocks(model))...,
)

function varshks(model::DFMModel)
    vcat(observed(model), factors(model), shocks(model))
end

allvars(model::DFMModel) = varshks(model)

function Base.getproperty(m::DFMModel, name::Symbol)
    if !hasfield(typeof(m), name)
        if name === :maxlead
            return 0
        elseif name === :maxlag
            return maximum(fb -> fb.maxlag, _blocks(m))
        elseif name === :shocks
            return shocks(m)
        elseif name === :observed
            return observed(m)
        elseif name === :nobserved
            return nobserved(m)
        elseif name === :observedshocks
            return observedshocks(m)
        elseif name === :nobservedshocks
            return nobservedshocks(m)
        elseif name === :factorshocks
            return factorshocks(m)
        elseif name === :factors
            return factors(m)
        elseif name === :varshks
            return varshks(m)
        elseif name in m.variables
            return m.variables[indexin([name], m.variables)[1]]
        elseif haskey(m.factorblocks, name)
            return m.factorblocks[name]
        end
    end
    return getfield(m, name)
end

function Base.propertynames(m::DFMModel)
    return tuple(fieldnames(typeof(m))...,
        :maxlag, :maxlead, :varshks,
        :observed, :observedshocks, 
        :nobserved, :nobservedshocks, 
        :factors, :factroshocks, 
        :shocks,
        (v.name for v in m.variables)...,
        (keys(m.factorblocks))...
    )
end

#######################################################################################


# function stateshks(fb::ARFactorBlock)
#     append!(ModelVariable[
#             Symbol(fb.name, "_", i) for i = 1:_nfactors(fb)
#         ],
#         ModelVariable[
#             to_shock(Symbol(fb.name, "_", i, "_shk")) for i = 1:_nfactors(fb)
#         ])
# end

function Base.getproperty(fb::ARFactorBlock, name::Symbol)
    if !hasfield(typeof(fb), name)
        if name === :maxlag
            return _order(fb)
        elseif name === :order
            return _order(fb)
        elseif name === :nfactors 
            return _nfactors(fb)
        elseif name === :arcoefs
            return _arcoefs(fb)
        elseif name === :loadings
            return _loadings(fb)
        elseif name === :observed
            return observed(fb)
        elseif name === :nobserved
            return nobserved(fb)
        elseif name === :maxlead
            return 0
        # elseif name === :stateshks
        #     return stateshks(fb)
        end
    end
    return getfield(fb, name)
end

Base.propertynames(fb::ARFactorBlock) = tuple(
    fieldnames(typeof(fb))...,
    :maxlag, :maxlead, :order, :nfactors,
    :arcoefs, :loadings, 
)

