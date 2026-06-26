##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################
#
# DFM subsystem.
#
# The `DFMModels` submodule imports a handful of names from the parent
# `ModelBaseEcon` module: `ModelVariable`, `shocks`, `nshocks`, `allvars`,
# `nallvars`, `eval_resid`, `eval_RJ`, `eval_R!`, `to_shock`, `isshock`,
# `AbstractModel`. The Symbolics-based core does not have a `ModelVariable` type
# (it works in plain `Symbol`s) and the DFM subsystem is entirely self-contained
# above the equation-kernel layer. So we port the slice of `ModelVariable` /
# `to_shock` / `isshock` / shock helpers / `AbstractModel` that the DFM actually
# uses into `modelvariable.jl` inside this submodule.
##################################################################################

module DFMModels

using LinearAlgebra
using OrderedCollections
using ComponentArrays
using SparseArrays

####################################################

const LittleDictVec{K,V} = LittleDict{K,V,Vector{K},Vector{V}}
const NamedList{V} = LittleDictVec{Symbol,V}

include("modelvariable.jl")

const Sym = Union{AbstractString,Symbol,ModelVariable}
const LikeVec{T} = Union{Vector{T},NTuple{N,T} where {N},NamedTuple{NT,NTuple{N,T} where {N}} where {NT}}
const SymVec = LikeVec{<:Sym}

const DiagonalF64 = Diagonal{Float64,Vector{Float64}}
const SymmetricF64 = Symmetric{Float64,Matrix{Float64}}

####################################################

include("types.jl")
include("utils.jl")
include("params.jl")
include("evals.jl")
include("dfm_type.jl")
include("constraints.jl")
include("show.jl")

end # module DFMModels
