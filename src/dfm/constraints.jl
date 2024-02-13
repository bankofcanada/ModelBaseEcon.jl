##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2024, Bank of Canada
# All rights reserved.
##################################################################################

function get_loading_ncons(model::DFMModel, params::DFMParams)
    ncons = 0
    for (on, ob) in model.observed
        op = getproperty(params, on)
        onc = _get_loading_ncons(ob, op)
        ncons = ncons + onc
    end
    return ncons
end

_n_loading_coefs(::Pair{Symbol,<:IdiosyncraticComponents}, ::DFMParams) = 0
_n_loading_coefs((nm, blk)::Pair{Symbol,<:CommonComponents}, loadings::DFMParams) = length(getproperty(loadings, nm))
function _get_loading_ncons(ob::ObservedBlock{MF}, op::DFMParams) where {MF<:MixedFrequency}
    ncm1 = mf_ncoefs(MF) - 1
    ncm1 == 0 && return 0
    opl = op.loadings
    ncons = 0
    for comp in ob.components
        ncons = ncons + ncm1 * _n_loading_coefs(comp, opl)
    end
    return ncons
end

function get_loading_cons!(W::AbstractMatrix{<:Real}, q::AbstractVector{<:Real},
    M::DFMModel, P::DFMParams)
    W = reshape(W, :, nobserved(M), DFMModels.nstates_with_lags(M))
    yinds = _enumerate_vars(observed(M))
    offset_ncons = 0
    for obss in M.observed
        offset_col = 0
        for (nm, blk) in M.components
            ncons = _get_loading_cons!(W, q, offset_ncons, yinds, offset_col, P, obss, nm => blk)
            offset_col = offset_col + lags(blk) * nstates(blk)
            offset_ncons = offset_ncons + ncons
        end
    end
    return
end

_get_loading_cons!(::AbstractArray{<:Real,3}, ::AbstractVector{<:Real},
    ::Integer, ::NamedTuple, ::Integer, ::DFMParams,
    ::Pair{Symbol,<:ObservedBlock},
    ::Pair{Symbol,<:IdiosyncraticComponents}) = 0

function _get_loading_cons!(W::AbstractArray{<:Real,3}, q::AbstractVector{<:Real},
    offset_ncons::Integer, yinds::NamedTuple, offset_col::Integer, P::DFMParams,
    (on, ob)::Pair{Symbol,ObservedBlock{MF}},
    (nm, blk)::Pair{Symbol,<:ComponentsBlock}) where {MF}
    crefs = get(ob.comp2vars, nm, nothing)
    isnothing(crefs) && return 0
    NC = mf_ncoefs(MF)
    NC == 1 && return 0
    N = nstates(blk)
    L = lags(blk)
    # bxinds(i, j) returns the column index in A corresponding to lag i-1 factor j of blk
    #  in other words, i = lag + 1 
    bxinds(lagp1, j) = (offset_col + (L - lagp1) * N) + j
    # byinds = Int[yinds[v] for v in keys(crefs)]
    ncons = 0
    C = mf_coefs(MF)
    for (var, cr) in crefs
        bi = yinds[var]
        for j = _inds_comp_refs(cr)
            for c = 2:NC
                ncons = ncons + 1
                W[offset_ncons+ncons, bi, bxinds(1, j)] = C[c]
                W[offset_ncons+ncons, bi, bxinds(c, j)] = -C[1]
                q[offset_ncons+ncons] = 0
            end
        end
    end
    return ncons
end



