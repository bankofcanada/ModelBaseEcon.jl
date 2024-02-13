##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2024, Bank of Canada
# All rights reserved.
##################################################################################

_getcoef(::ComponentsBlock, p::DFMParams, i::Integer=1) = @view p.coefs[:, :, i]
_getcoef(::IdiosyncraticComponents, p::DFMParams, i::Integer=1) = Diagonal(@view p.coefs[:, i])

_setcoef!(::ComponentsBlock, p::DFMParams, val, i::Integer=1) = (p.coefs[:, :, i] = val; val)
_setcoef!(::IdiosyncraticComponents, p::DFMParams, val, i::Integer=1) = (p.coefs[:, i] = diag(val); val)

_getloading(::Pair{Symbol,<:IdiosyncraticComponents}, var_comprefs::NamedList{_BlockComponentRef}, ::DFMParams) = I(length(var_comprefs))
function _getloading((name, blk)::Pair{Symbol,<:ComponentsBlock}, var_comprefs::NamedList{_BlockComponentRef}, p::DFMParams)
    pvals = getproperty(p.loadings, name)
    all(c -> c isa _BlockRef, values(var_comprefs)) && return pvals
    L = zeros(eltype(p), length(var_comprefs), blk.size)
    idx_p = 0
    for (i, cr) in enumerate(values(var_comprefs))
        nvals = _n_comp_refs(cr)
        L[i, _inds_comp_refs(cr)] = pvals[idx_p.+(1:nvals)]
        idx_p = idx_p + nvals
    end
    return L
end

_setloading!(::Pair{Symbol,<:IdiosyncraticComponents}, var_comprefs::NamedList{_BlockComponentRef}, ::DFMParams, val) = nothing
function _setloading!((name, _)::Pair{Symbol,<:ComponentsBlock}, var_comprefs::NamedList{_BlockComponentRef}, p::DFMParams, val)
    pl = p.loadings
    if all(c -> c isa _BlockRef, values(var_comprefs))
        setproperty!(pl, name, val)
    end
    pvals = getproperty(pl, name)
    idx_p = 0
    for (i, cr) in enumerate(values(var_comprefs))
        nvals = _n_comp_refs(cr)
        pvals[idx_p.+(1:nvals)] = val[i, _inds_comp_refs(cr)]
        idx_p = idx_p + nvals
    end
    return nothing
end

# export eval_RJ!

_alloc_R(b_or_m) = Vector{Float64}(undef, nendog(b_or_m))
_alloc_J(b::DFMBlockOrModel) = spzeros(nendog(b), (1 + lags(b) + leads(b)) * nvarshks(b))

function eval_resid(point::AbstractMatrix, bm::DFMBlockOrModel, p::DFMParams)
    R = _alloc_R(bm)
    eval_R!(R, point, bm, p)
    return R
end

function eval_RJ(point::AbstractMatrix, bm::DFMBlockOrModel, p::DFMParams)
    R = _alloc_R(bm)
    J = _alloc_J(bm)
    eval_RJ!(R, J, point, bm, p)
    return R, J
end



function _eval_dfm_R!(CR, Cpoint, blk::ComponentsBlock, p::DFMParams)
    vars = endog(blk)
    shks = shocks(blk)
    CR[vars] = Cpoint[end, vars] - Cpoint[end, shks]
    for i = 1:order(blk)
        C = _getcoef(blk, p, i)
        CR[vars] -= C * Cpoint[end-i, vars]
    end
    return CR
end

function _eval_dfm_RJ!(CR, CJ, Cpoint, blk::ComponentsBlock, p::DFMParams)
    nvars = nendog(blk)
    vars = endog(blk)
    shks = shocks(blk)
    CR[vars] = Cpoint[end, vars] - Cpoint[end, shks]
    CJ[vars, end, vars] = I(nvars)
    CJ[vars, end, shks] = -I(nvars)
    for i = 1:order(blk)
        C = _getcoef(blk, p, i)
        CR[vars] -= C * Cpoint[end-i, vars]
        CJ[vars, end-i, vars] = -C
    end
    return CR, CJ
end


function _eval_dfm_R!(CR, Cpoint, blk::ObservedBlock{MF}, p::DFMParams) where {MF}
    # nvars = nendog(blk)
    vars = endog(blk)         # all observed vars
    #! this uses implementation detail of LittleDict
    svars = blk.var2shk.keys  # observed vars with observation shocks
    sshks = blk.var2shk.vals
    CR[vars] = Cpoint[end, vars] - p.mean
    CR[svars] -= Cpoint[end, sshks]
    for (name, fblk) in blk.components
        # names of factors in this block
        fnames = endog(fblk)
        # names of observed vars that are loading the factors in this block
        comprefs = blk.comp2vars[name]
        onames = comprefs.keys
        Λ = _getloading(name => fblk, comprefs, p)
        C = mf_coefs(MF)
        for i = 1:mf_ncoefs(MF)
            CR[onames] -= C[i] * Λ * Cpoint[end-i+1, fnames]
        end
    end
    return CR
end


function _eval_dfm_RJ!(CR, CJ, Cpoint, blk::ObservedBlock{MF}, p::DFMParams) where {MF}
    nvars = nendog(blk)
    vars = endog(blk)
    #! this uses implementation detail of LittleDict
    svars = blk.var2shk.keys
    sshks = blk.var2shk.vals
    CR[vars] = Cpoint[end, vars] - p.mean
    CR[svars] -= Cpoint[end, sshks]
    CJ[vars, end, vars] = I(nvars)
    CJ[svars, end, sshks] = -I(length(sshks))
    for (name, fblk) in blk.components
        # names of factors in this block
        fnames = endog(fblk)
        # names of observed that are loading the factors in this block
        comprefs = blk.comp2vars[name]
        onames = comprefs.keys
        Λ = _getloading(name => fblk, comprefs, p)
        C = mf_coefs(MF)
        for i = 1:mf_ncoefs(MF)
            CR[onames] -= C[i] * Λ * Cpoint[end-i+1, fnames]
            CJ[onames, end-i+1, fnames] = -C[i] * Λ
        end
    end
    return CR, CJ
end

function _eval_dfm_R!(CR, Cpoint, m::DFMModel, p::DFMParams)
    for (name, block) in m.observed
        _eval_dfm_R!(CR, Cpoint, block, getproperty(p, name))
    end
    for (name, block) in m.components
        _eval_dfm_R!(CR, Cpoint, block, getproperty(p, name))
    end
    return CR
end

function _eval_dfm_RJ!(CR, CJ, Cpoint, m::DFMModel, p::DFMParams)
    fill!(CJ, 0)
    for (name, block) in m.observed
        _eval_dfm_RJ!(CR, CJ, Cpoint, block, getproperty(p, name))
    end
    for (name, block) in m.components
        _eval_dfm_RJ!(CR, CJ, Cpoint, block, getproperty(p, name))
    end
    return CR, CJ
end

function eval_R!(R::AbstractVector, point::AbstractMatrix, bm::DFMBlockOrModel, p::DFMParams)
    CR, _, Cpoint = _wrap_arrays(bm, R, nothing, point)
    _eval_dfm_R!(CR, Cpoint, bm, p)
    return R
end

function eval_RJ!(R::AbstractVector, J::AbstractMatrix, point::AbstractMatrix, bm::DFMBlockOrModel, p::DFMParams)
    CR, CJ, Cpoint = _wrap_arrays(bm, R, J, point)
    _eval_dfm_RJ!(CR, CJ, Cpoint, bm, p)
    # @assert R ≈ J * point[:]
    return R, J
end


######################################################
###   eval functions needed for the Kalman filter in StateSpaceEcon.Kalman 

function get_transition!(A::AbstractMatrix, bm::ComponentsBlock, params::DFMParams)
    L = lags(bm)
    O = order(bm)
    NS = nstates(bm)
    if L > 1
        fill!(A, 0)
        RA = reshape(A, NS, L, NS, L)
        for l = 1:L-1
            RA[:, l, :, l+1] = I(NS)
        end
        for l = 1:O
            RA[:, L, :, L-l+1] = _getcoef(bm, params, l)
        end
    else
        # L = 1 or 0.  We know that O <= L
        if O == 1
            A[:, :] = _getcoef(bm, params, 1)
            # elseif O == 0
            #     nothing
        end
    end
    return A
end

function get_transition!(A::AbstractMatrix, M::DFMModel, params::DFMParams)
    fill!(A, 0)
    offset = 0
    for (bname, blk) in M.components
        binds = offset .+ (1:lags(blk)*nstates(blk))
        get_transition!(view(A, binds, binds), blk, getproperty(params, bname))
        offset = last(binds)
    end
    return A
end

function _get_oblk_loading!(A::AbstractMatrix, yinds::NamedTuple, offset::Int, par::DFMParams,
    (obname, oblk)::Pair{Symbol,ObservedBlock{MF}},
    (cbname, cblk)::Pair{Symbol,<:ComponentsBlock}
) where {MF}
    crefs = get(oblk.comp2vars, cbname, nothing)
    isnothing(crefs) && return
    N = nstates(cblk)
    L = lags(cblk)
    bxinds(lagp1) = (offset + (L - lagp1) * N) .+ (1:N)
    NC = mf_ncoefs(MF)
    if NC > L
        error("Components block :$cbname does not have enough lags for observed block :$obname. Need $NC, have $L.")
    end
    # bxinds(i) returns the column indices in A corresponding to lag i-1 of the factors in cblk
    #  in other words, i = lag + 1

    byinds = Int[yinds[v] for v in keys(crefs)]
    Λ = _getloading(cbname => cblk, crefs, getproperty(par, obname))
    C = mf_coefs(MF)
    for i = 1:NC
        A[byinds, bxinds(i)] = C[i] * Λ
    end
    return
end

function get_loading!(A::AbstractMatrix, M::DFMModel, P::DFMParams)
    fill!(A, 0)
    yinds = _enumerate_vars(observed(M))
    offset = 0
    for (nm, blk) in M.components
        for obss in M.observed
            _get_oblk_loading!(A, yinds, offset, P, obss, nm => blk)
        end
        offset = offset + lags(blk) * nstates(blk)
    end
    return A
end

function get_mean!(mu::AbstractVector, M::DFMModel, P::DFMParams)
    nm_obs = M.observed
    if length(nm_obs) == 1
        on = first(keys(nm_obs))
        mu[:] = getproperty(P, on).mean
        return mu
    end
    yinds = _enumerate_vars(observed(M))
    for (on, ob) in nm_obs
        byinds = Int[yinds[Symbol(v)] for v in observed(ob)]
        mu[byinds] = getproperty(P, on).mean
    end
    return mu
end

function get_covariance!(A::AbstractMatrix, M::DFMModel, P::DFMParams, ::Val{:Observed})
    fill!(A, 0)
    yinds = _enumerate_vars(observed(M))
    for (on, ob) in M.observed
        shk_inds = Int[yinds[v] for v in keys(ob.var2shk)]
        A[shk_inds, shk_inds] = get_covariance(ob, getproperty(P, on))
    end
    return A
end

function get_covariance!(A::AbstractMatrix, M::DFMModel, P::DFMParams, ::Val{:State})
    fill!(A, 0)
    offset = 0
    for (bname, blk) in M.components
        L = lags(blk)
        N = nstates(blk)
        bxinds = (offset + (L - 1) * N) .+ (1:N)
        A[bxinds, bxinds] = get_covariance(blk, getproperty(P, bname))
        offset = last(bxinds)
    end
    return A
end

# The following functions do the opposite - update the parameter vector given the matrix

function _set_oblk_loading!(P::DFMParams, yinds::NamedTuple, offset::Int, A::AbstractMatrix,
    (obname, oblk)::Pair{Symbol,ObservedBlock{MF}},
    (cbname, cblk)::Pair{Symbol,<:ComponentsBlock}
) where {MF}
    N = nstates(cblk)
    L = lags(cblk)
    bxinds = (offset + (L - 1) * N) .+ (1:N)
    crefs = get(oblk.comp2vars, cbname, nothing)
    isnothing(crefs) && return last(bxinds)
    byinds = Int[yinds[v] for v in keys(crefs)]
    _setloading!(cbname => cblk, crefs, getproperty(P, obname), view(A, byinds, bxinds))
    return last(bxinds)
end

function set_loading!(P::DFMParams, M::DFMModel, A::AbstractMatrix)
    yinds = _enumerate_vars(observed(M))
    offset = 0
    for (nm, blk) in M.components
        for o_nm_blk in M.observed
            _set_oblk_loading!(P, yinds, offset, A, o_nm_blk, nm => blk)
        end
        offset = offset + lags(blk) * nstates(blk)
    end
    return P
end

function set_mean!(P::DFMParams, M::DFMModel, mu::AbstractVector)
    nm_obs = M.observed
    if length(nm_obs) == 1
        on = first(keys(nm_obs))
        getproperty(P, on).mean[:] = mu
        return mu
    end
    yinds = _enumerate_vars(observed(M))
    for (on, ob) in nm_obs
        byinds = Int[yinds[Symbol(v)] for v in observed(ob)]
        getproperty(P, on).mean[:] = mu[byinds]
    end
    return P
end

function set_transition!(P::DFMParams, M::DFMModel, A::AbstractMatrix)
    offset = 0
    for (bname, blk) in M.components
        binds = offset .+ (1:lags(blk)*nstates(blk))
        set_transition!(getproperty(P, bname), blk, view(A, binds, binds))
        offset = last(binds)
    end
    return P
end

function set_transition!(P::DFMParams, bm::ComponentsBlock, A::AbstractMatrix)
    L = lags(bm)
    O = order(bm)
    NS = nstates(bm)
    if L > 1
        RA = reshape(A, NS, L, NS, L)
        for l = 1:O
            _setcoef!(bm, P, view(RA, :, L, :, L - l + 1), l)
        end
    else
        # L = 1 or 0.  We know that O <= L
        if O == 1
            _setcoef!(bm, P, A, 1)
            # elseif O == 0
            #     nothing
        end
    end
    return P
end

function set_covariance!(P::DFMParams, M::DFMModel, A::AbstractMatrix, ::Val{:Observed})
    yinds = _enumerate_vars(observed(M))
    for (on, ob) in M.observed
        shk_inds = Int[yinds[v] for v in keys(ob.var2shk)]
        set_covariance!(getproperty(P, on), ob, view(A, shk_inds, shk_inds))
    end
    return P
end

function set_covariance!(P::DFMParams, M::DFMModel, A::AbstractMatrix, ::Val{:State})
    offset = 0
    for (bname, blk) in M.components
        L = lags(blk)
        N = nstates(blk)
        bxinds = (offset + (L - 1) * N) .+ (1:N)
        set_covariance!(getproperty(P, bname), blk, view(A, bxinds, bxinds))
        offset = last(bxinds)
    end
    return P
end
