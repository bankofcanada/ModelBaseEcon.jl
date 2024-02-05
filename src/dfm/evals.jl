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

_getloading(::IdiosyncraticComponents, var_comprefs::LittleDictVec{Symbol,_BlockComponentRef}, ::DFMParams, ::Symbol) = Diagonal(Ones(length(var_comprefs)))
function _getloading(blk::ComponentsBlock, var_comprefs::LittleDictVec{Symbol,_BlockComponentRef}, p::DFMParams, name::Symbol)
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

_setloading!(::IdiosyncraticComponents, var_comprefs::LittleDictVec{Symbol,_BlockComponentRef}, ::DFMParams, val, name::Symbol) = nothing
function _setloading!(::ComponentsBlock, var_comprefs::LittleDictVec{Symbol,_BlockComponentRef}, p::DFMParams, val, name::Symbol)
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
    for i = 1:lags(blk)
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
    for i = 1:lags(blk)
        C = _getcoef(blk, p, i)
        CR[vars] -= C * Cpoint[end-i, vars]
        CJ[vars, end-i, vars] = -C
    end
    return CR, CJ
end


function _eval_dfm_R!(CR, Cpoint, blk::ObservedBlock, p::DFMParams)
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
        Λ = _getloading(fblk, comprefs, p, name)
        CR[onames] -= Λ * Cpoint[end, fnames]
    end
    return CR
end


function _eval_dfm_RJ!(CR, CJ, Cpoint, blk::ObservedBlock, p::DFMParams)
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
        Λ = _getloading(fblk, comprefs, p, name)
        CJ[onames, end, fnames] = -Λ
        CR[onames] -= Λ * Cpoint[end, fnames]
    end
    return CR, CJ
end

function _eval_dfm_R!(CR, Cpoint, m::DFMModel, p::DFMParams)
    _eval_dfm_R!(CR, Cpoint, m.observed_block, p.observed)
    for (name, block) in m.components
        _eval_dfm_R!(CR, Cpoint, block, getproperty(p, name))
    end
    return CR
end

function _eval_dfm_RJ!(CR, CJ, Cpoint, m::DFMModel, p::DFMParams)
    fill!(CJ, 0)
    _eval_dfm_RJ!(CR, CJ, Cpoint, m.observed_block, p.observed)
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
    NS = nstates(bm)
    if L > 1
        fill!(A, 0)
        RA = reshape(A, NS, L, NS, L)
        for l = 1:L-1
            RA[:, l, :, l+1] = I(NS)
        end
        for l = 1:L
            RA[:, L, :, L-l+1] = _getcoef(bm, params, l)
        end
    else
        A[:, :] = _getcoef(bm, params, 1)
    end
    return A
end

function get_transition!(A::AbstractMatrix, M::DFMModel, params::DFMParams)
    fill!(A, 0)
    offset = 0
    for (bname, blk) in M.components
        binds = offset .+ (1:lags(blk)*nstates(blk))
        get_transition!(view(A, binds, binds), blk, params[bname])
        offset = last(binds)
    end
    return A
end

function get_loading!(A::AbstractMatrix, M::DFMModel, P::DFMParams)
    fill!(A, 0)
    obs = M.observed_block
    obs_comp = obs.components
    obs_c2v = obs.comp2vars
    par = P.observed
    yinds = _enumerate_vars(observed(obs))
    offset = 0
    for (bname, blk) in M.components
        L = lags(blk)
        N = nstates(blk)
        bxinds = (offset + (L - 1) * N) .+ (1:N)
        if haskey(obs_comp, bname)
            crefs = obs_c2v[bname]
            byinds = Int[yinds[v] for v in keys(crefs)]
            A[byinds, bxinds] = _getloading(blk, crefs, par, bname)
        end
        offset = last(bxinds)
    end
    return A
end

function get_mean!(mu::AbstractVector, ::DFMModel, P::DFMParams)
    mu[:] = P.observed.mean
end

function get_covariance!(A::AbstractMatrix, M::DFMModel, P::DFMParams, ::Val{:Observed})
    fill!(A, 0)
    bm = M.observed_block
    par = P.observed
    yinds = _enumerate_vars(observed(bm))
    shk_inds = Int[yinds[v] for v in keys(bm.var2shk)]
    A[shk_inds, shk_inds] = get_covariance(bm, par)
    return A
end

function get_covariance!(A::AbstractMatrix, M::DFMModel, P::DFMParams, ::Val{:State})
    fill!(A, 0)
    offset = 0
    for (bname, blk) in M.components
        L = lags(blk)
        N = nstates(blk)
        bxinds = (offset + (L - 1) * N) .+ (1:N)
        A[bxinds, bxinds] = get_covariance(blk, P[bname])
        offset = last(bxinds)
    end
    return A
end

# The following functions do the opposite - update the parameter vector given the matrix

function set_loading!(P::DFMParams, M::DFMModel, A::AbstractMatrix)
    obs = M.observed_block
    obs_comp = obs.components
    obs_c2v = obs.comp2vars
    par = P.observed
    yinds = _enumerate_vars(observed(obs))
    offset = 0
    for (bname, blk) in M.components
        L = lags(blk)
        N = nstates(blk)
        bxinds = (offset + (L - 1) * N) .+ (1:N)
        if haskey(obs_comp, bname)
            crefs = obs_c2v[bname]
            byinds = Int[yinds[v] for v in keys(crefs)]
            _setloading!(blk, crefs, par, view(A, byinds, bxinds), bname)
        end
        offset = last(bxinds)
    end
    return P.observed.loadings
end

function set_mean!(P::DFMParams, ::DFMModel, mu::AbstractVector)
    P.observed.mean[:] = mu
    return P.observed.mean
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
    NS = nstates(bm)
    if L > 1
        RA = reshape(A, NS, L, NS, L)
        for l = 1:L
            _setcoef!(bm, P, view(RA, :, L, :, L - l + 1), l)
        end
    else
        _setcoef!(bm, P, A, 1)
    end
    return P
end

function set_covariance!(P::DFMParams, M::DFMModel, A::AbstractMatrix, ::Val{:Observed})
    bm = M.observed_block
    par = P.observed
    yinds = _enumerate_vars(observed(bm))
    shk_inds = Int[yinds[v] for v in keys(bm.var2shk)]
    set_covariance!(par, bm, view(A, shk_inds, shk_inds))
    return par.covar
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
