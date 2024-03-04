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

function _getloading((name, blk)::Pair{Symbol,<:ComponentsBlock}, crefs::NamedList{_BlockComponentRef}, p::DFMParams)
    L = zeros(eltype(p), length(crefs), blk.size)
    _getloading!(L, name=>blk, crefs, p)
    return L
end

@inline function _getloading!(Λ::AbstractMatrix, ::Pair{Symbol,<:IdiosyncraticComponents}, crefs::NamedList{_BlockComponentRef}, ::DFMParams)
    col = 0
    for (row, c) in enumerate(values(crefs))
        c isa _NoCompRef && continue
        col = col + 1
        Λ[row, col] = 1
    end
    return Λ
end

function _getloading!(Λ::AbstractMatrix, (nm, b)::Pair{Symbol,<:ComponentsBlock}, crefs::NamedList{_BlockComponentRef}, p::DFMParams)
    pvals = getproperty(p.loadings, nm)
    if all(c -> isa(c, _BlockRef), values(crefs))
        Λ[:, :] = pvals
        return Λ
    end
    idx_p = 0
    for (i, cr) in enumerate(values(crefs))
        nvals = _n_comp_refs(cr)
        nvals == 0 && continue
        Λ[i, _inds_comp_refs(cr)] = pvals[idx_p.+(1:nvals)]
        idx_p = idx_p + nvals
    end
    return Λ
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

function get_loading!(A::AbstractMatrix{<:Number}, ob::ObservedBlock{MF}, par::DFMParams, (cn, cb)::Pair{Symbol,<:ComponentsBlock}) where {MF}
    crefs = get(ob.comp2vars, cn, nothing)
    if isnothing(crefs)
        return A
    end
    yinds =
        C = mf_coefs(MF)
    NC = length(C)
    L = lags(cb)
    if NC > L
        error("Components block :$cn does not have enough lags. Need $NC, have $L.")
    end
    NS = nstates(cb)
    @assert size(A) == (nobserved(ob), nstates_with_lags(cb))
    Λ = _getloading!(view(A, :, (L-1)*NS+1:L*NS), cn => cb, crefs, par)
    for i = 2:NC
        A[:, (L-i)*NS.+(1:NS)] = C[i] * Λ
    end
    lmul!(C[1], Λ)
    return A
end

function get_loading!(A::AbstractMatrix, ob::ObservedBlock, par::DFMParams)
    @assert size(A) == (nobserved(ob), sum(nstates_with_lags, ob.components, init=0))
    offset = 0
    for (cn, cb) in ob.components
        cols = offset .+ (1:nstates_with_lags(cb))
        get_loading!(view(A, :, cols), ob, par, cn => cb)
        offset = last(cols)
    end
    return A
end

function get_loading!(A::AbstractMatrix, M::DFMModel, P::DFMParams)
    fill!(A, 0)
    yinds = _enumerate_vars(observed(M))
    offset = 0
    for (cn, cb) in M.components
        cols = offset .+ (1:nstates_with_lags(cb))
        for (on, ob) in M.observed
            rows = Int[yinds[Symbol(v)] for v in observed(ob)]
            get_loading!(view(A, rows, cols), ob, getproperty(P, on), cn => cb)
        end
        offset = last(cols)
    end
    return A
end

function get_mean!(mu::AbstractVector, ::ObservedBlock, bpar::DFMParams)
    mu[:] = bpar.mean
    return mu
end

function get_mean!(mu::AbstractVector, M::DFMModel, P::DFMParams)
    nm_obs = M.observed
    if length(nm_obs) == 1
        on, ob = first(nm_obs)
        return get_mean!(mu, ob, getproperty(P, on))
    end
    yinds = _enumerate_vars(observed(M))
    for (on, ob) in nm_obs
        byinds = Int[yinds[Symbol(v)] for v in observed(ob)]
        get_mean!(view(mu, byinds), ob, getproperty(P, on))
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
