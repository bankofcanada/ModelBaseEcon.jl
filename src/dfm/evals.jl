##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2023, Bank of Canada
# All rights reserved.
##################################################################################

_getcoef(::ComponentsBlock, p::DFMParams, i::Integer=1) = @view p.coefs[:, :, i]
_getcoef(::IdiosyncraticComponents, p::DFMParams, i::Integer=1) = Diagonal(@view p.coefs[:, i])

_getloading(::ComponentsBlock, p::DFMParams, name::Symbol) = getproperty(p.loadings, name)
_getloading(blk::IdiosyncraticComponents, ::DFMParams, ::Symbol) = Diagonal(Ones(blk.size))

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
        onames = blk.comp2vars[name]
        Λ = _getloading(fblk, p, name)
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
        onames = blk.comp2vars[name]
        Λ = _getloading(fblk, p, name)
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

