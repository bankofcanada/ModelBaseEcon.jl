##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2024, Bank of Canada
# All rights reserved.
##################################################################################


struct MatrixConstraint{TW,Tq}
    ncols::Int # number of columns in the matrix being constrained
    W::TW
    q::Tq
end

struct LoadingConstraint{MF}
    blk::ObservedBlock{MF}
    nfixedcols::Int
    fixedcols::BitVector
    nestimcols::Int
    estimcols::BitVector
    estimblocks::NamedList{MatrixConstraint}
    LoadingConstraint(blk::ObservedBlock{MF}, args...) where {MF<:MixedFrequency} = new{MF}(blk, args...)
end

function loading_constraint(Λ::AbstractMatrix, ob::ObservedBlock{MF}) where {MF<:MixedFrequency}
    # NaN values in Λ indicate values that need to be estimated
    # non-NaN values are known and therefore must remain fixed as given
    NO, NS = size(Λ)
    NC = mf_ncoefs(MF)
    @assert NO == nobserved(ob)
    @assert NS == sum(nstates_with_lags, ob.components)
    estimcols = falses(NS)
    any!(isnan, estimcols, transpose(Λ))
    col_offset = 0
    estimblocks = NamedList{MatrixConstraint}()
    for (cn, cb) = ob.components
        bcols = col_offset .+ (1:nstates_with_lags(cb))
        if any(estimcols[bcols])
            NS = nstates(cb)
            # we take all columns of the components block;
            # we assume lags(cb) ≥ NC (check is done before calling us); 
            # we take NC lags -- lags between NC and lags(cb) always have their loadings = 0
            b_est_cols = bcols[end-NS*NC+1:end]
            W, q = loadingcons(view(Λ, :, b_est_cols), ob, cb)
            push!(estimblocks, cn => MatrixConstraint(NS * NC, W, q))
            # mark all b_est_cols columns to be estimated (entire block, constraints will be imposed)
            fill!(view(estimcols, b_est_cols), true)
        end
        col_offset = last(bcols)
    end
    nestimcols = sum(estimcols)
    fixedcols = .!estimcols
    nfixedcols = NS - nestimcols
    return LoadingConstraint(ob,
        nfixedcols, fixedcols,
        nestimcols, estimcols,
        estimblocks)
end

###################################################################

# these must not be called for IdiosyncraticComponents
nloadingcons(::AbstractMatrix, ::ObservedBlock, ic::IdiosyncraticComponents) = error("IdiosyncraticComponents don't have estimatable loadings")
loadingcons(::AbstractMatrix, ::ObservedBlock, ic::IdiosyncraticComponents) = error("IdiosyncraticComponents don't have estimatable loadings")
loadingcons!(::AbstractMatrix, ::AbstractVector, ::AbstractMatrix, ::ObservedBlock, ic::IdiosyncraticComponents) = error("IdiosyncraticComponents don't have estimatable loadings")

# implementations for CommonComponents
function nloadingcons(Λ::AbstractMatrix, ob::ObservedBlock{MF}, cb::CommonComponents) where {MF<:MixedFrequency}
    # free parameters can be found only in the columns corresponding to lag0
    # free parameters are indicated by NaN values
    NS = nstates(cb)
    NCOLS = size(Λ, 2)
    nfree = 0
    for i = NCOLS-NS+1:NCOLS
        nfree = nfree + sum(isnan, view(Λ, :, i))
    end
    # all non-free values in Λ must get a constraint
    return length(Λ) - nfree
end

function loadingcons(Λ::AbstractMatrix, ob::ObservedBlock, cb::CommonComponents)
    ncons = nloadingcons(Λ, ob, cb)
    W = spzeros(ncons, length(Λ))
    q = spzeros(ncons)
    loadingcons!(W, q, Λ, ob, cb)
    return W, q
end

function loadingcons!(W::AbstractMatrix, q::AbstractVector, Λ::AbstractMatrix, ob::ObservedBlock{MF}, cb::CommonComponents) where {MF<:MixedFrequency}
    (size(W, 1) > 0) || return 0
    NO = nobserved(ob)
    NS = nstates(cb)
    NL = lags(cb)
    NC = mf_ncoefs(MF)
    NLAM2 = size(Λ, 2)
    NP = (NLAM2 == NS * NC) ? NC : (NLAM2 == NS * NL) ? NL : error("Unexpected number of columns in loadings matrix")
    @assert size(Λ, 1) == NO
    @assert size(W, 2) == length(Λ)
    C = mf_coefs(MF)
    W3 = reshape(W, size(W, 1), NO, NLAM2)
    # start fresh
    fill!(W, zero(eltype(W)))
    fill!(q, zero(eltype(q)))
    # iterate over the entries of lag0. These are in the last NS columns of Λ
    con = 0
    for j = NLAM2-NS+1:NLAM2
        for i = 1:NO
            val = Λ[i, j]
            if isnan(val)
                # free parameter, no constraint equation
                nothing
            else
                con = con + 1
                W3[con, i, j] = 1
                q[con] = val
            end
            # set mixed-frequency constraints
            for lag = 1:NC-1  # lag coefficient is in C[lag + 1]
                con = con + 1
                W3[con, i, j] = C[lag+1]
                W3[con, i, j-lag*NS] = -C[1]
                # q[con] = 0
            end
            # the remaining lags have loadings equal to 0
            for lag = NC:NP-1
                con = con + 1
                W3[con, i, j-lag*NS] = 1
                # q[con] = 0
            end
        end
    end
    @assert con == size(W, 1)
    return con
end


