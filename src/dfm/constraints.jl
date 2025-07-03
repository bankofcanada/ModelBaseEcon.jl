##################################################################################
# This file is part of ModelBaseEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################


#  = How do constraints work =
#  
#  Say we are estimating a loadings matrix, Λ, but we know that some 
#  of its entries. For example, 
#  * some of the entries may be zero because of block structure
#  * some entries may be known by other means 
#  * some entries may depend on other entries due to mixed frequency
#  Generally, the constraints can be written in the form 
#       W * vec(Λ) = q
#  where W is a matrix and q is a vector. The columns of W correspond to 
#  the entries of Λ, the rows of W and entries of q correspond to constraint 
#  equations.
#
#  Here we provide the means of constructing W and q for a given `DFM`
#  The parameter vector in the DFM must have NaN for the coefficients that 
#  need to be estimated and non-NaN for the coefficients that are known. 
#  The constraints related to mixed-frequency and block-structure of the model 
#  are handled automatically.
#
#  The actual use of these constraints during estimation is implemented in StateSpaceEcon.
#


"""
    struct MatrixConstraint ... end

A data type representing a constraint on the entries of matrix Λ of the type W ⋅
vec(Λ) = q. Instances contain the matrix `W` and vector `q`.

"""
struct MatrixConstraint{TW<:AbstractMatrix{<:Real},Tq<:AbstractVector{<:Real}}
    ncols::Int # number of columns in the matrix being constrained
    W::TW
    q::Tq
end

"""
    struct LoadingConstraint ... end

A data structure that effectively contains a list of
[`MatrixConstraint`](@ref). It is necessary to speed up estimations with
constraints on large sparse loading matrices. Specifically, when a large
observed block loads several large components blocks and idiosyncratic blocks,
with mixed frequencies, the resulting loading matrix contains many structural
zeros. In this case we can run several small constraints on dense regions of the
loading matrix.

Construct instances by calling [`loading_constraint`](@ref)

"""
struct LoadingConstraint{MF<:MixedFrequency}
    blk::ObservedBlock{MF}      # reference to the observed block we're dealing with
    # "fixed" columns are ones that either are all zeros due to block-structure, 
    # or have all their entries already known by other other means before estimation.
    nfixedcols::Int         
    fixedcols::BitVector
    # columns that are not "fixed" will be estimated, subject to constraints
    nestimcols::Int
    estimcols::BitVector
    # columns that will be estimates are organized into blocks corresponding to
    # common components blocks. Each of these has an entry in the estimvlocks list.
    estimblocks::NamedList{MatrixConstraint}
    # inner constructor enforces the mixed frequency of the observed block
    LoadingConstraint(blk::ObservedBlock{MF}, args...) where {MF<:MixedFrequency} = new{MF}(blk, args...)
end

"""
    loading_constraint(Λ, ob::ObservedBlock)

Construct the [`LoadingConstraint`](@ref) data for the given observed block.
Some of the coefficients of Λ are known to be zero from the block structure of
the model and some are known to depend on other coefficients due to
mixed-frequency. For the rest of them, the Λ matrix must contain NaN where a
loading coefficient is to be estimated and non-NaN where it is known.
"""
function loading_constraint(Λ::AbstractMatrix, ob::ObservedBlock{MF}) where {MF<:MixedFrequency}
    # NaN values in Λ indicate values that need to be estimated
    # non-NaN values are known and therefore must remain fixed as given
    #
    # rows correspond to observed variables, columns to possibly lagged state variables
    NO, NS = size(Λ)  
    @assert NO == nobserved(ob)
    @assert NS == sum(nstates_with_lags, ob.components)
    #
    # number of mixed-frequency coefficients
    NC = mf_ncoefs(MF)
    #
    # estimcols[j] is `true` if column Λ[:, j] has anything to be estimated (NaN)
    estimcols = falses(NS)
    any!(isnan, estimcols, transpose(Λ))
    #
    # Loop over the components blocks that connect to ob and populate list 
    # of separate loading constraint for each block.
    # effectively, we split the rows of W matrix by component block
    col_offset = 0      # keep track of where we are along the columns of Λ
    estimblocks = NamedList{MatrixConstraint}()  # the list of matrix constraints we need to populate
    for (cn, cb) = ob.components
        # indices of columns of Λ corresponding the component block cb
        bcols = col_offset .+ (1:nstates_with_lags(cb))
        if any(estimcols[bcols])
            NScb = nstates(cb)
            # we take all columns of the components block;
            # we assume lags(cb) ≥ NC (check is done before calling us); 
            # we take NC lags -- lags between NC and lags(cb) always have their loadings set to 0
            b_est_cols = bcols[end-NScb*NC+1:end]
            W, q = loadingcons(view(Λ, :, b_est_cols), ob, cb)
            push!(estimblocks, cn => MatrixConstraint(NScb * NC, W, q))
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
# nloadingcons(::AbstractMatrix, ::ObservedBlock, ic::IdiosyncraticComponents) = error("IdiosyncraticComponents don't have estimable loadings")
# loadingcons(::AbstractMatrix, ::ObservedBlock, ic::IdiosyncraticComponents) = error("IdiosyncraticComponents don't have estimable loadings")
# loadingcons!(::AbstractMatrix, ::AbstractVector, ::AbstractMatrix, ::ObservedBlock, ic::IdiosyncraticComponents) = error("IdiosyncraticComponents don't have estimable loadings")

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
    #
    # N.B.: Reshape returns a view into W, in other words writing into 
    #       W3 writes into W. So, why do we do this? Here is why.
    #       The columns of W correspond to entries in Λ. This reshape means that
    #       W3[:, i, j] is the column of W that corresponds to entry Λ[i, j].
    #       There is a k, such that W[:, k] === W3[:, i, j], but by using W3 we
    #       offload the calculation of k to the Julia compiler. It makes for simpler
    #       code below and the runtime overhead is relatively small.
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


