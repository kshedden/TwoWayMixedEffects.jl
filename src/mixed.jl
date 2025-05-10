
struct TwoWayMixedEffectsModel <: RegressionModel

    # Design matrix for the regression model
    X::AbstractMatrix

    # Response vector for the regression model
    y::AbstractVector

    # Regression coefficient estimates
    params::AbstractVector

    # Sampleing variance/covariance matrix of the regression parameter estimates
    vcov::AbstractMatrix

    vc::TwoWayVarianceComponentsModel
end

# Find the boundaries of consecutive runs of constant values in ix[ii].
function boundaries(ix, ii)

    n = length(ix)

    # The start positions of all blocks
    b = [1]
    for j in 2:length(ii)
        if ix[ii[j]] != ix[ii[j-1]]
            push!(b, j)
        end
    end
    push!(b, n+1)

    return b
end

# Construct a symmetric sparse matrix that assigns variance or covariance `v` to
# all pairs of identical values in `ix`.
function sketchCov(ix, v)

    n = length(ix)
    ii = sortperm(ix)
    b = boundaries(ix, ii)

    I, J = Int[], Int[]
    for i in 1:length(b)-1
        j1, j2 = b[i], b[i+1]-1
        jj = ii[j1:j2]
        for k1 in 1:length(jj)
            for k2 in 1:k1
                (l1, l2) = jj[k1] < jj[k2] ? (jj[k1], jj[k2]) : (jj[k2], jj[k1])
                push!(I, l1)
                push!(J, l2)
            end
        end
    end

    M = Symmetric(sparse(I, J, fill(v, length(I))))
    return M
end

# Construct an approximation to the covariance among observations, using a random
# subsample of nsketch observations.
function sketchCov(rowv, colv, residv, row, col; nsketch=1000)
    @assert length(row) == length(col)
    n = length(row)
    nsketch = min(nsketch, n)

    ix = sample(1:n, nsketch; replace=false)
    sort!(ix)

    S0 = spdiagm(fill(residv, nsketch))
    SR = sketchCov(row[ix], rowv)
    SC = sketchCov(col[ix], colv)
    M = Symmetric(S0 + SR + SC)

    # Rescale to account for sketching
    M *= n / nsketch

    return M, ix
end

function vcov(m::TwoWayMixedEffectsModel)
    return m.vcov
end

function coef(m::TwoWayMixedEffectsModel)
    return m.params
end

function variances(m::TwoWayMixedEffectsModel)
    return variances(m.vc)
end

function fit(::Type{TwoWayMixedEffectsModel}, X::R, y::S, row::T, col::U; nsketch=1000) where{R<:AbstractMatrix, S<:AbstractVector, T<:AbstractVector, U<:AbstractVector}

    rowi = recode(row)
    coli = recode(col)

    if !(eltype(X)<:AbstractFloat && eltype(y)<:AbstractFloat)
        error("X and y must contain floating point values")
    end

    # OLS
    b = X \ y
    yhat = X * b
    resid = y - yhat

    # Estimate variance parameters using the residuals
    mr = fit(TwoWayVarianceComponentsModel, rowi, coli, resid)

    # Estimate the vcov matrix using sketching.
    rowv = coef(mr)[1]
    colv = coef(mr)[2]
    residv = coef(mr)[3]
    C, ix = sketchCov(rowv, colv, residv, mr.row, mr.col; nsketch=nsketch)
    XX = X[ix, :]
    M = XX' * C * XX
    B = X' * X
    vcov = B \ M / B

    return TwoWayMixedEffectsModel(X, y, b, vcov, mr)
end

function show(io::IO, mx::TwoWayMixedEffectsModel)
    (; vc) = mx
    (; params) = vc
    write(io, @sprintf("Row variance:      %.4f\n", params[1]))
    write(io, @sprintf("Column variance:   %.4f\n", params[2]))
    write(io, @sprintf("Residual variance: %.4f\n\n", params[3]))

    vm = vcov(mx)
    se = sqrt.(diag(vm))
    td = [coef(mx) se]
    pretty_table(io, td, tf=tf_markdown; header=["Estimate", "SE"])
end
