# https://arxiv.org/pdf/1510.04923


struct TwoWayVarianceComponentsModel <: RegressionModel

    # The row index of each observation
    row::Vector{Int}

    # The column index of each observation
    col::Vector{Int}

    # The data
    Y::Vector{Float64}

    # The number of observations in each row
    rowCounts::Vector{Int}

    # The number of observations in each column
    colCounts::Vector{Int}

    # The sum of observations in each row
    rowSums::Vector{Float64}

    # The sum of observations in each column
    colSums::Vector{Float64}

    # The mean centered sum of squares within each row
    rowSS::Vector{Float64}

    # The mean centered sum of squares within each column
    colSS::Vector{Float64}

    # The mean centered sum of fourth powers within each row
    rowSQ::Vector{Float64}

    # The mean centered sum of fourth powers within each column
    colSQ::Vector{Float64}

    # Second moment U-statistics
    ustats2::Vector{Float64}

    # Fourth moment U-statistics
    ustats4::Vector{Float64}

    # The variance estimates for row effects, column effects, and residual variance.
    params::Vector{Float64}

    # The kurtosis values for the row effects, column effects, and residual variance.
    kurt::Vector{Float64}

    # Provided row values
    rowLabels::AbstractVector

    # Provided column values
    colLabels::AbstractVector
end

function moments4(ii, Y, Mu)

    m = length(Mu)
    SQ = zeros(m)

    for (i,y) in zip(ii, Y)
        SQ[i] += (y - Mu[i])^4
    end

    return SQ
end

# Calculate the first moment and the second central moment of Y within each
# group defined by ii.  Also calculates the sample size within each group.
function moments12(ii, Y)

    @assert length(ii) == length(Y)
    n = length(ii)
    m = maximum(ii)
    N = zeros(m)
    T = zeros(Float64, m)
    S = zeros(Float64, m)

    for (i,y) in zip(ii, Y)
        if N[i] == 0
            N[i] = 1
            T[i] = y
        else
            N[i] += 1
            k = N[i]
            T[i] += y
            S[i] += (k*y - T[i])^2 / (k * (k - 1))
        end
    end

    return N, T, S
end

"""
    vcov(TwoWayVarianceComponentsModel)

vcov returns the estimated variance covariance for the sampling distribution
of the method of moments estimates of the two-way variance components.
"""
function vcov(m::TwoWayVarianceComponentsModel; kf=1.0)

    (; Y, params, kurt, rowCounts, colCounts) = m

    # This is the large sample version of Gao and Owen
    N = length(Y)
    v = zeros(3, 3)
    v[1, 1] = params[1]^2 * (kf*kurt[1] + 2) * sum(abs2, colCounts)
    v[2, 2] = params[2]^2 * (kf*kurt[2] + 2) * sum(abs2, rowCounts)
    v[3, 3] = params[3]^2 * (kf*kurt[3] + 2) * N
    v /= N^2

    return v
end

function coef(m::TwoWayVarianceComponentsModel)
    return m.params
end

function variances(m::TwoWayVarianceComponentsModel)
    (; params) = m
    return (rowvar=m.params[1], colvar=m.params[2], unexplainedvar=m.params[3])
end

function recode(ix)
    T = eltype(ix)
    uq = Dict{T,Int}()
    for x in sort(unique(ix))
        uq[x] = length(uq) + 1
    end
    return [uq[x] for x in ix]
end

function fit(::Type{TwoWayVarianceComponentsModel}, row::S, col::T, Y::U) where{S<:AbstractVector,T<:AbstractVector,U<:AbstractVector}

    if !(length(row) == length(col) == length(Y))
        error(@sprintf("row, col, and Y must have equal lengths (%d, %d, %d are not equal)", len(row), len(col), len(Y)))
    end

    if !(eltype(row)<:Integer && eltype(col)<:Integer)
        error("row and col must contain integers")
    end

    rowi = recode(row)
    coli = recode(col)

    rowCounts, rowSums, rowSS = moments12(rowi, Y)
    colCounts, colSums, colSS = moments12(coli, Y)

    resid = Y .- mean(Y)

    # The second order U-statistics
    Ua = sum(rowSS)
    Ub = sum(colSS)
    m2 = sum(abs2, resid)
    Ue = length(Y) * m2
    ustats2 = [Ua, Ub, Ue]

    N = length(Y)
    n = length(rowi)
    r = length(rowCounts)
    c = length(colCounts)
    M = zeros(3, 3)
    M[1, :] = [0, n-r, n-r]
    M[2, :] = [n-c, 0, n-c]
    M[3, :] = [n^2 - sum(abs2, rowCounts), n^2 - sum(abs2, colCounts), n*(n-1)]

    # Solve the second order moment estimating equations to estimate the variance parameters.
    params = M \ ustats2

    # The fourth order U-statistics
    rowSQ = moments4(rowi, Y, rowSums ./ rowCounts)
    colSQ = moments4(coli, Y, colSums ./ colCounts)
    m4 = sum(x->x^4, resid)
    Wa = sum(rowSQ) + 3*sum(rowSS.^2 ./ rowCounts)
    Wb = sum(colSQ) + 3*sum(colSS.^2 ./ colCounts)
    We = N*m4 + 3*m2^2
    ustats4 = [Wa, Wb, We]

    # Estimate the fourth moment statistics that are used to estimate the kurtosis
    ma = (3*params[2]^2 + 12*params[2]*params[3] + 3*params[3]^2) * (N - r)
    mb = (3*params[1]^2 + 12*params[1]*params[3] + 3*params[3]^2) * (N - c)
    me = (3*params[1]^2 + 12*params[1]*params[3]) * (N^2 - sum(abs2, rowCounts))
    me += (3*params[2]^2 + 12*params[2]*params[3]) * (N^2 - sum(abs2, colCounts))
    me += 3*params[3]^2*N*(N - 1)
    me += 12*params[1]*params[2]*(N^2 - sum(abs2, rowCounts) - sum(abs2, colCounts) + N)
    W = ustats4 - [ma, mb, me]
    mu4 = M \ W

    # Kurtosis estimate
    kurt = mu4 ./ params.^2 .- 3

    model = TwoWayVarianceComponentsModel(rowi, coli, Y, rowCounts, colCounts, rowSums, colSums, rowSS, colSS,
                                          rowSQ, colSQ, ustats2, ustats4, params, kurt, row, col)
    return model
end

function show(io::IO, vc::TwoWayVarianceComponentsModel)
    (; params) = vc
    write(io, @sprintf("Row variance:      %.4f\n", params[1]))
    write(io, @sprintf("Column variance:   %.4f\n", params[2]))
    write(io, @sprintf("Residual variance: %.4f\n", params[3]))
end
