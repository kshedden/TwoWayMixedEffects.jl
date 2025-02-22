using Test
using StableRNGs
using TwoWayMixedEffects
using Statistics

function gendat(rng, nrow, ncol, sdrow, sdcol, sdresid, ee)

    rowEffects = sdrow * randn(rng, nrow)
    colEffects = sdcol * randn(rng, ncol)

    Y = Float64[]
    row, col = Int[], Int[]
    for i in 1:nrow
        for j in 1:ncol
            if rand(rng) < ee
                push!(Y, rowEffects[i] + colEffects[j] + sdresid*randn(rng))
                push!(row, i)
                push!(col, j)
            end
        end
    end

    return Y, row, col
end

@testset "Basic test of two way covariance construction" begin

    sketchCov = TwoWayMixedEffects.sketchCov
    boundaries = TwoWayMixedEffects.boundaries

    ix = Int[3, 1, 2, 5, 6, 5]
    ii = sortperm(ix)
    b = boundaries(ix, ii)
    @test isapprox(ix[ii][b[1:end-1]], [1, 2, 3, 5, 6])

    rng = StableRNG(123)
    row = [3, 1, 2, 1]
    col = [1, 1, 3, 2]
    S, _ = sketchCov(1., 2., 3., row, col)
    S0 = Float64[6 2 0 0; 2 6 0 1; 0 0 6 0; 0 1 0 6]
    @test isapprox(S, S0)

    rng = StableRNG(123)
    row = [2, 1, 2, 1]
    col = [2, 1, 2, 1]
    S, _ = sketchCov(1., 2., 3., row, col)
    S0 = Float64[6 0 3 0; 0 6 0 3; 3 0 6 0; 0 3 0 6]
    @test isapprox(S, S0)
end

@testset "Basic test of mixed effects" begin

    rng = StableRNG(123)

    nrow = 400
    ncol = 500

    sda = 3.0
    sdb = 2.0
    sde = 1.0

    E, row, col = gendat(rng, nrow, ncol, sda, sdb, sde, 0.5)

    n = length(E)
    p = 4
    X = randn(rng, n, p)
    beta = Float64[1, -2, 0, 1]
    Ey = X * beta
    y = Ey + E

    m = fit(TwoWayMixedEffectsModel, X, y, row, col)
    println(m)
    bhat = coef(m)
    vc = vcov(m)
    c2 = (bhat - beta)' * (vc \ (bhat - beta))
    @test c2 < p + 2*sqrt(2*p)
end

@testset "Basic test of variance components" begin

    rng = StableRNG(123)

    nrow = 400
    ncol = 500

    sda = 3.0
    sdb = 2.0
    sde = 1.0

    Y, row, col = gendat(rng, nrow, ncol, sda, sdb, sde, 0.5)
    mm = fit(TwoWayVarianceComponentsModel, row, col, Y)

    n = length(Y)

    # Check the second order U-statistics
    EUa = (sdb^2 + sde^2) * (n - nrow)
    EUb = (sda^2 + sde^2) * (n - ncol)
    EUe = sda^2 * (n^2 - sum(abs2, mm.rowCounts)) + sdb^2 * (n^2 - sum(abs2, mm.colCounts)) + sde^2 * n * (n - 1)
    Eustats2 = [EUa, EUb, EUe]
    @test isapprox(Eustats2, mm.ustats2, atol=0.01, rtol=0.05)

    # Check some of the moments by direct calculation
    for i in 1:100
        ii = findall(row .== i)
        if length(ii) > 0
            YY = Y[ii]
            resid = YY .- mean(YY)
            @test isapprox(length(resid), mm.rowCounts[i])
            @test isapprox(sum(YY), mm.rowSums[i])
            @test isapprox(sum(x->x^2, resid), mm.rowSS[i])
            @test isapprox(sum(x->x^4, resid), mm.rowSQ[i])
        end
        ii = findall(col .== i)
        if length(ii) > 0
            YY = Y[ii]
            resid = YY .- mean(YY)
            @test isapprox(length(resid), mm.colCounts[i])
            @test isapprox(sum(YY), mm.colSums[i])
            @test isapprox(sum(x->x^2, resid), mm.colSS[i])
            @test isapprox(sum(x->x^4, resid), mm.colSQ[i])
        end
    end

    # Check the parameter estimates
    @test isapprox(mm.params, [sda^2, sdb^2, sde^2], rtol=0.05, atol=0.05)
    va = variances(mm)
    @test isapprox([va.rowvar, va.colvar, va.unexplainedvar], [sda^2, sdb^2, sde^2], rtol=0.05, atol=0.05)
end
