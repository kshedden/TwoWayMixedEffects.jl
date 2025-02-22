using TwoWayMixedEffects
using DataFrames
using LinearAlgebra
using Statistics
using StableRNGs

function gendat(rng, nrow, ncol, sdrow, sdcol, sdresid, ee)

    rowEffects = sdrow * randn(rng, nrow)
    colEffects = sdcol * randn(rng, ncol)

    E = Float64[]
    row, col = Int[], Int[]
    for i in 1:nrow
        for j in 1:ncol
            if rand(rng) < ee
                push!(E, rowEffects[i] + colEffects[j] + sdresid*randn(rng))
                push!(row, i)
                push!(col, j)
            end
        end
    end

    return E, row, col
end

rng = StableRNG(321)

nrow = 100
ncol = 200
sdrow = 1.0
sdcol = 1.0
sdresid = 1.0
ee = 0.1
p = 5

function simstudy_mixed(rng, p, sdrow, sdcol, sdresid, ee; nrep=1000)

    Z = zeros(nrep, p)
    B = zeros(nrep, p)
    beta = randn(rng, p)

    for i in 1:nrep
        E, row, col = gendat(rng, nrow, ncol, sdrow, sdcol, sdresid, ee)
        n = length(E)
        X = randn(rng, n, p)
        Ey = X * beta
        Y = Ey + E
        md = fit(TwoWayMixedEffectsModel, X, Y, row, col; nsketch=1000)
        bhat = coef(md)
        vc = vcov(md)
        Z[i, :] = (bhat - beta) ./ sqrt.(diag(vc))
        B[i, :] = bhat - beta
    end

    return Z, B
end

Z, B = simstudy_mixed(rng, p, sdrow, sdcol, sdresid, ee; nrep=1000)
