using MixedModels
using DataFrames
using LinearAlgebra
using TwoWayMixedEffects
using Statistics
using StableRNGs

function gendat(rng, nrow, ncol, sdrow, sdcol, sdresid, ee)

    rowEffects = sdrow * randn(rng, nrow)
    colEffects = sdcol * randn(rng, ncol)

    Y = Float64[]
    row, col = Int[], Int[]
    for i in 1:nrow
        for j in 1:ncol
            if rand(rng) < ee
                push!(Y, rowEffects[i] + colEffects[j] + sdresid*randn())
                push!(row, i)
                push!(col, j)
            end
        end
    end

    return Y, row, col
end

rng = StableRNG(123)

nrow = 200
ncol = 400
sdrow = 2.0
sdcol = 3.0
sdresid = 1.0
ee = 0.5

Y, row, col = gendat(rng, nrow, ncol, sdrow, sdcol, sdresid, ee)
da = DataFrame(Y=Y, row=row, col=col)

ml = fit(MixedModel, @formula(Y ~ (1 | row) + (1 | col)), da)
mm = fit(TwoWayVarianceComponentsModel, row, col, Y)

function time_mle(nrep=20)
    ti = time()
    for i in 1:nrep
        fit(MixedModel, @formula(Y ~ (1 | row) + (1 | col)), da)
    end
    return (time() - ti) / nrep
end

function time_mom(nrep=20)
    ti = time()
    for i in 1:nrep
        fit(TwoWayVarianceComponentsModel, row, col, Y)
    end
    return (time() - ti) / nrep
end

function run_mle(; nrep=100)
    params = zeros(nrep, 3)
    for i in 1:nrep
        Y, row, col = gendat(rng, nrow, ncol, sdrow, sdcol, sdresid, ee)
        da = DataFrame(Y=Y, row=row, col=col)
        m = fit(MixedModel, @formula(Y ~ (1 | row) + (1 | col)), da)
        vc = VarCorr(m).σρ
        # All three parameters are standard deviations.
        params[i, :] = [vc[:row][:σ][1], vc[:col][:σ][1], sqrt(varest(m))]
    end
    return params
end

function run_mom(; nrep=1000)
    params = zeros(nrep, 3)
    params_se = zeros(nrep, 3)
    for i in 1:nrep
        Y, row, col = gendat(rng, nrow, ncol, sdrow, sdcol, sdresid, ee)
        da = DataFrame(Y=Y, row=row, col=col)
        m = fit(TwoWayVarianceComponentsModel, row, col, Y)
        params[i, :] = sqrt.(m.params)
        params_se[i, :] = sqrt.(clamp.(diag(vcov(m)), 0, Inf))
    end
    return params, params_se
end

tl = time_mle()
tm = time_mom()

params_mle = run_mle(; nrep=100)
params_mom, params_mom_se = run_mom(; nrep=1000)
