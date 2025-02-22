module TwoWayMixedEffects

    import StatsAPI: fit, coef, vcov, RegressionModel
    import Base: show

    using Printf, LinearAlgebra, Statistics, SparseArrays, StatsBase, PrettyTables

    export fit, TwoWayVarianceComponentsModel, TwoWayMixedEffectsModel
    export vcov, variances, coef
    export show

    include("varcomp.jl")
    include("mixed.jl")
end
