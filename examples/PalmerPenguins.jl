# -*- coding: utf-8 -*-
using Pkg; Pkg.activate("../docs")
using PalmerPenguins, Plots, DataFrames, Random, StatsPlots
using KernelFunctions, AbstractGPs, GPLikelihoods
using EllipticalSliceSampling, Distributions
df = dropmissing(DataFrame(PalmerPenguins.load()))
df = df[randperm(nrow(df)), :]
df[!,:flipper_length_mm] = convert(Vector{Float64}, df[!,:flipper_length_mm])
df[!,:body_mass_g] = convert(Vector{Float64}, df[!,:body_mass_g])
first(df, 5)

# ## Binary classification using Bernoulli likelihood

unique(df[!,:sex])


x = RowVecs(Array(df[:, [:bill_length_mm, :bill_depth_mm, :flipper_length_mm, :body_mass_g]]))
y = y = df[!,:sex] .== "female"
scatter(
    df[!, :bill_length_mm],
    df[!, :bill_depth_mm],
    df[!, :flipper_length_mm],
    group = df[!, :sex],
    m = (0.5, [:+ :h :star7], 5),
)

x_train, y_train = x[1:266], y[1:266]
x_test, y_test = x[267:end], y[267:end];


# logpdf without any parameters

k = SqExponentialKernel()
f = LatentGP(GP(k), BernoulliLikelihood(), 0.001)
fx = f(x_test)
logpdf(fx, (f=rand(fx.fx), y=y_test))


function ℓ(params; x=x_train, y=y_train)
    kernel = ScaledKernel(
        KernelFunctions.transform(
            SqExponentialKernel(),
            ScaleTransform(exp(params[1]))
        ),
        exp(params[2])
    )
    f = LatentGP(GP(kernel), BernoulliLikelihood(), 0.1)
    fx = f(x)
#     return mean(logpdf(fx, (f=rand(fx.fx), y=y)) for _ in 1:1)
    return logpdf(fx, (f=rand(fx.fx), y=y))
end

contour(-4:0.1:4, -4:0.1:4, (x, y) -> ℓ([x,y]))

prior = MvNormal(2, 2)
ℓ(rand(prior)) # sanity check

samples = sample(ESSModel(prior, ℓ), ESS(), 10; progress=true)
samples_mat = reduce(hcat, samples)';

mean_params = mean(samples_mat; dims=1)

plt = histogram(samples_mat; layout=2, labels= "Param")
vline!(plt, mean_params; layout=2, label="Mean")

# +
# Multi class classification using Categorical likelihood - species and/or island
# -

scatter(
    df[:bill_length_mm],
    df[:bill_depth_mm],
    df[:flipper_length_mm],
    group = df[:species],
    m = (0.5, [:+ :h :star7], 5),
)
