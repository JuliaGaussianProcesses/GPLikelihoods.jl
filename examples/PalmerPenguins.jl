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


x = RowVecs(Array(df[:, [
                :bill_length_mm, 
                :bill_depth_mm, 
                :flipper_length_mm, 
                :body_mass_g
                ]]))
y = df[!,:sex] .== "female"
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
    return logpdf(fx, (f=params[3:end], y=y))
end

prior = MvNormal(2+length(x_train), 1)
ℓ(rand(prior)) # sanity check

# +
# contour(0:0.5:10, 0:0.5:10, (x, y) -> ℓ([x,y]))

# +
# (f_, y_) = rand(fx)
# plt = plot()
# plot!(plt, f_)
# scatter!(plt, y_.-0.5)
# -

samples = sample(ESSModel(prior, ℓ), ESS(), 1_000; progress=true)
samples_mat = reduce(hcat, samples)';

mean_params = mean(samples_mat; dims=1)

ℓ(mean_params)

plt = histogram(samples_mat[:, 1:2]; layout=2, labels= "Param", bins=10)
vline!(plt, mean_params[:, 1:2]; layout=2, label="Mean")

function posterior_mean(posterior_params)
    ys = Vector(undef, size(posterior_params, 1))
    for i in 1:size(posterior_params, 1)
        kernel = ScaledKernel(
                KernelFunctions.transform(
                    SqExponentialKernel(),
                    ScaleTransform(exp(posterior_params[i,1]))
                ),
                exp(posterior_params[i,2])
            )
        p_fx = posterior(GP(kernel)(x_train), posterior_params[i,3:end])
        l_p_fx = LatentGP(p_fx, BernoulliLikelihood(), 0.01)
        (_, ys[i]) = rand(l_p_fx(x))
    end
    mean(ys)
end

mean_ys = posterior_mean(samples_mat);

gr()
plt = scatter(
        df[!, :bill_length_mm],
        df[!, :bill_depth_mm],
        df[!, :flipper_length_mm],
        marker_z=mean_ys,
        m = (0.5, :h, 5),
        colorbar=:left
    )

# Accuracy
mean(((mean_ys .> 0.5) .== y))

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
