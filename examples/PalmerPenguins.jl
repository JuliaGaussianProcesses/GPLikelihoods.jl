# -*- coding: utf-8 -*-
using Pkg; Pkg.activate("../docs")
using PalmerPenguins, Plots, DataFrames, Random
using KernelFunctions, AbstractGPs, GPLikelihoods
using EllipticalSliceSampling, Distributions, MCMCChains
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
    m = (0.5, :h, 5),
)

x_train, y_train = x[1:266], y[1:266]
x_test, y_test = x[267:end], y[267:end];


# logpdf without any parameters

k = SqExponentialKernel()
f = LatentGP(GP(k), BernoulliLikelihood(), 0.1)
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

samples = sample(ESSModel(prior, ℓ), ESS(), 1_000; progress=true)
samples_mat = reduce(hcat, samples);

ess_rhat = ess(Chains(samples))
ess_plt = scatter(ess_rhat[:,[:ess]], label="", m=(0.8,:dot,3), color=:black)
hline!(ess_plt, [mean(ess_rhat[:,[:ess]])], label="Mean", yaxis="ESS", xaxis="Parameters", color=:red)
rhat_plt = scatter(ess_rhat[:,[:rhat]], label="", m=(0.8,:dot,3), color=:black)
hline!(rhat_plt, [mean(ess_rhat[:,[:rhat]])], label="Mean", yaxis="Rhat", xaxis="Parameters", color=:red)
plot(ess_plt, rhat_plt, size=(600,300),)

mean_params = mean(samples)
ℓ(mean_params)

plt = histogram(samples_mat[1:2,:]'; layout=2, labels= "Param", bins=10)
vline!(plt, mean_params[1:2]'; layout=2, label="Mean", size=(500,250))

function posterior_mean(posterior_params)
    ys = [
        begin
            kernel = ScaledKernel(
                    KernelFunctions.transform(
                        SqExponentialKernel(),
                        ScaleTransform(exp(params[1]))
                    ),
                    exp(params[2])
                )
            p_fx = posterior(GP(kernel)(x_train), params[3:end])
            l_p_fx = LatentGP(p_fx, BernoulliLikelihood(), 0.1)
            (_, y) = rand(l_p_fx(x))
            y
        end
        for params in eachcol(posterior_params)
    ]
    return mean(ys)
end
mean_ys = posterior_mean(samples_mat);

# train data accuracy
mean(((mean_ys[1:length(y_train)] .> 0.5) .== y_train))

# test data accuracy
mean(((mean_ys[end-length(y_test)+1:end] .> 0.5) .== y_test))

plt = scatter(
        df[!, :bill_length_mm],
        df[!, :bill_depth_mm],
        df[!, :flipper_length_mm],
        marker_z=mean_ys,
        m = (0.5, :h, 5),
        colorbar=:left
    )


