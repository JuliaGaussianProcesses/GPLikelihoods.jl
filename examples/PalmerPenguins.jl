# -*- coding: utf-8 -*-
# # Palmer Penguins

# Import necessery packages and PalmerPenguins dataset.

using Pkg; Pkg.activate("../docs")
using PalmerPenguins, Plots, DataFrames, Random
using KernelFunctions, AbstractGPs, GPLikelihoods
using EllipticalSliceSampling, Distributions, MCMCChains
df = dropmissing(DataFrame(PalmerPenguins.load()))
df = df[randperm(nrow(df)), :]
df[!,:flipper_length_mm] = convert(Vector{Float64}, df[!,:flipper_length_mm])
df[!,:body_mass_g] = convert(Vector{Float64}, df[!,:body_mass_g])
first(df, 5)

# The Palmer penguins dataset consists of measurements of 344 penguins from three islands in the Palmer Archipelago, Antarctica, that were collected by Dr. Kristen Gorman and the Palmer Station, Antarctica LTER (Gorman, Williams, & Fraser (2014)). The simplified version of the dataset contains at most seven measurements for each penguin, namely the species (Adelie, Chinstrap, and Gentoo), the island (Torgersen, Biscoe, and Dream), the bill length (measured in mm), the bill depth (measured in mm), the flipper length (measured in mm), the body mass (measured in g), and the sex (male and female). [Source](https://widmann.dev/blog/2020/07/palmerpenguins/)

# ## Binary classification using Bernoulli likelihood

# In this example we will demonstrate binary classfication on the sex of thee penguins based on their bill length & depth. flipper length and body mass.

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

# A visualization of three of the four input parameters.

# We split the dataset in train and test. Approximately 20% of the dataset is held out as test data.

x_train, y_train = x[1:266], y[1:266]
x_test, y_test = x[267:end], y[267:end];


# Defining log density function. 

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

# # Elliptical Slice Sampling

# We will use [elliptical slice sampling](http://proceedings.mlr.press/v9/murray10a/murray10a.pdf) to infer the kernel parameters along with the outputs. [EllipticalSliceSampling.jl](https://github.com/TuringLang/EllipticalSliceSampling.jl/) gives us a easy to use implementation of this.

prior = MvNormal(2+length(x_train), 1)
samples = sample(ESSModel(prior, ℓ), ESS(), 1_000)
samples_mat = reduce(hcat, samples);

# We visualize the effective sample size (ess) and Gelman-Rubin statistic ($\hat{R}$)

ess_rhat = ess(Chains(samples))
ess_plt = scatter(ess_rhat[:,[:ess]], label="", m=(0.8,:dot,3), color=:black)
hline!(ess_plt, [mean(ess_rhat[:,[:ess]])], label="Mean", yaxis="ess", xaxis="Parameters", color=:red)
rhat_plt = scatter(ess_rhat[:,[:rhat]], label="", m=(0.8,:dot,3), color=:black)
hline!(rhat_plt, [mean(ess_rhat[:,[:rhat]])], label="Mean", yaxis="Rhat", xaxis="Parameters", color=:red)
plot(ess_plt, rhat_plt, size=(600,300),)

mean_params = mean(samples)
ℓ(mean_params)

# A histogram of the posterior samples of the kernel parameters

plt = histogram(samples_mat[1:2,:]'; layout=2, labels= "", bins=10)
vline!(plt, mean_params[1:2]'; layout=2, label="Mean", size=(500,250))

# A helper function to compute the mean posterior predictions of the binary classification task. This gives the probability of a particular penguin being a female.

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

# Train Data Accuracy:

mean(((mean_ys[1:length(y_train)] .> 0.5) .== y_train))

# Test Data Accuracy:

mean(((mean_ys[end-length(y_test)+1:end] .> 0.5) .== y_test))

# A visualzation of the mean of the posterior predictions of the sex of each penguin. 

plt = scatter(
        df[!, :bill_length_mm],
        df[!, :bill_depth_mm],
        df[!, :flipper_length_mm],
        marker_z=mean_ys,
        m = (0.5, :h, 5),
        colorbar=:left
    )
