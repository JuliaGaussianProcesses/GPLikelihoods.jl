using PalmerPenguins, Plots
const TABLE = PalmerPenguins.load()

using DataFrames, Random
df = dropmissing(DataFrame(TABLE))
df[randperm(nrow(df)), :]
first(df, 5)

# Binary classification using Bernoulli likelihood - sex

unique(df[:sex])


using KernelFunctions, AbstractGPs, GPLikelihoods
x = RowVecs(Array(df[:, [:bill_length_mm, :bill_depth_mm, :flipper_length_mm, :body_mass_g]]))
y = [sex=="female" ? true : false for sex in df[:sex]]

scatter(
    df[:bill_length_mm],
    df[:bill_depth_mm],
    df[:flipper_length_mm],
    group = df[:species],
    m = (0.5, [:+ :h :star7], 5),
)

x_train, y_train = x[1:266], y[1:266]
x_test, y_test = x[267:end], y[267:end]


k = Matern52Kernel()
f = LatentGP(GP(k), BernoulliLikelihood(), 0.001)
fx = f(x_train)
logpdf(fx, rand(fx))


using EllipticalSliceSampling, Distributions
function logp(params; x=x_train, y=y_train)
    kernel = ScaledKernel(
        KernelFunctions.transform(
            Matern52Kernel(),
            ScaleTransform(exp(params[1]))
        ),
        exp(params[2])
    )
    f = LatentGP(GP(k), BernoulliLikelihood(), 0.001)
    fx = f(x_train)
    return logpdf(fx, rand(fx))
end

prior = MvNormal(2, 1)
logp(rand(prior))

samples = sample(ESSModel(prior, logp), ESS(), 100; progress=true)
samples_mat = reduce(hcat, samples)';

mean_params = mean(samples_mat; dims=1)

plt = histogram(samples_mat; layout=2, labels= "Param")
vline!(plt, mean_params; layout=2, label="Mean")


# Multi class classification using Categorical likelihood - species and/or island
unique(df[:species])

unique(df[:island])
