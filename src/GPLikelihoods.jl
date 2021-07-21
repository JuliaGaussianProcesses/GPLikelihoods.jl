module GPLikelihoods

using AbstractGPs
using Distributions
using Functors
using LogExpFunctions: logistic, softmax
using Random
using StatsFuns


export BernoulliLikelihood,
    CategoricalLikelihood,
    GaussianLikelihood, 
    HeteroscedasticGaussianLikelihood, 
    PoissonLikelihood,
    ExponentialLikelihood,
    GammaLikelihood
export Link,
    ChainLink,
    ExpLink,
    LogLink,
    InvLink,
    SqrtLink,
    SquareLink,
    LogitLink,
    LogisticLink,
    ProbitLink,
    NormalCDFLink,
    SoftMaxLink

# Links
include("links.jl")

# Likelihoods
include(joinpath("likelihoods", "bernoulli.jl"))
include(joinpath("likelihoods", "categorical.jl"))
include(joinpath("likelihoods", "gaussian.jl"))
include(joinpath("likelihoods", "poisson.jl"))
include(joinpath("likelihoods", "gamma.jl"))
include(joinpath("likelihoods", "exponential.jl"))

end # module
