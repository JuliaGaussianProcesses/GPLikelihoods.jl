module GPLikelihoods

using Distributions
using Functors
using InverseFunctions: InverseFunctions
using LinearAlgebra
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
abstract type AbstractLikelihood end
include("likelihoods/bernoulli.jl")
include("likelihoods/categorical.jl")
include("likelihoods/gaussian.jl")
include("likelihoods/poisson.jl")
include("likelihoods/gamma.jl")
include("likelihoods/exponential.jl")

end # module
