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
    GammaLikelihood,
    NegativeBinomialLikelihood
export Link,
    ChainLink,
    BijectiveSimplexLink,
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
export nlatent

# Links
include("links.jl")

# Likelihoods
abstract type AbstractLikelihood end

"""
    nlatent(::AbstractLikelihood)::Int

Returns the number of latent Gaussian processes needed to build the likelihood.
In other terms the input dimensionality passed to the likelihood from the GP perspective.
It is typically 1, but for some likelihoods like [`CategoricalLikelihood`](@ref) or 
[`HeteroscedasticGaussianLikelihood`](@ref) multiple latent GPs are necessary.
"""
nlatent(::AbstractLikelihood) = 1 # Default number of latent GPs required is 1

include("likelihoods/bernoulli.jl")
include("likelihoods/categorical.jl")
include("likelihoods/gaussian.jl")
include("likelihoods/poisson.jl")
include("likelihoods/gamma.jl")
include("likelihoods/exponential.jl")
include("likelihoods/negativebinomial.jl")

# TestInterface module
include("TestInterface.jl")

end # module
