module GPLikelihoods

using Distributions
using AbstractGPs
using Random
using Functors
using StatsFuns: logistic, softmax

import Distributions

export BernoulliLikelihood,
    CategoricalLikelihood,
    GaussianLikelihood, 
    HeteroscedasticGaussianLikelihood, 
    PoissonLikelihood,
    ExponentialLikelihood,
    GammaLikelihood

# Likelihoods
include("likelihoods/bernoulli.jl")
include("likelihoods/categorical.jl")
include("likelihoods/gaussian.jl")
include("likelihoods/poisson.jl")
include("likelihoods/gamma.jl")
include("likelihoods/exponential.jl")

end # module
