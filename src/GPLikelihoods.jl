module GPLikelihoods

using Distributions
using AbstractGPs
using Random
using Functors
using StatsFuns: logistic

import Distributions

export BernoulliLikelihood,
    GaussianLikelihood, 
    HeteroscedasticGaussianLikelihood, 
    PoissonLikelihood

# Likelihoods
include("likelihoods/bernoulli.jl")
include("likelihoods/gaussian.jl")
include("likelihoods/poisson.jl")

end # module
