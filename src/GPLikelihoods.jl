module GPLikelihoods

using Distributions
using AbstractGPs
using Random
using Functors
using StatsFuns: logistic

import Distributions

export BernoulliLikelihood,
    CategoricalLikelihood,
    GaussianLikelihood, 
    HeteroscedasticGaussianLikelihood, 
    PoissonLikelihood
export Link, LogisticLink

# Links
include("links.jl")
# Likelihoods
include("likelihoods/bernoulli.jl")
include("likelihoods/categorical.jl")
include("likelihoods/gaussian.jl")
include("likelihoods/poisson.jl")

end # module
