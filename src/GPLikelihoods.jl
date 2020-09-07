module GPLikelihoods

using Distributions
using AbstractGPs
using Random
using Functors

import Distributions

export BernoulliLikelihood, GaussianLikelihood, PoissonLikelihood

# Likelihoods
include("likelihoods/bernoulli.jl")
include("likelihoods/gaussian.jl")
include("likelihoods/poisson.jl")

end # module
