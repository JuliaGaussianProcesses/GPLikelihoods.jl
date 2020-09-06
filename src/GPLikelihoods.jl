module GPLikelihoods

using Distributions
using AbstractGPs
using Random
using Functors
using StatsFuns: logistic

import Distributions

export GaussianLikelihood, PoissonLikelihood
export Link, LogisticLink

# Links
include(joinpath("likelihoods", "link.jl"))
# Likelihoods
include(joinpath("likelihoods", "gaussian.jl"))
include(joinpath("likelihoods", "poisson.jl"))
end # module
