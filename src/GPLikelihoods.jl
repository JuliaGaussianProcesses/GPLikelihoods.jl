module GPLikelihoods

using Distributions
using AbstractGPs
using Random
using Functors
using StatsFuns: logistic

import Distributions

export GaussianLikelihood, PoissonLikelihood

# Likelihoods
include(joinpath("likelihoods", "gaussian.jl"))
include(joinpath("likelihoods", "poisson.jl"))
include(joinpath("likelihoods", "link.jl"))
end # module
