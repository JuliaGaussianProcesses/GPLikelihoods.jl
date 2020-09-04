module GPLikelihoods

using Distributions
using KernelFunctions
using AbstractGPs
using Random
using Functors

import Distributions

export GaussianLikelihood, PoissonLikelihood

# Likelihoods
include("likelihoods/gaussian.jl")
include("likelihoods/poisson.jl")

end # module
