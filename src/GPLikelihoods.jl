module GPLikelihoods

using Distributions
using AbstractGPs
using Random
using Functors

import Distributions

export GaussianLikelihood, 
    HeteroscedasticGaussianLikelihood, 
    PoissonLikelihood

# Likelihoods
include("likelihoods/gaussian.jl")
include("likelihoods/poisson.jl")

end # module
