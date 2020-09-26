module GPLikelihoods

using Distributions
using AbstractGPs
using Random
using Functors
using StatsFuns: logistic, softmax

import Distributions

export CategoricalLikelihood,
    GaussianLikelihood, 
    HeteroscedasticGaussianLikelihood, 
    PoissonLikelihood

# Likelihoods
include("likelihoods/categorical.jl")
include("likelihoods/gaussian.jl")
include("likelihoods/poisson.jl")

end # module
