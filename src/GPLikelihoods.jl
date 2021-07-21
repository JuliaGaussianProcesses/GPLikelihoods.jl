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
include(joinpath("likelihoods", "bernoulli.jl"))
include(joinpath("likelihoods", "categorical.jl"))
include(joinpath("likelihoods", "gaussian.jl"))
include(joinpath("likelihoods", "poisson.jl"))
include(joinpath("likelihoods", "gamma.jl"))
include(joinpath("likelihoods", "exponential.jl"))

end # module
