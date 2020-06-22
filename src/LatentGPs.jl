module LatentGPs

using Distributions
using KernelFunctions
using AbstractGPs
using LinearAlgebra
using Random
using Functors

import Statistics
import Distributions

export LatentGP

export logpdf, rand

export GaussianLikelihood, PoissonLikelihood


include("latent_gp.jl")

# Likelihoods
include("likelihoods/gaussian.jl")
include("likelihoods/poisson.jl")

end # module
