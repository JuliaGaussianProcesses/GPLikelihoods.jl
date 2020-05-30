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

export logpdf, rand, trainable, randf, randy

export GaussianLikelihood, PoissonLikelihood


include("latent_gp.jl")
include("likelihoods/likelihoods.jl")

end # module
