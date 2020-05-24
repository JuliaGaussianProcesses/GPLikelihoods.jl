module LatentGPs

using Distributions
using KernelFunctions
using AbstractGPs
using LinearAlgebra

import Statistics

export LatentGP
export log_density, logpdf, mean, cov
export gaussian


include("latent_gp.jl")
include("likelihoods.jl")


end # module
