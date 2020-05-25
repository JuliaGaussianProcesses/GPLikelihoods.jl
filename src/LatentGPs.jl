module LatentGPs

using Distributions
using KernelFunctions
using AbstractGPs
using LinearAlgebra
using Random

import Statistics

export LatentGP
export logpdf, rand
export gaussian


include("latent_gp.jl")
include("likelihoods.jl")


end # module
