module LatentGPs

using Distributions
using KernelFunctions
using AbstractGPs
using LinearAlgebra
using Random
using Requires

import Statistics
import Distributions

export LatentGP

export logpdf, rand, trainable

export GaussianLikelihood


include("latent_gp.jl")
include("likelihoods.jl")

function __init__()
    @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" include("trainable.jl")
end


end # module
