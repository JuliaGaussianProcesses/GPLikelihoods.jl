"""
    Likelihood

An abstract type for likelihoods. Likelihoods are used to model the uncertainities associated
with the data.
"""
abstract type Likelihood end

include("gaussian.jl")
include("poisson.jl")
