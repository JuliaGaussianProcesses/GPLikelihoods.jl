using GPLikelihoods
using AbstractGPs
using Test
using Random
using Functors
using Distributions

@testset "GPLikelihoods.jl" begin

    include("test_utils.jl")
    
    @testset "likelihoods" begin
        include(joinpath("likelihoods", "bernoulli.jl"))
        include(joinpath("likelihoods", "categorical.jl"))
        include(joinpath("likelihoods", "gaussian.jl"))
        include(joinpath("likelihoods", "poisson.jl"))
        include(joinpath("likelihoods", "gamma.jl"))
        include(joinpath("likelihoods", "exponential.jl"))
    end

end
