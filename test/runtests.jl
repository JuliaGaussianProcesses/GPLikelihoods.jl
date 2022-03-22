using GPLikelihoods
using GPLikelihoods.TestInterface: test_interface
using Test
using Random
using Functors
using Distributions
using StatsFuns

@testset "GPLikelihoods.jl" begin
    include("links.jl")
    @testset "likelihoods" begin
        include("likelihoods/bernoulli.jl")
        include("likelihoods/categorical.jl")
        include("likelihoods/gaussian.jl")
        include("likelihoods/poisson.jl")
        include("likelihoods/gamma.jl")
        include("likelihoods/exponential.jl")
        include("likelihoods/negativebinomial.jl")
    end
    @testset "expected_loglik" begin
        include("expected_loglik.jl")
    end
end
