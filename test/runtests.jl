using GPLikelihoods
using AbstractGPs
using Test
using Random
using Functors
using Distributions
using StatsFuns

@testset "GPLikelihoods.jl" begin

    include("test_utils.jl")
    include("links.jl")    
    @testset "likelihoods" begin
        include("likelihoods/bernoulli.jl")
        include("likelihoods/categorical.jl")
        include("likelihoods/gaussian.jl")
        include("likelihoods/poisson.jl")
        include("likelihoods/gamma.jl")
        include("likelihoods/exponential.jl")
    end

end
