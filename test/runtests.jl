using GPLikelihoods
using AbstractGPs
using Test
using Random
using Functors
using Distributions

@testset "GPLikelihoods.jl" begin

    include("test_utils.jl")
    
    @testset "likelihoods" begin
        include("likelihoods/bernoulli.jl")
        include("likelihoods/categorical.jl")
        include("likelihoods/gaussian.jl")
        include("likelihoods/poisson.jl")
    end

end
