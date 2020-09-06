using GPLikelihoods
using AbstractGPs
using Test
using Random
using Functors
using Distributions
using StatsFuns

@testset "GPLikelihoods.jl" begin

    @testset "likelihoods" begin
        include(joinpath("likelihoods", "gaussian.jl"))
        include(joinpath("likelihoods", "poisson.jl"))
        include(joinpath("likelihoods", "link.jl"))
    end

end
