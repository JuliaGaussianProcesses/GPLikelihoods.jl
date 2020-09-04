using GPLikelihoods
using KernelFunctions
using AbstractGPs
using Test
using Random
using Functors
using Distributions

@testset "GPLikelihoods.jl" begin

    @testset "likelihoods" begin
        include("likelihoods/gaussian.jl")
        include("likelihoods/poisson.jl")
    end

end
