using LatentGPs
using KernelFunctions
using AbstractGPs
using LinearAlgebra
using Test
using Random
using Functors
using Distributions

@testset "LatentGPs.jl" begin

    include("latent_gp.jl")

    @testset "likelihoods" begin
        include("likelihoods/gaussian.jl")
        include("likelihoods/poisson.jl")
    end

end
