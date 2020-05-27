using LatentGPs
using KernelFunctions
using AbstractGPs
using Flux
using LinearAlgebra
using Test
using Random
using Distributions

@testset "LatentGPs.jl" begin

    include("latent_gp.jl")

    @testset "likelihoods" begin
        include("likelihoods/gaussian.jl")
        include("likelihoods/poisson.jl")
    end

    include("trainable.jl")

end
