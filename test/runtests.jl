using LatentGPs
using KernelFunctions
using AbstractGPs
using LinearAlgebra
using Test

@testset "LatentGPs.jl" begin
    # Write your own tests here.

    include("latent_gp.jl")
    include("likelihoods.jl")
end
