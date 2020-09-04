using NonConjugateGPs
using KernelFunctions
using AbstractGPs
using Test
using Random
using Functors
using Distributions

@testset "NonConjugateGPs.jl" begin

    @testset "likelihoods" begin
        include("likelihoods/gaussian.jl")
        include("likelihoods/poisson.jl")
    end

end
