using GPLikelihoods
using GPLikelihoods: GaussHermiteExpectation, MonteCarloExpectation
using GPLikelihoods.TestInterface: test_interface

using Aqua
using Distributions
using Functors
using Random
using StatsFuns
using Test
using Zygote

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
    include("expectations.jl")
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(GPLikelihoods; ambiguities = false)
        Aqua.test_ambiguities([GPLikelihoods, Base, Core]; recursive=false)
    end
end
