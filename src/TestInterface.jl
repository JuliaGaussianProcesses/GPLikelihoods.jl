module TestInterface

using Functors
using Random
using Test

function test_interface(rng::AbstractRNG, lik, out_dist, D_in=nlatent(lik); functor_args=())
    N = 10
    T = Float64 # TODO test Float32 as well
    f, fs = if D_in == 1
        f = randn(rng, T)
        fs = randn(rng, T, 10)
        f, fs
    else
        f = randn(rng, T, D_in)
        fs = [randn(rng, T, D_in) for _ in 1:N]
        f, fs
    end
    # Check if likelihood produces the correct distribution
    # and is sampleable
    @test lik(f) isa out_dist
    @test_nowarn rand(rng, lik(f))

    # Check if the likelihood samples are of correct length
    @test length(rand(rng, lik(fs))) == N

    # Check if functor works properly
    xs, re = Functors.functor(lik)
    @test lik == re(xs)
    if isempty(functor_args)
        @test xs === ()
    else
        @test keys(xs) == functor_args
    end

    return nothing
end

@doc raw"""
    test_interface([rng::AbstractRNG,] lik, out_dist; functor_args=())

This function provides unified method to check the interface of the various likelihoods 
defined. It checks if the likelihood produces the correct distribution, length of likelihood 
samples is correct and if the functor works as intended.

...
## Arguments
- `lik`: the likelihood to test the interface of
- `out_dist::Type{<:Distribution}`: the type of distribution the likelihood should return
- `D_in::Int=1` : The input dimension of the likelihood

## Keyword arguments
- `functor_args=()`: a collection of symbols of arguments to match functor parameters with.
...
"""
function test_interface(lik, out_dist, D_in=nlatent(lik); kwargs...)
    return test_interface(Random.GLOBAL_RNG, lik, out_dist, D_in; kwargs...)
end

end
