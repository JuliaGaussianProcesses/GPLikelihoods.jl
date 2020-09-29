function test_interface(
    rng::AbstractRNG, lik, k::Kernel, x::AbstractVector; functor_args=(),
)
    gp = GP(k)
    lgp = LatentGP(gp, lik, 1e-5)
    lfgp = lgp(x)

    # Check if likelihood produces a distribution
    @test lik(rand(rng, lfgp.fx)) isa Distribution

    N = length(x)
    y = rand(rng, lfgp.fx)

    if x isa MOInput
        # TODO: replace with mo_inverse_transform
        N = length(x.x)
        y = [y[[i + j*N for j in 0:(x.out_dim - 1)]] for i in 1:N]
    end

    # Check if the likelihood samples are of correct length
    @test length(rand(rng, lik(y))) == N
    
    # Check if functor works properly
    if functor_args == ()
        @test Functors.functor(lik)[1] == functor_args
    else
        @test keys(Functors.functor(lik)[1]) == functor_args
    end
end

"""
    test_interface(lik, k::Kernel, x::AbstractVector; functor_args=())

This function provides unified method to check the interface of the various likelihoods 
defined. It checks if the likelihood produces a distribution, length of likelihood 
samples is correct and if the functor works as intended.  
...
# Arguments
- `lik`: the likelihood to test the interface of
- `k::Kernel`: the kernel to use for the GP
- `x::AbstractVector`: intputs to compute the likelihood on
- `functor_args=()`: a collection of symbols of arguments to match functor parameters with.
...
"""
function test_interface(
    lik,
    k::KernelFunctions.Kernel,
    x::AbstractVector;
    kwargs...
)
    test_interface(Random.GLOBAL_RNG, lik, k, x; kwargs...)
end
