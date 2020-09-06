@testset "GaussianLikelihood" begin
    rng = MersenneTwister(123)
    gp = GP(SqExponentialKernel())
    x = rand(rng, 10)
    y = rand(rng, 10)
    lik = GaussianLikelihood(1e-5)
    lgp = LatentGP(gp, lik, 1e-5)
    lfgp = lgp(x)

    @test typeof(lik(rand(rng, lfgp.fx))) <: Distribution
    @test length(rand(rng, lik(rand(rng, lfgp.fx)))) == 10
    @test keys(Functors.functor(lik)[1]) == (:σ²,)

    # Test default constructore
    l = GaussianLikelihood()
    @test l.σ² == [1e-6]
end
