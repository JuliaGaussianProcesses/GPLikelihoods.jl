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
end

@testset "HeteroscedasticGaussianLikelihood" begin
    rng = MersenneTwister(123)
    gp = GP(IndependentMOKernel(SqExponentialKernel()))
    IN_DIM = 3
    OUT_DIM = 2 # one for the mean the other for the log-standard deviation
    N = 10
    x = [rand(rng, IN_DIM) for _=1:N]
    X = MOInput(x, OUT_DIM)
    lik = HeteroscedasticGaussianLikelihood()
    lgp = LatentGP(gp, lik, 1e-5)
    lfgp = lgp(X)

    Y = rand(rng, lfgp.fx)

    y = [Y[[i + j*N for j in 0:(OUT_DIM - 1)]] for i in 1:N]
    # Replace with mo_inverse_transform once it is merged

    @test typeof(lik(y)) <: Distribution
    @test length(rand(rng, lik(y))) == 10
    @test Functors.functor(lik)[1] == ()
end
