@testset "GaussianLikelihood" begin
    gp = GP(SqExponentialKernel())
    x = rand(10)
    y = rand(10)
    fx = gp(x, 1e-6)
    lik = GaussianLikelihood(first(fx.Σy))
    lgp = LatentGP(fx, lik)
    
    @test typeof(lik(rand(fx))) <: Distribution
    @test length(rand(lik(rand(fx)))) == 10
end
