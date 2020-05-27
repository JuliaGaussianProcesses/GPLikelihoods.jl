@testset "PoissonLikelihood" begin
    gp = GP(SqExponentialKernel())
    x = rand(10)
    y = rand(10)
    fx = gp(x, 1e-6)
    lik = PoissonLikelihood()
    lgp = LatentGP(fx, lik)
    
    @test typeof(lik(rand(fx))) <: Distribution
    @test length(rand(lik(rand(fx)))) == 10
end