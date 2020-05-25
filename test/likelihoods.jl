@testset "likelihoods" begin
    @testset "GaussianLikelihood" begin
        gp = GP(SqExponentialKernel())
        x = rand(10)
        y = rand(10)
        fx = gp(x, 1e-6)
        lik = GaussianLikelihood(first(fx.Σy))
        lgp = LatentGP(fx, lik)
        
        @test typeof(logpdf(lik, y, mean(fx))) <: Real
    end
end
