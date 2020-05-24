@testset "likelihoods" begin
    @testset "gaussian" begin
        f = GP(SqExponentialKernel())
        x = rand(10)
        y = rand(10)
        fx = LatentGP(f, x, 0.1)
        
        @test gaussian(fx, y) ≈ AbstractGPs.logpdf(AbstractGPs.FiniteGP(f, x, 0.1), y) atol=1e-5
    end
end
