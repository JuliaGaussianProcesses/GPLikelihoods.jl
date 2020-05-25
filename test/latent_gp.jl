@testset "latent_gp" begin
    gp = GP(SqExponentialKernel())
    x = rand(10)
    y = rand(10)
    fx = gp(x)

    lik(v) = Product(Normal.(v, 1e-5))
    
    lgp1 = LatentGP(fx, lik)
    @test typeof(lgp1) <: LatentGP
    @test typeof(lgp1.fx) <: AbstractGPs.FiniteGP

    lgp2 = LatentGP(gp, x, lik)
    @test typeof(lgp2) <: LatentGP
    @test typeof(lgp2.fx) <: AbstractGPs.FiniteGP
    @test lgp2.fx.x ≈ x atol=1e-5

end
