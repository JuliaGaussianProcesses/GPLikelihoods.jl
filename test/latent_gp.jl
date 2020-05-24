@testset "latent_gp" begin
    gp = GP(SqExponentialKernel())
    x = rand(10)
    y = rand(10)
    lik = LatentGPs.gaussian_likelihood
    fx = LatentGP(gp, x, lik)

    #TODO: Add tests.
end
