@testset "CategoricalLikelihood" begin
    rng = MersenneTwister(123)
    gp = GP(IndependentMOKernel(SqExponentialKernel()))
    IN_DIM = 3
    OUT_DIM = 4
    N = 10
    x = [rand(rng, IN_DIM) for _=1:N]
    X = MOInput(x, OUT_DIM)
    lik = CategoricalLikelihood()
    lgp = LatentGP(gp, lik, 1e-5)
    lfgp = lgp(X)

    Y = rand(rng, lfgp.fx)
    
    y = [Y[[i + j*N for j in 0:(OUT_DIM - 1)]] for i in 1:N]
    # Replace with mo_inverse_transform once it is merged

    @test lik(y) isa Distribution
    @test length(rand(rng, lik(y))) == 10
    @test Functors.functor(lik)[1] == ()
end
