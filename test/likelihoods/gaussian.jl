@testset "GaussianLikelihood" begin
    lik = GaussianLikelihood(1e-5)
    test_interface(lik, Normal; functor_args=(:σ²,))
end

@testset "HeteroscedasticGaussianLikelihood" begin
    for args in ((), (exp,), (ExpLink(),))
        lik = HeteroscedasticGaussianLikelihood(args...)
        @test lik isa HeteroscedasticGaussianLikelihood{ExpLink}
    end

    lik = HeteroscedasticGaussianLikelihood()
    N = 10
    test_interface(lik, Normal)
    @test nlatent(lik) == 2
end
