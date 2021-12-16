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
    IN_DIM = 3
    OUT_DIM = 2 # one for the mean the other for the log-standard deviation
    N = 10
    # X = MOInput([rand(IN_DIM) for _ in 1:N], OUT_DIM)
    test_interface(lik, Normal, 2)
end
