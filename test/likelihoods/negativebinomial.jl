@testset "NegativeBinomialLikelihood" begin
    for args in
        ((), (1.0,), (logistic,), (LogisticLink(),), (1.0, logistic), (1.0, LogisticLink()))
        lik = NegativeBinomialLikelihood(args...)
        @test lik isa NegativeBinomialLikelihood{Float64,LogisticLink}
        @test lik.r == 1
    end

    lik = NegativeBinomialLikelihood(1.0)
    test_interface(lik, NegativeBinomial; functor_args=(:r, :invlink))
end
