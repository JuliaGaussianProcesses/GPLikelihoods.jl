@testset "NegBinomialLikelihood" begin
    for args in ((), (1.0,), (logistic,), (LogisticLink(),), (1.0, logistic), (1.0, LogisticLink()))
        lik = NegBinomialLikelihood(args...)
        @test lik isa NegBinomialLikelihood{Float64,LogisticLink}
        @test lik.r == 1
    end

    lik = NegBinomialLikelihood(1.0)
    test_interface(lik, NegativeBinomial; functor_args=(:r, :invlink))
end