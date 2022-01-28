@testset "NegativeBinomialLikelihood" begin
    for args in
        ((), (1,), (logistic,), (LogisticLink(),), (1, logistic), (1, LogisticLink()))
        lik = NegativeBinomialLikelihood(args...)
        @test lik isa NegativeBinomialLikelihood{LogisticLink,Int}
        @test lik.r == 1
    end

    lik = NegativeBinomialLikelihood(1.0)
    test_interface(lik, NegativeBinomial; functor_args=(:r, :invlink))
end
