@testset "NegativeBinomialLikelihood" begin
    for args in ((), (logistic,), (LogisticLink(),)), kwargs in ((), (; successes=1))
        lik = NegativeBinomialLikelihood(args...; kwargs...)
        @test lik isa NegativeBinomialLikelihood{LogisticLink,Int}
        @test lik.successes == 1
    end

    for args in ((normcdf,), (NormalCDFLink(),)), kwargs in ((; successes=2.0),)
        lik = NegativeBinomialLikelihood(args...; kwargs...)
        @test lik isa NegativeBinomialLikelihood{NormalCDFLink,Float64}
        @test lik.successes == 2
    end

    lik = NegativeBinomialLikelihood()
    test_interface(lik, NegativeBinomial; functor_args=(:successes, :invlink))
end
