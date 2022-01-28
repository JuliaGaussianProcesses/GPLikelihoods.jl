@testset "NegativeBinomialLikelihood" begin
    for args in ((), (logistic,), (LogisticLink(),)), kwargs in ((), (; r=1))
        lik = NegativeBinomialLikelihood(args...; kwargs...)
        @test lik isa NegativeBinomialLikelihood{LogisticLink,Int}
        @test lik.r == 1
    end

    for args in ((normcdf,), (NormalCDFLink(),)), kwargs in ((; r=2.0),)
        lik = NegativeBinomialLikelihood(args...; kwargs...)
        @test lik isa NegativeBinomialLikelihood{NormalCDFLink,Float64}
        @test lik.r == 2
    end

    lik = NegativeBinomialLikelihood()
    test_interface(lik, NegativeBinomial; functor_args=(:r, :invlink))
end
