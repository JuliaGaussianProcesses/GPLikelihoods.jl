@testset "NegativeBinomialLikelihood" begin
    # Test based on p success
    for nbparam in (NBParamSuccess, NBParamFailure)
        for r in (10, 9.5) # Test both input types
            @testset "$(nameof(nbparam)), r=$r" begin
                lik = NegativeBinomialLikelihood(nbparam(r))
                @test lik isa NegativeBinomialLikelihood{<:nbparam,LogisticLink}
                lik = NegativeBinomialLikelihood(nbparam(r), logistic)
                @test lik isa NegativeBinomialLikelihood{<:nbparam,LogisticLink}
                test_interface(lik, NegativeBinomial; functor_args=(:params, :invlink))
            end
        end
    end
    # Test based on mean = link(f)
    for (nbparam, args) in
        ((NBParamI, (2.345,)), (NBParamII, (3.456,)), (NBParamPower, (2.34, 3.21)))
        @testset "$(nameof(nbparam))" begin
            lik = NegativeBinomialLikelihood(nbparam(args...))
            @test lik isa NegativeBinomialLikelihood{<:nbparam,ExpLink}
            lik = NegativeBinomialLikelihood(nbparam(args...), exp)
            @test lik isa NegativeBinomialLikelihood{<:nbparam,ExpLink}
            x = randn()
            @test mean(lik(x)) ≈ exp(x)
            test_interface(lik, NegativeBinomial; functor_args=(:params, :invlink))
        end
    end
    struct NBParamFoo <: GPLikelihoods.NBParam end
    @test_throws ErrorException NegativeBinomialLikelihood(NBParamFoo(), logistic)(2.0)
end
