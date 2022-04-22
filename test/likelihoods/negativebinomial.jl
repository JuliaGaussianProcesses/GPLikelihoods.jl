@testset "NegativeBinomialLikelihood" begin
    rs = (10, 9.5)
    for nbparam in (NBParamI, NBParamII, NBParamIII)
        for r in (10, 9.5) # Test both input types
            @testset "$(nameof(nbparam)), r=$r" begin
                lik = NegativeBinomialLikelihood(nbparam(r), logistic)
                @test lik isa NegativeBinomialLikelihood{<:nbparam}
                test_interface(lik, NegativeBinomial; functor_args=(:params, :invlink))
            end
        end
    end
    struct NBParamFoo <: GPLikelihoods.NBParam end
    @test_throws ErrorException NegativeBinomialLikelihood(NBParamFoo(), logistic)(2.0)
end
