@testset "NegativeBinomialLikelihood" begin
    for (nbparam, kwarg) in
        ((NBParamI, :successes), (NBParamII, :failures), (NBParamIII, :successes))
        @testset "$(nameof(nbparam))" begin
            @eval begin
                for args in ((logistic,), (LogisticLink(),)), kwargs in ((), (; $(kwarg)=1))
                    lik = NegativeBinomialLikelihood{$(nbparam)}(args...; kwargs...)
                    @test lik isa NegativeBinomialLikelihood{
                        $(nbparam),LogisticLink,<:NamedTuple{<:Any,<:Tuple{Int}}
                    }
                end

                for args in ((normcdf,), (NormalCDFLink(),)), kwargs in ((; $(kwarg)=2.0),)
                    lik = NegativeBinomialLikelihood{$(nbparam)}(args...; kwargs...)
                    @test lik isa NegativeBinomialLikelihood{
                        $(nbparam),NormalCDFLink,<:NamedTuple{<:Any,<:Tuple{Float64}}
                    }
                end

                lik = NegativeBinomialLikelihood{$(nbparam)}()
                test_interface(lik, NegativeBinomial; functor_args=(:params, :invlink))
            end
        end
    end
    struct NBParamFoo <: GPLikelihoods.NBParam end
    function NegativeBinomialLikelihood{NBParamFoo}(l=logistic; bar=1)
        return NegativeBinomialLikelihood{NBParamFoo}((; bar), GPLikelihoods.link(l))
    end
    @test_throws ErrorException NegativeBinomialLikelihood{NBParamFoo}()(2.0)
end
