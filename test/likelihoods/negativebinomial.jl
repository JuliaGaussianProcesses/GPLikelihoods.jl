@testset "NegativeBinomialLikelihood" begin
    for (nbparam, kwarg) in
        ((NBParamI, :successes), (NBParamII, :failures), (NBParamIII, :successes))
        @testset "$(nameof(nbparam))" begin
            @eval begin
                sym_kwarg = $(Meta.quot(kwarg))
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
end
