@testset "BernoulliLikelihood" begin
    for args in ((), (logistic,), (LogisticLink(),))
        @test BernoulliLikelihood(args...) isa BernoulliLikelihood{LogisticLink}
    end

    lik = BernoulliLikelihood()
    test_interface(lik, Bernoulli)
end
