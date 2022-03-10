@testset "CategoricalLikelihood" begin
    nclass = 4
    @test CategoricalLikelihood(nclass) isa
        CategoricalLikelihood{<:GPLikelihoods.BijectiveSimplexLink}
    @test CategoricalLikelihood(nclass, softmax) isa CategoricalLikelihood{SoftMaxLink}
    @test CategoricalLikelihood(nclass, SoftMaxLink()) isa
        CategoricalLikelihood{SoftMaxLink}

    lik_bijective = CategoricalLikelihood(nclass)
    test_interface(lik_bijective, Categorical)
    @test nlatent(lik_bijective) == nclass - 1
    lik_nonbijective = CategoricalLikelihood(nclass, softmax)
    test_interface(lik_nonbijective, Categorical)
    @test nlatent(lik_nonbijective) == nclass
end
