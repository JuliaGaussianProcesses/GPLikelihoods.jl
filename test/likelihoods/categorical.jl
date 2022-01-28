@testset "CategoricalLikelihood" begin
    @test CategoricalLikelihood() isa CategoricalLikelihood{<:GPLikelihoods.BijectiveSimplexLink}

    @test CategoricalLikelihood(softmax) isa CategoricalLikelihood{SoftMaxLink}
    @test CategoricalLikelihood(SoftMaxLink()) isa CategoricalLikelihood{SoftMaxLink}

    OUT_DIM = 4
    lik_bijective = CategoricalLikelihood()
    test_interface(lik_bijective, Categorical, OUT_DIM)
    lik_nonbijective = CategoricalLikelihood(softmax)
    test_interface(lik_nonbijective, Categorical, OUT_DIM)
end
