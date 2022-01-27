@testset "CategoricalLikelihood" begin
    for args in ((), (softmax,), (softmax, true), (SoftMaxLink(),))
        @test CategoricalLikelihood(args...) isa
            CategoricalLikelihood{true,SoftMaxLink}
    end
    @test CategoricalLikelihood(softmax, false) isa
        CategoricalLikelihood{false,SoftMaxLink}

    lik_bijective = CategoricalLikelihood()
    OUT_DIM = 4
    test_interface(lik_bijective, Categorical, OUT_DIM)
    lik_nonbijective = CategoricalLikelihood(softmax, false)
    test_interface(lik_nonbijective, Categorical, OUT_DIM)
end
