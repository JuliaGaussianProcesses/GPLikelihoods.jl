@testset "CategoricalLikelihood" begin
    for args in ((), (softmax,), (SoftMaxLink(),))
        @test CategoricalLikelihood(args...) isa
            CategoricalLikelihood{SimplexVariant,SoftMaxLink}
    end
    @test CategoricalLikelihood(softmax, CurvedVariant()) isa
        CategoricalLikelihood{CurvedVariant,SoftMaxLink}

    lik_simplex = CategoricalLikelihood()
    OUT_DIM = 4
    test_interface(lik, Categorical, OUT_DIM)
    lik_curved = CategoricalLikelihood(softmax, CurvedVariant())
    test_interface(lik, Categorical, OUT_DIM)
end
