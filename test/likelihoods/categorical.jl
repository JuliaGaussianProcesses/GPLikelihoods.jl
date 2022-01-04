@testset "CategoricalLikelihood" begin
    for args in ((), (softmax,), (SoftMaxLink(),))
        @test CategoricalLikelihood(args...) isa CategoricalLikelihood{SoftMaxLink}
    end

    lik = CategoricalLikelihood()
    OUT_DIM = 4
    test_interface(lik, Categorical, OUT_DIM)
end
