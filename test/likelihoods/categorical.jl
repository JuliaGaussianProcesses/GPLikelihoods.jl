@testset "CategoricalLikelihood" begin
    Catbij = CategoricalLikelihood{true,SoftMaxLink}
    @test CategoricalLikelihood() isa Catbij
    @test CategoricalLikelihood(; bijective=true) isa Catbij
    @test CategoricalLikelihood(softmax) isa Catbij
    @test CategoricalLikelihood(SoftMaxLink()) isa Catbij
    @test CategoricalLikelihood(softmax; bijective=true) isa Catbij
    @test CategoricalLikelihood(SoftMaxLink(); bijective=true) isa Catbij

    Catnonbij = CategoricalLikelihood{false,SoftMaxLink}
    @test CategoricalLikelihood(; bijective=false) isa Catnonbij
    @test CategoricalLikelihood(softmax; bijective=false) isa Catnonbij
    @test CategoricalLikelihood(SoftMaxLink(); bijective=false) isa Catnonbij
    @test CategoricalLikelihood(softmax; bijective=Val(false)) isa Catnonbij
    @test CategoricalLikelihood(SoftMaxLink(); bijective=Val(false)) isa Catnonbij

    lik_bijective = CategoricalLikelihood()
    OUT_DIM = 4
    test_interface(lik_bijective, Categorical, OUT_DIM)
    lik_nonbijective = CategoricalLikelihood(softmax, false)
    test_interface(lik_nonbijective, Categorical, OUT_DIM)
end
