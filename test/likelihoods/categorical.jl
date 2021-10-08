@testset "CategoricalLikelihood" begin
    for args in ((), (softmax,), (SoftMaxLink(),))
        @test CategoricalLikelihood(args...) isa CategoricalLikelihood{SoftMaxLink}
    end

    lik = CategoricalLikelihood()
    IN_DIM = 3
    OUT_DIM = 4
    N = 10
    X = MOInput([rand(IN_DIM) for _ in 1:N], OUT_DIM)
    test_interface(lik, IndependentMOKernel(SqExponentialKernel()), X)
end
