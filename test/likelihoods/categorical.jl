@testset "CategoricalLikelihood" begin
    lik = CategoricalLikelihood()
    IN_DIM = 3
    OUT_DIM = 2 # one for the mean the other for the log-standard deviation
    N = 10
    X = MOInput([rand(IN_DIM) for _ in 1:N], OUT_DIM)
    test_interface(lik, IndependentMOKernel(SqExponentialKernel()), X)
end
