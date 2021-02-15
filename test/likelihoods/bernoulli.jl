@testset "BernoulliLikelihood" begin
    lik = BernoulliLikelihood()
    test_interface(lik, SqExponentialKernel(), rand(10))
end
