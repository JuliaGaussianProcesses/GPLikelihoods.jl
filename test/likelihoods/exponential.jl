@testset "ExponentialLikelihood" begin
    lik = ExponentialLikelihood()
    test_interface(lik, SqExponentialKernel(), rand(10))
end
