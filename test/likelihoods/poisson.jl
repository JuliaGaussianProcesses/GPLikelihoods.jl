@testset "PoissonLikelihood" begin
    lik = PoissonLikelihood()
    test_interface(lik, SqExponentialKernel(), rand(10))
end
