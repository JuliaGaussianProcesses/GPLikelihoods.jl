@testset "GammaLikelihood" begin
    lik = GammaLikelihood(1.0)
    test_interface(lik, SqExponentialKernel(), rand(10); functor_args=(:α, :invlink))
end
