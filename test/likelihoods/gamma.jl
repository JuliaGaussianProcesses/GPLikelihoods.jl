@testset "GammaLikelihood" begin
    for args in ((), (1.0,), (exp,), (ExpLink(),), (1.0, exp), (1.0, ExpLink()))
        lik = GammaLikelihood(args...)
        @test lik isa GammaLikelihood{Float64,ExpLink}
        @test lik.α == 1
    end

    lik = GammaLikelihood(1.0)
    test_interface(lik, SqExponentialKernel(), rand(10); functor_args=(:α, :invlink))
end
