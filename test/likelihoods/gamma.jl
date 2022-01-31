@testset "GammaLikelihood" begin
    for args in ((), (1.0,), (exp,), (ExpLink(),), (1.0, exp), (1.0, ExpLink()))
        lik = GammaLikelihood(args...)
        @test lik isa GammaLikelihood{ExpLink,Float64}
        @test lik.α == 1
    end

    lik = GammaLikelihood(1.0)
    test_interface(lik, Gamma; functor_args=(:α, :invlink))
end
