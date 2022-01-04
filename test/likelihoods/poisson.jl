@testset "PoissonLikelihood" begin
    for args in ((), (exp,), (ExpLink(),))
        @test PoissonLikelihood(args...) isa PoissonLikelihood{ExpLink}
    end

    for lik in (PoissonLikelihood(), PoissonLikelihood(log1pexp))
        test_interface(lik, Poisson)
    end
end
