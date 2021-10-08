@testset "PoissonLikelihood" begin
    for args in ((), (exp,), (ExpLink(),))
        @test PoissonLikelihood(args...) isa PoissonLikelhood{ExpLink}
    end

    for lik in (PoissonLikelihood(), PoissonLikelihood(log1pexp))
        test_interface(lik, SqExponentialKernel(), rand(10))
    end
end
