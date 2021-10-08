@testset "PoissonLikelihood" begin
    @test PoissonLikelihood() isa PoissonLikelhood{ExpLink}
    @test PoissonLikelihood(ExpLink()) isa PoissonLikelihood{ExpLink}

    for lik in (PoissonLikelihood(), PoissonLikelihood(log1pexp))
        test_interface(lik, SqExponentialKernel(), rand(10))
    end
end
