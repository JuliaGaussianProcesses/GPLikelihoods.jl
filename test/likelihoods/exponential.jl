@testset "ExponentialLikelihood" begin
    for args in ((), (exp,), (ExpLink(),))
        @test ExponentialLikelihood(args...) isa ExponentialLikelihood{ExpLink}
    end

    lik = ExponentialLikelihood()
    test_interface(lik, Exponential)
end
