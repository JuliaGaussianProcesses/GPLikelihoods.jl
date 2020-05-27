@testset "trainable" begin
    σ² = 1.0
    function test_params(likelihood, reference)
        params_likelihood = Flux.params(likelihood)
        params_reference = Flux.params(reference)

        @test length(params_likelihood) == length(params_reference)
        @test all(p == q for (p, q) in zip(params_likelihood, params_reference))
    end

    likGaussian = GaussianLikelihood(σ²)
    test_params(likGaussian, ([σ²],))
end
