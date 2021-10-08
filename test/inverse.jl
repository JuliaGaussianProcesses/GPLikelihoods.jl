@testset "inverse.jl" begin
    @testset "square" begin
        x = randn()
        @test GPLikelihoods.square(x) == x^2
    end

    @testset "inverse" begin
        x = rand()
        for f in (
            exp,
            log,
            inv,
            log1pexp,
            logexpm1,
            sqrt,
            GPLikelihoods.square,
            logit,
            logistic,
            normcdf,
            norminvcdf,
        )
            g = GPLikelihoods.inverse(f)

            # check that definitions are correct
            @test g(f(x)) ≈ x
            @test f(g(x)) ≈ x

            # check that definitions are complete
            @test GPLikelihoods.inverse(g) === f
        end
    end
end
