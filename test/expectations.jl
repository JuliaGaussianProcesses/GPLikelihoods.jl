@testset "expected_loglikelihood" begin
    # Test that the various methods of computing expectations return the same
    # result.
    rng = MersenneTwister(123456)
    q_f = Normal.(zeros(10), ones(10))

    likelihoods_to_test = [
        ExponentialLikelihood(),
        GammaLikelihood(),
        PoissonLikelihood(),
        GaussianLikelihood(),
    ]

    @testset "testing all analytic implementations" begin
        # Test that we're not missing any analytic implementation in `likelihoods_to_test`!
        implementation_types = [
            (; quadrature=m.sig.types[2], lik=m.sig.types[5]) for
            m in methods(GPLikelihoods.expected_loglikelihood)
        ]
        analytic_likelihoods = [
            m.lik for m in implementation_types if
            m.quadrature == GPLikelihoods.Analytic && m.lik != Any
        ]
        for lik_type in analytic_likelihoods
            @test any(lik isa lik_type for lik in likelihoods_to_test)
        end
    end

    @testset "$(nameof(typeof(lik)))" for lik in likelihoods_to_test
        methods = [GaussHermite(100), MonteCarlo(1e7)]
        def = GPLikelihoods._default_quadrature(lik)
        if def isa Analytic
            push!(methods, def)
        end
        y = rand.(rng, lik.(zeros(10)))

        results = map(m -> GPLikelihoods.expected_loglikelihood(m, y, q_f, lik), methods)
        @test all(x -> isapprox(x, results[end]; atol=1e-6, rtol=1e-3), results)
    end

    @test GPLikelihoods.expected_loglikelihood(
        MonteCarlo(), zeros(10), q_f, GaussianLikelihood()
    ) isa Real
    @test GPLikelihoods.expected_loglikelihood(
        GaussHermite(), zeros(10), q_f, GaussianLikelihood()
    ) isa Real
    @test GPLikelihoods._default_quadrature(θ -> Normal(0, θ)) isa GaussHermite

    @testset "testing Zygote compatibility with GaussHermite" begin # see https://github.com/JuliaGaussianProcesses/ApproximateGPs.jl/issues/82
        N = 10
        gh = GaussHermite(12)
        μs = randn(rng, N)
        σs = rand(rng, N)
        # Test differentiation with variational parameters
        for lik in likelihoods_to_test
            y = rand.(rng, lik.(rand.(Normal.(μs, σs))))
            gμ, glogσ = Zygote.gradient(μs, log.(σs)) do μ, logσ
                GPLikelihoods.expected_loglikelihood(gh, y, Normal.(μ, exp.(logσ)), lik)
            end
            @test all(isfinite, gμ)
            @test all(isfinite, glogσ)
        end
        # Test differentiation with likelihood parameters
        # Test GaussianLikelihood parameter
        σ = 1.0
        y = randn(rng, N)
        glogσ = only(
            Zygote.gradient(log(σ)) do x
                GPLikelihoods.expected_loglikelihood(
                    gh, y, Normal.(μs, σs), GaussianLikelihood(exp(x))
                )
            end,
        )
        @test isfinite(glogσ)
        # Test GammaLikelihood parameter
        α = 2.0
        y = rand.(rng, Gamma.(α, rand(N)))
        glogα = only(
            Zygote.gradient(log(α)) do x
                GPLikelihoods.expected_loglikelihood(
                    gh, y, Normal.(μs, σs), GammaLikelihood(exp(x))
                )
            end,
        )
        @test isfinite(glogα)
    end
end
