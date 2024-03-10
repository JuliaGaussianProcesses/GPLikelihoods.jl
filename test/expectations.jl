@testset "expectations" begin
    rng = MersenneTwister(123456)
    q_f = Normal.(zeros(10), ones(10))

    likelihoods_to_test = [
        BernoulliLikelihood(),
        ExponentialLikelihood(),
        GammaLikelihood(),
        GaussianLikelihood(),
        NegativeBinomialLikelihood(NBParamSuccess(1.0)),
        NegativeBinomialLikelihood(NBParamFailure(1.0)),
        NegativeBinomialLikelihood(NBParamI(1.0)),
        NegativeBinomialLikelihood(NBParamII(1.0)),
        PoissonLikelihood(),
    ]

    @testset "testing all analytic implementations" begin
        # Test that we're not missing any analytic implementation in `likelihoods_to_test`!
        implementation_types = [
            (; quadrature=m.sig.types[2], lik=m.sig.types[3]) for
            m in methods(GPLikelihoods.expected_loglikelihood)
        ]
        analytic_likelihoods = [
            m.lik for m in implementation_types if
            m.quadrature == GPLikelihoods.AnalyticExpectation && m.lik != Any
        ]
        for lik_type in analytic_likelihoods
            lik_type_instances = filter(lik -> isa(lik, lik_type), likelihoods_to_test)
            @test !isempty(lik_type_instances)
            lik = first(lik_type_instances)
            @test GPLikelihoods.default_expectation_method(lik) isa
                GPLikelihoods.AnalyticExpectation
        end
    end

    @testset "testing consistency of different expectation methods" begin
        @testset "$(nameof(typeof(lik)))" for lik in likelihoods_to_test
            # Test that the various methods of computing expectations return the same
            # result.
            methods = [
                GaussHermiteExpectation(100),
                MonteCarloExpectation(1e7),
                GPLikelihoods.DefaultExpectationMethod(),
            ]
            def = GPLikelihoods.default_expectation_method(lik)
            if def isa GPLikelihoods.AnalyticExpectation
                push!(methods, def)
            end
            y = rand.(rng, lik.(zeros(10)))

            results = map(
                m -> GPLikelihoods.expected_loglikelihood(m, lik, q_f, y), methods
            )
            @test all(x -> isapprox(x, results[end]; atol=1e-6, rtol=1e-3), results)
        end
    end

    @testset "testing return types and type stability" begin
        @test GPLikelihoods.expected_loglikelihood(
            MonteCarloExpectation(1), GaussianLikelihood(), q_f, zeros(10)
        ) isa Real
        @test GPLikelihoods.expected_loglikelihood(
            GaussHermiteExpectation(1), GaussianLikelihood(), q_f, zeros(10)
        ) isa Real
        @test GPLikelihoods.default_expectation_method(θ -> Normal(0, θ)) isa
            GaussHermiteExpectation

        @testset "$(nameof(typeof(lik)))" for lik in likelihoods_to_test
            # Test that `expectec_loglikelihood` is type-stable
            y = rand.(rng, lik.(zeros(10)))
            for method in [
                MonteCarloExpectation(100),
                GaussHermiteExpectation(100),
                GPLikelihoods.DefaultExpectationMethod(),
            ]
                @test (@inferred expected_loglikelihood(method, lik, q_f, y)) isa Real
            end
        end
    end

    # see https://github.com/JuliaGaussianProcesses/ApproximateGPs.jl/issues/82
    @testset "testing Zygote compatibility with GaussHermiteExpectation" begin
        N = 10
        gh = GaussHermiteExpectation(12)
        μs = randn(rng, N)
        σs = rand(rng, N)

        # Test differentiation with variational parameters
        for lik in likelihoods_to_test
            y = rand.(rng, lik.(rand.(Normal.(μs, σs))))
            gμ, glogσ = Zygote.gradient(μs, log.(σs)) do μ, logσ
                GPLikelihoods.expected_loglikelihood(gh, lik, Normal.(μ, exp.(logσ)), y)
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
                    gh, GaussianLikelihood(exp(x)), Normal.(μs, σs), y
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
                    gh, GammaLikelihood(exp(x)), Normal.(μs, σs), y
                )
            end,
        )
        @test isfinite(glogα)
    end

    @testset "non-constant likelihood" begin
        @testset "$(nameof(typeof(liks[1])))" for liks in (
            NegativeBinomialLikelihood.(NBParamII.(rand(10))),
        )
            # Test that the various methods of computing expectations return the same
            # result.
            methods = [
                GaussHermiteExpectation(100),
                MonteCarloExpectation(1e7),
                GPLikelihoods.DefaultExpectationMethod(),
            ]
            y = [rand(rng, lik(0.)) for lik in liks]

            results = map(
                m -> GPLikelihoods.expected_loglikelihood(m, liks, q_f, y), methods
            )
            @test all(x -> isapprox(x, results[end]; atol=1e-6, rtol=1e-3), results)
        end
    end
end
