using FastGaussQuadrature: gausshermite
using SpecialFunctions: loggamma
using ChainRulesCore: ChainRulesCore
using IrrationalConstants: sqrt2, invsqrtπ

struct DefaultExpectationMethod end

struct AnalyticExpectation end

struct GaussHermiteExpectation
    xs::Vector{Float64}
    ws::Vector{Float64}
end
GaussHermiteExpectation(n::Integer) = GaussHermiteExpectation(gausshermite(n)...)

ChainRulesCore.@non_differentiable gausshermite(n)

struct MonteCarloExpectation
    n_samples::Int
end

default_expectation_method(_) = GaussHermiteExpectation(20)

"""
    expected_loglikelihood(
        quadrature,
        lik,
        q_f::AbstractVector{<:Normal},
        y::AbstractVector,
    )

This function computes the expected log likelihood:

```math
    ∫ q(f) log p(y | f) df
```
where `p(y | f)` is the process likelihood. This is described by `lik`, which should be a
callable that takes `f` as input and returns a Distribution over `y` that supports
`loglikelihood(lik(f), y)`.

`q(f)` is an approximation to the latent function values `f` given by:
```math
    q(f) = ∫ p(f | u) q(u) du
```
where `q(u)` is the variational distribution over inducing points.
The marginal distributions of `q(f)` are given by `q_f`.

`quadrature` determines which method is used to calculate the expected log
likelihood.

# Extended help

`q(f)` is assumed to be an `MvNormal` distribution and `p(y | f)` is assumed to
have independent marginals such that only the marginals of `q(f)` are required.
"""
expected_loglikelihood(quadrature, lik, q_f, y)

"""
    expected_loglikelihood(::DefaultExpectationMethod, lik, q_f::AbstractVector{<:Normal}, y::AbstractVector)

The expected log likelihood, using the default quadrature method for the given likelihood.
(The default quadrature method is defined by `default_expectation_method(lik)`, and should
be the closed form solution if it exists, but otherwise defaults to Gauss-Hermite
quadrature.)
"""
function expected_loglikelihood(
    ::DefaultExpectationMethod, lik, q_f::AbstractVector{<:Normal}, y::AbstractVector
)
    quadrature = default_expectation_method(lik)
    return expected_loglikelihood(quadrature, lik, q_f, y)
end

function expected_loglikelihood(
    mc::MonteCarloExpectation, lik, q_f::AbstractVector{<:Normal}, y::AbstractVector
)
    # take `n_samples` reparameterised samples
    r = randn(typeof(mean(first(q_f))), length(q_f), mc.n_samples)
    lls = _mc_exp_loglikelihood_kernel.(_maybe_ref(lik), q_f, y, r)
    return sum(lls) / mc.n_samples
end

function _mc_exp_loglikelihood_kernel(lik, q_f, y, r)
    f = mean(q_f) + std(q_f) * r
    return loglikelihood(lik(f), y)
end

# Compute the expected_loglikelihood over a collection of observations and marginal distributions
function expected_loglikelihood(
    gh::GaussHermiteExpectation, lik, q_f::AbstractVector{<:Normal}, y::AbstractVector
)
    # Compute the expectation via Gauss-Hermite quadrature
    # using a reparameterisation by change of variable
    # (see e.g. en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature)
    # PR #90 introduces eager instead of lazy broadcast over observations
    # and Gauss-Hermit points and weights in order to make the function
    # type stable. Compared to other type stable implementations, e.g.
    # using a custom two-argument pairwise sum, this is faster to
    # differentiate using Zygote.
    A = _gh_exp_loglikelihood_kernel.(_maybe_ref(lik), q_f, y, gh.xs', gh.ws')
    return invsqrtπ * sum(A)
end

function _gh_exp_loglikelihood_kernel(lik, q_f, y, x, w)
    return loglikelihood(lik(sqrt2 * std(q_f) * x + mean(q_f)), y) * w
end

function expected_loglikelihood(
    ::AnalyticExpectation, lik, q_f::AbstractVector{<:Normal}, y::AbstractVector
)
    return error(
        "No analytic solution exists for $(typeof(lik)). Use `DefaultExpectationMethod`, `GaussHermiteExpectation` or `MonteCarloExpectation` instead.",
    )
end

_maybe_ref(lik) = Ref(lik)
_maybe_ref(lik::AbstractArray) = lik
