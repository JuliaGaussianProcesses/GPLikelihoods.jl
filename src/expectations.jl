using FastGaussQuadrature: gausshermite
using SpecialFunctions: loggamma
using ChainRulesCore: ChainRulesCore
using IrrationalConstants: sqrt2, invsqrtπ

abstract type QuadratureMethod end
struct DefaultQuadrature <: QuadratureMethod end
struct Analytic <: QuadratureMethod end

struct GaussHermite <: QuadratureMethod
    n_points::Int
end
GaussHermite() = GaussHermite(20)

struct MonteCarlo <: QuadratureMethod
    n_samples::Int
end
MonteCarlo() = MonteCarlo(20)

_default_quadrature(_) = GaussHermite()

"""
    expected_loglikelihood(
        quadrature::QuadratureMethod,
        lik,
        q_f::AbstractVector{<:Normal},
        y::AbstractVector,
    )

This function computes the expected log likelihood:

```math
    ∫ q(f) log p(y | f) df
```
where `p(y | f)` is the process likelihood. This is described by `lik`, which should be a callable that takes `f` as input and returns a Distribution over `y` that supports `loglikelihood(lik(f), y)`.

`q(f)` is an approximation to the latent function values `f` given by:
```math
    q(f) = ∫ p(f | u) q(u) du
```
where `q(u)` is the variational distribution over inducing points (see
[`elbo`](@ref)). The marginal distributions of `q(f)` are given by `q_f`.

`quadrature` determines which method is used to calculate the expected log
likelihood - see [`elbo`](@ref) for more details.

# Extended help

`q(f)` is assumed to be an `MvNormal` distribution and `p(y | f)` is assumed to
have independent marginals such that only the marginals of `q(f)` are required.
"""
expected_loglikelihood(quadrature, lik, q_f, y)

"""
    expected_loglikelihood(::DefaultQuadrature, lik, q_f::AbstractVector{<:Normal}, y::AbstractVector)

The expected log likelihood.
Defaults to a closed form solution if it exists, otherwise defaults to
Gauss-Hermite quadrature.
"""
function expected_loglikelihood(
    ::DefaultQuadrature, lik, q_f::AbstractVector{<:Normal}, y::AbstractVector
)
    quadrature = _default_quadrature(lik)
    return expected_loglikelihood(quadrature, lik, q_f, y)
end

function expected_loglikelihood(
    mc::MonteCarlo, lik, q_f::AbstractVector{<:Normal}, y::AbstractVector
)
    # take `n_samples` reparameterised samples
    f_μ = mean.(q_f)
    fs = f_μ .+ std.(q_f) .* randn(eltype(f_μ), length(q_f), mc.n_samples)
    lls = loglikelihood.(lik.(fs), y)
    return sum(lls) / mc.n_samples
end

# Compute the expected_loglikelihood over a collection of observations and marginal distributions
function expected_loglikelihood(
    gh::GaussHermite, lik, q_f::AbstractVector{<:Normal}, y::AbstractVector
)
    # Compute the expectation via Gauss-Hermite quadrature
    # using a reparameterisation by change of variable
    # (see e.g. en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature)
    xs, ws = gausshermite(gh.n_points)
    return sum(
        Broadcast.instantiate(
            Broadcast.broadcasted(y, q_f) do yᵢ, q_fᵢ  # Loop over every pair
                # of marginal distribution q(fᵢ) and observation yᵢ
                expected_loglikelihood(gh, lik, q_fᵢ, yᵢ, (xs, ws))
            end,
        )
    )
end

# Compute the expected_loglikelihood for one observation and a marginal distributions
function expected_loglikelihood(
    gh::GaussHermite, lik, q_f::Normal, y, (xs, ws)=gausshermite(gh.n_points)
)
    μ = mean(q_f)
    σ̃ = sqrt2 * std(q_f)
    return invsqrtπ * sum(Broadcast.instantiate(
        Broadcast.broadcasted(xs, ws) do x, w # Loop over every
            # pair of Gauss-Hermite point x with weight w
            f = σ̃ * x + μ
            loglikelihood(lik(f), y) * w
        end,
    ))
end

ChainRulesCore.@non_differentiable gausshermite(n)

function expected_loglikelihood(
    ::Analytic, lik, q_f::AbstractVector{<:Normal}, y::AbstractVector
)
    return error(
        "No analytic solution exists for ",
        typeof(lik),
        ". Use `DefaultQuadrature()`, `GaussHermite()` or `MonteCarlo()` instead.",
    )
end

# The closed form solution for independent Gaussian noise
function expected_loglikelihood(
    ::Analytic,
    lik::GaussianLikelihood,
    q_f::AbstractVector{<:Normal},
    y::AbstractVector{<:Real},
)
    return sum(
        -0.5 * (log(2π) .+ log.(lik.σ²) .+ ((y .- mean.(q_f)) .^ 2 .+ var.(q_f)) / lik.σ²)
    )
end

_default_quadrature(::GaussianLikelihood) = Analytic()

# The closed form solution for a Poisson likelihood with an exponential inverse link function
function expected_loglikelihood(
    ::Analytic,
    ::PoissonLikelihood{ExpLink},
    q_f::AbstractVector{<:Normal},
    y::AbstractVector{<:Real},
)
    f_μ = mean.(q_f)
    return sum((y .* f_μ) - exp.(f_μ .+ (var.(q_f) / 2)) - loggamma.(y .+ 1))
end

_default_quadrature(::PoissonLikelihood{ExpLink}) = Analytic()

# The closed form solution for an Exponential likelihood with an exponential inverse link function
function expected_loglikelihood(
    ::Analytic,
    ::ExponentialLikelihood{ExpLink},
    q_f::AbstractVector{<:Normal},
    y::AbstractVector{<:Real},
)
    f_μ = mean.(q_f)
    return sum(-f_μ - y .* exp.((var.(q_f) / 2) .- f_μ))
end

_default_quadrature(::ExponentialLikelihood{ExpLink}) = Analytic()

# The closed form solution for a Gamma likelihood with an exponential inverse link function
function expected_loglikelihood(
    ::Analytic,
    lik::GammaLikelihood{ExpLink},
    q_f::AbstractVector{<:Normal},
    y::AbstractVector{<:Real},
)
    f_μ = mean.(q_f)
    return sum(
        (lik.α - 1) * log.(y) .- y .* exp.((var.(q_f) / 2) .- f_μ) .- lik.α * f_μ .-
        loggamma(lik.α),
    )
end

_default_quadrature(::GammaLikelihood{ExpLink}) = Analytic()
