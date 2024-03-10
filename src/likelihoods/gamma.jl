"""
    GammaLikelihood(α::Real=1.0, l=exp)

Gamma likelihood with fixed shape `α`.

```math
    p(y|f) = \\operatorname{Gamma}(y | α, l(f))
```
On calling, this returns a Gamma distribution with shape `α` and scale `invlink(f)`.
"""
struct GammaLikelihood{Tl<:AbstractLink,T<:Real} <: AbstractLikelihood
    α::T    # shape parameter
    invlink::Tl
end

GammaLikelihood(l) = GammaLikelihood(1.0, l)
GammaLikelihood(α::Real=1.0, l=exp) = GammaLikelihood(α, link(l))

@functor GammaLikelihood

(l::GammaLikelihood)(f::Real) = Gamma(l.α, l.invlink(f))

(l::GammaLikelihood)(fs::AbstractVector{<:Real}) = product_distribution(map(l, fs))

function expected_loglikelihood(
    ::AnalyticExpectation,
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

default_expectation_method(::GammaLikelihood{ExpLink}) = AnalyticExpectation()
