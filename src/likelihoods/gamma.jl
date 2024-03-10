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

(l::GammaLikelihood)(fs::AbstractVector{<:Real}) = Product(map(l, fs))

function expected_loglikelihood(
    ::AnalyticExpectation,
    lik::GammaLikelihood{ExpLink},
    q_f::AbstractVector{<:Normal},
    y::AbstractVector{<:Real},
)
    return sum(_gamma_exp_loglikelihood_kernel.(lik.α, q_f, y))
end

function expected_loglikelihood(
    ::AnalyticExpectation,
    liks::AbstractVector{<:GammaLikelihood{ExpLink}},
    q_f::AbstractVector{<:Normal},
    y::AbstractVector{<:Real},
)
    return sum(_gamma_exp_loglikelihood_kernel.(getfield.(liks, :α), q_f, y))
end

function _gamma_exp_loglikelihood_kernel(α, q_f, y)
    return (α - 1) * log(y) - y * exp((var(q_f) / 2) - mean(q_f)) - α * mean(q_f) -
        loggamma(α)
end

default_expectation_method(::GammaLikelihood{ExpLink}) = AnalyticExpectation()
