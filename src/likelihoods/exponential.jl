"""
    ExponentialLikelihood(l=exp)

Exponential likelihood with scale given by `l(f)`.

```math
    p(y|f) = \\operatorname{Exponential}(y | l(f))
```
"""
struct ExponentialLikelihood{Tl<:AbstractLink} <: AbstractLikelihood
    invlink::Tl
end

ExponentialLikelihood(l=exp) = ExponentialLikelihood(link(l))

(l::ExponentialLikelihood)(f::Real) = Exponential(l.invlink(f))

(l::ExponentialLikelihood)(fs::AbstractVector{<:Real}) = Product(map(l, fs))

function expected_loglikelihood(
    ::AnalyticExpectation,
    ::ExponentialLikelihood{ExpLink},
    q_f::AbstractVector{<:Normal},
    y::AbstractVector{<:Real},
)
    return sum(_exp_exp_loglikelihood_kernel.(q_f, y))
end

function expected_loglikelihood(
    ::AnalyticExpectation,
    ::AbstractVector{<:ExponentialLikelihood{ExpLink}},
    q_f::AbstractVector{<:Normal},
    y::AbstractVector{<:Real},
)
    return sum(_exp_exp_loglikelihood_kernel.(q_f, y))
end

_exp_exp_loglikelihood_kernel(q_f, y) = -mean(q_f) - y * exp((var(q_f) / 2) - mean(q_f))

default_expectation_method(::ExponentialLikelihood{ExpLink}) = AnalyticExpectation()
