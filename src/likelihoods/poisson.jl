"""
    PoissonLikelihood(l=exp)

Poisson likelihood with rate defined as `l(f)`.

```math
    p(y|f) = \\operatorname{Poisson}(y | θ=l(f))
```

This is to be used if  we assume that the uncertainity associated
with the data follows a Poisson distribution.
"""
struct PoissonLikelihood{L<:AbstractLink} <: AbstractLikelihood
    invlink::L
end

PoissonLikelihood(l=exp) = PoissonLikelihood(link(l))

(l::PoissonLikelihood)(f::Real) = Poisson(l.invlink(f))

(l::PoissonLikelihood)(fs::AbstractVector{<:Real}) = Product(Poisson.(l.invlink.(fs)))

function expected_loglikelihood(
    ::AnalyticExpectation,
    ::PoissonLikelihood{ExpLink},
    q_f::AbstractVector{<:Normal},
    y::AbstractVector{<:Real},
)
    f_μ = mean.(q_f)
    return sum((y .* f_μ) - exp.(f_μ .+ (var.(q_f) / 2)) - loggamma.(y .+ 1))
end

default_expectation_method(::PoissonLikelihood{ExpLink}) = AnalyticExpectation()
