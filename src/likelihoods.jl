"""
    Likelihood

An abstract type for likelihoods. Likelihoods are used to model the uncertainities associated
with the data.
"""
abstract type Likelihood end

"""
    logpdf(::Likelihood, y, f)

Given log probability density.
```math
    logp(y|F)
```
"""
Distributions.logpdf(::Likelihood, y, f) = error("Not implemented")

"""
    GaussianLikelihood(σ²)

Gaussian likelihood with `σ²` variance. This is to be used if we assume that the uncertainity 
associated with the data follows a Gaussian distribution.

```math
    p(y|f) = Normal(y | f, σ²)
```
On calling, this would return a normal distribution with mean `f` and variance σ².
"""
struct GaussianLikelihood{T} <: Likelihood
    σ²::T
end

GaussianLikelihood() = GaussianLikelihood(1e-6)

(l::GaussianLikelihood)(f::Real) = Normal(f, l.σ²)

(l::GaussianLikelihood)(fs::AbstractVector{<:Real}) = Product([Normal(f, l.σ²) for f in fs])

function Distributions.logpdf(l::GaussianLikelihood, y::Real, f::Real)
    return Distributions.logpdf(l(f), y)
end

function Distributions.logpdf(
    l::GaussianLikelihood, 
    ys::AbstractVector{<:Real}, 
    fs::AbstractVector{<:Real}
    )
    return Distributions.logpdf(l(fs), ys)
end
