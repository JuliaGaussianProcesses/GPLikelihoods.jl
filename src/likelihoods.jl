"""
    Likelihood

An abstract type for likelihoods. Likelihoods are used to model the uncertainities associated
with the data.
"""
abstract type Likelihood end

"""
    GaussianLikelihood(σ²)

Gaussian likelihood with `σ²` variance. This is to be used if we assume that the uncertainity 
associated with the data follows a Gaussian distribution.

```math
    p(y|f) = Normal(y | f, σ²)
```
On calling, this would return a normal distribution with mean `f` and variance σ².
"""
struct GaussianLikelihood{T<:Real} <: Likelihood
    σ²::Vector{T}
    function GaussianLikelihood(σ²::T) where {T<:Real}
        new{typeof(σ²)}([σ²])
    end
end

GaussianLikelihood() = GaussianLikelihood(1e-6)

(l::GaussianLikelihood)(f::Real) = Normal(f, first(l.σ²))

(l::GaussianLikelihood)(fs::AbstractVector{<:Real}) = Product([Normal(f, first(l.σ²)) for f in fs])

"""
    PoissonLikelihood()

Poisson likelihood with rate as exponential of samples from GP `f`. This is to be used if
we assume that the uncertainity associated with the data follows a Poisson distribution.
"""
struct PoissonLikelihood <: Likelihood end

(l::PoissonLikelihood)(f::Real) = Poisson(exp(f))

(l::PoissonLikelihood)(fs::AbstractVector{<:Real}) = Product(Poisson.(exp.(fs)))
