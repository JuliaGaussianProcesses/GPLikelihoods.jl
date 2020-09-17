"""
    GaussianLikelihood(σ²)

Gaussian likelihood with `σ²` variance. This is to be used if we assume that the 
uncertainity associated with the data follows a Gaussian distribution.

```math
    p(y|f) = Normal(y | f, σ²)
```
On calling, this would return a normal distribution with mean `f` and variance σ².
"""
struct GaussianLikelihood{T<:Real}
    σ²::T
end

GaussianLikelihood() = GaussianLikelihood(1e-6)

@functor GaussianLikelihood

(l::GaussianLikelihood)(f::Real) = Normal(f, sqrt(l.σ²))

(l::GaussianLikelihood)(fs::AbstractVector{<:Real}) = MvNormal(fs, sqrt(l.σ²))

"""
    HeteroscedasticGaussianLikelihood(σ²)

Heteroscedastic Gaussian likelihood. This is to be used if we assume that the 
uncertainity associated with the data follows a Gaussian distribution where the 
variance varies according to the input.

```math
    p(y|[f, σ]) = Normal(y | f, σ)
```
On calling, this would return a normal distribution with mean `f` and variance σ.
"""
struct HeteroscedasticGaussianLikelihood end

@functor HeteroscedasticGaussianLikelihood

(::HeteroscedasticGaussianLikelihood)(f::AbstractVector{<:Real}) = Normal(f[1], exp(f[2]))

(::HeteroscedasticGaussianLikelihood)(fs::AbstractVector) = MvNormal([f[1]  for f in fs], exp.([f[2]  for f in fs]))
